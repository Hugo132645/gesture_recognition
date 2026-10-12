import cv2
import numpy as np
import mediapipe as mp
import math
from collections import deque

# Config
USE_WORLD_LANDMARKS = True   
SMOOTH_ALPHA = 0.35           # EMA smoothing for angles (0=no smoothing, 1=very sluggish)
SHOW_BOTH_HANDS = False

# Servo mapping (will be overwritten by calibration if done)
SERVO_MIN, SERVO_MAX = 0.0, 180.0  # servo target range in degrees (experimental)
# Default per-finger linear maps: servo = gain * (angle_deg - offset)
calib = {
    "THUMB": {"gain": 2.0, "offset": 0.0},
    "INDEX": {"gain": 2.0, "offset": 0.0},
    "MIDDLE": {"gain": 2.0, "offset": 0.0},
    "RING": {"gain": 2.0, "offset": 0.0},
    "PINKY": {"gain": 2.0, "offset": 0.0},
}

import os, json

# Try to load previous calibration if it exists
if os.path.exists("servo_calib.txt"):
    try:
        with open("servo_calib.txt", "r") as f:
            data = json.load(f)
            calib = data.get("calib", calib)
        print("Loaded calibration from servo_calib.txt")
    except Exception as e:
        print("Could not load calibration:", e)

# Quick calibration storage (press 'o' with open hand, 'f' with closed fist, 'r' to reset, 'w' to write out)
open_pose = {}
fist_pose = {}

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# Helpers
FINGERS = [
    ("THUMB", 2, 3),   # (name, MCP idx, next joint idx)  thumb MCP=2, IP=3 | try 3,4 later 
    ("INDEX", 5, 6),   # MCP=5, PIP=6
    ("MIDDLE", 9, 10),
    ("RING", 13, 14),
    ("PINKY", 17, 18),
]

def to_np_point(lm, w, h, use_z=True):
    """Convert a landmark to numpy point; z is in image-depth units (negative toward camera)."""
    if use_z and hasattr(lm, 'z'):
        return np.array([lm.x, lm.y, lm.z], dtype=np.float32)
    else:
        return np.array([lm.x, lm.y, 0.0], dtype=np.float32)

def norm(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps: return v * 0.0
    return v / n

def angle_to_plane(vector, plane_normal):
    """
    Returns angle (deg) between vector and plane.
    If vector lies in the plane => ~0°, if aligned with normal => ~90°.
    """
    v = norm(vector)
    n = norm(plane_normal)
    # angle to normal
    cos_theta = np.clip(np.abs(np.dot(v, n)), 0.0, 1.0)
    theta = math.degrees(math.acos(cos_theta))
    # angle to plane
    return 90.0 - theta

def build_palm_frame(landmarks, use_world=True):
    """
    Palm plane and normal:
      Use wrist (0), index_mcp (5), pinky_mcp (17)
      n = (wrist->index) x (wrist->pinky)
    Returns origin, n (normal), and a palm X axis (roughly wrist->index direction projected onto plane).
    """
    w = landmarks[0]
    i = landmarks[5]
    p = landmarks[17]

    W = to_np_point(w, 1, 1, use_world)
    I = to_np_point(i, 1, 1, use_world)
    P = to_np_point(p, 1, 1, use_world)

    wi = I - W
    wp = P - W
    n = norm(np.cross(wi, wp))  # palm normal

    # X axis along wrist->index projected to plane
    x_axis = norm(wi - np.dot(wi, n) * n)
    # Y axis completes right-handed frame
    y_axis = norm(np.cross(n, x_axis))

    return W, x_axis, y_axis, n

def project_to_plane(v, n):
    return v - np.dot(v, n) * n

def ema(prev, new, alpha):
    if prev is None: return new
    return alpha * prev + (1 - alpha) * new

# Smoothers per finger + wrist
smooth_angles = {name: None for name, _, _ in FINGERS}
smooth_servo  = {name: None for name, _, _ in FINGERS}
smooth_wrist  = {"pitch": None, "roll": None}

def finger_flexion_deg(landmarks, name, mcp_idx, next_idx, use_world=True):
    """
    Finger flexion from proximal phalanx (MCP->PIP or MCP->IP for thumb) vs palm plane.
    """
    W, x_axis, y_axis, n = build_palm_frame(landmarks, use_world)
    mcp = landmarks[mcp_idx]
    nxt = landmarks[next_idx]
    MCP = to_np_point(mcp, 1, 1, use_world)
    NXT = to_np_point(nxt, 1, 1, use_world)

    proximal = NXT - MCP
    # If you want to ignore ab/adduction, we already take angle to the PALM PLANE (not x/y axes),
    # which is robust to side/back views.
    flex = angle_to_plane(proximal, n)
    # Clamp to [0, 90] as a sane range (open≈0, curled≈90)
    return float(np.clip(flex, 0.0, 90.0))

def wrist_pitch_roll(landmarks, use_world=True):
    """
    Approx wrist orientation proxy: pitch/roll of the PALM NORMAL relative to camera axes.
    pitch ~ tilt up/down; roll ~ rotate around camera Z. This is not anatomical flex/ulnar dev,
    but a useful on-screen indicator.
    """
    _, _, _, n = build_palm_frame(landmarks, use_world)
    # Camera frame: x→right, y→down, z→into camera (MP z is negative toward camera).
    # Define pitch (tilt toward/away): arcsin of -n_y (up is negative y)
    pitch = math.degrees(math.asin(np.clip(-n[1], -1.0, 1.0)))
    # Define roll from x/z components
    roll = math.degrees(math.atan2(n[0], -n[2]))  # -z so that facing camera ~0
    return pitch, roll

def servo_map(name, angle_deg):
    """Linear map via calibration: servo = clamp(gain*(angle-offset))."""
    g = calib[name]["gain"]
    off = calib[name]["offset"]
    val = g * (angle_deg - off)
    return float(np.clip(val, SERVO_MIN, SERVO_MAX))

def capture_pose_snapshot(landmarks, use_world=True):
    snapshot = {}
    for (nm, mcp, nxt) in FINGERS:
        snapshot[nm] = finger_flexion_deg(landmarks, nm, mcp, nxt, use_world)
    return snapshot

def update_calibration(open_pose, fist_pose):
    """
    Make a simple linear map so that:
      angle_open -> SERVO_MIN
      angle_fist -> SERVO_MAX
    for each finger independently.
    """
    for nm in open_pose.keys():
        a0 = open_pose[nm]
        a1 = fist_pose[nm]
        if abs(a1 - a0) < 1e-3:
            # avoid div-by-zero; keep previous
            continue
        gain = (SERVO_MAX - SERVO_MIN) / (a1 - a0)
        offset = a0  # so that at a0, value=SERVO_MIN
        calib[nm]["gain"] = gain
        calib[nm]["offset"] = offset

# Main loop

print("Controls:")
print("  o = capture OPEN pose (hand flat)")
print("  f = capture FIST pose (hand closed)")
print("  r = reset calibration to defaults")
print("  w = write current calibration to 'servo_calib.txt'")
print("  q/ESC = quit")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Failed to read camera.")
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    overlay = frame.copy()
    if res.multi_hand_landmarks:
        # process either only the first hand or both
        hand_indices = range(len(res.multi_hand_landmarks)) if SHOW_BOTH_HANDS else [0]
        for hi in hand_indices:
            hand_landmarks = res.multi_hand_landmarks[hi]
            lms = hand_landmarks.landmark

            # Draw skeleton
            mp_draw.draw_landmarks(
                overlay, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            # Compute wrist orientation (pitch/roll)
            try:
                pitch, roll = wrist_pitch_roll(lms, use_world=USE_WORLD_LANDMARKS)
                smooth_wrist["pitch"] = ema(smooth_wrist["pitch"], pitch, SMOOTH_ALPHA)
                smooth_wrist["roll"]  = ema(smooth_wrist["roll"],  roll,  SMOOTH_ALPHA)
                cv2.putText(overlay, f"Wrist pitch: {smooth_wrist['pitch']:.1f} deg",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
                cv2.putText(overlay, f"Wrist pitch: {smooth_wrist['pitch']:.1f} deg",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
                cv2.putText(overlay, f"Wrist roll:  {smooth_wrist['roll']:.1f} deg",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
                cv2.putText(overlay, f"Wrist roll:  {smooth_wrist['roll']:.1f} deg",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
            except Exception:
                pass

            # Palm MCP points for labeling positions
            mcp_points_2d = {}
            idxs_to_draw = [2,5,9,13,17]
            for idx in idxs_to_draw:
                lm = lms[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                mcp_points_2d[idx] = (cx, cy)

            # Finger flexion and servo mapping
            for (nm, mcp_idx, nxt_idx) in FINGERS:
                try:
                    ang = finger_flexion_deg(lms, nm, mcp_idx, nxt_idx, use_world=USE_WORLD_LANDMARKS)
                except Exception:
                    ang = np.nan

                if not np.isnan(ang):
                    smooth_angles[nm] = ema(smooth_angles[nm], ang, SMOOTH_ALPHA)
                    servo = servo_map(nm, smooth_angles[nm])
                    smooth_servo[nm] = ema(smooth_servo[nm], servo, SMOOTH_ALPHA)

                    # Label near MCP
                    pos = mcp_points_2d.get(mcp_idx, (20, 100))
                    x, y = pos
                    text1 = f"{nm[:1]}: {smooth_angles[nm]:.1f} deg"
                    text2 = f"-> servo {smooth_servo[nm]:.0f} deg"
                    cv2.putText(overlay, text1, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
                    cv2.putText(overlay, text1, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
                    cv2.putText(overlay, text2, (x-10, y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3)
                    cv2.putText(overlay, text2, (x-10, y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    # UI footer
    cv2.putText(overlay, "o:Open  f:Fist  r:Reset  w:Write  q:Quit",
                (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
    cv2.putText(overlay, "o:Open  f:Fist  r:Reset  w:Write  q:Quit",
                (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    cv2.imshow("Hand → Servo (Angles & Mapping)", overlay)
    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):  # ESC or q
        break
    elif key == ord('o'):
        if res.multi_hand_landmarks:
            lms = res.multi_hand_landmarks[0].landmark
            open_pose = capture_pose_snapshot(lms, USE_WORLD_LANDMARKS)
            print("Captured OPEN pose:", open_pose)
    elif key == ord('f'):
        if res.multi_hand_landmarks:
            lms = res.multi_hand_landmarks[0].landmark
            fist_pose = capture_pose_snapshot(lms, USE_WORLD_LANDMARKS)
            print("Captured FIST pose:", fist_pose)
            if open_pose and fist_pose:
                update_calibration(open_pose, fist_pose)
                print("Calibration updated:", calib)
    elif key == ord('r'):
        # reset to defaults
        for nm in calib:
            calib[nm]["gain"] = 2.0
            calib[nm]["offset"] = 0.0
        open_pose = {}
        fist_pose = {}
        print("Calibration reset.")
    elif key == ord('w'):
        import json
        with open("servo_calib.txt", "w") as f:
            f.write(json.dumps({"calib": calib, "open": open_pose, "fist": fist_pose}, indent=2))
        print("Wrote calibration to servo_calib.txt")

cap.release()
cv2.destroyAllWindows()
