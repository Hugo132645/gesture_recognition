import cv2
import numpy as np
import mediapipe as mp
import math
from collections import deque
import os, json

# ===== Config =====
USE_WORLD_LANDMARKS = True
SMOOTH_ALPHA = 0.35          # EMA smoothing (fallback if One-Euro off)
SHOW_BOTH_HANDS = False
USE_ONE_EURO = True
ONE_EURO = dict(freq=30.0, min_cutoff=1.0, beta=0.04, dcutoff=1.0)

# PnP (optional if you have camera intrinsics)
USE_PNP = False
CAMERA_MATRIX = None
DIST_COEFFS  = None

# Servo mapping (overwritten by calibration)
SERVO_MIN, SERVO_MAX = 0.0, 180.0
calib = {
    "THUMB":  {"gain": 2.0, "offset": 0.0},
    "INDEX":  {"gain": 2.0, "offset": 0.0},
    "MIDDLE": {"gain": 2.0, "offset": 0.0},
    "RING":   {"gain": 2.0, "offset": 0.0},
    "PINKY":  {"gain": 2.0, "offset": 0.0},
}

# Load previous calibration if present
if os.path.exists("servo_calib.txt"):
    try:
        with open("servo_calib.txt", "r") as f:
            data = json.load(f)
            calib = data.get("calib", calib)
        print("Loaded servo_calib.txt")
    except Exception as e:
        print("Calibration load error:", e)

open_pose = {}
fist_pose = {}

# ===== MediaPipe =====
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

# ===== Helpers =====
FINGERS = [
    ("THUMB", 2, 3),
    ("INDEX", 5, 6),
    ("MIDDLE", 9, 10),
    ("RING", 13, 14),
    ("PINKY", 17, 18),
]

def to_np_point(lm, w, h, use_z=True):
    """MediaPipe landmark -> np.array([x,y,z])."""
    x = float(lm.x); y = float(lm.y)
    z = float(getattr(lm, 'z', 0.0)) if use_z else 0.0
    return np.array([x, y, z], dtype=np.float32)

def norm(v, eps=1e-8):
    n = np.linalg.norm(v)
    return v * 0.0 if n < eps else v / n

def angle_to_plane(vector, plane_normal):
    """Angle (deg) between vector and plane (0=in-plane, 90=along normal)."""
    v = norm(vector)
    n = norm(plane_normal)
    cos_theta = np.clip(abs(np.dot(v, n)), 0.0, 1.0)
    theta = math.degrees(math.acos(cos_theta))
    return 90.0 - theta

def build_palm_frame(landmarks, use_world=True):
    """Palm normal from (wrist->index) x (wrist->pinky)."""
    Wp = to_np_point(landmarks[0], 1, 1, use_world)
    Ip = to_np_point(landmarks[5], 1, 1, use_world)
    Pp = to_np_point(landmarks[17], 1, 1, use_world)
    wi = Ip - Wp
    wp = Pp - Wp
    n = norm(np.cross(wi, wp))
    x_axis = norm(wi - np.dot(wi, n) * n)   # project to plane
    y_axis = norm(np.cross(n, x_axis))
    return Wp, x_axis, y_axis, n

def project_to_plane(v, n):
    return v - np.dot(v, n) * n

def ema(prev, new, alpha):
    if prev is None: return new
    return alpha * prev + (1 - alpha) * new

# ---- Robust axes + YPR helpers ----
def pca_axis(points2d):
    pts = np.asarray(points2d, dtype=np.float32)
    mu = pts.mean(axis=0, keepdims=True)
    X = pts - mu
    cov = (X.T @ X) / max(len(pts)-1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    return axis / (np.linalg.norm(axis) + 1e-9), mu.ravel()

def rotation_from_axes(x_hat, y_hat):
    x = x_hat / (np.linalg.norm(x_hat) + 1e-9)
    y = y_hat - x * (x @ y_hat)
    y = y / (np.linalg.norm(y) + 1e-9)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)  # columns are axes

def rmat_to_ypr(R):
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        yaw   = math.degrees(math.atan2(R[1,0], R[0,0]))
        pitch = math.degrees(math.atan2(-R[2,0], sy))
        roll  = math.degrees(math.atan2(R[2,1], R[2,2]))
    else:
        yaw   = math.degrees(math.atan2(-R[0,1], R[1,1]))
        pitch = math.degrees(math.atan2(-R[2,0], sy))
        roll  = 0.0
    return np.array([yaw, pitch, roll], dtype=np.float32)

class OneEuro:
    """Low-lag filter."""
    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.04, dcutoff=1.0):
        self.freq=freq; self.min_cutoff=min_cutoff; self.beta=beta; self.dcutoff=dcutoff
        self.x_prev=None; self.dx_prev=None
    def _alpha(self, cutoff):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / max(self.freq, 1e-6)
        return 1.0 / (1.0 + tau/te)
    def __call__(self, x):
        x = np.asarray(x, dtype=np.float32)
        if self.x_prev is None:
            self.x_prev = x; self.dx_prev = np.zeros_like(x); return x
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.dcutoff)
        dx_hat = a_d*dx + (1-a_d)*self.dx_prev
        cutoff = self.min_cutoff + self.beta*np.abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a*x + (1-a)*self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat

def wrist_yaw_pitch_roll(landmarks, use_world=True):
    """Yaw/Pitch/Roll from MCP axis (X) and wrist->middle (Y)."""
    pts = np.array([to_np_point(lm, 1, 1, use_world) for lm in landmarks], dtype=np.float32)
    mcp2d = pts[[5,9,13,17], :2]
    x2d, _ = pca_axis(mcp2d)
    x3 = np.array([x2d[0], x2d[1], 0.0], dtype=np.float32)
    y3 = pts[9] - pts[0]
    if not use_world or abs(y3[2]) < 1e-6:
        y3[2] = 0.0
    R = rotation_from_axes(x3, y3)
    return rmat_to_ypr(R)

# Smoothers
smooth_angles = {name: None for name, _, _ in FINGERS}
smooth_servo  = {name: None for name, _, _ in FINGERS}
smooth_wrist  = {"pitch": None, "roll": None}
one_euro = OneEuro(**ONE_EURO) if USE_ONE_EURO else None
handedness_lock = {'label': None, 'streak': 0}

# Fallback yaw (when landmarks drop)
bgsub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=False)
def fallback_inplane_yaw(frame_bgr):
    mask = bgsub.apply(frame_bgr)
    mask = cv2.medianBlur(mask, 5)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 20: return None
    el = cv2.fitEllipse(c)
    ang = float(el[2])
    if ang > 90: ang -= 180
    return ang

def finger_flexion_deg(landmarks, name, mcp_idx, nxt_idx, use_world=True):
    """Finger flexion = angle between proximal segment and palm plane."""
    _, _, _, n = build_palm_frame(landmarks, use_world)
    MCP = to_np_point(landmarks[mcp_idx], 1, 1, use_world)
    NXT = to_np_point(landmarks[nxt_idx], 1, 1, use_world)
    proximal = NXT - MCP
    flex = angle_to_plane(proximal, n)
    return float(np.clip(flex, 0.0, 90.0))

def wrist_pitch_roll(landmarks, use_world=True):
    """Legacy: pitch/roll from palm normal."""
    _, _, _, n = build_palm_frame(landmarks, use_world)
    pitch = math.degrees(math.asin(np.clip(-n[1], -1.0, 1.0)))
    roll = math.degrees(math.atan2(n[0], -n[2]))
    return pitch, roll

def servo_map(name, angle_deg):
    """servo = gain * (angle - offset), clamped."""
    g = calib[name]["gain"]
    off = calib[name]["offset"]
    val = g * (angle_deg - off)
    return float(np.clip(val, SERVO_MIN, SERVO_MAX))

def capture_pose_snapshot(landmarks, use_world=True):
    snap = {}
    for (nm, mcp, nxt) in FINGERS:
        snap[nm] = finger_flexion_deg(landmarks, nm, mcp, nxt, use_world)
    return snap

def update_calibration(open_pose, fist_pose):
    """Map open->SERVO_MIN, fist->SERVO_MAX per finger."""
    for nm in open_pose.keys():
        a0 = open_pose[nm]; a1 = fist_pose[nm]
        if abs(a1 - a0) < 1e-3: continue
        gain = (SERVO_MAX - SERVO_MIN) / (a1 - a0)
        calib[nm]["gain"] = gain
        calib[nm]["offset"] = a0

# ===== UI =====
print("Keys: o=open  f=fist  r=reset  w=write  q=quit  (focus the window)")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Camera read failed.")
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    overlay = frame.copy()
    if res.multi_hand_landmarks:
        hand_indices = range(len(res.multi_hand_landmarks)) if SHOW_BOTH_HANDS else [0]
        for hi in hand_indices:
            # image landmarks (for drawing)
            hand_img = res.multi_hand_landmarks[hi]
            lms_img = hand_img.landmark
            # world landmarks (for math) if available
            lms_world_list = getattr(res, 'multi_hand_world_landmarks', None)
            lms_math = lms_world_list[hi].landmark if (USE_WORLD_LANDMARKS and lms_world_list) else lms_img

            # draw
            mp_draw.draw_landmarks(
                overlay, hand_img, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            # yaw/pitch/roll (robust for side/back)
            try:
                ypr = wrist_yaw_pitch_roll(lms_math, use_world=USE_WORLD_LANDMARKS)
                ypr = one_euro(ypr) if one_euro else ypr
                cv2.putText(overlay, f"Yaw:   {ypr[0]:.1f} deg", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
                cv2.putText(overlay, f"Yaw:   {ypr[0]:.1f} deg", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
                cv2.putText(overlay, f"Pitch: {ypr[1]:.1f} deg", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
                cv2.putText(overlay, f"Pitch: {ypr[1]:.1f} deg", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
                cv2.putText(overlay, f"Roll:  {ypr[2]:.1f} deg", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
                cv2.putText(overlay, f"Roll:  {ypr[2]:.1f} deg", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
            except Exception:
                pass

            # label anchor points in image coords
            mcp_points_2d = {}
            for idx in [2,5,9,13,17]:
                lm = lms_img[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                mcp_points_2d[idx] = (cx, cy)

            # finger flexion + servo map (safe formatting)
            for (nm, mcp_idx, nxt_idx) in FINGERS:
                try:
                    ang = finger_flexion_deg(lms_math, nm, mcp_idx, nxt_idx, use_world=USE_WORLD_LANDMARKS)
                except Exception:
                    ang = np.nan

                if np.isfinite(ang):
                    smooth_angles[nm] = ema(smooth_angles[nm], ang, SMOOTH_ALPHA)
                    servo = servo_map(nm, smooth_angles[nm])
                    smooth_servo[nm] = ema(smooth_servo[nm], servo, SMOOTH_ALPHA)

                    pos = mcp_points_2d.get(mcp_idx, (20, 100))
                    x, y = pos
                    cv2.putText(overlay, f"{nm[0]}: {smooth_angles[nm]:.1f} deg",
                                (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
                    cv2.putText(overlay, f"{nm[0]}: {smooth_angles[nm]:.1f} deg",
                                (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
                    cv2.putText(overlay, f"-> servo {smooth_servo[nm]:.0f}",
                                (x-10, y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3)
                    cv2.putText(overlay, f"-> servo {smooth_servo[nm]:.0f}",
                                (x-10, y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

            # handedness lock (reduce L/R flapping)
            if hasattr(res, 'multi_handedness'):
                try:
                    label = res.multi_handedness[hi].classification[0].label  # 'Left'/'Right'
                    if handedness_lock['label'] == label:
                        handedness_lock['streak'] = min(handedness_lock['streak']+1, 1000)
                    elif handedness_lock['streak'] < 3:
                        handedness_lock['label'] = label; handedness_lock['streak'] += 1
                    lbl = handedness_lock['label'] or label
                    cv2.putText(overlay, f"Hand: {lbl}", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
                    cv2.putText(overlay, f"Hand: {lbl}", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                except Exception:
                    pass
    else:
        # fallback yaw if no landmarks this frame
        fb_yaw = fallback_inplane_yaw(frame)
        if fb_yaw is not None:
            if USE_ONE_EURO:
                fb = one_euro(np.array([fb_yaw, 0.0, 0.0], dtype=np.float32))
                fb_yaw = float(fb[0])
            cv2.putText(overlay, f"Fallback yaw: {fb_yaw:.1f} deg",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
            cv2.putText(overlay, f"Fallback yaw: {fb_yaw:.1f} deg",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

    # footer
    cv2.putText(overlay, "o:Open  f:Fist  r:Reset  w:Write  q:Quit",
                (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
    cv2.putText(overlay, "o:Open  f:Fist  r:Reset  w:Write  q:Quit",
                (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    cv2.imshow("Hand → Servo (Angles & Mapping)", overlay)
    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):
        break
    elif key == ord('o'):
        if res.multi_hand_landmarks:
            lms = (res.multi_hand_world_landmarks[0].landmark
                   if (USE_WORLD_LANDMARKS and getattr(res, 'multi_hand_world_landmarks', None))
                   else res.multi_hand_landmarks[0].landmark)
            open_pose = capture_pose_snapshot(lms, USE_WORLD_LANDMARKS)
            print("Open pose captured.")
    elif key == ord('f'):
        if res.multi_hand_landmarks:
            lms = (res.multi_hand_world_landmarks[0].landmark
                   if (USE_WORLD_LANDMARKS and getattr(res, 'multi_hand_world_landmarks', None))
                   else res.multi_hand_landmarks[0].landmark)
            fist_pose = capture_pose_snapshot(lms, USE_WORLD_LANDMARKS)
            print("Fist pose captured.")
            if open_pose and fist_pose:
                update_calibration(open_pose, fist_pose)
                print("Calibration updated.")
    elif key == ord('r'):
        for nm in calib:
            calib[nm]["gain"] = 2.0
            calib[nm]["offset"] = 0.0
        open_pose = {}
        fist_pose = {}
        print("Calibration reset.")
    elif key == ord('w'):
        with open("servo_calib.txt", "w") as f:
            f.write(json.dumps({"calib": calib, "open": open_pose, "fist": fist_pose}, indent=2))
        print("Saved servo_calib.txt")

cap.release()
cv2.destroyAllWindows()
