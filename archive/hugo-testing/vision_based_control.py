import argparse
import time
import math
import json
import os
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    import serial
    from serial import SerialException
except ImportError:
    serial = None

    class SerialException(Exception):
        pass


FINGERS = [
    ("THUMB", 1, 3),     # CMC → MCP (unchanged)
    ("INDEX", 5, 7),     # PIP → DIP (improved responsiveness)
    ("MIDDLE", 9, 11),  # PIP → DIP (improved responsiveness)
    ("RING", 13, 15),    # PIP → DIP (improved responsiveness)
    ("PINKY", 17, 19),   # PIP → DIP (improved responsiveness)
]

SERVO_MIN = 0.0
SERVO_MAX = 180.0
CALIB_FILE = "servo_calib.txt"

THUMB_OPEN, THUMB_CLOSE = 0.0, 100.0
INDEX_OPEN, INDEX_CLOSE = 0.0, 110.0
MIDDLE_OPEN, MIDDLE_CLOSE = 0.0, 90.0
RING_OPEN, RING_CLOSE = 0.0, 110.0 
PINKY_OPEN, PINKY_CLOSE = 0.0, 100.0

WRIST_NEUT = 135.0
WRIST_LEFT = 180.0
WRIST_RIGHT = 0.0

FINGER_SERVO_RANGES = {
    "THUMB": {"servo_open": THUMB_OPEN, "servo_close": THUMB_CLOSE},
    "INDEX": {"servo_open": INDEX_OPEN, "servo_close": INDEX_CLOSE},
    "MIDDLE": {"servo_open": MIDDLE_OPEN, "servo_close": MIDDLE_CLOSE},
    "RING": {"servo_open": RING_OPEN, "servo_close": RING_CLOSE},
    "PINKY": {"servo_open": PINKY_OPEN, "servo_close": PINKY_CLOSE},
}

mp_hands_module = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
HAND_CONNECTIONS = mp_hands_module.HAND_CONNECTIONS


def to_np_point(lm, use_z=True):
    x = float(lm.x)
    y = float(lm.y)
    if use_z and hasattr(lm, "z"):
        z = float(lm.z)
    else:
        z = 0.0
    return np.array([x, y, z], dtype=np.float32)


def norm_vec(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n


def build_palm_normal(landmarks, use_world=True):
    """
    IMPROVED: Calculate palm normal using best-fit plane through 4 finger MCP joints.
    More stable than 3-point method - individual finger movements have less impact.
    Uses PCA to find the plane that best fits all metacarpal joints.
    Excludes thumb to prevent thumb movements from affecting palm calculation.
    """
    # Get the 4 finger MCP joint positions (excludes thumb)
    mcp_indices = [0, 5, 9, 13, 17]  # Index, Middle, Ring, Pinky MCPs only
    points = np.array([to_np_point(landmarks[i], use_world) for i in mcp_indices])
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    cov_matrix = np.dot(centered.T, centered) / len(points)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    normal = eigenvectors[:, 0]
    wrist = to_np_point(landmarks[0], use_world)
    to_wrist = wrist - centroid
    if np.dot(normal, to_wrist) > 0:
        normal = -normal
    return norm_vec(normal)


def build_offset_palm_normal(landmarks, offset_cm=2.2, blend_factor=0.9, tilt_factor=-0.2, use_world=True):
    """
    Create an offset palm plane specifically for thumb tracking.
    
    IMPROVED: Two-stage optimization for thumb tracking:
    1. POSITION: Offset centroid toward thumb base (blend_factor)
    2. ORIENTATION: Tilt plane normal toward thumb rotation axis (tilt_factor)
    
    This creates maximum angular separation as the thumb moves.
    
    Args:
        landmarks: Hand landmarks
        offset_cm: Distance to offset the palm plane in centimeters (default 2.2cm)
        blend_factor: How much to orient position toward thumb (0.0=pure normal, 1.0=toward thumb, default 0.9)
        tilt_factor: How much to tilt plane normal toward thumb axis (0.0=no tilt, 1.0=fully tilted, default 0.0)
        use_world: Whether to use world coordinates
    
    Returns:
        Tuple of (offset_normal, offset_centroid)
        - offset_normal: Normal vector (potentially tilted for better thumb tracking)
        - offset_centroid: Centroid positioned toward thumb base for better tracking
    """
    # Get the 4 finger MCP joint positions (excludes thumb)
    mcp_indices = [0, 5, 9, 13, 17]  # Wrist, Index, Middle, Ring, Pinky MCPs
    points = np.array([to_np_point(landmarks[i], use_world) for i in mcp_indices])
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    cov_matrix = np.dot(centered.T, centered) / len(points)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    normal = eigenvectors[:, 0]
    
    # Ensure normal points away from wrist
    wrist = to_np_point(landmarks[0], use_world)
    to_wrist = wrist - centroid
    if np.dot(normal, to_wrist) > 0:
        normal = -normal
    
    # Normalize the normal vector
    normal = norm_vec(normal)
    
    # Get thumb joints for analyzing rotation axis
    thumb_cmc = to_np_point(landmarks[2], use_world)  # Thumb CMC joint (base)
    thumb_mcp = to_np_point(landmarks[3], use_world)  # Thumb MCP joint
    
    # OPTION 2: TILT THE PLANE NORMAL (new feature!)
    # Calculate thumb's primary direction (its rotation axis)
    thumb_direction = thumb_mcp - thumb_cmc
    thumb_direction_norm = norm_vec(thumb_direction)
    
    # Tilt the normal toward being perpendicular to thumb direction
    # This makes the plane more aligned with thumb's rotation
    if tilt_factor > 0.0:
        # Project thumb direction onto the plane defined by current normal
        thumb_on_plane = thumb_direction_norm - np.dot(thumb_direction_norm, normal) * normal
        thumb_on_plane_norm = norm_vec(thumb_on_plane)
        
        # Rotate normal toward thumb's perpendicular axis
        # This makes the plane "face" the thumb's movement more directly
        tilted_normal = norm_vec(
            (1.0 - tilt_factor) * normal + tilt_factor * thumb_on_plane_norm
        )
        normal = tilted_normal
    
    # OPTION 1: POSITION THE CENTROID (existing feature)
    # Direction from centroid toward thumb base
    to_thumb_base = thumb_cmc - centroid
    to_thumb_base_norm = norm_vec(to_thumb_base)
    
    # Blend between palm normal and direction to thumb
    offset_direction = norm_vec(
        (1.0 - blend_factor) * normal + blend_factor * to_thumb_base_norm
    )
    
    # Convert offset from cm to MediaPipe units
    offset_meters = offset_cm / 100.0 if use_world else offset_cm * 0.01
    
    # Create offset centroid along the blended direction
    offset_centroid = centroid + (offset_direction * offset_meters)
    
    return normal, offset_centroid


def angle_to_plane(v, plane_normal):
    v = np.asarray(v, dtype=np.float32)
    n = norm_vec(plane_normal)
    v_norm = norm_vec(v)
    cos_theta = np.clip(np.dot(v_norm, n), -1.0, 1.0)
    theta = math.degrees(math.acos(cos_theta))
    return max(0.0, min(90.0, 90.0 - theta))


def angle_to_offset_plane(point_a, point_b, plane_normal, plane_point):
    """
    Calculate the angle between a vector (from point_a to point_b) and an offset plane.
    
    This differs from angle_to_plane by accounting for where the plane is positioned in space,
    not just its orientation. This is crucial for thumb tracking where we use an offset palm plane.
    
    Args:
        point_a: Starting point of the vector (e.g., MCP joint)
        point_b: Ending point of the vector (e.g., next joint)
        plane_normal: Normal vector of the plane
        plane_point: A point on the plane (e.g., offset centroid)
    
    Returns:
        Angle in degrees (0-90)
    """
    point_a = np.asarray(point_a, dtype=np.float32)
    point_b = np.asarray(point_b, dtype=np.float32)
    plane_point = np.asarray(plane_point, dtype=np.float32)
    
    # Vector from point_a to point_b
    v = point_b - point_a
    v_norm = norm_vec(v)
    
    # Normalize plane normal
    n = norm_vec(plane_normal)
    
    # Calculate angle between vector and plane normal
    cos_theta = np.clip(np.dot(v_norm, n), -1.0, 1.0)
    theta = math.degrees(math.acos(cos_theta))
    
    # Convert from angle to normal to angle to plane
    # (90° - angle_to_normal = angle_to_plane)
    return max(0.0, min(90.0, 90.0 - theta))


def finger_flexion_deg(landmarks, mcp_idx, next_idx, use_world=True):
    n = build_palm_normal(landmarks, use_world)
    MCP = to_np_point(landmarks[mcp_idx], use_world)
    NXT = to_np_point(landmarks[next_idx], use_world)
    proximal = NXT - MCP
    return angle_to_plane(proximal, n)


def finger_flexion_deg_offset(landmarks, mcp_idx, next_idx, offset_cm=3.0, blend_factor=0.7, tilt_factor=-0.2, use_world=True):
    """
    Calculate finger flexion using an offset palm plane.
    
    This is specifically designed for thumb tracking where the thumb rotates
    in a different plane than the other fingers. By using an offset palm plane,
    we can more accurately track the thumb's angle changes.
    
    Args:
        landmarks: Hand landmarks
        mcp_idx: Index of the MCP joint
        next_idx: Index of the next joint (e.g., PIP or DIP)
        offset_cm: Distance to offset the palm plane in centimeters (default 3.0cm)
        blend_factor: How much to orient toward thumb (0.0=pure normal, 1.0=toward thumb, default 0.7)
        tilt_factor: How much to tilt plane normal (0.0=no tilt, 1.0=perpendicular to thumb, default 0.0)
        use_world: Whether to use world coordinates
    
    Returns:
        Angle in degrees (0-90)
    """
    # Get the offset palm plane (normal and centroid)
    plane_normal, plane_centroid = build_offset_palm_normal(landmarks, offset_cm, blend_factor, tilt_factor, use_world)
    
    # Get joint positions
    MCP = to_np_point(landmarks[mcp_idx], use_world)
    NXT = to_np_point(landmarks[next_idx], use_world)
    
    # Calculate angle using offset plane
    return angle_to_offset_plane(MCP, NXT, plane_normal, plane_centroid)


def wrist_pitch_roll(landmarks, use_world=True):
    n = build_palm_normal(landmarks, use_world)
    pitch = math.degrees(math.asin(np.clip(-n[1], -1.0, 1.0)))
    roll = math.degrees(math.atan2(n[0], -n[2]))
    return pitch, roll


def wrist_roll_to_servo(roll_deg):
    r = max(-45.0, min(45.0, roll_deg))
    t = (r + 45.0) / 90.0
    val = WRIST_RIGHT + t * (WRIST_LEFT - WRIST_RIGHT)
    return float(np.clip(val, SERVO_MIN, SERVO_MAX))


def default_calibration():
    calib = {}
    for name in [f[0] for f in FINGERS]:
        ranges = FINGER_SERVO_RANGES[name]
        calib[name] = {
            "open_angle": 0.0,
            "fist_angle": 90.0,  # IMPROVED: All fingers use improved palm normal
            "servo_open": ranges["servo_open"],
            "servo_close": ranges["servo_close"],
        }
    return calib


def load_calibration():
    calib = default_calibration()
    open_pose = {}
    fist_pose = {}
    if os.path.exists(CALIB_FILE):
        try:
            with open(CALIB_FILE, "r") as f:
                data = json.load(f)
            saved_calib = data.get("calib", {})
            for name in calib.keys():
                if name in saved_calib and isinstance(saved_calib[name], dict):
                    calib[name].update(saved_calib[name])
            open_pose = data.get("open", {})
            fist_pose = data.get("fist", {})
            print("Loaded calibration from", CALIB_FILE)
        except Exception as e:
            print("Could not load calibration:", e)
    return calib, open_pose, fist_pose


def save_calibration(calib, open_pose, fist_pose):
    data = {"calib": calib, "open": open_pose, "fist": fist_pose}
    with open(CALIB_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print("Saved calibration to", CALIB_FILE)


def update_calibration(calib, open_pose, fist_pose):
    for name, _, _ in FINGERS:
        a_open = open_pose.get(name)
        a_fist = fist_pose.get(name)
        if a_open is None or a_fist is None:
            continue
        calib[name]["open_angle"] = float(a_open)
        calib[name]["fist_angle"] = float(a_fist)
    print("Updated calibration from open/fist poses")


def servo_map(calib, name, angle_deg):
    conf = calib[name]
    a_open = conf["open_angle"]
    a_fist = conf["fist_angle"]
    s_open = conf["servo_open"]
    s_close = conf["servo_close"]

    if abs(a_fist - a_open) < 1e-3:
        t = 0.0
    else:
        t = (angle_deg - a_open) / (a_fist - a_open)

    t = max(0.0, min(1.0, t))
    val = s_open + t * (s_close - s_open)
    return float(np.clip(val, SERVO_MIN, SERVO_MAX))


def nflex(angle_deg):
    if angle_deg is None:
        return None
    return float(angle_deg) / 90.0


def draw_progress_bar(percent, width=10):
    """Draw ASCII progress bar: [=====     ] 50%"""
    percent = max(0.0, min(100.0, percent))
    filled = int((percent / 100.0) * width)
    bar = "=" * filled + " " * (width - filled)
    return f"[{bar}] {percent:3.0f}%"


def qdeg(x, q, lo=SERVO_MIN, hi=SERVO_MAX):
    if x is None:
        return None
    y = round(float(x) / q) * q
    return float(min(max(y, lo), hi))


def open_serial(port, baud, dry_run):
    if dry_run:
        print("Dry run: serial disabled")
        return None
    if serial is None:
        print("pyserial not installed, running without serial")
        return None
    try:
        ser = serial.Serial(port, baudrate=baud, timeout=0.01)
        time.sleep(2.0)
        print(f"Connected to {port} at {baud}")
        return ser
    except SerialException as e:
        print(f"Could not open serial port {port}: {e}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Hand tracker to robotic arm")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--quantize-threshold", type=float, default=4.0) # Quantizer
    parser.add_argument("--keepalive-sec", type=float, default=2.0) # KeepAlive
    parser.add_argument("--thumb-offset", type=float, default=3.0, help="Offset distance for thumb palm plane in cm (default 3.0)")
    parser.add_argument("--thumb-blend", type=float, default=0.7, help="Blend factor for thumb offset direction (0.0=pure normal, 1.0=toward thumb, default 0.7)")
    parser.add_argument("--thumb-tilt", type=float, default=-0.25, help="Tilt factor for plane normal (0.0=no tilt, 1.0=perpendicular to thumb, default 0.0)")
    parser.add_argument("--serial-port", type=str, default="COM4")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log", type=str, default=None)
    parser.add_argument("--show-3d", action="store_true")
    return parser.parse_args()


def setup_logging(log_base):
    if not log_base:
        return None, None, None, None
    from csv import writer

    base = Path(log_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    raw_path = base.parent / (base.name + "_raw.csv")
    quant_path = base.parent / (base.name + "_quant.csv")
    log_raw = open(raw_path, "w", newline="", buffering=1)
    log_quant = open(quant_path, "w", newline="", buffering=1)
    writer_raw = writer(log_raw)
    writer_quant = writer(log_quant)
    writer_raw.writerow(
        ["t", "wrist_pitch", "wrist_roll"]
        + [f"{nm}_angle" for nm, _, _ in FINGERS]
    )
    writer_quant.writerow(
        ["t", "wrist_pitch_servo", "wrist_roll_servo"]
        + [f"{nm}_servo" for nm, _, _ in FINGERS]
    )
    print("Logging to", raw_path, "and", quant_path)
    return log_raw, log_quant, writer_raw, writer_quant


def setup_3d(show_3d):
    if not show_3d:
        return None, None
    plt.ion()
    fig = plt.figure("Hand 3D")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.view_init(elev=20, azim=-60)
    return fig, ax


def update_3d(ax, lms_world):
    if ax is None:
        return
    xs = np.array([lm.x for lm in lms_world], dtype=np.float32)
    ys = np.array([lm.y for lm in lms_world], dtype=np.float32)
    zs = np.array([lm.z for lm in lms_world], dtype=np.float32)

    ax.cla()
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.view_init(elev=20, azim=-60)

    ax.scatter(xs, zs, -ys, s=10)
    for i, j in HAND_CONNECTIONS:
        x_pair = [xs[i], xs[j]]
        y_pair = [zs[i], zs[j]]
        z_pair = [-ys[i], -ys[j]]
        ax.plot(x_pair, y_pair, z_pair, linewidth=1)

    ax.set_xlim(xs.min() - 0.05, xs.max() + 0.05)
    ax.set_ylim(zs.min() - 0.05, zs.max() + 0.05)
    ax.set_zlim(-ys.max() - 0.05, -ys.min() + 0.05)

    plt.draw()
    plt.pause(0.001)


def main():
    args = parse_args()

    calib, open_pose, fist_pose = load_calibration()
    ser = open_serial(args.serial_port, args.baud, args.dry_run)

    log_raw, log_quant, writer_raw, writer_quant = setup_logging(args.log)
    fig3d, ax3d = setup_3d(args.show_3d)

    mp_hands = mp_hands_module.Hands(
        static_image_mode=False,
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print("Could not open camera", args.camera_index)
        return

    FLEX_PCT_TH = 0.10
    SERVO_DEG_TH = args.quantize_threshold
    KEEPALIVE_S = args.keepalive_sec

    last_tx_time = 0.0
    last_flex = {name: None for name, _, _ in FINGERS}
    last_servo_q = {name: None for name, _, _ in FINGERS}
    last_time = time.monotonic()
    
    verification_mode = False

    print("Keys: o=open  f=fist  v=verify  r=reset  w=write  q/ESC=quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_hands.process(frame_rgb)
        h, w, _ = frame.shape
        overlay = frame.copy()

        now = time.monotonic()
        dt = max(now - last_time, 1e-3)
        freq = 1.0 / dt
        last_time = now

        have_hand = res.multi_hand_landmarks is not None

        finger_angles = {}
        wrist_pitch = 0.0
        wrist_roll = 0.0
        lms_world = None

        if have_hand:
            use_world =False
            if getattr(res, "multi_hand_world_landmarks", None):
                use_world = True
                lms_world = res.multi_hand_world_landmarks[0].landmark
                lms_for_angles = lms_world
            else:
                lms_for_angles = res.multi_hand_landmarks[0].landmark

            for name, mcp_idx, nxt_idx in FINGERS:
                try:
                    if name == "THUMB":
                        # IMPROVED: Use offset palm plane for thumb tracking
                        # This provides better accuracy since thumb rotates differently
                        ang = finger_flexion_deg_offset(lms_for_angles, mcp_idx, nxt_idx, 
                                                        offset_cm=args.thumb_offset, 
                                                        blend_factor=args.thumb_blend,
                                                        tilt_factor=args.thumb_tilt,
                                                        use_world=use_world)
                    else:
                        # IMPROVED: Other fingers use improved palm normal (best-fit plane)
                        ang = finger_flexion_deg(lms_for_angles, mcp_idx, nxt_idx, use_world=use_world)
                except Exception:
                    ang = 0.0
                finger_angles[name] = ang

            wrist_pitch, wrist_roll = wrist_pitch_roll(lms_for_angles, use_world=use_world)

            mp_drawing.draw_landmarks(
                overlay,
                res.multi_hand_landmarks[0],
                HAND_CONNECTIONS,
            )

        if args.show_3d and lms_world is not None:
            update_3d(ax3d, lms_world)

        if finger_angles:
            # No smoothing - use raw angles directly
            thumb_a = finger_angles["THUMB"]
            index_a = finger_angles["INDEX"]
            middle_a = finger_angles["MIDDLE"]
            ring_a = finger_angles["RING"]
            pinky_a = finger_angles["PINKY"]

            servo_vals = {}
            flex_now = {}
            servo_q_now = {}

            for name, angle in zip(
                ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"],
                [thumb_a, index_a, middle_a, ring_a, pinky_a],
            ):
                servo_angle = servo_map(calib, name, angle)
                servo_vals[name] = servo_angle
                flex_now[name] = nflex(angle)
                servo_q_now[name] = qdeg(servo_angle, q=1.0)

            chg_flex = 0.0
            chg_servo = 0.0
            for name in servo_vals.keys():
                f_new = flex_now[name]
                s_new = servo_q_now[name]
                f_old = last_flex[name]
                s_old = last_servo_q[name]
                if f_old is None or s_old is None:
                    chg_flex = float("inf")
                    chg_servo = float("inf")
                    break
                chg_flex = max(chg_flex, abs(f_new - f_old))
                chg_servo = max(chg_servo, abs(s_new - s_old))

            t_now = time.time()
            # Send on any detected change or keepalive
            should_send = (
                (chg_flex >= FLEX_PCT_TH)
                or ((now - last_tx_time) >= KEEPALIVE_S)
            )

            if should_send:
                wrist_servo = wrist_roll_to_servo(wrist_roll)
                wrist_pitch_servo = max(SERVO_MIN, min(SERVO_MAX, 90.0 + wrist_pitch))

                if writer_raw is not None:
                    writer_raw.writerow(
                        [t_now, wrist_pitch, wrist_roll]
                        + [finger_angles[k] for k, _, _ in FINGERS]
                    )
                if writer_quant is not None:
                    writer_quant.writerow(
                        [t_now, wrist_pitch_servo, wrist_servo]
                        + [servo_q_now[k] for k, _, _ in FINGERS]
                    )

                if ser is not None:
                    try:
                        data_string = "<{},{},{},{},{},{},{}>\n".format(
                            int(servo_q_now["THUMB"]),
                            int(servo_q_now["INDEX"]),
                            int(servo_q_now["MIDDLE"]),
                            int(servo_q_now["RING"]),
                            int(servo_q_now["PINKY"]),
                            int(wrist_servo),
                            int(wrist_pitch_servo),
                        )
                        ser.write(data_string.encode("ascii"))
                        print("Sent", data_string.strip())
                    except Exception as e:
                        print("Serial send error:", e)

                last_tx_time = now
                last_flex.update(flex_now)
                last_servo_q.update(servo_q_now)

            y0 = 30
            wrist_text = f"WRoll: {wrist_roll:.1f}  WPitch: {wrist_pitch:.1f}"
            cv2.putText(
                overlay,
                wrist_text,
                (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            y0 += 30

            for name in ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]:
                angle = finger_angles.get(name, 0.0)
                servo_angle = servo_q_now.get(name, 0.0)  # Show quantized value (what's actually sent)
                text = f"{name[0]}: {angle:.1f} -> {servo_angle:.0f}"
                cv2.putText(
                    overlay,
                    text,
                    (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                y0 += 20

            # Verification mode - show progress bars
            if verification_mode:
                # Position at bottom right
                bar_width = 350  # Width of the text area
                y0 = h - 160  # Start from bottom
                x0 = w - bar_width - 10  # Right side
                
                cv2.putText(
                    overlay,
                    "[VERIFICATION MODE]",
                    (x0, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),  # Black
                    2,
                )
                y0 += 25
                
                for name in ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]:
                    angle = finger_angles.get(name, 0.0)
                    servo = servo_q_now.get(name, 0.0)  # Show quantized value (what's actually sent)
                    # Calculate percentage based on calibration
                    # Handle both increasing and decreasing angles
                    a_open = calib[name]["open_angle"]
                    a_fist = calib[name]["fist_angle"]
                    
                    if abs(a_fist - a_open) > 1e-3:
                        # Calculate how far we are from open to fist (0% to 100%)
                        raw_pct = ((angle - a_open) / (a_fist - a_open)) * 100.0
                        # If the range is inverted (fist < open), the percentage already works correctly
                        # because the negative denominator flips the sign appropriately
                        pct = raw_pct
                    else:
                        pct = 0.0
                    # Clamp to 0-100%
                    pct = max(0.0, min(100.0, pct))
                    
                    bar = draw_progress_bar(pct, width=10)
                    text = f"{name[0]}: {bar} -> {servo:.0f}deg"
                    cv2.putText(
                        overlay,
                        text,
                        (x0, y0),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),  # Black
                        2,
                    )
                    y0 += 25

        cv2.putText(
            overlay,
            "o=open  f=fist  v=verify  r=reset  w=write  q=quit",
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        cv2.imshow("HandAngleTrack", overlay)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q")):
            break
        elif key == ord("o") and have_hand:
            if getattr(res, "multi_hand_world_landmarks", None):
                lms = res.multi_hand_world_landmarks[0].landmark
            else:
                lms = res.multi_hand_landmarks[0].landmark
            # IMPROVED: Thumb uses offset palm, other fingers use improved palm normal
            open_pose = {}
            for name, mcp, nxt in FINGERS:
                if name == "THUMB":
                    open_pose[name] = finger_flexion_deg_offset(lms, mcp, nxt, 
                                                                offset_cm=args.thumb_offset,
                                                                blend_factor=args.thumb_blend,
                                                                tilt_factor=args.thumb_tilt,
                                                                use_world=True)
                else:
                    open_pose[name] = finger_flexion_deg(lms, mcp, nxt, use_world=True)
            print("Open pose captured")
        elif key == ord("f") and have_hand:
            if getattr(res, "multi_hand_world_landmarks", None):
                lms = res.multi_hand_world_landmarks[0].landmark
            else:
                lms = res.multi_hand_landmarks[0].landmark
            # IMPROVED: Thumb uses offset palm, other fingers use improved palm normal
            fist_pose = {}
            for name, mcp, nxt in FINGERS:
                if name == "THUMB":
                    fist_pose[name] = finger_flexion_deg_offset(lms, mcp, nxt, 
                                                                offset_cm=args.thumb_offset,
                                                                blend_factor=args.thumb_blend,
                                                                tilt_factor=args.thumb_tilt,
                                                                use_world=True)
                else:
                    fist_pose[name] = finger_flexion_deg(lms, mcp, nxt, use_world=True)
            print("Fist pose captured")
            if open_pose and fist_pose:
                update_calibration(calib, open_pose, fist_pose)
        elif key == ord("v"):
            verification_mode = not verification_mode
            mode_str = "ON" if verification_mode else "OFF"
            print(f"Verification mode: {mode_str}")
        elif key == ord("r"):
            calib = default_calibration()
            open_pose = {}
            fist_pose = {}
            print("Calibration reset")
        elif key == ord("w"):
            save_calibration(calib, open_pose, fist_pose)

    cap.release()
    mp_hands.close()
    if ser is not None:
        ser.close()
        print("Serial closed")
    if log_raw is not None:
        log_raw.close()
    if log_quant is not None:
        log_quant.close()
    cv2.destroyAllWindows()
    if fig3d is not None:
        plt.ioff()
        plt.close(fig3d)


if __name__ == "__main__":
    main()