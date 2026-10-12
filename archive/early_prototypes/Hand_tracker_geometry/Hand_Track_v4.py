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


# configuration constants

FINGERS = [
    ("THUMB", 1, 3),
    ("INDEX", 5, 7),
    ("MIDDLE", 9, 11),
    ("RING", 13, 15),
    ("PINKY", 17, 19),
]

SERVO_MIN = 0.0
SERVO_MAX = 180.0
CALIB_FILE = "servo_calib.txt"

THUMB_OPEN, THUMB_CLOSE = 0.0, 100.0
INDEX_OPEN, INDEX_CLOSE = 0.0, 110.0
MIDDLE_OPEN, MIDDLE_CLOSE = 0.0, 90.0
RING_OPEN, RING_CLOSE = 0.0, 110.0
PINKY_OPEN, PINKY_CLOSE = 0.0, 100.0

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


# Smoothing filters

class OneEuro:
    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.04, dcutoff=1.0):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.x_prev = None
        self.dx_prev = None

    def _alpha(self, cutoff):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / max(self.freq, 1e-6)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float32)
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            return x

        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.dcutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff)

        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


def ema(prev, new, alpha):
    if prev is None:
        return new
    return alpha * new + (1.0 - alpha) * prev


# Geometry and angle calculations

def to_np_point(lm, use_world=True):
    x = float(lm.x)
    y = float(lm.y)
    z = float(lm.z) if (use_world and hasattr(lm, "z")) else 0.0
    return np.array([x, y, z], dtype=np.float32)


def norm_vec(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n


def build_palm_normal(landmarks, use_world=True):
    # wrist + MCPs: 0,5,9,13,17
    idx = [0, 5, 9, 13, 17]
    pts = np.array([to_np_point(landmarks[i], use_world) for i in idx])
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    cov = (centered.T @ centered) / len(pts)
    _, evecs = np.linalg.eigh(cov)
    normal = evecs[:, 0]

    wrist = to_np_point(landmarks[0], use_world)
    if np.dot(normal, wrist - centroid) > 0:
        normal = -normal

    return norm_vec(normal)


def build_offset_thumb_plane(landmarks, offset_cm, blend, tilt, use_world):
    normal = build_palm_normal(landmarks, use_world)

    idx = [0, 5, 9, 13, 17]
    pts = np.array([to_np_point(landmarks[i], use_world) for i in idx])
    centroid = np.mean(pts, axis=0)

    thumb_cmc = to_np_point(landmarks[0], use_world) # (2,3) initially best
    thumb_mcp = to_np_point(landmarks[4], use_world)
    thumb_dir = norm_vec(thumb_mcp - thumb_cmc)

    # tilt normal toward thumb motion axis
    if tilt != 0.0:
        thumb_on_plane = thumb_dir - np.dot(thumb_dir, normal) * normal
        thumb_on_plane = norm_vec(thumb_on_plane)
        normal = norm_vec((1.0 - tilt) * normal + tilt * thumb_on_plane)

    # move plane centroid toward thumb base
    to_thumb = norm_vec(thumb_cmc - centroid)
    direction = norm_vec((1.0 - blend) * normal + blend * to_thumb)
    offset_m = offset_cm / 100.0
    centroid = centroid + direction * offset_m

    return normal, centroid


def angle_to_plane(v, normal):
    v = norm_vec(np.asarray(v, dtype=np.float32))
    n = norm_vec(np.asarray(normal, dtype=np.float32))
    cos_t = np.clip(np.dot(v, n), -1.0, 1.0)
    theta = math.degrees(math.acos(cos_t))
    return max(0.0, min(90.0, 90.0 - theta))


def finger_flexion_deg(landmarks, mcp, nxt, use_world=True):
    n = build_palm_normal(landmarks, use_world)
    a = to_np_point(landmarks[mcp], use_world)
    b = to_np_point(landmarks[nxt], use_world)
    return angle_to_plane(b - a, n)


def thumb_flexion_deg_offset(landmarks, mcp, nxt, offset_cm, blend, tilt, use_world=True):
    n, _centroid = build_offset_thumb_plane(landmarks, offset_cm, blend, tilt, use_world)
    a = to_np_point(landmarks[mcp], use_world)
    b = to_np_point(landmarks[nxt], use_world)
    return angle_to_plane(b - a, n)


def wrist_pitch_roll(landmarks, use_world=True):
    n = build_palm_normal(landmarks, use_world)
    pitch = math.degrees(math.asin(np.clip(-n[1], -1.0, 1.0)))
    roll = math.degrees(math.atan2(n[0], -n[2]))
    return pitch, roll


# Calibration and mapping

def default_calibration():
    calib = {}
    for name, _, _ in FINGERS:
        r = FINGER_SERVO_RANGES[name]
        calib[name] = {
            "open_angle": 0.0,
            "fist_angle": 90.0,
            "servo_open": r["servo_open"],
            "servo_close": r["servo_close"],
        }
    return calib


def load_calibration():
    calib = default_calibration()
    open_pose = {}
    fist_pose = {}

    if os.path.exists(CALIB_FILE):
        try:
            with open(CALIB_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            saved_calib = data.get("calib", {})
            for k, v in saved_calib.items():
                if k in calib and isinstance(v, dict):
                    calib[k].update(v)
            open_pose = data.get("open", {})
            fist_pose = data.get("fist", {})
            print("Loaded calibration from", CALIB_FILE)
        except Exception as e:
            print("Could not load calibration:", e)

    return calib, open_pose, fist_pose


def save_calibration(calib, open_pose, fist_pose):
    data = {"calib": calib, "open": open_pose, "fist": fist_pose}
    with open(CALIB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print("Saved calibration to", CALIB_FILE)


def update_calibration(calib, open_pose, fist_pose):
    for name, _, _ in FINGERS:
        calib[name]["open_angle"] = float(open_pose[name])
        calib[name]["fist_angle"] = float(fist_pose[name])
    print("Updated calibration (open/fist -> mapping)")


def servo_map(calib, name, angle_deg):
    c = calib[name]
    a_open = c["open_angle"]
    a_fist = c["fist_angle"]
    s_open = c["servo_open"]
    s_close = c["servo_close"]

    if abs(a_fist - a_open) < 1e-3:
        t = 0.0
    else:
        t = (angle_deg - a_open) / (a_fist - a_open)

    t = max(0.0, min(1.0, t))
    val = s_open + t * (s_close - s_open)
    return float(np.clip(val, SERVO_MIN, SERVO_MAX))


def qdeg(x, q=1.0, lo=SERVO_MIN, hi=SERVO_MAX):
    y = round(float(x) / q) * q
    return float(min(max(y, lo), hi))


def wrist_roll_to_servo(roll_deg):
    r = max(-45.0, min(45.0, roll_deg))
    t = (r + 45.0) / 90.0
    val = WRIST_RIGHT + t * (WRIST_LEFT - WRIST_RIGHT)
    return float(np.clip(val, SERVO_MIN, SERVO_MAX))


def wrist_pitch_to_servo(pitch_deg):
    return float(np.clip(90.0 + pitch_deg, SERVO_MIN, SERVO_MAX))


def draw_progress_bar(percent, width=10):
    percent = max(0.0, min(100.0, percent))
    filled = int((percent / 100.0) * width)
    bar = "=" * filled + " " * (width - filled)
    return f"[{bar}] {percent:3.0f}%"


# Serial communication

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


# logging

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

    writer_raw.writerow(["t", "wrist_pitch", "wrist_roll"] + [f"{nm}_angle" for nm, _, _ in FINGERS])
    writer_quant.writerow(["t", "wrist_pitch_servo", "wrist_roll_servo"] + [f"{nm}_servo" for nm, _, _ in FINGERS])

    print("Logging to", raw_path, "and", quant_path)
    return log_raw, log_quant, writer_raw, writer_quant


# Helpers

def compute_angles(landmarks, use_world, thumb_offset, thumb_blend, thumb_tilt):
    finger_angles = {}
    for name, mcp, nxt in FINGERS:
        if name == "THUMB":
            ang = thumb_flexion_deg_offset(
                landmarks, mcp, nxt,
                offset_cm=thumb_offset,
                blend=thumb_blend,
                tilt=thumb_tilt,
                use_world=use_world,
            )
        else:
            ang = finger_flexion_deg(landmarks, mcp, nxt, use_world=use_world)
        finger_angles[name] = float(ang)

    pitch, roll = wrist_pitch_roll(landmarks, use_world=use_world)
    return finger_angles, float(pitch), float(roll)


# args for CLI flags

def parse_args():
    p = argparse.ArgumentParser(description="Hand tracking (geometry + smoothing) + Arduino control (locked until calibrated)")

    p.add_argument("--camera-index", type=int, default=0)

    p.add_argument("--smoothing", choices=["one-euro", "ema", "off"], default="one-euro")
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--min-cutoff", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--dcutoff", type=float, default=1.0)

    p.add_argument("--thumb-offset", type=float, default=3.0)
    p.add_argument("--thumb-blend", type=float, default=0.7)
    p.add_argument("--thumb-tilt", type=float, default=-0.25)

    p.add_argument("--serial-port", type=str, default="COM4")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--dry-run", action="store_true")

    p.add_argument("--log", type=str, default=None)
    p.add_argument("--show-3d", action="store_true")  # kept for compatibility; not used here (future work)

    return p.parse_args()


# Main function

def main():
    args = parse_args()

    calib, open_pose, fist_pose = load_calibration()

    # movement lock
    calibrated = bool(open_pose and fist_pose)
    if calibrated:
        print("Calibration loaded from file. Movement ENABLED.")
    else:
        print("No valid calibration found. Movement LOCKED. Do: o -> f -> w")

    ser = open_serial(args.serial_port, args.baud, args.dry_run)
    log_raw, log_quant, writer_raw, writer_quant = setup_logging(args.log)

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

    one_euro = None
    ema_prev = None

    if args.smoothing == "one-euro":
        one_euro = OneEuro(
            freq=30.0,
            min_cutoff=args.min_cutoff,
            beta=args.beta,
            dcutoff=args.dcutoff,
        )

    verification_mode = False
    last_time = time.monotonic()

    print("Keys: o=open  f=fist  v=verify  w=write(+CALIB_DONE)  r=reset  q/ESC=quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        overlay = frame.copy()
        h, w, _ = overlay.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_hands.process(rgb)

        now = time.monotonic()
        dt = max(now - last_time, 1e-3)
        last_time = now
        if one_euro is not None:
            one_euro.freq = 1.0 / dt

        have_hand = res.multi_hand_landmarks is not None

        # Default display values
        finger_angles = None
        pitch = 0.0
        roll = 0.0

        # If we have a hand, compute angles
        if have_hand:
            use_world = False
            if getattr(res, "multi_hand_world_landmarks", None):
                use_world = True
                lms_for_angles = res.multi_hand_world_landmarks[0].landmark
            else:
                lms_for_angles = res.multi_hand_landmarks[0].landmark

            finger_angles, pitch, roll = compute_angles(
                lms_for_angles,
                use_world=use_world,
                thumb_offset=args.thumb_offset,
                thumb_blend=args.thumb_blend,
                thumb_tilt=args.thumb_tilt,
            )

            # Draw landmarks using 2D landmarks (always available when have_hand)
            mp_drawing.draw_landmarks(
                overlay,
                res.multi_hand_landmarks[0],
                HAND_CONNECTIONS,
            )

        # Show locked warning
        if not calibrated:
            cv2.putText(
                overlay,
                "CALIBRATION REQUIRED - Press o, f, then w",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        # If we have angles, apply smoothing and compute servo values (but only send if calibrated)
        if finger_angles is not None:
            vec = np.array([
                finger_angles["THUMB"],
                finger_angles["INDEX"],
                finger_angles["MIDDLE"],
                finger_angles["RING"],
                finger_angles["PINKY"],
                roll,
                pitch,
            ], dtype=np.float32)

            if args.smoothing == "off":
                sm = vec
            elif args.smoothing == "ema":
                ema_prev = ema(ema_prev, vec, args.alpha)
                sm = ema_prev
            else:
                sm = one_euro(vec) if one_euro is not None else vec

            thumb_a, index_a, middle_a, ring_a, pinky_a, roll_s, pitch_s = sm.tolist()

            smooth_angles = {
                "THUMB": thumb_a,
                "INDEX": index_a,
                "MIDDLE": middle_a,
                "RING": ring_a,
                "PINKY": pinky_a,
            }

            # Live servo values (quantized 1 degree increments)
            servo_q = {}
            for name in ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]:
                s = servo_map(calib, name, smooth_angles[name])
                servo_q[name] = qdeg(s, q=1.0)

            wrist_roll_servo = qdeg(wrist_roll_to_servo(roll_s), q=1.0)
            wrist_pitch_servo = qdeg(wrist_pitch_to_servo(pitch_s), q=1.0)

            # display wrist angles and servo values
            cv2.putText(
                overlay,
                f"WRoll: {roll_s:+.1f} deg -> {wrist_roll_servo:.0f} | WPitch: {pitch_s:+.1f} deg -> {wrist_pitch_servo:.0f} | smooth={args.smoothing}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # display finger angles and servo values
            y0 = 70
            for name in ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]:
                ang = smooth_angles[name]
                s = servo_q[name]
                cv2.putText(
                    overlay,
                    f"{name[0]}: {ang:5.1f} deg -> {s:3.0f}",
                    (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )
                y0 += 26

            # verification mode display (shows progress bars and mapping info)
            if verification_mode:
                xv = w - 390
                yv = h - 170
                cv2.putText(overlay, "[VERIFICATION MODE]", (xv, yv), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                yv += 25
                for name in ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]:
                    a_open = calib[name]["open_angle"]
                    a_fist = calib[name]["fist_angle"]
                    ang = smooth_angles[name]
                    if abs(a_fist - a_open) > 1e-3:
                        pct = ((ang - a_open) / (a_fist - a_open)) * 100.0
                    else:
                        pct = 0.0
                    pct = max(0.0, min(100.0, pct))
                    bar = draw_progress_bar(pct, width=10)
                    cv2.putText(
                        overlay,
                        f"{name[0]}: {bar} -> {servo_q[name]:.0f}deg",
                        (xv, yv),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2,
                    )
                    yv += 25

            # logging
            if writer_raw is not None:
                t_now = time.time()
                writer_raw.writerow([t_now, pitch_s, roll_s] + [smooth_angles[nm] for nm, _, _ in FINGERS])
            if writer_quant is not None:
                t_now = time.time()
                writer_quant.writerow([t_now, wrist_pitch_servo, wrist_roll_servo] + [servo_q[nm] for nm, _, _ in FINGERS])

            # it will only send to serial if calibrated
            if calibrated and ser is not None:
                try:
                    data = "<{},{},{},{},{},{},{}>\n".format(
                        int(servo_q["THUMB"]),
                        int(servo_q["INDEX"]),
                        int(servo_q["MIDDLE"]),
                        int(servo_q["RING"]),
                        int(servo_q["PINKY"]),
                        int(wrist_roll_servo),
                        int(wrist_pitch_servo),
                    )
                    ser.write(data.encode("ascii"))
                except Exception as e:
                    print("Serial send error:", e)

        # footer instructions
        cv2.putText(
            overlay,
            "o=open  f=fist  v=verify  w=write(+CALIB_DONE)  r=reset  q/ESC=quit",
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

        elif key == ord("v"):
            verification_mode = not verification_mode
            print("Verification mode:", "ON" if verification_mode else "OFF")

        elif key == ord("o") and have_hand:
            # capture open pose using the same logic as runtime
            if getattr(res, "multi_hand_world_landmarks", None):
                lms = res.multi_hand_world_landmarks[0].landmark
                use_world = True
            else:
                lms = res.multi_hand_landmarks[0].landmark
                use_world = False

            open_pose = {}
            f_angles, _p, _r = compute_angles(
                lms, use_world,
                thumb_offset=args.thumb_offset,
                thumb_blend=args.thumb_blend,
                thumb_tilt=args.thumb_tilt
            )
            open_pose.update(f_angles)
            print("Open pose captured")

        elif key == ord("f") and have_hand:
            if getattr(res, "multi_hand_world_landmarks", None):
                lms = res.multi_hand_world_landmarks[0].landmark
                use_world = True
            else:
                lms = res.multi_hand_landmarks[0].landmark
                use_world = False

            fist_pose = {}
            f_angles, _p, _r = compute_angles(
                lms, use_world,
                thumb_offset=args.thumb_offset,
                thumb_blend=args.thumb_blend,
                thumb_tilt=args.thumb_tilt
            )
            fist_pose.update(f_angles)
            print("Fist pose captured")

        elif key == ord("w"):
            if open_pose and fist_pose:
                update_calibration(calib, open_pose, fist_pose)
                save_calibration(calib, open_pose, fist_pose)
                calibrated = True
                print("Calibration complete. Movement ENABLED.")

                if ser is not None:
                    try:
                        ser.write(b"<CALIB_DONE>\n")
                        print("Sent <CALIB_DONE> to Arduino")
                    except Exception as e:
                        print("Serial send error:", e)
            else:
                print("Calibration incomplete. Do: o then f, then w")

        elif key == ord("r"):
            calibrated = False
            open_pose = {}
            fist_pose = {}
            print("Calibration reset. Movement LOCKED again.")

    # cleanup
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


if __name__ == "__main__":
    main()