import argparse
import cv2
import numpy as np
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

FINGERTIP_IDS = [4, 8, 12, 16, 20]


def detect_skin_mask(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 30, 50], dtype=np.uint8)
    upper = np.array([25, 200, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.medianBlur(mask, 5)
    return mask


def compute_object_mask(skin_mask, hand_landmarks_2d, w, h, margin=20):
    if hand_landmarks_2d is None:
        return np.zeros_like(skin_mask)
    us = [int(lm.x * w) for lm in hand_landmarks_2d]
    vs = [int(lm.y * h) for lm in hand_landmarks_2d]
    xmin = max(min(us) - margin, 0)
    xmax = min(max(us) + margin, w - 1)
    ymin = max(min(vs) - margin, 0)
    ymax = min(max(vs) + margin, h - 1)
    not_skin = cv2.bitwise_not(skin_mask)
    obj_mask = np.zeros_like(skin_mask)
    obj_mask[ymin:ymax + 1, xmin:xmax + 1] = not_skin[ymin:ymax + 1, xmin:xmax + 1]
    return obj_mask


def detect_contact_points(obj_mask, hand_landmarks_2d, w, h,
                          radius=8, frac_thresh=0.2):
    contacts = set()
    if hand_landmarks_2d is None:
        return contacts
    for idx in FINGERTIP_IDS:
        lm = hand_landmarks_2d[idx]
        u = int(lm.x * w)
        v = int(lm.y * h)
        if u < 0 or u >= w or v < 0 or v >= h:
            continue
        v0 = max(0, v - radius)
        v1 = min(h, v + radius)
        u0 = max(0, u - radius)
        u1 = min(w, u + radius)
        patch = obj_mask[v0:v1, u0:u1]
        if patch.size == 0:
            continue
        obj_pixels = np.count_nonzero(patch)
        total = patch.size
        frac_obj = obj_pixels / float(total)
        if frac_obj >= frac_thresh:
            contacts.add(idx)
    return contacts


def build_delaunay(points, w, h):
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)

    # insert points safely inside image
    safe_points = []
    for (x, y) in points:
        xs = float(min(max(x, 1), w - 2))
        ys = float(min(max(y, 1), h - 2))
        safe_points.append((xs, ys))
        try:
            subdiv.insert((xs, ys))
        except cv2.error:
            # if insert fails we just skip this point
            pass

    triangle_list = []
    try:
        triangle_list = subdiv.getTriangleList()
    except cv2.error:
        return []

    pts = np.array(safe_points, dtype=np.float32)
    tris_idx = []

    for t in triangle_list:
        tri_coords = np.array(
            [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])],
            dtype=np.float32,
        )
        idxs = []
        for (x, y) in tri_coords:
            d = np.sum((pts - np.array([x, y], dtype=np.float32)) ** 2, axis=1)
            idxs.append(int(np.argmin(d)))
        if len(set(idxs)) == 3:
            tris_idx.append(tuple(idxs))
    return tris_idx


def main():
    parser = argparse.ArgumentParser(
        description="Live hand + object mesh overlay with contact highlighting"
    )
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print("Could not open camera", args.camera_index)
        return

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    print("Press q or ESC to quit")

    cached_tris = None
    cached_points = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res_hands = hands.process(frame_rgb)

        overlay = frame.copy()
        skin_mask = detect_skin_mask(frame)

        hand_image = None
        if res_hands.multi_hand_landmarks:
            hand_image = res_hands.multi_hand_landmarks[0].landmark
            mp_drawing.draw_landmarks(
                overlay,
                res_hands.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS,
            )

        obj_mask = compute_object_mask(skin_mask, hand_image, w, h)
        contact_indices = detect_contact_points(obj_mask, hand_image, w, h)

        if hand_image is not None:
            pts_2d = []
            for lm in hand_image:
                u = lm.x * w
                v = lm.y * h
                u = min(max(u, 1.0), w - 2.0)
                v = min(max(v, 1.0), h - 2.0)
                pts_2d.append((u, v))

            pts_arr = np.array(pts_2d, dtype=np.float32)

            recompute = False
            if cached_tris is None or cached_points is None:
                recompute = True
            else:
                mean_shift = np.mean(
                    np.linalg.norm(pts_arr - cached_points, axis=1)
                )
                if mean_shift > 10.0:
                    recompute = True

            if recompute:
                cached_tris = build_delaunay(pts_2d, w, h)
                cached_points = pts_arr.copy()

            mesh_img = np.zeros_like(frame)

            for (i, j, k) in cached_tris:
                p0 = (int(pts_2d[i][0]), int(pts_2d[i][1]))
                p1 = (int(pts_2d[j][0]), int(pts_2d[j][1]))
                p2 = (int(pts_2d[k][0]), int(pts_2d[k][1]))
                poly = np.array([p0, p1, p2], dtype=np.int32)

                if (i in contact_indices) or (j in contact_indices) or (k in contact_indices):
                    color = (0, 0, 255)  # red for contact
                else:
                    color = (255, 220, 180)  # light mesh color (feel free to adjust)

                cv2.fillConvexPoly(mesh_img, poly, color)

            cv2.addWeighted(mesh_img, 0.5, overlay, 0.5, 0, overlay)

            for idx in contact_indices:
                lm = hand_image[idx]
                u = int(min(max(lm.x * w, 0), w - 1))
                v = int(min(max(lm.y * h, 0), h - 1))
                cv2.circle(overlay, (u, v), 6, (0, 0, 255), -1)

        if np.any(obj_mask):
            contours, _ = cv2.findContours(
                obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                overlay,
                contours,
                -1,
                (160, 160, 160),
                thickness=-1,
            )

        obj_bgr = cv2.cvtColor(obj_mask, cv2.COLOR_GRAY2BGR)
        small_h = h // 4
        small_w = w // 4
        small_obj = cv2.resize(
            obj_bgr, (small_w, small_h), interpolation=cv2.INTER_NEAREST
        )
        overlay[0:small_h, 0:small_w] = small_obj

        cv2.putText(
            overlay,
            "Red = contact regions",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Hand + Object Mesh", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
