import cv2
import os
import re
from glob import glob

gestures = ["open_palm", "fist", "thumbs_up", "two_fingers", "point_left", "point_right"] # type of gesture
base_dir = "/path1" # place to save frames for each gesture
SAVE_EVERY = 5
CAM_INDEX = 0

for g in gestures:
    os.makedirs(os.path.join(base_dir, g), exist_ok=True)

print("Available gestures:", gestures)
gesture = input("Enter the gesture you want to record: ").strip()

if gesture not in gestures:
    print("Invalid gesture. Please run again and choose from the list.")
    raise SystemExit

save_dir = os.path.join(base_dir, gesture)

pattern = re.compile(rf"^{re.escape(gesture)}_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
existing_files = [os.path.basename(p) for p in glob(os.path.join(save_dir, "*"))]

existing_nums = []
for fname in existing_files:
    m = pattern.match(fname)
    if m:
        try:
            existing_nums.append(int(m.group(1)))
        except ValueError:
            pass

if existing_nums:
    last_num = max(existing_nums)
    count = last_num + SAVE_EVERY # continue at the next step
else:
    count = 0

cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    print("Could not open camera. Try a different CAM_INDEX or check permissions.")
    raise SystemExit

print(f"Recording gesture : {gesture} (press 'q' to stop)")
print(f"Saving into: {save_dir}")
print(f"Starting from index: {count}")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame grab failed; stopping.")
        break

    cv2.imshow("Capture", frame)

    # we will save every nth frame (5th here)
    if frame_idx % SAVE_EVERY == 0:
        img_path = os.path.join(save_dir, f"{gesture}_{count}.jpg")
        
        while os.path.exists(img_path):
            count += SAVE_EVERY
            img_path = os.path.join(save_dir, f"{gesture}_{count}.jpg")
            
        cv2.imwrite(img_path, frame)
        print(f"Saved {img_path}")
        count += SAVE_EVERY

    frame_idx += 1

    # quit button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
