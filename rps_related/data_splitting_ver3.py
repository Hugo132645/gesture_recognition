import cv2
import os
import re
import math
import shutil
import random
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_DIR = Path('data/merged_raw')   # Your source data 
OUTPUT_DIR = Path('data/final_split_dataset')

PAD = 50                         # Padding around the hand (pixels)
SPLIT_RATIO_TRAIN = 0.70
SPLIT_RATIO_VAL   = 0.15
# Test gets the remainder (0.15)

def process_and_split_mediapipe():
    print(f"Initializing MediaPipe Pipeline...")
    
    # 1. Setup MediaPipe
    mp_hands = mp.solutions.hands
    # min_detection_confidence=0.5 is standard and robust
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

    # 2. Setup Directories
    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    
    for split in ['train', 'val', 'test']:
        for cls in ['Rock', 'Paper', 'Scissor']:
            (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    # 3. Iterate Classes
    class_map = {
        'rock': 'Rock', 'Rock': 'Rock',
        'paper': 'Paper', 'Paper': 'Paper',
        'scissor': 'Scissor', 'scissors': 'Scissor', 'Scissor': 'Scissor'
    }

    for subfolder in INPUT_DIR.iterdir():
        if not subfolder.is_dir(): continue
        
        clean_name = class_map.get(subfolder.name.lower(), None)
        if not clean_name: continue

        # Group by Video
        pattern = re.compile(r'(.+)_(\d+)\.(jpg|jpeg|png)$', re.IGNORECASE)
        videos = {}
        for img_path in subfolder.glob('*'):
            m = pattern.match(img_path.name)
            if m:
                vid_id = m.group(1)
                frame_idx = int(m.group(2))
                if vid_id not in videos: videos[vid_id] = []
                videos[vid_id].append((frame_idx, img_path))
        
        # Split Videos
        video_ids = list(videos.keys())
        random.shuffle(video_ids)
        
        n_total = len(video_ids)
        n_train = int(n_total * SPLIT_RATIO_TRAIN)
        n_val = int(n_total * SPLIT_RATIO_VAL)
        
        train_ids = set(video_ids[:n_train])
        val_ids = set(video_ids[n_train : n_train + n_val])
        
        print(f"Processing {clean_name}: {n_total} videos...")

        # 4. Process
        for vid_id, frames in tqdm(videos.items()):
            frames.sort(key=lambda x: x[0]) 
            
            if vid_id in train_ids: subset = 'train'
            elif vid_id in val_ids: subset = 'val'
            else: subset = 'test'
            
            dest_folder = OUTPUT_DIR / subset / clean_name
            
            for idx, (frame_num, img_path) in enumerate(frames):
                image = cv2.imread(str(img_path))
                if image is None: continue
                h_img, w_img, _ = image.shape
                
                # MediaPipe Inference
                # Must convert to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                
                if results.multi_hand_landmarks:
                    # Get the first hand found
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    # Convert normalized coordinates to pixels
                    x_list = [lm.x * w_img for lm in hand_landmarks.landmark]
                    y_list = [lm.y * h_img for lm in hand_landmarks.landmark]
                    
                    # Calculate Bounding Box
                    x_min, x_max = min(x_list), max(x_list)
                    y_min, y_max = min(y_list), max(y_list)
                    
                    center_x = (x_min + x_max) / 2
                    center_y = (y_min + y_max) / 2
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    # Square the box (makes it easier for ResNet)
                    box_size = max(width, height) + PAD
                    
                    x1 = int(center_x - box_size / 2)
                    y1 = int(center_y - box_size / 2)
                    x2 = int(center_x + box_size / 2)
                    y2 = int(center_y + box_size / 2)
                    
                    # Clamp
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(w_img, x2); y2 = min(h_img, y2)
                    
                    crop = image[y1:y2, x1:x2]
                    
                    if crop.size > 0:
                        cv2.imwrite(str(dest_folder / img_path.name), crop)

    print(f"\nDone! Dataset saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_and_split_mediapipe()