import os
import shutil

# All datasets' paths
source_dirs = ["/path1", "/path2", "/path3"]   # change to your folder names
target_dir = "/pathU"  # unified dataset folder

os.makedirs(target_dir, exist_ok=True)

subject_id = 1
for src in source_dirs:
    for gesture in os.listdir(src):
        src_folder = os.path.join(src, gesture)
        if not os.path.isdir(src_folder):
            continue
        
        dest_folder = os.path.join(target_dir, gesture)
        os.makedirs(dest_folder, exist_ok=True)

        for fname in os.listdir(src_folder):
            if not fname.endswith((".jpg", ".png")):
                continue
            new_name = f"{gesture}_s{subject_id}_{fname.split('_')[-1]}"
            shutil.copy(os.path.join(src_folder, fname),
                        os.path.join(dest_folder, new_name))
    subject_id += 1

print("All data merged into", target_dir)
