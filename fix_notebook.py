import json

with open('baseline_training.ipynb', 'r') as f:
    nb = json.load(f)

# Cell 2 is index 2
robust_code = '''import os
import shutil

KAGGLE_INPUT_DIR = "/kaggle/input"
WORKING_DIR = "/kaggle/working/data"
TARGET_CROPS = ["Corn", "Tomato", "Grape", "Potato", "Maize"]

print("Finding dataset directories...")
train_src = None
valid_src = None
for root, dirs, files in os.walk(KAGGLE_INPUT_DIR):
    # Only pick leaf nodes that are actual dataset splits, avoiding hidden or extra dirs
    if os.path.basename(root) == "train" and not train_src:
        train_src = root
    if os.path.basename(root) == "valid" and not valid_src:
        valid_src = root

print(f"Found train dir: {train_src}")
print(f"Found valid dir: {valid_src}")

if train_src and valid_src:
    for split_name, src_split in [("train", train_src), ("valid", valid_src)]:
        dst_split = os.path.join(WORKING_DIR, split_name)
        os.makedirs(dst_split, exist_ok=True)
        for class_name in os.listdir(src_split):
            if any(crop in class_name for crop in TARGET_CROPS):
                src_class = os.path.join(src_split, class_name)
                dst_class = os.path.join(dst_split, class_name)
                if not os.path.exists(dst_class):
                    shutil.copytree(src_class, dst_class)
else:
    raise FileNotFoundError("Could not find train/valid directories in input")
print("Filtering complete. Data ready in /kaggle/working/data")
'''

nb['cells'][2]['source'] = [line + '\n' for line in robust_code.split('\n')]
# Keep the fix for Keras warning
with open('baseline_training.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)
