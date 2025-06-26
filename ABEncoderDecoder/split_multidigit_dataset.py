import os
import shutil
import random

# Paths
root_dir = ".data/multidigit_mnist"
images_dir = os.path.join(root_dir, "images")
labels_file = os.path.join(root_dir, "labels", "labels.txt")

output_dir = "data/multidigit"
train_img_dir = os.path.join(output_dir, "train", "images")
val_img_dir = os.path.join(output_dir, "val", "images")
train_lbl_file = os.path.join(output_dir, "train", "labels.txt")
val_lbl_file = os.path.join(output_dir, "val", "labels.txt")

# Create output directories
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)

# Load all labels into a dict: {img_filename: [digits]}
label_dict = {}
with open(labels_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            img_file = parts[0]
            digits = parts[1:]
            label_dict[img_file] = digits

# Get all image filenames and shuffle
image_filenames = list(label_dict.keys())
random.shuffle(image_filenames)

# Split (80% train, 20% val)
split_idx = int(0.8 * len(image_filenames))
train_files = image_filenames[:split_idx]
val_files = image_filenames[split_idx:]

# Copy files and write new label files
def copy_and_log(files, img_dest, label_dest):
    os.makedirs(img_dest, exist_ok=True)
    with open(label_dest, "w") as out_f:
        for fname in files:
            src_img = os.path.join(images_dir, fname)
            dst_img = os.path.join(img_dest, fname)
            shutil.copy2(src_img, dst_img)
            out_f.write(fname + " " + " ".join(label_dict[fname]) + "\n")

copy_and_log(train_files, train_img_dir, train_lbl_file)
copy_and_log(val_files, val_img_dir, val_lbl_file)

print(f"✅ Done! Train: {len(train_files)} | Val: {len(val_files)}")