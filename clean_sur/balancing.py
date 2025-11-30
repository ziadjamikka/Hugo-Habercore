import os
import random
import shutil
from collections import defaultdict
import cv2
import albumentations as A

# -------- CONFIG --------
DATA_DIR = r"D:\DEPIMariam\Final\data\final_datasetB\train\images"  # train images
LABELS_DIR = r"D:\DEPIMariam\Final\data\final_datasetB\train\labels"
OUTPUT_DIR = r"D:\DEPIMariam\Final\data\final_datasetB\train_balanced"

CLASSES = ["clean_surface", "dirty_surface", "insect", "Rats", "trash"]
TARGET_CLASS = "Rats"       # class we want to oversample
AUG_COPIES = 1              # number of augmented copies per image

# Safe augmentations
augment = A.Compose([
    A.HorizontalFlip(p=0.6),
    A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2),
    A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=10, p=0.5),
    A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
    A.OneOf([
        A.GaussianBlur(blur_limit=3, p=0.5),
        A.MotionBlur(blur_limit=3, p=0.3),
    ], p=0.2)
], bbox_params=None)

# Ensure output dirs
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

# Load image → classes mapping
image_classes = defaultdict(set)
for label_file in os.listdir(LABELS_DIR):
    if not label_file.endswith(".txt"):
        continue
    label_path = os.path.join(LABELS_DIR, label_file)
    with open(label_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            cls_id = int(line.split()[0])
            image_classes[label_file.replace(".txt", ".jpg")].add(CLASSES[cls_id])

# Copy original train images first
for img_file in os.listdir(DATA_DIR):
    if not img_file.lower().endswith(".jpg"):
        continue
    src_img = os.path.join(DATA_DIR, img_file)
    src_lbl = os.path.join(LABELS_DIR, img_file.replace(".jpg", ".txt"))
    dst_img = os.path.join(OUTPUT_DIR, "images", img_file)
    dst_lbl = os.path.join(OUTPUT_DIR, "labels", img_file.replace(".jpg", ".txt"))
    shutil.copy(src_img, dst_img)
    shutil.copy(src_lbl, dst_lbl)

# Augment Rats images
rats_images = [img for img, cls_set in image_classes.items() if TARGET_CLASS in cls_set]
print(f"Found {len(rats_images)} Rats images to augment...")

for img_file in rats_images:
    img_path = os.path.join(DATA_DIR, img_file)
    lbl_path = os.path.join(LABELS_DIR, img_file.replace(".jpg", ".txt"))

    image = cv2.imread(img_path)
    if image is None:
        print(f"Warning: cannot read {img_path}")
        continue

    with open(lbl_path, "r") as f:
        label_lines = f.readlines()

    for i in range(AUG_COPIES):
        transformed = augment(image=image)
        aug_img = transformed["image"]

        new_name = img_file.replace(".jpg", f"_aug{i}.jpg")
        new_lbl_name = new_name.replace(".jpg", ".txt")

        out_img_path = os.path.join(OUTPUT_DIR, "images", new_name)
        out_lbl_path = os.path.join(OUTPUT_DIR, "labels", new_lbl_name)

        cv2.imwrite(out_img_path, aug_img)
        with open(out_lbl_path, "w") as f:
            f.writelines(label_lines)

print("✅ Rats augmentation complete.")

# Optional: summary
def count_per_class():
    counts = defaultdict(int)
    for lbl_file in os.listdir(os.path.join(OUTPUT_DIR, "labels")):
        if not lbl_file.endswith(".txt"):
            continue
        with open(os.path.join(OUTPUT_DIR, "labels", lbl_file)) as f:
            seen = set()
            for line in f:
                cls = int(line.split()[0])
                seen.add(CLASSES[cls])
            for c in seen:
                counts[c] += 1
    return counts

counts = count_per_class()
print("\n=== BALANCED TRAIN SUMMARY ===")
for c in CLASSES:
    print(f"{c:15} : {counts.get(c,0)} images")
