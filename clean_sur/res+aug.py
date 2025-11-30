import os
import random
import shutil
from collections import defaultdict
import cv2
import albumentations as A

# -------- CONFIG --------
DATA_DIR = r"D:\DEPIMariam\Final\data\annotated\photos"
LABELS_DIR = r"D:\DEPIMariam\Final\data\annotated\labels\obj_train_data"
OUTPUT_DIR = r"D:\DEPIMariam\Final\data\final_dataset"

CLASSES = ["clean_surface", "dirty_surface", "insect", "Rats", "trash"]

TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

MIN_VAL = 10     # required images per class in val
MIN_TEST = 10    # required images per class in test

SCENE_CLASSES = ["clean_surface", "dirty_surface"]   # augment these ONLY
OBJECT_CLASSES = ["insect", "Rats", "trash"]

# Number of augmented variants to create **per scene image in train**
AUG_PER_SCENE_IMAGE = 1

# Safe augmentations for scene classes (no vertical flip, small rotations only)
augment = A.Compose([
    A.HorizontalFlip(p=0.6),
    A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2),
    A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=10, p=0.5),
    A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
    # small blur occasionally
    A.OneOf([
        A.GaussianBlur(blur_limit=3, p=0.5),
        A.MotionBlur(blur_limit=3, p=0.3),
    ], p=0.2)
], bbox_params=None)  # no bbox augmentation since we keep same labels

# ----------------------------------------

def load_image_label_pairs():
    """List images and classes appearing in them"""
    image_classes = defaultdict(set)

    for label_file in os.listdir(LABELS_DIR):
        if not label_file.endswith(".txt"):
            continue
        label_path = os.path.join(LABELS_DIR, label_file)

        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls_id = int(line.split()[0])
                # guard for unexpected ids
                if 0 <= cls_id < len(CLASSES):
                    image_classes[label_file.replace(".txt", ".jpg")].add(CLASSES[cls_id])

    return image_classes


def ensure_dirs():
    for split in ["train", "val", "test"]:
        for sub in ["images", "labels"]:
            os.makedirs(os.path.join(OUTPUT_DIR, split, sub), exist_ok=True)


def copy_pair(img, split):
    img_src = os.path.join(DATA_DIR, img)
    lbl_src = os.path.join(LABELS_DIR, img.replace(".jpg", ".txt"))

    dst_img = os.path.join(OUTPUT_DIR, split, "images", img)
    dst_lbl = os.path.join(OUTPUT_DIR, split, "labels", img.replace(".jpg", ".txt"))

    shutil.copy(img_src, dst_img)
    shutil.copy(lbl_src, dst_lbl)


def augment_scene_image(img, count, split):
    """Generate augmented images for scene classes ONLY (train split only)"""
    if split != "train":
        return  # do not augment val/test

    img_src = os.path.join(DATA_DIR, img)
    lbl_src = os.path.join(LABELS_DIR, img.replace(".jpg", ".txt"))

    image = cv2.imread(img_src)
    if image is None:
        print(f"Warning: cannot read image {img_src}")
        return

    with open(lbl_src, "r") as f:
        lines = f.readlines()

    for i in range(count):
        transformed = augment(image=image)
        aug_img = transformed["image"]

        # create a unique name and save
        new_name = img.replace(".jpg", f"_aug{i}.jpg")
        new_lbl = new_name.replace(".jpg", ".txt")

        out_img_path = os.path.join(OUTPUT_DIR, split, "images", new_name)
        out_lbl_path = os.path.join(OUTPUT_DIR, split, "labels", new_lbl)

        cv2.imwrite(out_img_path, aug_img)
        with open(out_lbl_path, "w") as f:
            f.writelines(lines)


# ----------------------------------------------------------

# Load mapping
image_classes = load_image_label_pairs()
all_images = list(image_classes.keys())

# STRATIFIED SPLIT at image level
random.shuffle(all_images)

splits = {"train": [], "val": [], "test": []}

# First fill VAL + TEST to meet minimum requirement per class
for cls in CLASSES:
    imgs = [img for img in all_images if cls in image_classes[img]]
    random.shuffle(imgs)

    # how many of this class are already assigned
    already_val = sum(1 for img in splits["val"] if cls in image_classes[img])
    already_test = sum(1 for img in splits["test"] if cls in image_classes[img])

    needed_val = max(0, MIN_VAL - already_val)
    needed_test = max(0, MIN_TEST - already_test)

    # take images for val/test from this class (avoid index error)
    splits["val"].extend(imgs[:needed_val])
    splits["test"].extend(imgs[needed_val:needed_val + needed_test])

# remaining → random split
remaining = [img for img in all_images if img not in splits["val"] and img not in splits["test"]]
random.shuffle(remaining)
n_train = int(len(remaining) * TRAIN_RATIO)
n_val = int(len(remaining) * VAL_RATIO)

splits["train"].extend(remaining[:n_train])
splits["val"].extend(remaining[n_train:n_train + n_val])
splits["test"].extend(remaining[n_train + n_val:])

# Create dirs
ensure_dirs()

# Copy + augment (augment only for scene images in train)
for split in splits:
    for img in splits[split]:
        copy_pair(img, split)

        if split == "train" and any(cls in SCENE_CLASSES for cls in image_classes[img]):
            augment_scene_image(img, count=AUG_PER_SCENE_IMAGE, split=split)

# Summary: count images per split
print("\n--- FINAL SUMMARY ---")
for split in ["train", "val", "test"]:
    imgs_count = len(os.listdir(os.path.join(OUTPUT_DIR, split, 'images')))
    print(f"{split.upper()}: {imgs_count} images")

# Optional: print per-class image counts (image-level presence)
def count_per_class():
    result = {s: defaultdict(int) for s in ["train", "val", "test"]}
    for s in ["train", "val", "test"]:
        imgs = [f for f in os.listdir(os.path.join(OUTPUT_DIR, s, "images")) if f.lower().endswith(".jpg")]
        for img in imgs:
            label_path = os.path.join(OUTPUT_DIR, s, "labels", img.replace(".jpg", ".txt"))
            if not os.path.exists(label_path):
                continue
            seen = set()
            with open(label_path) as fh:
                for line in fh:
                    cls = int(line.split()[0])
                    seen.add(CLASSES[cls])
            for c in seen:
                result[s][c] += 1
    return result

counts = count_per_class()
print()
for s in ["train", "val", "test"]:
    print(f"=== {s} ===")
    for c in CLASSES:
        print(f"  {c:15} : {counts[s].get(c,0)} images")
