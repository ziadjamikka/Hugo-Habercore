import os
import yaml
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans

DATA_DIR = r"/data"
BEST_MODEL_PATH = r"runs\behavior_detection\weights\best.ptt"
OUTPUT_MODEL_DIR = os.path.join(DATA_DIR, "improved_models")
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

def clean_labels(min_area_ratio=0.002):
    print("\n Starting label cleaning...")

    stats = {"fixed": 0, "ignored_small": 0, "invalid": 0}

    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(DATA_DIR, split, "images")
        lbl_dir = os.path.join(DATA_DIR, split, "labels")

        for lbl_file in os.listdir(lbl_dir):
            if not lbl_file.endswith(".txt"):
                continue

            img_path = os.path.join(img_dir, lbl_file.replace(".txt", ".jpg"))
            if not os.path.exists(img_path):
                img_path = img_path.replace(".jpg", ".png")
            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            area_img = w * h

            new_lines = []
            with open(os.path.join(lbl_dir, lbl_file), "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    stats["invalid"] += 1
                    continue

                cls, xc, yc, bw, bh = map(float, parts)

                xc, yc = max(0, min(xc, 1)), max(0, min(yc, 1))
                bw, bh = max(0, min(bw, 1)), max(0, min(bh, 1))

                box_area = (bw * w) * (bh * h)

                if box_area < area_img * min_area_ratio:
                    stats["ignored_small"] += 1
                    continue

                new_lines.append(f"{int(cls)} {xc} {yc} {bw} {bh}\n")
                stats["fixed"] += 1

            with open(os.path.join(lbl_dir, lbl_file), "w") as f:
                f.writelines(new_lines)

    print(f"\n Label cleaning results:")
    print(f"    Fixed boxes: {stats['fixed']}")
    print(f"    Ignored small boxes: {stats['ignored_small']}")
    print(f"    Invalid entries removed: {stats['invalid']}")
    return stats


def compute_best_anchors(k=9):
    print("\n Calculating optimal anchors...")

    boxes = []

    for split in ["train", "valid"]:
        lbl_dir = os.path.join(DATA_DIR, split, "labels")

        for lbl_file in os.listdir(lbl_dir):
            if lbl_file.endswith(".txt"):
                with open(os.path.join(lbl_dir, lbl_file), "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            _, _, _, bw, bh = map(float, parts)
                            boxes.append([bw, bh])

    boxes = np.array(boxes)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(boxes)
    anchors = np.round(kmeans.cluster_centers_ * 640).astype(int)

    print(" Best anchors:\n", anchors)
    return anchors.tolist()


def update_yaml(anchors):
    yaml_path = os.path.join(DATA_DIR, "dataset.yaml")

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    data["anchors"] = anchors

    with open(yaml_path, "w") as f:
        yaml.dump(data, f)

    print(" dataset.yaml updated with new anchors")


def train_updated_model():
    print("\n Starting fine-tuning with improved settings...")

    model = YOLO(BEST_MODEL_PATH)

    output_name = "improved_behavior_model"

    model.train(
        data=os.path.join(DATA_DIR, "dataset.yaml"),
        epochs=20,
        batch=16,
        imgsz=640,
         workers=0,

        device=0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
        project=OUTPUT_MODEL_DIR,
        name=output_name,
        exist_ok=True,
        augment=True,
        hsv_h=0.02,
        hsv_s=0.4,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.0,
        scale=0.5,       
        shear=0.0,
        flipud=0.0,
        fliplr=0.3,
        mosaic=0.0,      
        copy_paste=0.0,   

        save=True,
        save_period=1
    )

    print(f"\n New improved model saved in: {OUTPUT_MODEL_DIR}/{output_name}")


if __name__ == "__main__":
    clean_labels()
    anchors = compute_best_anchors()
    update_yaml(anchors)
    train_updated_model()

    print("\n Model update pipeline completed successfully!")
