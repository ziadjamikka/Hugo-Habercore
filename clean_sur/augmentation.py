import os
import cv2
import albumentations as A

image_folder = r"G:\Database\train\images"
label_folder = r"G:\Database\train\labels"

output_image_folder = r"G:\Database\train_aug\images"
output_label_folder = r"G:\Database\train_aug\labels"

os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_label_folder, exist_ok=True)


transform = A.Compose(
    [
        A.HorizontalFlip(p=1.0),

        A.Rotate(limit=5, border_mode=cv2.BORDER_REFLECT101, p=1.0),

        A.ShiftScaleRotate(
            shift_limit=0.03,
            scale_limit=0.07,
            rotate_limit=0,
            border_mode=cv2.BORDER_REFLECT101,
            p=1.0
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.10,
            p=1.0
        ),

        A.HueSaturationValue(
            hue_shift_limit=5,
            sat_shift_limit=10,
            val_shift_limit=10,
            p=1.0
        ),

        A.GaussianBlur(blur_limit=(1,1), p=0.05),
        A.GaussNoise(var_limit=(2,5), p=0.05),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.01  
    ),
)

augmentations_per_image = 2  

for img_name in os.listdir(image_folder):
    if not img_name.lower().endswith((".jpg", ".jpeg")):
        continue

    image_path = os.path.join(image_folder, img_name)
    label_path = os.path.join(label_folder, img_name.replace(".jpg", ".txt"))

    if not os.path.exists(label_path):
        open(label_path, "w").close()

    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    boxes = []
    classes = []

    with open(label_path, "r") as f:
        for line in f.readlines():
            if line.strip() == "":
                continue
            cls, x, y, bw, bh = line.strip().split()
            boxes.append([float(x), float(y), float(bw), float(bh)])
            classes.append(int(cls))

    for i in range(augmentations_per_image):
        augmented = transform(
            image=image,
            bboxes=boxes,
            class_labels=classes
        )

        aug_img = augmented["image"]
        aug_bboxes = augmented["bboxes"]
        aug_classes = augmented["class_labels"]
        if len(aug_bboxes) == 0:
            continue
        new_name = img_name.replace(".jpg", f"_aug{i}.jpg")
        new_label = img_name.replace(".jpg", f"_aug{i}.txt")

        cv2.imwrite(os.path.join(output_image_folder, new_name), aug_img)

        with open(os.path.join(output_label_folder, new_label), "w") as f:
            for cls, bbox in zip(aug_classes, aug_bboxes):
                x, y, bw, bh = bbox
                f.write(f"{cls} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")

print(f" Augmentation completed: {augmentations_per_image} per image for all images!")



