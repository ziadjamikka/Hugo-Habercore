import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import json
import os

def main():
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU")

    model = YOLO("yolo11s.pt")

    results = model.train(
        data=r"data.yaml",
        epochs=120,
        imgsz=640,
        batch=8,
        device=0,
        augment=True,      
        mosaic=0.8,
        mixup=0.2,
        copy_paste=0.1,
        erasing=0.4,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.8,
        shear=5.0,
        fliplr=0.5,
        flipud=0.1,
        perspective=0.0008,
        optimizer="AdamW",
        lr0=0.001,
        dropout=0.2,
        save_period=1,
        project="G:/Database/runs/",
        name="behavior_aug_train",
        pretrained=True
    )


    results_file = "G:/Database/runs/behavior_aug_train/results.json"
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            history = json.load(f)

        train_loss = history["train/box_loss"]
        val_loss = history["val/box_loss"]

        plt.figure(figsize=(10, 5))
        plt.plot(train_loss, label="Train Loss")
        plt.plot(val_loss, label="Val Loss")
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
    else:
        print("Results file not found:", results_file)

if __name__ == "__main__":
    main()
