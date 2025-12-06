"""
YOLO Stage-2 Fine-Tuning Script for Kitchen PPE Detection
Improves model accuracy based on Stage-1 results
"""

from ultralytics import YOLO
import torch
import yaml
import os
from pathlib import Path

class Config:
    BASE_MODEL = "yolo_output/ppe_detection/weights/epoch45.pt"
    DATA_YAML = "data.yaml"

    EPOCHS = 20                     
    BATCH = 8
    IMG = 640

    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    PROJECT = "yolo_output"
    NAME = "ppe_detection_stage2"   


cfg = Config()


def train_stage2():
    print("\n==============================================")
    print("         YOLO Stage-2 Fine-Tuning")
    print("==============================================")

    print(f"Using Stage-1 model: {cfg.BASE_MODEL}")
    print(f"Training for {cfg.EPOCHS} epochs")

    model = YOLO(cfg.BASE_MODEL)

    results = model.train(
        data=cfg.DATA_YAML,
        epochs=cfg.EPOCHS,
        batch=cfg.BATCH,
        imgsz=cfg.IMG,
        device=cfg.DEVICE,

        project=cfg.PROJECT,
        name=cfg.NAME,
        exist_ok=True,

        optimizer="SGD",         
        lr0=0.0005,            
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0004,

        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        translate=0.12,
        scale=0.6,
        fliplr=0.5,
        degrees=5,
        mixup=0.1,           
        mosaic=0.8,             
        close_mosaic=15,

        pretrained=True,
        cos_lr=True,           
        warmup_epochs=2,
        patience=20,
        amp=True,
        seed=0,
        deterministic=True,
        overlap_mask=True,
        dropout=0.05,           
        val=True,
        save=True,
        save_period=1,
        plots=True
    )

    print("\nTraining Complete!")
    print("##############################################")

    best_model_path = Path(cfg.PROJECT) / cfg.NAME / "weights" / "best.pt"
    print(f"Stage-2 Best Model: {best_model_path}\n")

    return best_model_path


if __name__ == "__main__":
    train_stage2()
