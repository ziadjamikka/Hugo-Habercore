"""
YOLO Training Script for Kitchen PPE Detection
Simple and efficient training with YOLOv8
"""

from ultralytics import YOLO
import torch
import yaml
import os
from pathlib import Path

class Config:
    MODEL_SIZE = 'm'  # n, s, m, l, x (m is balanced)
    
    EPOCHS = 50
    BATCH_SIZE = 8
    IMAGE_SIZE = 640
    
    DEVICE = 0 if torch.cuda.is_available() else 'cpu'
    
    DATA_YAML = 'data.yaml'
    
    PROJECT = 'yolo_output'
    NAME = 'ppe_detection'

config = Config()

print("="*60)
print("YOLO Training - Kitchen PPE Detection")
print("="*60)
print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Model: YOLOv8{config.MODEL_SIZE}")
print(f"Epochs: {config.EPOCHS}")
print(f"Batch Size: {config.BATCH_SIZE}")
print(f"Image Size: {config.IMAGE_SIZE}")
print("="*60)


def train():
    """Train YOLO model"""
    
    print("\nLoading YOLOv8 model...")
    model = YOLO(f'yolov8{config.MODEL_SIZE}.pt')
    
    print("Model loaded successfully!")
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    results = model.train(
        data=config.DATA_YAML,
        epochs=config.EPOCHS,
        batch=config.BATCH_SIZE,
        imgsz=config.IMAGE_SIZE,
        device=config.DEVICE,
        project=config.PROJECT,
        name=config.NAME,
        
        optimizer='AdamW',
        workers=2,
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        
        patience=10,
        save=True,
        save_period=1,
        cache=False,
        exist_ok=True,
        pretrained=True,
        verbose=True,
        seed=0,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        amp=True,
        fraction=1.0,
        profile=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        plots=True
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    best_model = Path(config.PROJECT) / config.NAME / 'weights' / 'best.pt'
    last_model = Path(config.PROJECT) / config.NAME / 'weights' / 'last.pt'
    
    print(f"\nBest model: {best_model}")
    print(f"Last model: {last_model}")
    print(f"Results: {Path(config.PROJECT) / config.NAME}")
    
    print("\n" + "="*60)
    print("Validating Best Model")
    print("="*60)
    
    model = YOLO(str(best_model))
    metrics = model.val()
    
    print(f"\nValidation Results:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    
    return model, results


def test_model(model_path=None):
    """Test model on test set"""
    
    if model_path is None:
        model_path = Path(config.PROJECT) / config.NAME / 'weights' / 'best.pt'
    
    print("\n" + "="*60)
    print("Testing Model")
    print("="*60)
    
    model = YOLO(str(model_path))
    
    test_images = Path('test/images')
    if test_images.exists():
        results = model.predict(
            source=str(test_images),
            save=True,
            save_txt=True,
            save_conf=True,
            conf=0.25,
            iou=0.45,
            project=config.PROJECT,
            name=f'{config.NAME}_test',
            exist_ok=True
        )
        
        print(f"\nTest predictions saved to: {Path(config.PROJECT) / f'{config.NAME}_test'}")
    else:
        print("Test images folder not found!")


def export_model(model_path=None, format='onnx'):
    """Export model to different formats"""
    
    if model_path is None:
        model_path = Path(config.PROJECT) / config.NAME / 'weights' / 'best.pt'
    
    print("\n" + "="*60)
    print(f"Exporting Model to {format.upper()}")
    print("="*60)
    
    model = YOLO(str(model_path))
    
    # Export
    model.export(format=format)
    
    print(f"\nModel exported successfully!")


def main():
    """Main function"""
    
    if not os.path.exists(config.DATA_YAML):
        print(f"Error: {config.DATA_YAML} not found!")
        return
    
    with open(config.DATA_YAML, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"\nDataset Configuration:")
    print(f"  Classes: {data_config['names']}")
    print(f"  Number of classes: {data_config['nc']}")
    print(f"  Train: {data_config.get('train', 'N/A')}")
    print(f"  Val: {data_config.get('val', 'N/A')}")
    print(f"  Test: {data_config.get('test', 'N/A')}")
    
    model, results = train()
    
    test_model()
    
    print("\n" + "="*60)
    print("Export Options")
    print("="*60)
    print("To export model, run:")
    print(f"  python train_yolo.py --export onnx")
    print(f"  python train_yolo.py --export torchscript")
    print(f"  python train_yolo.py --export engine")
    
    print("\n" + "="*60)
    print("All Done! 🎉")
    print("="*60)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--export':
        format = sys.argv[2] if len(sys.argv) > 2 else 'onnx'
        export_model(format=format)
    else:
        main()


