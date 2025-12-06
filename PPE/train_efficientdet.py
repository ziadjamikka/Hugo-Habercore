"""
EfficientDet Training Script - PyTorch Implementation
Kitchen PPE Detection with GPU support and augmentation
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import yaml
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Config:
    MODEL_NAME = 'efficientdet-d2'
    NUM_CLASSES = 3
    IMAGE_SIZE = 768  # 768 for D2, 896 for D3
    
    BATCH_SIZE = 8
    EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    BBOX_LOSS_WEIGHT = 0.1  
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    DATA_YAML = 'data.yaml'
    TRAIN_IMAGES = 'train/images'
    TRAIN_LABELS = 'train/labels'
    VALID_IMAGES = 'valid/images'
    VALID_LABELS = 'valid/labels'
    
    OUTPUT_DIR = 'efficientdet_output'
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
    
    NUM_WORKERS = 4

config = Config()

os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

print("="*60)
print("EfficientDet Training - PyTorch")
print("="*60)
print(f"Device: {config.DEVICE}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Model: {config.MODEL_NAME}")
print(f"Image Size: {config.IMAGE_SIZE}")
print(f"Batch Size: {config.BATCH_SIZE}")
print(f"Epochs: {config.EPOCHS}")
print("="*60)


class PPEDataset(Dataset):
    """Dataset for PPE Detection"""
    
    def __init__(self, images_dir, labels_dir, image_size, augment=False):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_size = image_size
        self.augment = augment
        
        self.image_files = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
        
        self.valid_samples = []
        for img_path in self.image_files:
            img_name = os.path.basename(img_path)
            label_name = img_name.replace('.jpg', '.txt')
            label_path = os.path.join(labels_dir, label_name)
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 0:
                        self.valid_samples.append((img_path, label_path))
        
        print(f"Found {len(self.valid_samples)} valid samples")
        
        if augment:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.HueSaturationValue(p=0.3),
                A.RandomGamma(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    
    def __len__(self):
        return len(self.valid_samples)
    
    def yolo_to_pascal(self, yolo_box, img_width, img_height):
        """Convert YOLO format to Pascal VOC format [xmin, ymin, xmax, ymax]"""
        x_center, y_center, width, height = yolo_box
        
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height
        
        xmin = x_center - width / 2
        ymin = y_center - height / 2
        xmax = x_center + width / 2
        ymax = y_center + height / 2
        
        return [xmin, ymin, xmax, ymax]
    
    def __getitem__(self, idx):
        img_path, label_path = self.valid_samples[idx]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width = image.shape[:2]
        
        boxes = []
        classes = []
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    yolo_box = [float(x) for x in parts[1:]]
                    box = self.yolo_to_pascal(yolo_box, img_width, img_height)
                    boxes.append(box)
                    classes.append(class_id)
        
        if len(boxes) > 0:
            box = boxes[0]
            class_id = classes[0]
        else:
            box = [0, 0, img_width, img_height]
            class_id = 0
        
        transformed = self.transform(
            image=image,
            bboxes=[box],
            class_labels=[class_id]
        )
        
        image = transformed['image']
        
        if len(transformed['bboxes']) > 0:
            box = transformed['bboxes'][0]
            class_id = transformed['class_labels'][0]
        else:
            box = [0, 0, self.image_size, self.image_size]
            class_id = 0
        
        box = torch.tensor(box, dtype=torch.float32)
        class_id = torch.tensor(class_id, dtype=torch.long)
        
        return image, box, class_id


class EfficientDetModel(nn.Module):
    """EfficientDet-like model using EfficientNet backbone"""
    
    def __init__(self, num_classes, pretrained=True):
        super(EfficientDetModel, self).__init__()
        
        self.backbone = models.efficientnet_b2(pretrained=pretrained)
        
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Linear(512, num_classes)
        
        self.bbox_regressor = nn.Linear(512, 4)
    
    def forward(self, x):
        features = self.backbone(x)
        features = self.feature_extractor(features)
        
        class_logits = self.classifier(features)
        
        bbox_pred = self.bbox_regressor(features)
        
        return class_logits, bbox_pred


class Trainer:
    """Training manager"""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.classification_loss = nn.CrossEntropyLoss()
        self.bbox_loss = nn.SmoothL1Loss()
        self.bbox_loss_weight = config.BBOX_LOSS_WEIGHT
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # History
        self.history = {
            'train_loss': [],
            'train_cls_loss': [],
            'train_bbox_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_cls_loss': [],
            'val_bbox_loss': [],
            'val_acc': []
        }
        
        self.best_val_loss = float('inf')
    
    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        
        total_loss = 0
        total_cls_loss = 0
        total_bbox_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, boxes, classes in pbar:
            # Move to device
            images = images.to(self.config.DEVICE)
            boxes = boxes.to(self.config.DEVICE)
            classes = classes.to(self.config.DEVICE)
            
            # Forward pass
            class_logits, bbox_pred = self.model(images)
            
            # Calculate losses
            cls_loss = self.classification_loss(class_logits, classes)
            bbox_loss = self.bbox_loss(bbox_pred, boxes)
            loss = cls_loss + (self.bbox_loss_weight * bbox_loss)  # Weighted bbox loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_bbox_loss += bbox_loss.item()
            
            _, predicted = torch.max(class_logits, 1)
            total += classes.size(0)
            correct += (predicted == classes).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls': f'{cls_loss.item():.4f}',
                'bbox': f'{bbox_loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_cls_loss = total_cls_loss / len(self.train_loader)
        avg_bbox_loss = total_bbox_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, avg_cls_loss, avg_bbox_loss, accuracy
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        
        total_loss = 0
        total_cls_loss = 0
        total_bbox_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for images, boxes, classes in pbar:
                # Move to device
                images = images.to(self.config.DEVICE)
                boxes = boxes.to(self.config.DEVICE)
                classes = classes.to(self.config.DEVICE)
                
                # Forward pass
                class_logits, bbox_pred = self.model(images)
                
                # Calculate losses
                cls_loss = self.classification_loss(class_logits, classes)
                bbox_loss = self.bbox_loss(bbox_pred, boxes)
                loss = cls_loss + (self.bbox_loss_weight * bbox_loss)  # Weighted bbox loss
                
                # Statistics
                total_loss += loss.item()
                total_cls_loss += cls_loss.item()
                total_bbox_loss += bbox_loss.item()
                
                _, predicted = torch.max(class_logits, 1)
                total += classes.size(0)
                correct += (predicted == classes).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / total:.2f}%'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        avg_cls_loss = total_cls_loss / len(self.val_loader)
        avg_bbox_loss = total_bbox_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, avg_cls_loss, avg_bbox_loss, accuracy
    
    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(
            self.config.CHECKPOINT_DIR,
            f'model_epoch_{epoch:02d}_loss_{val_loss:.4f}.pth'
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }, checkpoint_path)
        
        print(f" Checkpoint saved: {checkpoint_path}")
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        for epoch in range(self.config.EPOCHS):
            print(f"\nEpoch {epoch + 1}/{self.config.EPOCHS}")
            print("-" * 60)
            
            # Train
            train_loss, train_cls_loss, train_bbox_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_cls_loss, val_bbox_loss, val_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_cls_loss'].append(train_cls_loss)
            self.history['train_bbox_loss'].append(train_bbox_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_cls_loss'].append(val_cls_loss)
            self.history['val_bbox_loss'].append(val_bbox_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch + 1, val_loss)
                
                # Save as best model
                best_model_path = os.path.join(self.config.OUTPUT_DIR, 'best_model.pth')
                torch.save(self.model.state_dict(), best_model_path)
                print(f"🏆 New best model saved!")
        
        # Save final model
        final_model_path = os.path.join(self.config.OUTPUT_DIR, 'final_model.pth')
        torch.save(self.model.state_dict(), final_model_path)
        print(f"\n Final model saved: {final_model_path}")
        
        return self.history


def plot_history(history, output_dir):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Total loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train')
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation')
    axes[0, 1].set_title('Classification Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Classification loss
    axes[1, 0].plot(epochs, history['train_cls_loss'], 'b-', label='Train')
    axes[1, 0].plot(epochs, history['val_cls_loss'], 'r-', label='Validation')
    axes[1, 0].set_title('Classification Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(epochs, history['train_bbox_loss'], 'b-', label='Train')
    axes[1, 1].plot(epochs, history['val_bbox_loss'], 'r-', label='Validation')
    axes[1, 1].set_title('BBox Regression Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n Training plot saved: {plot_path}")
    plt.close()


def main():
    """Main training function"""
    
    # Load YAML config
    with open(config.DATA_YAML, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"\nDataset Classes: {data_config['names']}")
    print(f"Number of Classes: {data_config['nc']}")
    
    # Create datasets
    print("\n" + "="*60)
    print("Loading Datasets")
    print("="*60)
    
    train_dataset = PPEDataset(
        config.TRAIN_IMAGES,
        config.TRAIN_LABELS,
        config.IMAGE_SIZE,
        augment=True  # Enable augmentation for training
    )
    
    val_dataset = PPEDataset(
        config.VALID_IMAGES,
        config.VALID_LABELS,
        config.IMAGE_SIZE,
        augment=False  # No augmentation for validation
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Build model
    print("\n" + "="*60)
    print("Building Model")
    print("="*60)
    
    model = EfficientDetModel(num_classes=config.NUM_CLASSES, pretrained=True)
    model = model.to(config.DEVICE)
    
    print(f"Model moved to: {config.DEVICE}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config)
    
    # Train
    history = trainer.train()
    
    # Plot history
    plot_history(history, config.OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Output directory: {config.OUTPUT_DIR}")
    print(f"Best model: {os.path.join(config.OUTPUT_DIR, 'best_model.pth')}")
    print(f"Final model: {os.path.join(config.OUTPUT_DIR, 'final_model.pth')}")
    print(f"Checkpoints: {config.CHECKPOINT_DIR}")


if __name__ == '__main__':
    main()
