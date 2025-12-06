import os
import shutil
from pathlib import Path

DATA_DIR = r"G:\data"

def remove_empty_labels():
    """Remove images with empty label files"""
    print("\n" + "="*50)
    print("🧹 Cleaning Empty Labels")
    print("="*50)
    
    removed_count = 0
    
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(DATA_DIR, split, "images")
        lbl_dir = os.path.join(DATA_DIR, split, "labels")
        
        if not os.path.exists(lbl_dir):
            continue
            
        for lbl_file in os.listdir(lbl_dir):
            if not lbl_file.endswith(".txt"):
                continue
                
            lbl_path = os.path.join(lbl_dir, lbl_file)
            
            with open(lbl_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            
            if not content:
                os.remove(lbl_path)
                
                base_name = os.path.splitext(lbl_file)[0]
                for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                    img_path = os.path.join(img_dir, base_name + ext)
                    if os.path.exists(img_path):
                        os.remove(img_path)
                        removed_count += 1
                        print(f"   Removed: {split}/{base_name}")
                        break
    
    print(f"\n Removed {removed_count} empty images")
    return removed_count


def validate_and_fix_boxes(min_box_size=0.01):
    """Validate and fix bounding boxes (remove too small boxes)"""
    print("\n" + "="*50)
    print(" Validating and Fixing Bounding Boxes")
    print("="*50)
    
    stats = {
        "total_boxes": 0,
        "fixed_boxes": 0,
        "removed_small": 0,
        "fixed_coords": 0
    }
    
    for split in ["train", "valid", "test"]:
        lbl_dir = os.path.join(DATA_DIR, split, "labels")
        
        if not os.path.exists(lbl_dir):
            continue
            
        for lbl_file in os.listdir(lbl_dir):
            if not lbl_file.endswith(".txt"):
                continue
                
            lbl_path = os.path.join(lbl_dir, lbl_file)
            
            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            new_lines = []
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                cls, xc, yc, w, h = parts
                cls = int(cls)
                xc, yc, w, h = float(xc), float(yc), float(w), float(h)
                
                stats["total_boxes"] += 1
                
                original = (xc, yc, w, h)
                xc = max(0.0, min(1.0, xc))
                yc = max(0.0, min(1.0, yc))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))
                
                if (xc, yc, w, h) != original:
                    stats["fixed_coords"] += 1
                
                if w < min_box_size or h < min_box_size:
                    stats["removed_small"] += 1
                    continue
                
                new_lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                stats["fixed_boxes"] += 1
            
            with open(lbl_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
    
    print(f"\n Results:")
    print(f"  • Total boxes: {stats['total_boxes']}")
    print(f"  • Boxes kept: {stats['fixed_boxes']}")
    print(f"  • Small boxes removed: {stats['removed_small']}")
    print(f"  • Coordinates fixed: {stats['fixed_coords']}")
    
    return stats


def update_dataset_yaml():
    """Update dataset.yaml with correct paths"""
    print("\n" + "="*50)
    print(" Updating dataset.yaml")
    print("="*50)
    
    yaml_path = os.path.join(DATA_DIR, "dataset.yaml")
    
    yaml_content = f"""# Dataset Configuration for Behavior Detection
# Updated for higher resolution training

# Paths
train: {os.path.join(DATA_DIR, 'train', 'images')}
val: {os.path.join(DATA_DIR, 'valid', 'images')}
test: {os.path.join(DATA_DIR, 'test', 'images')}

# Classes
nc: 3
names: ['eating', 'face_touching', 'smoking']

# Training settings (optional - can be overridden in training script)
# These are optimized for small object detection
"""
    
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    
    print(f" Updated: {yaml_path}")


def print_final_stats():
    """Print final dataset statistics"""
    print("\n" + "="*50)
    print(" Final Dataset Statistics")
    print("="*50)
    
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(DATA_DIR, split, "images")
        lbl_dir = os.path.join(DATA_DIR, split, "labels")
        
        img_count = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(img_dir) else 0
        lbl_count = len([f for f in os.listdir(lbl_dir) if f.endswith('.txt')]) if os.path.exists(lbl_dir) else 0
        
        print(f"\n{split.upper()}:")
        print(f"  • Images: {img_count}")
        print(f"  • Labels: {lbl_count}")
        print(f"  • Matched: {'done' if img_count == lbl_count else 'mismatch'}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print(" Starting Data Cleaning and Preparation")
    print("="*60)
    
    remove_empty_labels()
    
    validate_and_fix_boxes(min_box_size=0.01)
    
    update_dataset_yaml()
    
    print_final_stats()
    
    print("\n" + "="*60)
    print(" Data Cleaning Completed Successfully!")
    print("="*60)
