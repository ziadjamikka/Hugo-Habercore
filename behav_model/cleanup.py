import os

DATA_DIR = r"G:\Ai\DEPI\final project\behav\data"

MIN_BOX_SIZE = 0.05 
MAX_BOX_SIZE = 0.9  

def adjust_bbox(xc, yc, w, h):
    w = max(w, MIN_BOX_SIZE)
    h = max(h, MIN_BOX_SIZE)
    w = min(w, MAX_BOX_SIZE)
    h = min(h, MAX_BOX_SIZE)
    return xc, yc, w, h

def process_folder(split):
    images_path = os.path.join(DATA_DIR, split, "images")
    labels_path = os.path.join(DATA_DIR, split, "labels")
    
    num_images = len([f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    empty_label_files = 0
    small_bboxes = 0
    
    if os.path.exists(labels_path):
        for lbl_file in os.listdir(labels_path):
            if lbl_file.endswith(".txt"):
                path = os.path.join(labels_path, lbl_file)
                with open(path, "r") as f:
                    lines = f.readlines()
                
                if len(lines) == 0:
                    empty_label_files += 1
                    continue
                
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, xc, yc, w, h = parts
                        xc, yc, w, h = map(float, (xc, yc, w, h))
                        if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
                            small_bboxes += 1
                        xc, yc, w, h = adjust_bbox(xc, yc, w, h)
                        new_lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                
                with open(path, "w") as f:
                    f.writelines(new_lines)
    
    return num_images, empty_label_files, small_bboxes

folders = ["train", "valid", "test"]
total_images = total_empty = total_small = 0

print(" Dataset Bounding Box Adjustment\n")

for folder in folders:
    images, empty, small = process_folder(folder)
    print(f" {folder} folder:")
    print(f"   🖼 Images: {images}")
    print(f"   ⚠ Empty label files: {empty}")
    print(f"   ⚠ Small bounding boxes detected and adjusted: {small}\n")
    
    total_images += images
    total_empty += empty
    total_small += small

print("════════════════════════════════")
print(f" Total images: {total_images}")
print(f" Total empty label files: {total_empty}")
print(f" Total small bounding boxes adjusted: {total_small}")
print(" Dataset adjustment completed (No files were removed)")

