import os

DATA_DIR = r"data"

folders = ["train", "valid", "test"]

for folder in folders:
    images_path = os.path.join(DATA_DIR, folder, "images")
    labels_path = os.path.join(DATA_DIR, folder, "labels")
    
    num_images = len([f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    num_labels = len([f for f in os.listdir(labels_path) if f.lower().endswith('.txt')]) if os.path.exists(labels_path) else 0
    
    print(f"📂 {folder} folder:")
    print(f"   🖼 Images: {num_images}")
    print(f"   🏷 Labels: {num_labels}\n")
