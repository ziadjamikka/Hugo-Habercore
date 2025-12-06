import os
from collections import defaultdict

DATA_ROOT = "Database"  
splits = ["train", "val", "test" ]
classes = ["clean_surface", "dirty_surface", "Rats", "insect"]


images_per_class = {s: defaultdict(int) for s in splits}
instances_per_class = {s: defaultdict(int) for s in splits}
total_images = {s: 0 for s in splits}

for s in splits:
    imgs_dir = os.path.join(DATA_ROOT, s, "images")
    labs_dir = os.path.join(DATA_ROOT, s, "labels")
    if not os.path.isdir(imgs_dir):
        print(f"Warning: missing {imgs_dir}")
        continue
    files = [f for f in os.listdir(imgs_dir) if f.lower().endswith((".jpg",".jpeg",".png"))]
    total_images[s] = len(files)
    for img in files:
        base = os.path.splitext(img)[0]
        lab_path = os.path.join(labs_dir, base + ".txt")
        if not os.path.exists(lab_path):
            continue
        seen_classes = set()
        with open(lab_path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                cls = int(parts[0])
                instances_per_class[s][classes[cls]] += 1
                seen_classes.add(classes[cls])
        for c in seen_classes:
            images_per_class[s][c] += 1

print("TOTAL IMAGES per split:", total_images)
print()
for s in splits:
    print("=== SPLIT:", s, " (", total_images[s], "images ) ===")
    for c in classes:
        imgs = images_per_class[s].get(c, 0)
        inst = instances_per_class[s].get(c, 0)
        print(f"  {c:15s} | images: {imgs:4d} | instances: {inst:4d}")
    print()
