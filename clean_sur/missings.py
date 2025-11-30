"""import os

jpg_folder = r"D:\DEPIMariam\Final\data\final_dataset\val\images"
txt_folder = r"D:\DEPIMariam\Final\data\final_dataset\val\labels"

jpg_files = {os.path.splitext(f)[0] for f in os.listdir(jpg_folder) if f.lower().endswith(".jpg")}
txt_files = {os.path.splitext(f)[0] for f in os.listdir(txt_folder) if f.lower().endswith(".txt")}

# Find jpg files that DO NOT have a matching txt
missing_txt = jpg_files - txt_files

if missing_txt:
    print("These JPG files have no matching TXT:")
    for name in sorted(missing_txt):
        print(name)
else:
    print(" All JPG files have matching TXT files!")


"""
import os

jpg_folder = r"D:\DEPIMariam\Final\data\final_dataset\train\images"
txt_folder = r"D:\DEPIMariam\Final\data\final_dataset\train\labels"

jpg_files = {os.path.splitext(f)[0] for f in os.listdir(jpg_folder) if f.lower().endswith(".jpg")}
txt_files = {os.path.splitext(f)[0] for f in os.listdir(txt_folder) if f.lower().endswith(".txt")}

# Find TXT files that DO NOT have a matching JPG
missing_jpg = txt_files - jpg_files

if missing_jpg:
    print("These TXT files refer to missing JPG images:")
    for name in sorted(missing_jpg):
        print(name)
else:
    print(" All TXT files have matching JPG files!")

