import cv2
import os
from PIL import Image
import imagehash

# ---------------------------------------
# SETTINGS
# ---------------------------------------

input_frames_dir = r"D:\DEPIMariam\Final\data\ratfar\4"
output_frames_dir = r"D:\DEPIMariam\Final\data\ratfar\4\FilteredFrames"

os.makedirs(output_frames_dir, exist_ok=True)

HASH_THRESHOLD = 5 # Hamming distance threshold (lower = stricter)

# ---------------------------------------
# MAIN LOGIC
# ---------------------------------------

saved_hashes = []     # store hashes of saved frames
count_saved = 0

for filename in sorted(os.listdir(input_frames_dir)):
    if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    frame_path = os.path.join(input_frames_dir, filename)

    # Load image with PIL for hashing
    img = Image.open(frame_path)

    # Compute perceptual hash (pHash)
    phash = imagehash.phash(img)

    # Check similarity to previous saved frames
    is_duplicate = False
    for h in saved_hashes:
        if abs(phash - h) <= HASH_THRESHOLD:
            is_duplicate = True
            break

    # If unique → save it
    if not is_duplicate:
        output_path = os.path.join(output_frames_dir, filename)
        img.save(output_path)
        saved_hashes.append(phash)
        count_saved += 1

print(f"Done! Saved {count_saved} unique frames in '{output_frames_dir}' folder.")
