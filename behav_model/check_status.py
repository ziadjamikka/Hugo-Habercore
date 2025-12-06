import os
import glob

DATA_DIR = r"G:\data"

def check_dataset_status():
    """Check current dataset status"""
    print("\n" + "="*60)
    print(" Dataset Status Check")
    print("="*60)
    
    splits = ["train", "valid", "test"]
    total_images = 0
    total_labels = 0
    
    for split in splits:
        img_dir = os.path.join(DATA_DIR, split, "images")
        lbl_dir = os.path.join(DATA_DIR, split, "labels")
        
        img_count = len(glob.glob(os.path.join(img_dir, "*.*"))) if os.path.exists(img_dir) else 0
        lbl_count = len(glob.glob(os.path.join(lbl_dir, "*.txt"))) if os.path.exists(lbl_dir) else 0
        
        total_images += img_count
        total_labels += lbl_count
        
        status = "Match" if img_count == lbl_count else " Mismatch"
        print(f"\n{split.upper()}:")
        print(f"  Images: {img_count}")
        print(f"  Labels: {lbl_count}")
        print(f"  Status: {status}")
    
    print(f"\n{'='*60}")
    print(f"TOTAL: {total_images} images, {total_labels} labels")
    print(f"Status: {' All matched' if total_images == total_labels else ' Mismatch detected'}")
    print("="*60)


def check_models():
    """Check available models"""
    print("\n" + "="*60)
    print(" Available Models")
    print("="*60)
    
    model_paths = [
        r"G:\data\runs\behavior_detection\weights\best.pt",
        r"G:\data\improved_models\improved_behavior_model\weights\best.pt",
        r"G:\data\improved_models\pretrained_behavior_model\weights\best.pt",
    ]
    
    found_models = []
    
    for path in model_paths:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            found_models.append((path, size_mb))
            print(f"\n Found: {path}")
            print(f"   Size: {size_mb:.2f} MB")
    
    if not found_models:
        print("\n No trained models found")
        print("   Run: python train_improved.py")
    
    high_res_dirs = glob.glob(os.path.join(DATA_DIR, "high_res_training_*"))
    if high_res_dirs:
        print(f"\n High-Resolution Training Runs: {len(high_res_dirs)}")
        for dir_path in high_res_dirs:
            best_pt = os.path.join(dir_path, "behavior_detection_1280", "weights", "best.pt")
            if os.path.exists(best_pt):
                size_mb = os.path.getsize(best_pt) / (1024 * 1024)
                print(f"    {os.path.basename(dir_path)}: {size_mb:.2f} MB")
    
    print("="*60)


def check_videos():
    """Check available videos"""
    print("\n" + "="*60)
    print("🎥 Available Videos")
    print("="*60)
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    videos = []
    
    for ext in video_extensions:
        videos.extend(glob.glob(os.path.join(DATA_DIR, f"*{ext}")))
    
    if videos:
        print(f"\nFound {len(videos)} video(s):")
        for video in videos:
            size_mb = os.path.getsize(video) / (1024 * 1024)
            print(f"  • {os.path.basename(video)} ({size_mb:.2f} MB)")
    else:
        print("\n No videos found in data directory")
    
    print("="*60)


def check_inference_results():
    """Check inference results"""
    print("\n" + "="*60)
    print(" Inference Results")
    print("="*60)
    
    results_dir = os.path.join(DATA_DIR, "inference_results")
    
    if not os.path.exists(results_dir):
        print("\n No inference results yet")
        print("   Run: python video_inference.py")
    else:
        videos = glob.glob(os.path.join(results_dir, "output_*.mp4"))
        jsons = glob.glob(os.path.join(results_dir, "results_*.json"))
        
        print(f"\nOutput Videos: {len(videos)}")
        print(f"JSON Reports: {len(jsons)}")
        
        if videos:
            print("\nLatest outputs:")
            for video in sorted(videos)[-3:]:
                print(f"  • {os.path.basename(video)}")
    
    print("="*60)


def print_next_steps():
    """Print recommended next steps"""
    print("\n" + "="*60)
    print(" Next Steps")
    print("="*60)
    
    train_imgs = len(glob.glob(os.path.join(DATA_DIR, "train", "images", "*.*")))
    train_lbls = len(glob.glob(os.path.join(DATA_DIR, "train", "labels", "*.txt")))
    
    if train_imgs != train_lbls:
        print("\n1.  Clean the dataset first:")
        print("   python clean_and_prepare.py")
        return
    
    high_res_dirs = glob.glob(os.path.join(DATA_DIR, "high_res_training_*"))
    high_res_model = None
    
    for dir_path in high_res_dirs:
        best_pt = os.path.join(dir_path, "behavior_detection_1280", "weights", "best.pt")
        if os.path.exists(best_pt):
            high_res_model = best_pt
            break
    
    if not high_res_model:
        print("\n1.  Train the improved model:")
        print("   python train_improved.py")
        print("\n2.  Wait for training to complete (2-4 hours with GPU)")
    else:
        print("\n High-resolution model found!")
        print(f"   {high_res_model}")
        print("\n1.  Test on videos:")
        print("   python video_inference.py")
        print("\n2.  Check results in: inference_results/")
    
    print("="*60)


if __name__ == "__main__":
    print("\n" + "="*60)
    print(" Behavior Detection - Status Check")
    print("="*60)
    
    check_dataset_status()
    check_models()
    check_videos()
    check_inference_results()
    print_next_steps()
    
    print("\n" + "="*60)
    print(" Status Check Complete")
    print("="*60 + "\n")
