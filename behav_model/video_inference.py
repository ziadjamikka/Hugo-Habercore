import cv2
import os
import json
from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime

# Configuration
# Update this path after training completes
MODEL_PATH = r"G:\data\improved_models\improved_behavior_model\weights\best.pt"
VIDEO_PATH = r"G:\data\VideoProject1.mp4"  # Change this to your video path
OUTPUT_DIR = r"G:\data\inference_results"
CONFIDENCE_THRESHOLD = 0.20  # Lowered for better recall on small objects

# Class names
CLASS_NAMES = {
    0: "eating",
    1: "face_touching", 
    2: "smoking"
}

# Colors for visualization (BGR format)
COLORS = {
    0: (0, 255, 0),      # Green for eating
    1: (255, 0, 0),      # Blue for face_touching
    2: (0, 0, 255)       # Red for smoking
}


def process_video(video_path, model_path, conf_threshold=0.25, save_video=True, show_video=False):
    """
    Process video and detect behaviors
    
    Args:
        video_path: Path to input video
        model_path: Path to YOLO model
        conf_threshold: Confidence threshold for detections
        save_video: Whether to save output video
        show_video: Whether to display video during processing
    
    Returns:
        Dictionary with detection statistics
    """
    
    print("\n" + "="*60)
    print("🎥 Starting Video Inference")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        print("Please train the model first or update MODEL_PATH")
        return None
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"❌ Error: Video not found at {video_path}")
        return None
    
    # Load model
    print(f"📦 Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Open video
    print(f"📹 Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Error: Cannot open video file!")
        return None
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Video Info:")
    print(f"   • Resolution: {width}x{height}")
    print(f"   • FPS: {fps}")
    print(f"   • Total Frames: {total_frames}")
    print(f"   • Duration: {total_frames/fps:.2f} seconds")
    
    # Prepare output video writer
    video_writer = None
    if save_video:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"output_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"💾 Output video will be saved to: {output_path}")
    
    # Initialize statistics
    detection_stats = {
        "eating": {"count": 0, "confidences": []},
        "face_touching": {"count": 0, "confidences": []},
        "smoking": {"count": 0, "confidences": []}
    }
    
    frame_detections = []  # Store detections per frame
    frame_count = 0
    
    print(f"\n🔍 Processing video with confidence threshold: {conf_threshold}")
    print("="*60)
    
    # Process video frame by frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run inference
        results = model(frame, conf=conf_threshold, verbose=False)
        
        # Process detections
        frame_detection_info = {
            "frame": frame_count,
            "detections": []
        }
        
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = CLASS_NAMES[cls_id]
            
            # Update statistics
            detection_stats[class_name]["count"] += 1
            detection_stats[class_name]["confidences"].append(conf)
            
            # Store detection info
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            frame_detection_info["detections"].append({
                "class": class_name,
                "confidence": round(conf, 3),
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })
            
            # Draw bounding box
            color = COLORS[cls_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with confidence
            label = f"{class_name}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw background for text
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw text
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        frame_detections.append(frame_detection_info)
        
        # Add frame counter and stats overlay
        info_text = f"Frame: {frame_count}/{total_frames} | Eating: {detection_stats['eating']['count']} | Face Touch: {detection_stats['face_touching']['count']} | Smoking: {detection_stats['smoking']['count']}"
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save frame to output video
        if video_writer:
            video_writer.write(frame)
        
        # Display frame
        if show_video:
            cv2.imshow("Behavior Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⚠️ Processing interrupted by user")
                break
        
        # Progress indicator
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)", end='\r')
    
    # Cleanup
    cap.release()
    if video_writer:
        video_writer.release()
    if show_video:
        cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("✅ Video Processing Complete!")
    print("="*60)
    
    # Calculate average confidences
    for class_name in detection_stats:
        if detection_stats[class_name]["confidences"]:
            avg_conf = sum(detection_stats[class_name]["confidences"]) / len(detection_stats[class_name]["confidences"])
            detection_stats[class_name]["avg_confidence"] = round(avg_conf, 3)
            detection_stats[class_name]["max_confidence"] = round(max(detection_stats[class_name]["confidences"]), 3)
            detection_stats[class_name]["min_confidence"] = round(min(detection_stats[class_name]["confidences"]), 3)
        else:
            detection_stats[class_name]["avg_confidence"] = 0.0
            detection_stats[class_name]["max_confidence"] = 0.0
            detection_stats[class_name]["min_confidence"] = 0.0
    
    # Print summary
    print("\n📊 Detection Summary:")
    print("="*60)
    for class_name in ["eating", "face_touching", "smoking"]:
        stats = detection_stats[class_name]
        print(f"\n{class_name.upper()}:")
        print(f"  • Total Detections: {stats['count']}")
        print(f"  • Average Confidence: {stats['avg_confidence']:.3f}")
        print(f"  • Max Confidence: {stats['max_confidence']:.3f}")
        print(f"  • Min Confidence: {stats['min_confidence']:.3f}")
    
    # Save results to JSON
    if save_video:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(OUTPUT_DIR, f"results_{timestamp}.json")
        
        results_data = {
            "video_info": {
                "path": video_path,
                "resolution": f"{width}x{height}",
                "fps": fps,
                "total_frames": total_frames,
                "duration_seconds": round(total_frames/fps, 2)
            },
            "model_info": {
                "path": model_path,
                "confidence_threshold": conf_threshold
            },
            "detection_summary": detection_stats,
            "frame_detections": frame_detections
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {json_path}")
    
    return detection_stats


def batch_process_videos(video_folder, model_path, conf_threshold=0.25):
    """
    Process multiple videos in a folder
    
    Args:
        video_folder: Path to folder containing videos
        model_path: Path to YOLO model
        conf_threshold: Confidence threshold
    
    Returns:
        Dictionary with results for all videos
    """
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    video_files = [f for f in os.listdir(video_folder) 
                   if os.path.splitext(f)[1].lower() in video_extensions]
    
    if not video_files:
        print(f"❌ No video files found in {video_folder}")
        return None
    
    print(f"\n📁 Found {len(video_files)} video(s) to process")
    
    all_results = {}
    
    for i, video_file in enumerate(video_files, 1):
        video_path = os.path.join(video_folder, video_file)
        print(f"\n{'='*60}")
        print(f"Processing video {i}/{len(video_files)}: {video_file}")
        print(f"{'='*60}")
        
        results = process_video(video_path, model_path, conf_threshold, 
                               save_video=True, show_video=False)
        
        if results:
            all_results[video_file] = results
    
    return all_results


if __name__ == "__main__":
    # Single video processing
    results = process_video(
        video_path=VIDEO_PATH,
        model_path=MODEL_PATH,
        conf_threshold=CONFIDENCE_THRESHOLD,
        save_video=True,
        show_video=False  # Set to True if you want to see the video while processing
    )
    
    # Uncomment below to process multiple videos in a folder
    # batch_results = batch_process_videos(
    #     video_folder=r"G:\data\videos",
    #     model_path=MODEL_PATH,
    #     conf_threshold=CONFIDENCE_THRESHOLD
    # )
