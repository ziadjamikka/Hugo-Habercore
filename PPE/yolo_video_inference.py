"""
YOLO Video Inference Script
Process videos with trained YOLO model
"""

from ultralytics import YOLO
import cv2
import argparse
from pathlib import Path

def process_video(model_path, video_path, output_path=None, conf=0.25, show=False):
    """Process video with YOLO model"""
    
    print("="*60)
    print("YOLO Video Inference")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Video: {video_path}")
    print(f"Confidence: {conf}")
    print("="*60)
    
    # Load model
    model = YOLO(model_path)
    
    # Process video
    results = model.predict(
        source=video_path,
        save=True,
        conf=conf,
        iou=0.45,
        show=show,
        stream=True,
        verbose=True
    )
    
    # Process results
    for r in results:
        pass  # Results are automatically saved
    
    print("\n✅ Video processing complete!")


def process_webcam(model_path, conf=0.25):
    """Process webcam feed"""
    
    print("="*60)
    print("YOLO Webcam Inference")
    print("="*60)
    print(f"Model: {model_path}")
    print("Press 'q' to quit")
    print("="*60)
    
    # Load model
    model = YOLO(model_path)
    
    # Process webcam
    results = model.predict(
        source=0,
        conf=conf,
        iou=0.45,
        show=True,
        stream=True,
        verbose=False
    )
    
    for r in results:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    parser = argparse.ArgumentParser(description='YOLO Video Inference')
    
    parser.add_argument(
        '--model',
        type=str,
        default='yolo_output/ppe_detection/weights/best.pt',
        help='Path to YOLO model'
    )
    
    parser.add_argument(
        '--video',
        type=str,
        help='Path to input video'
    )
    
    parser.add_argument(
        '--webcam',
        action='store_true',
        help='Use webcam'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold'
    )
    
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show video while processing'
    )
    
    args = parser.parse_args()
    
    if args.webcam:
        process_webcam(args.model, args.conf)
    elif args.video:
        process_video(args.model, args.video, conf=args.conf, show=args.show)
    else:
        print("Error: Specify --video or --webcam")


if __name__ == '__main__':
    main()
