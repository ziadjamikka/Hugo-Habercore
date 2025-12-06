import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

model_path = r'G:\datasets\yolo_output\ppe_detection_stage2\weights\Epoch19.pt'
model = YOLO(model_path)

video_path = r'G:\datasets\chef2.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f"Video not found: {video_path}")

print("Close the window to stop")

fig, ax = plt.subplots(figsize=(12, 8))
plt.ion()  

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.predict(frame, conf=0.25, save=False, verbose=False)
        
        annotated_frame = results[0].plot()
        
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        ax.clear()
        ax.imshow(annotated_frame_rgb)
        ax.axis('off')
        plt.pause(0.001)
        
        if not plt.fignum_exists(fig.number):
            break

except KeyboardInterrupt:
    print("\nStopped by user")

cap.release()
plt.close('all')

print("Done!")
