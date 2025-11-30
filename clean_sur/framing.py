import cv2
import os

# Video file
video_path = r'D:\DEPIMariam\Final\data\insects\rats\istockphoto-1989623726-640_adpp_is.mp4'

# Directory to save frames
output_dir = 'ratfar/4'
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

vidcap = cv2.VideoCapture(video_path)
count = 0

while True:
    success, image = vidcap.read()  # Read a frame
    if not success:               # Stop if no more frames
        break
    
    # Save frame in the directory
    frame_path = os.path.join(output_dir, f"frame{count}.jpg")
    cv2.imwrite(frame_path, image)
    
    count += 1

vidcap.release()
cv2.destroyAllWindows()
print(f"Saved {count} frames in '{output_dir}' folder.")
