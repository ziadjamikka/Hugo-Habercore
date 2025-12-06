import cv2
from ultralytics import YOLO


model_path = r'G:\Database\stage2_pests\stage2_surfaces3\weights\epoch37.pt' 
model = YOLO(model_path)


image_path = r'G:\Database\elfar.jpg'  
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")


results = model.predict(source=image_path, conf=0.25, save=False)  # conf=confidence threshold



for result in results:
# result.boxes.xyxy -> bounding boxes
# result.boxes.conf -> confidence
# result.boxes.cls -> class indices
    annotated_frame = result.plot() 


cv2.imshow("YOLO Prediction", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

