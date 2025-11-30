import cv2
from ultralytics import YOLO


model = YOLO(r"weights\best.pt")

video_path = r"VideoProject1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("[ERROR] Cannot open video file!")
    exit()


ACTIONS = {
    "face_touching": "Face Touching",
    "eating": "Eating",
    "smoking": "Smoking"
}

action_counts = {
    "face_touching": 0,
    "eating": 0,
    "smoking": 0
}

COLOR = (0, 255, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ended.")
        break

    results = model(frame, conf=0.45)

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id]

        if class_name in ACTIONS:
            action_counts[class_name] += 1

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, 2)

            label = f"{ACTIONS[class_name]} ({conf:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR, 2)

            print(f"[DETECTED] {ACTIONS[class_name]} | conf={conf:.2f}")

    cv2.imshow("Real-Time Action Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n==========================")
print("     ACTION SUMMARY")
print("==========================")
print(f"Face Touching: {action_counts['face_touching']}")
print(f"Eating:        {action_counts['eating']}")
print(f"Smoking:       {action_counts['smoking']}")
print("==========================\n")
