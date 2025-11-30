from ultralytics import YOLO
import os

def main():
    DATASET_PATH = r"data"

    DATA_YAML = os.path.join(DATASET_PATH, "dataset.yaml")

    if not os.path.exists(DATA_YAML):
        yaml_content = f"""
train: {os.path.join(DATASET_PATH, 'train', 'images')}
val: {os.path.join(DATASET_PATH, 'valid', 'images')}
test: {os.path.join(DATASET_PATH, 'test', 'images')}

nc: 3
names: ['eating', 'face_touching', 'smoking']
"""
        with open(DATA_YAML, "w") as f:
            f.write(yaml_content)
        print(f" dataset.yaml created at {DATA_YAML}")

    MODEL_SIZE = "yolov8s.pt"

    model = YOLO(MODEL_SIZE)



    model.train(
        data=DATA_YAML,
        epochs=100,
        batch=16,
        imgsz=640,
        device=0,
        project=os.path.join(DATASET_PATH, "runs"),
        name="behavior_detection",
        exist_ok=True,
        save_period=10,
        workers=0  
    )


    print("Training started! Check 'runs' folder for progress and best weights.")

if __name__ == "__main__":
    main()
