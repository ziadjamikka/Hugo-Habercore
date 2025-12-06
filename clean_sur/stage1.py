from ultralytics import YOLO
import torch


model = YOLO("yolo11s.pt")


results = model.train(
    data="data.yaml",  
    project="G:/Database/stage1_pests",     
    name="stage1",                          

    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    workers=0,  
    optimizer="AdamW",
    lr0=0.0003,

    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    flipud=0.0,
    fliplr=0.0,
    mosaic=0.0,
    mixup=0.0,

    save_period=1,
    exist_ok=True,    

    pretrained=True,
)
