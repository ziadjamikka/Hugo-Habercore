import torch
from ultralytics import YOLO

if __name__ == "__main__":
    torch.cuda.empty_cache()

    model = YOLO(r"G:\Database\stage1_pests\stage1\weights\last.pt")

    
    results = model.train(
        data=r"data.yaml",
        epochs=50,         
        imgsz=640,
        batch=8,

        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        translate=0.10,
        degrees=5.0,
        scale=0.50,

        mosaic=0.30,

        mixup=0.0,
        copy_paste=0.0,
        erasing=0.0,
        flipud=0.0,
        shear=0.0,
        perspective=0.0,

        optimizer="AdamW",
        lr0=0.0001,     
        dropout=0.10,    

        resume=False,     

        save=True,
        save_period=1,
        project=r"G:/Database/stage2_pests/",
        name="stage2_surfaces",

        pretrained=False,
        cache=True
    )

