import os
import csv
from datetime import datetime
from ultralytics import YOLO
import argparse
import cv2
import torch

cv2.setNumThreads(1)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

DATA_DIR = r"C:\Users\jamikka_\Desktop\data"
RUNS = [
    r"G:\data\improved_models\improved_behavior_model",
    r"G:\data\improved_models_20251129_122246\improved_behavior_model",
]

def read_results(path):
    rows = []
    if not os.path.isfile(path):
        return rows, []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            if not headers:
                return rows, []
            for r in reader:
                if len(r) != len(headers):
                    continue
                rows.append({headers[i]: r[i] for i in range(len(headers))})
            return rows, headers
    except Exception:
        return rows, []

def best_epoch(rows, headers):
    if not rows or not headers:
        return None
    mcols = [h for h in headers if "mAP50-95" in h]
    if mcols:
        col = mcols[0]
        best = max(rows, key=lambda r: float(r.get(col, 0) or 0))
    else:
        m50 = [h for h in headers if "mAP50" in h]
        if m50:
            col = m50[0]
            best = max(rows, key=lambda r: float(r.get(col, 0) or 0))
        else:
            vbl = [h for h in headers if "val/box_loss" in h]
            if vbl:
                col = vbl[0]
                best = min(rows, key=lambda r: float(r.get(col, 1e9) or 1e9))
            else:
                return None
    try:
        ep = int(float(best.get("epoch", 0)))
    except Exception:
        ep = None
    return {
        "epoch": ep,
        "row": best,
    }

def find_weights(run_dir, epoch):
    wdir = os.path.join(run_dir, "weights")
    if epoch is not None:
        cand = os.path.join(wdir, f"epoch{epoch}.pt")
        if os.path.isfile(cand):
            return cand
    b = os.path.join(wdir, "best.pt")
    if os.path.isfile(b):
        return b
    l = os.path.join(wdir, "last.pt")
    if os.path.isfile(l):
        return l
    return None

def select_best_run(run_dirs):
    picks = []
    for rd in run_dirs:
        rcsv = os.path.join(rd, "results.csv")
        rows, headers = read_results(rcsv)
        be = best_epoch(rows, headers)
        ep = be["epoch"] if be else None
        wp = find_weights(rd, ep)
        metric_val = None
        if be and headers:
            mcols = [h for h in headers if "mAP50-95" in h]
            if mcols:
                metric_val = float(be["row"].get(mcols[0], 0) or 0)
            else:
                m50 = [h for h in headers if "mAP50" in h]
                if m50:
                    metric_val = float(be["row"].get(m50[0], 0) or 0)
        picks.append({"run": rd, "epoch": ep, "weights": wp, "metric": metric_val})
    best_pick = None
    for p in picks:
        if best_pick is None:
            best_pick = p
        else:
            a = best_pick.get("metric")
            b = p.get("metric")
            if b is not None and (a is None or b > a):
                best_pick = p
    if best_pick and best_pick.get("weights") is None:
        for p in picks:
            if p.get("weights"):
                best_pick = p
                break
    return best_pick, picks

def _device():
    try:
        return 0 if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cpu'

def pretrain(weights_path, output_root, data_yaml, epochs=50, batch=8, imgsz=640):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(output_root, f"pretrained_realtime_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    name = "pretrained_behavior_model"
    m = YOLO(weights_path)
    try:
        m.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=_device(),
            workers=0,
            project=out_dir,
            name=name,
            exist_ok=True,
            save=True,
            save_period=1,
            augment=True,
            hsv_h=0.02,
            hsv_s=0.4,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.0,
            scale=0.4,
            shear=0.0,
            flipud=0.0,
            fliplr=0.3,
            mosaic=0.0,
            copy_paste=0.0,
            cos_lr=True,
            patience=50,
            rect=True,
            cache=False,
        )
    except Exception:
        nb = max(1, batch // 2)
        ni = max(320, imgsz // 2)
        try:
            m.train(
                data=data_yaml,
                epochs=epochs,
                batch=nb,
                imgsz=ni,
                device=_device(),
                workers=0,
                project=out_dir,
                name=name,
                exist_ok=True,
                save=True,
                save_period=1,
                augment=True,
                hsv_h=0.02,
                hsv_s=0.4,
                hsv_v=0.4,
                degrees=0.0,
                translate=0.0,
                scale=0.4,
                shear=0.0,
                flipud=0.0,
                fliplr=0.3,
                mosaic=0.0,
                copy_paste=0.0,
                cos_lr=True,
                patience=50,
                rect=True,
                cache=False,
            )
        except Exception:
            m.train(
                data=data_yaml,
                epochs=epochs,
                batch=1,
                imgsz=256,
                device=_device(),
                workers=0,
                project=out_dir,
                name=name,
                exist_ok=True,
                save=True,
                save_period=1,
                augment=True,
                hsv_h=0.02,
                hsv_s=0.4,
                hsv_v=0.4,
                degrees=0.0,
                translate=0.0,
                scale=0.4,
                shear=0.0,
                flipud=0.0,
                fliplr=0.3,
                mosaic=0.0,
                copy_paste=0.0,
                cos_lr=True,
                patience=50,
                rect=True,
                cache=False,
            )
    best_out = os.path.join(out_dir, name, "weights", "best.pt")
    if os.path.isfile(best_out):
        try:
            m = YOLO(best_out)
            m.export(format="onnx", dynamic=True, simplify=True)
        except Exception:
            pass
    return out_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--resume-dir", type=str, default=None)
    parser.add_argument("--resume-epoch", type=int, default=None)
    parser.add_argument("--remaining-epochs", type=int, default=None)
    args = parser.parse_args()
    data_yaml = os.path.join(DATA_DIR, "dataset.yaml")
    if args.resume_dir:
        w = None
        if args.resume_epoch is not None:
            cand = os.path.join(args.resume_dir, "weights", f"epoch{args.resume_epoch}.pt")
            if os.path.isfile(cand):
                w = cand
        if not w:
            cand = os.path.join(args.resume_dir, "weights", "last.pt")
            if os.path.isfile(cand):
                w = cand
        if not w:
            cand = os.path.join(args.resume_dir, "weights", "best.pt")
            if os.path.isfile(cand):
                w = cand
        if not w:
            print("No resume weights found")
            return
        m = YOLO(w)
        proj = os.path.dirname(args.resume_dir)
        name = os.path.basename(args.resume_dir)
        re = args.remaining_epochs or 1
        try:
            m.train(
                data=data_yaml,
                epochs=re,
                batch=args.batch,
                imgsz=args.imgsz,
                device=_device(),
                workers=0,
                project=proj,
                name=name,
                exist_ok=True,
                save=True,
                save_period=1,
                augment=True,
                hsv_h=0.02,
                hsv_s=0.4,
                hsv_v=0.4,
                degrees=0.0,
                translate=0.0,
                scale=0.4,
                shear=0.0,
                flipud=0.0,
                fliplr=0.3,
                mosaic=0.0,
                copy_paste=0.0,
                cos_lr=True,
                patience=50,
                rect=True,
                cache=False,
            )
        except Exception:
            nb = max(1, args.batch // 2)
            ni = max(320, args.imgsz // 2)
            try:
                m.train(
                    data=data_yaml,
                    epochs=re,
                    batch=nb,
                    imgsz=ni,
                    device=_device(),
                    workers=0,
                    project=proj,
                    name=name,
                    exist_ok=True,
                    save=True,
                    save_period=1,
                    augment=True,
                    hsv_h=0.02,
                    hsv_s=0.4,
                    hsv_v=0.4,
                    degrees=0.0,
                    translate=0.0,
                    scale=0.4,
                    shear=0.0,
                    flipud=0.0,
                    fliplr=0.3,
                    mosaic=0.0,
                    copy_paste=0.0,
                    cos_lr=True,
                    patience=50,
                    rect=True,
                    cache=False,
                )
            except Exception:
                m.train(
                    data=data_yaml,
                    epochs=re,
                    batch=1,
                    imgsz=256,
                    device=_device(),
                    workers=0,
                    project=proj,
                    name=name,
                    exist_ok=True,
                    save=True,
                    save_period=1,
                    augment=True,
                    hsv_h=0.02,
                    hsv_s=0.4,
                    hsv_v=0.4,
                    degrees=0.0,
                    translate=0.0,
                    scale=0.4,
                    shear=0.0,
                    flipud=0.0,
                    fliplr=0.3,
                    mosaic=0.0,
                    copy_paste=0.0,
                    cos_lr=True,
                    patience=50,
                    rect=True,
                    cache=False,
                )
        return
    best, allp = select_best_run(RUNS)
    print("Runs Summary:")
    for p in allp:
        print(f"run={p['run']}, epoch={p['epoch']}, metric={p['metric']}, weights={p['weights']}")
    if not best or not best.get("weights"):
        print("No valid weights found")
        return
    print(f"Selected: run={best['run']}, epoch={best['epoch']}, metric={best['metric']}, weights={best['weights']}")
    if args.no_train:
        return
    out_dir = pretrain(best["weights"], os.path.join(DATA_DIR, "improved_models"), data_yaml, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)
    print(f"Output: {out_dir}")

if __name__ == "__main__":
    main()
