import os
import sys
from collections import Counter, defaultdict
import random
import cv2
import numpy as np

def list_files(base, split):
    img_dir = os.path.join(base, split, "images")
    lbl_dir = os.path.join(base, split, "labels")
    img_exts = {".jpg", ".jpeg", ".png"}
    images = []
    if os.path.isdir(img_dir):
        for name in os.listdir(img_dir):
            p = os.path.join(img_dir, name)
            ext = os.path.splitext(p)[1].lower()
            if os.path.isfile(p) and ext in img_exts:
                images.append(p)
    labels = []
    if os.path.isdir(lbl_dir):
        for name in os.listdir(lbl_dir):
            p = os.path.join(lbl_dir, name)
            if os.path.isfile(p) and name.endswith(".txt"):
                labels.append(p)
    return images, labels

def base_name(path):
    return os.path.splitext(os.path.basename(path))[0]

def pairs_stats(images, labels):
    img_bases = set(base_name(p) for p in images)
    lbl_bases = set(base_name(p) for p in labels)
    images_without_label = sum(1 for b in img_bases if b not in lbl_bases)
    labels_without_image = sum(1 for b in lbl_bases if b not in img_bases)
    return {
        "images": len(images),
        "labels": len(labels),
        "images_without_label": images_without_label,
        "labels_without_image": labels_without_image,
    }

def read_label_file(path):
    try:
        lines = open(path, "r", encoding="utf-8", errors="ignore").read().splitlines()
    except Exception:
        return [("bad", None)]
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) != 5:
            out.append(("bad", None))
        else:
            cls = parts[0]
            try:
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                out.append((cls, (xc, yc, w, h)))
            except Exception:
                out.append(("bad", None))
    return out

def class_and_bbox_stats(labels):
    class_counts = Counter()
    bad = 0
    oor = 0
    min_w = 1.0
    min_h = 1.0
    max_w = 0.0
    max_h = 0.0
    per_image_class = defaultdict(list)
    for p in labels:
        bn = base_name(p)
        entries = read_label_file(p)
        for cls, vals in entries:
            if cls == "bad":
                bad += 1
                continue
            class_counts[cls] += 1
            if vals:
                xc, yc, w, h = vals
                if (xc < 0 or xc > 1 or yc < 0 or yc > 1 or w < 0 or w > 1 or h < 0 or h > 1):
                    oor += 1
                if w < min_w:
                    min_w = w
                if h < min_h:
                    min_h = h
                if w > max_w:
                    max_w = w
                if h > max_h:
                    max_h = h
                per_image_class[bn].append(cls)
    return class_counts, {
        "bad": bad,
        "oor": oor,
        "min_w": min_w,
        "min_h": min_h,
        "max_w": max_w,
        "max_h": max_h,
    }, per_image_class

def duplicates_between_splits(bases_a, bases_b):
    return len(bases_a & bases_b)

def balanced_plan(class_counts):
    keys = ["0", "1", "2"]
    maxc = max(class_counts.get(k, 0) for k in keys) if keys else 0
    plan = {}
    for k in keys:
        c = class_counts.get(k, 0)
        plan[k] = {"current": c, "target": maxc, "add": max(0, maxc - c)}
    return plan

def read_image(path):
    return cv2.imread(path)

def write_image(path, img):
    cv2.imwrite(path, img)

def transform_img(img, kind):
    if kind == "hflip":
        return cv2.flip(img, 1)
    if kind == "vflip":
        return cv2.flip(img, 0)
    if kind == "hvflip":
        return cv2.flip(img, -1)
    if kind == "brightness":
        beta = random.randint(-30, 30)
        return cv2.convertScaleAbs(img, alpha=1.0, beta=beta)
    if kind == "contrast":
        alpha = 0.8 + random.random() * 0.4
        return cv2.convertScaleAbs(img, alpha=alpha, beta=0)
    return img

def transform_labels(entries, kind):
    out = []
    for cls, vals in entries:
        if cls == "bad":
            continue
        if not vals:
            continue
        xc, yc, w, h = vals
        if kind in ("hflip", "hvflip"):
            xc = 1.0 - xc
        if kind in ("vflip", "hvflip"):
            yc = 1.0 - yc
        out.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return out

def clean_label_entries(entries, min_w=0.02, min_h=0.02):
    out_lines = []
    stats = {
        "boxes_total": 0,
        "boxes_kept": 0,
        "boxes_dropped": 0,
        "boxes_fixed": 0,
        "unknown_cls": 0,
        "bad_lines": 0,
    }
    for cls, vals in entries:
        if cls == "bad":
            stats["bad_lines"] += 1
            continue
        if cls not in ("0", "1", "2"):
            stats["unknown_cls"] += 1
            continue
        if not vals:
            stats["bad_lines"] += 1
            continue
        xc, yc, w, h = vals
        stats["boxes_total"] += 1
        ox, oy, ow, oh = xc, yc, w, h
        if xc < 0.0:
            xc = 0.0
        if xc > 1.0:
            xc = 1.0
        if yc < 0.0:
            yc = 0.0
        if yc > 1.0:
            yc = 1.0
        if w < 0.0:
            w = 0.0
        if w > 1.0:
            w = 1.0
        if h < 0.0:
            h = 0.0
        if h > 1.0:
            h = 1.0
        changed = (ox != xc or oy != yc or ow != w or oh != h)
        x1 = xc - w * 0.5
        y1 = yc - h * 0.5
        x2 = xc + w * 0.5
        y2 = yc + h * 0.5
        if x1 < 0.0 or x2 > 1.0:
            xc = max(w * 0.5, min(1.0 - w * 0.5, xc))
            changed = True
        if y1 < 0.0 or y2 > 1.0:
            yc = max(h * 0.5, min(1.0 - h * 0.5, yc))
            changed = True
        max_w = 2.0 * min(xc, 1.0 - xc)
        max_h = 2.0 * min(yc, 1.0 - yc)
        if w > max_w:
            w = max_w
            changed = True
        if h > max_h:
            h = max_h
            changed = True
        if w <= 0.0 or h <= 0.0 or w < min_w or h < min_h:
            stats["boxes_dropped"] += 1
            continue
        if changed:
            stats["boxes_fixed"] += 1
        out_lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        stats["boxes_kept"] += 1
    return out_lines, stats

def clean_label_file(lbl_path, min_w=0.02, min_h=0.02):
    entries = read_label_file(lbl_path)
    out_lines, st = clean_label_entries(entries, min_w=min_w, min_h=min_h)
    changed = False
    try:
        original = open(lbl_path, "r", encoding="utf-8", errors="ignore").read().splitlines()
    except Exception:
        original = []
    if original != out_lines:
        with open(lbl_path, "w", encoding="utf-8") as f:
            for ln in out_lines:
                f.write(ln + "\n")
        changed = True
    return {"changed": changed, **st}

def clean_split(base, split, min_w=0.02, min_h=0.02):
    lbl_dir = os.path.join(base, split, "labels")
    files = []
    if os.path.isdir(lbl_dir):
        for name in os.listdir(lbl_dir):
            p = os.path.join(lbl_dir, name)
            if os.path.isfile(p) and name.endswith(".txt"):
                files.append(p)
    agg = {
        "files": len(files),
        "changed": 0,
        "boxes_total": 0,
        "boxes_kept": 0,
        "boxes_dropped": 0,
        "boxes_fixed": 0,
        "unknown_cls": 0,
        "bad_lines": 0,
    }
    for lp in files:
        st = clean_label_file(lp, min_w=min_w, min_h=min_h)
        if st.get("changed"):
            agg["changed"] += 1
        for k in ("boxes_total", "boxes_kept", "boxes_dropped", "boxes_fixed", "unknown_cls", "bad_lines"):
            agg[k] += st.get(k, 0)
    return agg

def next_aug_base(bn, img_dir, lbl_dir):
    i = 1
    while True:
        cand = f"{bn}_aug{i}"
        if not os.path.exists(os.path.join(img_dir, cand + ".jpg")) and \
           not os.path.exists(os.path.join(img_dir, cand + ".png")) and \
           not os.path.exists(os.path.join(img_dir, cand + ".jpeg")) and \
           not os.path.exists(os.path.join(lbl_dir, cand + ".txt")):
            return cand
        i += 1

def augment_file(img_path, lbl_path, kind, out_base):
    img = read_image(img_path)
    if img is None:
        return False
    entries = read_label_file(lbl_path)
    new_img = transform_img(img, kind)
    img_dir = os.path.dirname(img_path)
    lbl_dir = os.path.dirname(lbl_path)
    ext = os.path.splitext(img_path)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        ext = ".jpg"
    out_img = os.path.join(img_dir, out_base + ext)
    out_lbl = os.path.join(lbl_dir, out_base + ".txt")
    write_image(out_img, new_img)
    lines = transform_labels(entries, kind)
    with open(out_lbl, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")
    return True

def image_class_sets(lbl_paths):
    m = {}
    for lp in lbl_paths:
        bn = base_name(lp)
        entries = read_label_file(lp)
        s = set()
        for cls, vals in entries:
            if cls == "bad":
                continue
            s.add(cls)
        m[bn] = s
    return m

def balance_split(base, split):
    img_dir = os.path.join(base, split, "images")
    lbl_dir = os.path.join(base, split, "labels")
    images, labels = list_files(base, split)
    lbl_bases = set(base_name(p) for p in labels)
    img_paths = {base_name(p): p for p in images}
    lbl_paths = {base_name(p): p for p in labels}
    cls_sets = image_class_sets(labels)
    class_to_images = {"0": [], "1": [], "2": []}
    for bn, s in cls_sets.items():
        for k in ("0", "1", "2"):
            if k in s:
                class_to_images[k].append(bn)
    counts = {k: len(class_to_images[k]) for k in ("0", "1", "2")}
    target = max(counts.values()) if counts else 0
    order = ["hflip", "vflip", "brightness", "contrast"]
    for k in ("0", "1", "2"):
        need = target - counts.get(k, 0)
        if need <= 0:
            continue
        pure = [bn for bn in class_to_images[k] if cls_sets.get(bn, set()) == {k}]
        mixed = [bn for bn in class_to_images[k] if bn not in pure]
        pool = pure + mixed
        if not pool:
            continue
        idx = 0
        created = 0
        while created < need:
            bn = pool[idx % len(pool)]
            idx += 1
            ip = img_paths.get(bn)
            lp = lbl_paths.get(bn)
            if not ip or not lp:
                continue
            kind = order[created % len(order)]
            out_base = next_aug_base(bn, img_dir, lbl_dir)
            ok = augment_file(ip, lp, kind, out_base)
            if ok:
                created += 1
                class_to_images[k].append(out_base)
                cls_sets[out_base] = cls_sets.get(bn, set())
                img_paths[out_base] = os.path.join(img_dir, out_base + os.path.splitext(ip)[1].lower())
                lbl_paths[out_base] = os.path.join(lbl_dir, out_base + ".txt")
        counts[k] = len(class_to_images[k])
    return counts

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    splits = ["train", "valid", "test"]
    clean_stats = {}
    for s in splits:
        clean_stats[s] = clean_split(base, s, min_w=0.02, min_h=0.02)
    print("Label Cleaning")
    for s in splits:
        cs = clean_stats[s]
        print(f"{s}: files={cs['files']}, changed={cs['changed']}, boxes_total={cs['boxes_total']}, boxes_kept={cs['boxes_kept']}, boxes_dropped={cs['boxes_dropped']}, fixed={cs['boxes_fixed']}, unknown_cls={cs['unknown_cls']}, bad_lines={cs['bad_lines']}")
    all_bases = {}
    summary = {}
    class_summaries = {}
    bbox_summaries = {}
    balance_plans = {}
    for s in splits:
        images, labels = list_files(base, s)
        summary[s] = pairs_stats(images, labels)
        bases_img = set(base_name(p) for p in images)
        all_bases[s] = bases_img
        class_counts, bbox_stats, per_image_class = class_and_bbox_stats(labels)
        class_summaries[s] = class_counts
        bbox_summaries[s] = bbox_stats
        balance_plans[s] = balanced_plan(class_counts)
    dup_tv = duplicates_between_splits(all_bases.get("train", set()), all_bases.get("valid", set()))
    dup_tt = duplicates_between_splits(all_bases.get("train", set()), all_bases.get("test", set()))
    dup_vt = duplicates_between_splits(all_bases.get("valid", set()), all_bases.get("test", set()))
    print("Dataset Review")
    for s in splits:
        ss = summary[s]
        print(f"{s}: images={ss['images']}, labels={ss['labels']}, images_without_label={ss['images_without_label']}, labels_without_image={ss['labels_without_image']}")
        cc = class_summaries[s]
        ev = cc.get("0", 0)
        fv = cc.get("1", 0)
        sv = cc.get("2", 0)
        print(f"{s} class boxes: eating={ev}, face_touching={fv}, smoking={sv}")
        bs = bbox_summaries[s]
        print(f"{s} label stats: bad_lines={bs['bad']}, out_of_range={bs['oor']}, min_w={bs['min_w']:.6f}, min_h={bs['min_h']:.6f}, max_w={bs['max_w']:.6f}, max_h={bs['max_h']:.6f}")
        bp = balance_plans[s]
        print(f"{s} balance plan: add_eating={bp['0']['add']}, add_face_touching={bp['1']['add']}, add_smoking={bp['2']['add']}")
    print(f"duplicates train-valid={dup_tv}, train-test={dup_tt}, valid-test={dup_vt}")
    ypath = os.path.join(base, "dataset.yaml")
    if os.path.isfile(ypath):
        try:
            lines = open(ypath, "r", encoding="utf-8", errors="ignore").read().splitlines()
            d = {}
            for ln in lines:
                t = ln.strip()
                if t.startswith("train:"):
                    d["train"] = t.split(":", 1)[1].strip()
                if t.startswith("val:"):
                    d["val"] = t.split(":", 1)[1].strip()
                if t.startswith("test:"):
                    d["test"] = t.split(":", 1)[1].strip()
            for k in ["train", "val", "test"]:
                p = d.get(k, "")
                exists = os.path.isdir(p)
                print(f"dataset.yaml {k} path exists={exists} path={p}")
        except Exception:
            print("dataset.yaml parse failed")
    for s in splits:
        balance_split(base, s)
    print("Images Per Class After Balancing")
    for s in splits:
        images, labels = list_files(base, s)
        lbl_paths = labels
        sets_map = image_class_sets(lbl_paths)
        c0 = sum(1 for v in sets_map.values() if "0" in v)
        c1 = sum(1 for v in sets_map.values() if "1" in v)
        c2 = sum(1 for v in sets_map.values() if "2" in v)
        print(f"{s}: eating={c0}, face_touching={c1}, smoking={c2}")
    print("Summary")
    for s in splits:
        images, labels = list_files(base, s)
        print(f"{s}: images={len(images)}, labels={len(labels)}")

if __name__ == "__main__":
    main()
