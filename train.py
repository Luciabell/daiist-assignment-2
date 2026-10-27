"""uv run python main.py train

1. Download the Roboflow dataset (fails loudly if it can't - no silent fallback)
2. Finetune YOLO11n
3. Evaluate on the TEST split (per class) + baselines
4. Build the case table (score, truth) and compute the cost sweep / review band
5. Save everything to weights/ so app.py never trains or recomputes the model
"""
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

import costing as C
from api_key import load_api_key

# ---- Config (record these in REPORT.md) ----
WORKSPACE = "gbc-tdplg"
PROJECT = "helmet-no-helmet"   # must match the dataset URL on Universe
VERSION = 1
BASE_MODEL = "yolo11n.pt"
EPOCHS = int(os.environ.get("EPOCHS", 30))   # quick test: EPOCHS=2 uv run python main.py train
IMGSZ = 640
BATCH = 16
SEED = 0
POSITIVE_NAME = "head"      # bare head = no hardhat = the violation
COMPLIANT_NAME = "helmet"   # worker wearing a hardhat
# Other classes (e.g. "person") are still trained on, but are not counted as cases.


def iou(a, b):
    """IoU between one box a (4,) and many boxes b (n,4); xyxy."""
    x1 = np.maximum(a[0], b[:, 0]); y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2]); y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a + area_b - inter + 1e-9)


def load_gt(label_file):
    if not label_file.exists() or label_file.stat().st_size == 0:
        return np.zeros((0, 4)), np.zeros(0, int)
    gt = np.loadtxt(label_file, ndmin=2)
    xc, yc, w, h = gt[:, 1], gt[:, 2], gt[:, 3], gt[:, 4]
    xyxy = np.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], axis=1)
    return xyxy, gt[:, 0].astype(int)


def dataset_stats(root, names):
    """Instances per class, images per split, object size (share of image area)."""
    out = {}
    for split in ["train", "valid", "test"]:
        lab_dir = Path(root, split, "labels")
        if not lab_dir.exists():
            continue
        counts, areas = Counter(), []
        for f in lab_dir.glob("*.txt"):
            for line in f.read_text().splitlines():
                p = line.split()
                if len(p) >= 5:
                    counts[names[int(p[0])]] += 1
                    areas.append(float(p[3]) * float(p[4]))
        out[split] = {
            "images": len(list(Path(root, split, "images").glob("*.*"))),
            "instances": dict(counts),
            "median_box_area_pct_of_image": round(100 * float(np.median(areas)), 2) if areas else None,
        }
    return out


def build_cases(model, test_dir, pos_id, comp_id):
    """One case per ground-truth worker + one per false alarm on background.
    score = highest 'no-hardhat' confidence overlapping that worker (IoU>=0.5)."""
    scores, labels = [], []
    imgs = sorted(str(p) for p in Path(test_dir, "images").glob("*.*"))
    for r in model.predict(imgs, conf=0.05, iou=0.6, stream=True, verbose=False):
        gt_xyxy, gt_cls = load_gt(Path(test_dir, "labels", Path(r.path).stem + ".txt"))
        is_worker = np.isin(gt_cls, [pos_id, comp_id])   # ignore e.g. "person" boxes
        gt_xyxy, gt_cls = gt_xyxy[is_worker], gt_cls[is_worker]
        b = r.boxes
        pxy, pconf, pcls = b.xyxyn.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)
        keep = pcls == pos_id                      # only 'no-hardhat' predictions
        pxy, pconf = pxy[keep], pconf[keep]
        matched = np.zeros(len(pxy), bool)
        for g, c in zip(gt_xyxy, gt_cls):
            s = 0.0
            if len(pxy):
                hit = iou(g, pxy) >= 0.5
                matched |= hit
                if hit.any():
                    s = float(pconf[hit].max())
            scores.append(s)
            labels.append(bool(c == pos_id))
        for s in pconf[~matched]:                  # false alarms on background
            scores.append(float(s))
            labels.append(False)
    return np.array(scores), np.array(labels, bool)


def main():
    # 1. Dataset (no fallback: if this fails, the pipeline must fail)
    from roboflow import Roboflow
    rf = Roboflow(api_key=load_api_key())
    ds = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download("yolov11")
    root = Path(ds.location)
    data_yaml = str(root / "data.yaml")
    cfg = yaml.safe_load(open(data_yaml))
    names = cfg["names"]
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    print("Classes:", names)

    def norm(s):
        return "".join(ch for ch in s.lower() if ch.isalnum())

    matches = [i for i, n in enumerate(names) if norm(n) == norm(POSITIVE_NAME)]
    assert matches, f"No class matches '{POSITIVE_NAME}'. Classes are: {names}"
    pos_id = matches[0]
    comp_matches = [i for i, n in enumerate(names) if norm(n) == norm(COMPLIANT_NAME)]
    assert comp_matches, f"No class matches '{COMPLIANT_NAME}'. Classes are: {names}"
    comp_id = comp_matches[0]
    test_dir = root / "test"
    assert test_dir.exists(), "Dataset has no test split"
    print("Classes:", names, "| positive class:", names[pos_id])

    # 2. Train
    model = YOLO(BASE_MODEL)
    model.train(data=data_yaml, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, seed=SEED,
                project="runs/detect", name="hardhat_yolo11n", exist_ok=True)
    Path("weights").mkdir(exist_ok=True)
    shutil.copy(Path(model.trainer.save_dir) / "weights" / "best.pt", "weights/best.pt")
    best = YOLO("weights/best.pt")

    # 3. Evaluate on TEST split, per class
    m = best.val(data=data_yaml, split="test", conf=0.001, plots=True,
                 project="runs/detect", name="test_eval", exist_ok=True)
    per_class = {}
    for i, ci in enumerate(m.box.ap_class_index):
        per_class[names[int(ci)]] = {
            "precision": float(m.box.p[i]), "recall": float(m.box.r[i]),
            "AP50": float(m.box.ap50[i]), "mAP50-95": float(m.box.ap[i])}
    overall = {"mAP50": float(m.box.map50), "mAP50-95": float(m.box.map)}
    cm_png = Path(m.save_dir) / "confusion_matrix.png"
    if cm_png.exists():
        shutil.copy(cm_png, "weights/confusion_matrix.png")

    # Baseline 1: pretrained COCO, zero-shot, on the same test images
    coco = YOLO("yolo11n.pt")
    shared = sorted({n.lower() for n in coco.names.values()} & {n.lower() for n in names})
    imgs = sorted(str(p) for p in Path(test_dir, "images").glob("*.*"))
    n_det = 0
    for r in coco.predict(imgs, conf=0.25, stream=True, verbose=False):
        n_det += sum(coco.names[int(c)].lower() in shared for c in r.boxes.cls)
    coco_baseline = {"classes_shared_with_dataset": shared,
                     "detections_of_dataset_classes": n_det,
                     "mAP50": 0.0 if not shared else None}

    # 4. Cases + cost sweep + review band (all from real predictions)
    scores, is_viol = build_cases(best, test_dir, pos_id, comp_id)
    t_single, c_single = C.best_single(scores, is_viol)
    t_low, t_high, c_band = C.best_band(scores, is_viol)
    ts, costs = C.sweep(scores, is_viol)
    band = C.evaluate(scores, is_viol, t_low, t_high)
    n_rev, hours, people = C.staffing(band["review_share"])

    artifacts = {
        "config": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION,
                   "model": BASE_MODEL, "epochs": EPOCHS, "imgsz": IMGSZ, "batch": BATCH,
                   "seed": SEED, "positive_class": names[pos_id], "compliant_class": names[comp_id]},
        "classes": names,
        "positive_class_id": pos_id,
        "dataset_stats": dataset_stats(root, names),
        "overall": overall,
        "per_class": per_class,
        "baselines": {"coco_zero_shot": coco_baseline,
                      "predict_nothing": {"recall_positive": 0.0,
                                          "cost_per_1000": C.BASE_RATE * 1000 * C.C_FN}},
        "assumptions": {"base_rate": C.BASE_RATE, "c_fn": C.C_FN, "c_fp": C.C_FP,
                        "c_review": C.C_REVIEW, "cases_per_day": C.CASES_PER_DAY,
                        "min_per_review": C.MIN_PER_REVIEW},
        "recommended": {"single_threshold": t_single, "single_cost_per_1000": c_single,
                        "t_low": t_low, "t_high": t_high, "band_cost_per_1000": c_band,
                        "review_share": band["review_share"], "reviews_per_day": n_rev,
                        "review_hours_per_day": hours, "people_needed": people},
        "cost_curve": {"thresholds": ts.tolist(), "cost_per_1000": costs.tolist()},
        "cases": {"scores": np.round(scores, 4).tolist(), "is_violation": is_viol.tolist()},
    }
    with open("weights/eval_artifacts.json", "w") as f:
        json.dump(artifacts, f, indent=2)

    print("\n=== RESULTS (copy these into REPORT.md) ===")
    print("Overall:", overall)
    for k, v in per_class.items():
        print(k, v)
    print("COCO zero-shot:", coco_baseline)
    print(f"Best single threshold: {t_single} -> {c_single:.0f} EUR / 1000 cases")
    print(f"Best band: low={t_low}, high={t_high} -> {c_band:.0f} EUR / 1000 cases")
    print(f"Review: {band['review_share']:.1%} of cases = {n_rev:.0f}/day, "
          f"{hours:.1f} h/day, {people:.2f} people")


if __name__ == "__main__":
    main()