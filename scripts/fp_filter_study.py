"""False-positive filter study for the classical method.

Step 1 (slow, cached): for each image file, run the classical detector once, form instances for a few
link radii, and record per instance: area, darkness (1 - median flattened intensity), elongation,
distance from disk centre, and its best IoU against EVERY annotator entry of that file.
Results are cached in data/_fpstudy/<split>.pkl, so re-runs only process new files.

Step 2 (instant): grid-search a simple keep-rule on the TRAIN files to maximise dataset-level PQ,
then report it on the held-out VALIDATION files next to the unfiltered baseline.

    python scripts/fp_filter_study.py --n-train 200
"""
import argparse
import itertools
import json
import os
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")  # leave the CPU cores to the U-Net training

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.setNumThreads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from filseg import CocoSet, read_image  # noqa: E402
from filseg.classical import segment_classical  # noqa: E402
from filseg.metrics import overlap_matrix, to_idx  # noqa: E402
from filseg.postprocess import PostParams, to_instances  # noqa: E402
from filseg.preprocess import preprocess  # noqa: E402
from run import load_cp, load_pp, split_ids  # noqa: E402

LINK_RADII = (12, 20, 32)
BASE_MIN = 150  # smallest instance kept before the rule (rule can raise it)


def features(m, flat, disk):
    ys, xs = np.nonzero(m)
    area = len(ys)
    dark = float(1.0 - np.median(flat[ys, xs]))
    cov = np.cov(np.vstack([xs, ys])) if area > 2 else np.eye(2)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    elong = float(np.sqrt(ev[0] / max(ev[1], 1e-6)))
    rho = float(np.hypot(xs.mean() - disk.cx, ys.mean() - disk.cy) / disk.r)
    return area, dark, elong, rho


def process_file(f, entries, coco, cp, pp):
    img = read_image(f)
    flat, disk = preprocess(img)
    sem = segment_classical(flat, disk, cp)
    gts = {e: to_idx(coco.instance_masks(e)) for e in entries}
    out = {"n_gt": {e: len(g) for e, g in gts.items()}, "inst": {}}
    for lr in LINK_RADII:
        p = PostParams(**{**pp.__dict__, "link_radius": lr, "min_instance": BASE_MIN})
        inst = to_instances(sem, p)
        rows = []
        pr = to_idx(inst)
        best = {}
        for e, g in gts.items():
            if not g or not pr:
                best[e] = np.zeros(len(pr))
                continue
            inter, ga, pa = overlap_matrix(g, pr)
            iou = inter / np.maximum(ga[:, None] + pa[None, :] - inter, 1)
            best[e] = iou.max(0)
        for k, m in enumerate(inst):
            rows.append((*features(m, flat, disk), {e: float(best[e][k]) for e in entries}))
        out["inst"][lr] = rows
    return out


def build(coco, ids, path, cp, pp, label):
    by_file = {}
    for i in ids:
        by_file.setdefault(coco.path_of[i], []).append(i)
    data = pickle.load(open(path, "rb")) if os.path.exists(path) else {}
    todo = [f for f in by_file if f not in data]
    t0 = time.time()
    for n, f in enumerate(todo):
        data[f] = process_file(f, by_file[f], coco, cp, pp)
        if n % 10 == 0 or n == len(todo) - 1:
            pickle.dump(data, open(path, "wb"))
            el = time.time() - t0
            print(f"[{label}] {n + 1}/{len(todo)} files  {el / (n + 1):.1f} s/file", flush=True)
    return {f: data[f] for f in by_file}


def to_arrays(data, lr):
    """Flatten to arrays: one row per (instance, annotator entry)."""
    A, D, E, R, IOU, EID = [], [], [], [], [], []
    n_gt = {}
    for f, d in data.items():
        n_gt.update(d["n_gt"])
        for area, dark, elong, rho, best in d["inst"][lr]:
            for e, v in best.items():
                A.append(area); D.append(dark); E.append(elong); R.append(rho); IOU.append(v); EID.append(e)
    return (np.array(A), np.array(D), np.array(E), np.array(R), np.array(IOU), np.array(EID, dtype=object),
            n_gt)


def pq(keep, iou, n_gt):
    tp_mask = keep & (iou > 0.5)
    tp, fp = int(tp_mask.sum()), int((keep & ~(iou > 0.5)).sum())
    fn = sum(n_gt.values()) - tp
    d = tp + 0.5 * fp + 0.5 * fn
    return (iou[tp_mask].sum() / d if d else 0.0), tp, fp, fn


GRID = {
    "min_area": [150, 400, 1000, 2000],
    "min_dark": [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15],
    "min_elong": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
    "big_area": [10 ** 9, 20000, 10000, 5000],
    "max_rho": [1.0, 0.97, 0.94],
}


def rule(arr, r):
    A, D, E, R = arr[:4]
    return (A >= r["min_area"]) & (R <= r["max_rho"]) & (((D >= r["min_dark"]) & (E >= r["min_elong"]))
                                                       | (A >= r["big_area"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=200, help="training FILES used to fit the rule")
    a = ap.parse_args()
    os.makedirs("data/_fpstudy", exist_ok=True)
    coco = CocoSet("data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
                   "data/MAGFiLO_1.0_Kaggle_2026/train/train_images")
    tr, va = split_ids(coco)
    cp, pp = load_cp(), load_pp("classical")
    tr_files = sorted({coco.path_of[i] for i in tr})
    sel = set(np.random.default_rng(0).permutation(tr_files)[: a.n_train])
    tr_ids = [i for i in tr if coco.path_of[i] in sel]
    dva = build(coco, va, "data/_fpstudy/val.pkl", cp, pp, "val")
    dtr = build(coco, tr_ids, "data/_fpstudy/train.pkl", cp, pp, "train")

    summary = {}
    for lr in LINK_RADII:
        T, V = to_arrays(dtr, lr), to_arrays(dva, lr)
        base = {"min_area": 400, "min_dark": 0, "min_elong": 0, "big_area": 10 ** 9, "max_rho": 1.0}
        best, best_pq = None, -1
        for vals in itertools.product(*GRID.values()):
            r = dict(zip(GRID, vals))
            s = pq(rule(T, r), T[4], T[6])[0]
            if s > best_pq:
                best, best_pq = r, s
        bt, bv = pq(rule(T, base), T[4], T[6]), pq(rule(V, base), V[4], V[6])
        ft, fv = pq(rule(T, best), T[4], T[6]), pq(rule(V, best), V[4], V[6])
        print(f"\nlink_radius={lr}  rule={best}")
        print(f"  TRAIN  baseline PQ={bt[0]:.4f} (TP {bt[1]}, FP {bt[2]}, FN {bt[3]})  "
              f"filtered PQ={ft[0]:.4f} (TP {ft[1]}, FP {ft[2]}, FN {ft[3]})")
        print(f"  VAL    baseline PQ={bv[0]:.4f} (TP {bv[1]}, FP {bv[2]}, FN {bv[3]})  "
              f"filtered PQ={fv[0]:.4f} (TP {fv[1]}, FP {fv[2]}, FN {fv[3]})")
        summary[lr] = {"rule": best, "train_base": bt, "train_filtered": ft, "val_base": bv, "val_filtered": fv}
    json.dump({str(k): v for k, v in summary.items()}, open("data/_fpstudy/summary.json", "w"), indent=2,
              default=float)
    print("\nsaved data/_fpstudy/summary.json")


if __name__ == "__main__":
    main()
