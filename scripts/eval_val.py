"""Evaluate one model on ALL validation files (the 15% held-out split, seed 0).

Each image file is predicted once and scored against every annotator's version of it, exactly as
pipeline.evaluate() would score each entry (same predict_instances / score_image calls).
Writes:
  weights/eval_<tag>.json        summary (PQ, SQ, RQ, TP/FP/FN, Dice, 1:n, n:1, sec/image)
  weights/eval_<tag>_pairs.csv   every overlapping (GT, prediction) pair: IoU and Dice, for histograms

Usage:
  python scripts/eval_val.py --method unet --weights weights/unet.pt --pp weights/postprocess_unet.json --tag unet
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from filseg import CocoSet, ClassicalParams, PostParams  # noqa: E402
from filseg import pipeline  # noqa: E402
from filseg.io import read_image  # noqa: E402
from filseg.metrics import overlap_matrix, score_image, summarize, to_idx  # noqa: E402
from run import split_ids  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="unet", choices=["classical", "unet"])
    ap.add_argument("--weights", default="weights/unet.pt")
    ap.add_argument("--pp", required=True, help="post-processing json")
    ap.add_argument("--cp", default="weights/classical_params.json")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-files", type=int, default=0, help="0 = all validation files")
    ap.add_argument("--train-json", default="data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ap.add_argument("--train-dir", default="data/MAGFiLO_1.0_Kaggle_2026/train/train_images")
    a = ap.parse_args()

    coco = CocoSet(a.train_json, a.train_dir)
    _, va = split_ids(coco)
    pp = PostParams(**json.load(open(a.pp)))
    cp = ClassicalParams(**json.load(open(a.cp))) if os.path.exists(a.cp) else ClassicalParams()
    model = None
    if a.method == "unet":
        from filseg import unet
        model = unet.load_model(a.weights)

    by_file = {}
    for i in va:
        by_file.setdefault(coco.path_of[i], []).append(i)
    files = list(by_file)[: a.max_files or None]
    print(f"{len(files)} validation files, {sum(len(by_file[f]) for f in files)} annotator entries", flush=True)

    scores, pairs, rows, org_1n, org_n1, t_pred = [], [], [], 0, 0, 0.0
    for k, f in enumerate(files):
        img = read_image(f)
        t0 = time.time()
        pred = pipeline.predict_instances(img, a.method, model, pp, cp)
        t_pred += time.time() - t0
        pr = to_idx(pred)
        for e in by_file[f]:
            gt = to_idx(coco.instance_masks(e))
            s = score_image(gt, pr)
            scores.append(s)
            rows.append({"file": os.path.basename(f), "entry": e, "TP": s.tp, "FP": s.fp, "FN": s.fn,
                         "iou_sum": s.iou_sum, "n_gt": len(gt), "n_pred": len(pr)})
            if gt and pr:
                inter, ga, pa = overlap_matrix(gt, pr)
                ov = inter > 0
                org_1n += int((ov.sum(1) > 1).sum())   # organisers: GT touched by >1 prediction
                org_n1 += int((ov.sum(0) > 1).sum())   # organisers: prediction touching >1 GT
                for g, p in zip(*np.nonzero(ov)):
                    u = ga[g] + pa[p] - inter[g, p]
                    pairs.append({"file": os.path.basename(f), "entry": e, "iou": inter[g, p] / u,
                                  "dice": 2 * inter[g, p] / (ga[g] + pa[p])})
        if (k + 1) % 10 == 0 or k + 1 == len(files):
            r = summarize(scores)
            print(f"{k + 1}/{len(files)} files  PQ_dataset so far {r['PQ_dataset']:.4f}  "
                  f"({t_pred / (k + 1):.1f} s/image)", flush=True)

    res = summarize(scores)
    res.update({"files": len(files), "entries": len(scores), "sec_per_image": t_pred / max(len(files), 1),
                "org_one_to_many_iou0": org_1n, "org_many_to_one_iou0": org_n1,
                "mean_Dice_all_overlapping_pairs": float(np.mean([p["dice"] for p in pairs])) if pairs else 0.0,
                "method": a.method, "weights": a.weights if a.method == "unet" else None, "pp": a.pp})
    # uncertainty: 95% bootstrap interval of dataset PQ, resampling image FILES (all annotators of a file together)
    ent = pd.DataFrame(rows)
    per_file = ent.groupby("file")[["TP", "FP", "FN", "iou_sum"]].sum().to_numpy(float)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(1000):
        t = per_file[rng.integers(0, len(per_file), len(per_file))].sum(0)
        d = t[0] + 0.5 * t[1] + 0.5 * t[2]
        boot.append(t[3] / d if d else np.nan)
    res["PQ_dataset_ci95"] = [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))]
    os.makedirs("weights", exist_ok=True)
    json.dump(res, open(f"weights/eval_{a.tag}.json", "w"), indent=2)
    pd.DataFrame(pairs).to_csv(f"weights/eval_{a.tag}_pairs.csv", index=False)
    ent.to_csv(f"weights/eval_{a.tag}_entries.csv", index=False)
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()})


if __name__ == "__main__":
    main()
