"""Post-processing lab: score outline-shaping ("humanising") and model-combination variants on the
106 validation files, using probability maps saved by scripts/cache_probs.py (no network is run).

Nothing here changes the pipeline. Every variant is scored with dataset-level PQ (the organisers'
way) on all 106 files AND with 2-fold cross-validation over files (settings chosen on one half,
scored on the other half), so a gain that only comes from fitting these images is visible.

  python scripts/postproc_lab.py baseline     # must reproduce weights/eval_unet_r34.json exactly
  python scripts/postproc_lab.py humanise     # outline shaping on the ResNet-34 maps
  python scripts/postproc_lab.py ensemble     # ResNet-34 + small U-Net averages
  python scripts/postproc_lab.py combo --tags gpu,r50   # each model alone and their average
Map tags: gpu = ResNet-34 (Kaggle Version 3), r50 = ResNet-50 (Version 4), r50rot = ResNet-50 with 6 views.
Writes weights/lab_<experiment>.csv
"""
import argparse
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from filseg import CocoSet, PostParams  # noqa: E402
from filseg.metrics import score_image, summarize, to_idx  # noqa: E402
from filseg.postprocess import to_instances  # noqa: E402
from cache_probs import IMGS, JSON  # noqa: E402
from run import split_ids  # noqa: E402

PP_KEYS = ("prob_threshold", "min_fragment", "hole_size", "link_radius", "min_instance", "work_size", "grow")
G = {}


def _init():
    coco = CocoSet(JSON, IMGS)
    _, va = split_ids(coco)
    by_file = {}
    for e in va:
        by_file.setdefault(coco.path_of[e], []).append(e)
    G["gts"] = {f: [to_idx(coco.instance_masks(e)) for e in es] for f, es in by_file.items()}


def files_in_order():
    coco = CocoSet(JSON, IMGS)
    _, va = split_ids(coco)
    return list(dict.fromkeys(coco.path_of[e] for e in va))


def _prob(tag, f):
    stem = os.path.splitext(os.path.basename(f))[0]
    return np.load(os.path.join("data", f"_probs_{tag}", stem + ".npz"))["prob"]


def _shape(m, fill, close):
    """Optional human-like outline: close small notches (radius `close` px) and fill all holes."""
    if not (fill or close):
        return m
    ys, xs = np.nonzero(m)
    pad = close + 2
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, m.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, m.shape[1])
    c = m[y0:y1, x0:x1].astype(np.uint8)
    if close:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close + 1, 2 * close + 1))
        c = cv2.morphologyEx(c, cv2.MORPH_CLOSE, k)
    c = c.astype(bool)
    if fill:
        c = ndimage.binary_fill_holes(c)
    out = np.zeros_like(m)
    out[y0:y1, x0:x1] = c
    return out


def predict(f, cfg, probs):
    models = cfg["models"]
    if len(models) == 1:
        prob = probs[next(iter(models))]                       # exactly the saved map
    else:
        wsum = float(sum(models.values()))
        prob = sum(np.float32(w / wsum) * probs[t] for t, w in models.items())
    if cfg.get("sigma", 0):
        prob = cv2.GaussianBlur(prob, (0, 0), cfg["sigma"])
    pp = PostParams(**{k: cfg[k] for k in PP_KEYS if k in cfg})
    inst = to_instances(prob >= pp.prob_threshold, pp)
    return [_shape(m, cfg.get("fill", False), cfg.get("close", 0)) for m in inst]


def _work(args):
    f, cfgs = args
    tags = sorted({t for c in cfgs for t in c["models"]})
    probs = {t: _prob(t, f) for t in tags}
    out = []
    for c in cfgs:
        pred = to_idx(predict(f, c, probs))
        out.append([score_image(gt, pred) for gt in G["gts"][f]])
    return f, out


def run(cfgs, workers):
    files = files_in_order()
    t0 = time.time()
    per_cfg = [dict() for _ in cfgs]     # cfg index -> {file: [ImageScore per annotator entry]}
    with Pool(workers, initializer=_init) as pool:
        for n, (f, out) in enumerate(pool.imap_unordered(_work, [(f, cfgs) for f in files]), 1):
            for k, sc in enumerate(out):
                per_cfg[k][f] = sc
            if n % 20 == 0:
                print(f"{n}/{len(files)} files  {time.time() - t0:.0f} s", flush=True)
    return files, per_cfg


def report(name, cfgs, files, per_cfg):
    rows = []
    for c, sc in zip(cfgs, per_cfg):
        s = summarize([x for f in files for x in sc[f]])
        rows.append({**{k: v for k, v in c.items() if k != "models"},
                     "models": json.dumps(c["models"]), **s})
    df = pd.DataFrame(rows)
    # 2-fold CV over files: choose the best setting on one half, score it on the other half
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(files))
    folds = [set(np.array(files)[perm[: len(files) // 2]]), set(np.array(files)[perm[len(files) // 2:]])]
    held = []
    for k in (0, 1):
        tune, test = folds[1 - k], folds[k]
        pq = [summarize([x for f in files if f in tune for x in sc[f]])["PQ_dataset"] for sc in per_cfg]
        best = int(np.argmax(pq))
        held += [x for f in files if f in test for x in per_cfg[best][f]]
        print(f"fold {k}: chosen on the other half -> setting #{best}")
    cv = summarize(held)
    df = df.sort_values("PQ_dataset", ascending=False, kind="stable")
    df.to_csv(f"weights/lab_{name}.csv", index=False)
    cols = [c for c in df.columns if c not in ("PQ_mean_per_image", "mean_IoU_matched")]
    print(df[cols].head(12).to_string(index=False))
    print(f"\n2-fold CV estimate (choose on half, score on other half): PQ {cv['PQ_dataset']:.4f} "
          f"(TP {cv['TP']}, FP {cv['FP']}, FN {cv['FN']})")
    print(f"-> weights/lab_{name}.csv")


GPU_BASE = {"models": {"gpu": 1.0}, "prob_threshold": 0.65, "min_fragment": 60, "hole_size": 200,
            "link_radius": 0, "min_instance": 400, "work_size": 1024}


def experiments(name):
    if name == "baseline":
        return [dict(GPU_BASE)]
    if name == "humanise":
        return [{**GPU_BASE, "grow": g, "sigma": s, "fill": fl, "close": cl}
                for g, s, fl, cl in itertools.product([0, 1, 2, 3], [0, 1.5, 3.0], [False, True], [0, 3])]
    if name == "ensemble":
        return [{**GPU_BASE, "models": {"gpu": w, "cpu": round(1 - w, 2)} if w < 1 else {"gpu": 1.0},
                 "prob_threshold": t, "link_radius": r, "min_instance": mi}
                for w, t, r, mi in itertools.product([1.0, 0.8, 0.7, 0.6, 0.5], [0.5, 0.65],
                                                     [0, 6, 12], [400, 1000])]
    raise ValueError(name)


def combo_experiments(tags):
    """Every single model and the equal-weight average of every group of 2+ models, each with a grid
    of threshold x fragment linking x minimum instance size."""
    mixes = [{t: 1.0} for t in tags]
    for k in range(2, len(tags) + 1):
        mixes += [{t: 1.0 for t in group} for group in itertools.combinations(tags, k)]
    return [{**GPU_BASE, "models": m, "prob_threshold": t, "link_radius": r, "min_instance": mi}
            for m in mixes for t, r, mi in itertools.product([0.4, 0.5, 0.6, 0.7], [0, 12, 20], [400, 1000])]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["baseline", "humanise", "ensemble", "combo"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--tags", default="gpu,r50", help="combo: comma-separated probability-map tags")
    a = ap.parse_args()
    cfgs = combo_experiments(a.tags.split(",")) if a.experiment == "combo" else experiments(a.experiment)
    print(f"{len(cfgs)} settings x 106 files, {a.workers} workers", flush=True)
    files, per_cfg = run(cfgs, a.workers)
    name = f"combo_{a.tags.replace(',', '_')}" if a.experiment == "combo" else a.experiment
    report(name, cfgs, files, per_cfg)
