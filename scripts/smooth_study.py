"""'Humanise' study: reshape each predicted filament like a human-drawn outline and measure PQ.

Annotators draw one smooth, filled polygon per filament. For every predicted instance we try:
  close_r : morphological closing radius (px at 2048) - bridges small gaps and notches
  fill    : fill interior holes
  eps     : polygon simplification of the outer contour (px), like a hand-drawn polygon
  grow    : dilate (+) or erode (-) the final outline by this many px
Each setting is scored on ALL validation entries with dataset-level PQ (the organisers' metric).

Semantic masks are cached in data/_sem_classical/ (packed bits), so later studies are fast.

    python scripts/smooth_study.py
"""
import itertools
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")  # leave CPU cores to the U-Net training

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.setNumThreads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from filseg import CocoSet, read_image  # noqa: E402
from filseg.metrics import score_image, summarize, to_idx  # noqa: E402
from filseg.pipeline import semantic_classical  # noqa: E402
from filseg.postprocess import to_instances  # noqa: E402
from run import load_cp, load_pp, split_ids  # noqa: E402

SEM_DIR = "data/_sem_classical"


def semantic_cached(path, cp):
    out = os.path.join(SEM_DIR, os.path.splitext(os.path.basename(path))[0] + ".npy")
    if os.path.exists(out):
        return np.unpackbits(np.load(out))[: 2048 * 2048].reshape(2048, 2048).astype(bool)
    sem = semantic_classical(read_image(path), cp)
    np.save(out, np.packbits(sem.ravel()))
    return sem


def humanise(m, close_r=0, fill=False, eps=0.0, grow=0):
    """Reshape one instance mask (bool 2048x2048). Works on a padded bounding box for speed."""
    if not (close_r or fill or eps or grow):
        return m
    pad = max(close_r, abs(grow)) + 2
    ys, xs = np.nonzero(m)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, m.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, m.shape[1])
    c = m[y0:y1, x0:x1].astype(np.uint8)
    if close_r:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_r + 1, 2 * close_r + 1))
        c = cv2.morphologyEx(c, cv2.MORPH_CLOSE, k)
    if fill:  # redraw each piece's outer outline, filled (optionally simplified like a hand polygon)
        cnts, _ = cv2.findContours(c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        out = np.zeros_like(c)
        for cn in cnts:
            if eps:
                cn = cv2.approxPolyDP(cn, eps, True)
            cv2.drawContours(out, [cn], -1, 1, thickness=-1)
        c = out
    if grow:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(grow) + 1, 2 * abs(grow) + 1))
        c = cv2.dilate(c, k) if grow > 0 else cv2.erode(c, k)
    r = np.zeros_like(m)
    r[y0:y1, x0:x1] = c.astype(bool)
    return r


GRID = {"close_r": [0, 3, 6, 12], "fill": [False, True], "eps": [0.0, 2.0, 4.0], "grow": [-1, 0, 1, 2]}


def main():
    os.makedirs(SEM_DIR, exist_ok=True)
    coco = CocoSet("data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
                   "data/MAGFiLO_1.0_Kaggle_2026/train/train_images")
    _, va = split_ids(coco)
    cp, pp = load_cp(), load_pp("classical")
    by_file = {}
    for i in va:
        by_file.setdefault(coco.path_of[i], []).append(i)
    data, t0 = [], time.time()
    for n, (f, entries) in enumerate(by_file.items()):
        inst = to_instances(semantic_cached(f, cp), pp)
        data.append((inst, [to_idx(coco.instance_masks(e)) for e in entries]))
        if n % 10 == 0:
            print(f"semantic {n + 1}/{len(by_file)}  {(time.time() - t0) / (n + 1):.1f} s/file", flush=True)

    rows = []
    for vals in itertools.product(*GRID.values()):
        prm = dict(zip(GRID, vals))
        if not prm["fill"] and prm["eps"]:
            continue  # simplification only makes sense on a filled outline
        sc = []
        for inst, gts in data:
            pr = to_idx([humanise(m, **prm) for m in inst])
            sc += [score_image(g, pr) for g in gts]
        s = summarize(sc)
        rows.append({**prm, **s})
        print(prm, "PQ=%.4f SQ=%.3f TP=%d FP=%d FN=%d" % (s["PQ_dataset"], s["SQ"], s["TP"], s["FP"], s["FN"]),
              flush=True)
    import pandas as pd
    df = pd.DataFrame(rows).sort_values("PQ_dataset", ascending=False)
    df.to_csv("data/_sem_classical/smooth_study.csv", index=False)
    print("\nBEST:\n", df.head(5).to_string())


if __name__ == "__main__":
    main()
