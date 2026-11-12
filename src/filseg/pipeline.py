"""End-to-end helpers: predict instances, tune post-processing, write the submission CSV."""
from __future__ import annotations

import itertools
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from .classical import ClassicalParams, segment_classical  # noqa: F401
from .io import image_id_from_path, read_image
from .metrics import score_image, summarize, to_idx
from .postprocess import PostParams, to_instances
from .preprocess import preprocess
from .rle import encode_rle

# Tuning target. The organisers' scorer (self-evaluation notebook) computes PQ over ALL filaments of
# all annotator-image entries at once, so we select settings by that dataset-level PQ.
TUNE_KEY = "PQ_dataset"


def semantic_classical(img, cp: ClassicalParams = ClassicalParams()):
    """Raw image -> binary mask with the classical method."""
    flat, disk = preprocess(img)
    return segment_classical(flat, disk, cp)


def semantic_unet(img, model, tta=True):
    """Raw image -> probability map with the U-Net (thresholded later).
    `model` may also be a list of U-Nets: their maps are averaged with equal weights."""
    if isinstance(model, (list, tuple)):
        w = np.float32(1.0 / len(model))
        return sum(w * semantic_unet(img, m, tta) for m in model)
    from .unet import predict_prob

    prob, disk = predict_prob(model, img, tta=tta)
    prob[~disk.mask(img.shape[0], shrink=0.01)] = 0
    return prob


def predict_instances(img, method: str, model=None, pp: PostParams = PostParams(),
                      cp: ClassicalParams = ClassicalParams()):
    flat = disk = None
    if method == "classical":
        flat, disk = preprocess(img)          # same steps as semantic_classical()
        sem = segment_classical(flat, disk, cp)
    elif method == "unet":
        sem = semantic_unet(img, model, tta=True if pp.tta == "flip" else pp.tta) >= pp.prob_threshold
        if pp.filter_on():
            flat, disk = preprocess(img)
    else:
        raise ValueError(method)
    return to_instances(sem, pp, flat, disk)


def tune_filter(coco, ids, method, model=None, pp: PostParams = PostParams(), cp=ClassicalParams(),
                grid=None):
    """Stage 2 tuning: with linking/size/threshold fixed (pp), choose the false-positive filter
    and outline growth that maximise dataset-level PQ. Each image FILE is processed once and the
    result reused for all its annotator entries; the filter grid itself is then instant."""
    from .postprocess import _link, grow_instance, instance_features

    grid = grid or {"min_dark": [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12],
                    "min_elong": [0.0, 1.5, 2.0, 2.5, 3.0],
                    "max_rho": [9.0, 0.97, 0.94],
                    "grow": [0, 1, 2]}
    by_file = {}
    for i in ids:
        by_file.setdefault(coco.path_of[i], []).append(i)
    feats, ious, n_gt = [], {g: [] for g in grid["grow"]}, 0
    for f, entries in by_file.items():
        img = read_image(f)
        flat, disk = preprocess(img)
        if method == "classical":
            sem = segment_classical(flat, disk, cp)
        else:
            sem = semantic_unet(img, model) >= pp.prob_threshold
        inst = _link(sem, pp)
        gts = [to_idx(coco.instance_masks(e)) for e in entries]
        n_gt += sum(len(g) for g in gts)
        for m in inst:
            ft = instance_features(m, flat, disk)
            for g in grid["grow"]:
                q = to_idx([grow_instance(m, g) if g else m])[0]
                for gt in gts:  # best IoU of this instance against each annotator's filaments
                    best = 0.0
                    for gi in gt:
                        if len(gi) and gi[0] <= q[-1] and q[0] <= gi[-1]:
                            inter = np.intersect1d(gi, q, assume_unique=True).size
                            best = max(best, inter / (len(gi) + len(q) - inter))
                    ious[g].append(best)
            for _ in gts:
                feats.append(ft)
    F = np.array(feats).reshape(-1, 3)
    results = []
    for md, me, mr, g in itertools.product(grid["min_dark"], grid["min_elong"], grid["max_rho"], grid["grow"]):
        iou = np.array(ious[g])
        keep = (F[:, 0] >= md) & (F[:, 1] >= me) & (F[:, 2] <= mr)
        if md == 0 and me == 0 and mr >= 9:
            keep[:] = True  # filter off
        hit = keep & (iou > 0.5)
        tp, fp = int(hit.sum()), int((keep & ~(iou > 0.5)).sum())
        fn = n_gt - tp
        d = tp + 0.5 * fp + 0.5 * fn
        results.append({"min_dark": md, "min_elong": me, "max_rho": mr, "grow": g,
                        "PQ_dataset": iou[hit].sum() / d if d else 0.0, "TP": tp, "FP": fp, "FN": fn})
    df = pd.DataFrame(results).sort_values("PQ_dataset", ascending=False, kind="stable")
    b = df.iloc[0]
    best = replace(pp, min_dark=float(b.min_dark), min_elong=float(b.min_elong),
                   max_rho=float(b.max_rho), grow=int(b.grow))
    return best, df


def evaluate(coco, ids, method, model=None, pp=PostParams(), cp=ClassicalParams(), verbose=True):
    scores, t0 = [], time.time()
    for i in ids:
        img = read_image(coco.path_of[i])
        pred = predict_instances(img, method, model, pp, cp)
        scores.append(score_image(coco.instance_masks(i), pred))
    res = summarize(scores)
    res["sec_per_image"] = (time.time() - t0) / max(len(ids), 1)
    if verbose:
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()})
    return res, scores


def tune_postprocess(coco, ids, method, model=None, cp=ClassicalParams(), grid=None):
    """Grid-search post-processing on a validation split to maximise PQ.

    The expensive semantic step runs once per image; only the cheap instance step is repeated.
    """
    grid = grid or {
        "prob_threshold": [0.35, 0.5, 0.65] if method == "unet" else [0.5],
        "link_radius": [0, 6, 12, 20, 32],
        "min_instance": [150, 400, 1000],
    }
    cache = []  # compact: semantic map as uint8, ground truth as pixel-index arrays
    for i in ids:
        img = read_image(coco.path_of[i])
        if method == "classical":
            sem = semantic_classical(img, cp).astype(np.uint8) * 255
        else:
            sem = np.round(semantic_unet(img, model) * 255).astype(np.uint8)
        cache.append((sem, to_idx(coco.instance_masks(i))))
    keys = list(grid)
    results = []
    for vals in itertools.product(*grid.values()):
        pp = replace(PostParams(), **dict(zip(keys, vals)))
        sc = []
        for sem, gt in cache:
            mask = sem >= max(1, round(pp.prob_threshold * 255))
            sc.append(score_image(gt, to_instances(mask, pp)))
        s = summarize(sc)
        results.append({**dict(zip(keys, vals)), **s})
        print(dict(zip(keys, vals)), "PQ=%.4f" % s[TUNE_KEY], flush=True)
    df = pd.DataFrame(results).sort_values(TUNE_KEY, ascending=False, kind="stable")
    best = replace(PostParams(), **{k: df.iloc[0][k] for k in keys})
    best.link_radius, best.min_instance = int(best.link_radius), int(best.min_instance)
    return best, df


def tune_classical(coco, ids, grid=None, pp: PostParams = PostParams()):
    """Grid-search the classical detector's contrast thresholds / boundary level for PQ."""
    grid = grid or {"k_low": [2.0, 3.0, 4.0], "k_high": [4.5, 5.5, 7.0], "edge_frac": [0.35, 0.5, 0.65]}
    cache = []
    for i in ids:
        flat, disk = preprocess(read_image(coco.path_of[i]))
        cache.append((flat.astype(np.float16), disk, to_idx(coco.instance_masks(i))))
    keys, results = list(grid), []
    for vals in itertools.product(*grid.values()):
        cp = replace(ClassicalParams(), **dict(zip(keys, vals)))
        if cp.k_high <= cp.k_low:
            continue
        sc = [score_image(gt, to_instances(segment_classical(flat.astype(np.float32), disk, cp), pp))
              for flat, disk, gt in cache]
        s = summarize(sc)
        results.append({**dict(zip(keys, vals)), **s})
        print(dict(zip(keys, vals)), "PQ=%.4f" % s[TUNE_KEY], flush=True)
    df = pd.DataFrame(results).sort_values(TUNE_KEY, ascending=False, kind="stable")
    best = replace(ClassicalParams(), **{k: float(df.iloc[0][k]) for k in keys})
    return best, df


def make_submission(image_paths, method, out_csv, model=None, pp=PostParams(), cp=ClassicalParams()):
    rows, t0 = [], time.time()
    for n, path in enumerate(image_paths):
        iid = image_id_from_path(path)
        for k, m in enumerate(predict_instances(read_image(path), method, model, pp, cp), start=1):
            rows.append({"filament_id": f"{iid}_{k}", "segmentation_rle": encode_rle(m)})
        if n % 25 == 0:
            print(f"{n}/{len(image_paths)} images", flush=True)
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    df.to_csv(out_csv, index=False)
    print(f"wrote {len(df)} filaments for {len(image_paths)} images -> {out_csv} "
          f"({(time.time() - t0) / max(len(image_paths), 1):.2f} s/image)")
    return df
