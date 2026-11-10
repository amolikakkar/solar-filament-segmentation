"""Turn a binary (semantic) mask into individual filament instances.

Panoptic Quality rewards one predicted segment per ground-truth filament, so how
fragments are grouped matters as much as the pixels themselves:
  * too little linking -> one filament split into pieces (one-to-many, FPs)
  * too much linking   -> neighbouring filaments merged (many-to-one, FNs)
We link fragments whose gap is below `link_radius` (dilate -> label -> map back),
then drop instances that are too small to be real filaments. Both values are
tuned on the training set by `tune.py` to maximise PQ.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PostParams:
    prob_threshold: float = 0.5   # only used for network probabilities
    min_fragment: int = 60        # px at 2048 – remove specks before linking
    hole_size: int = 200          # px at 2048 – fill small holes
    link_radius: int = 12         # px at 2048 – fragments closer than ~2x this are merged
    min_instance: int = 400       # px at 2048 – drop tiny final instances
    work_size: int = 1024         # linking is done at this resolution for speed
    # False-positive filter (off by default). An instance is kept only if it is dark enough
    # (1 - median flattened intensity), elongated enough (sqrt of covariance eigenvalue ratio)
    # and its centre lies within max_rho of the disk radius. Needs the flattened image + disk.
    min_dark: float = 0.0
    min_elong: float = 0.0
    max_rho: float = 9.0
    # "Human-like" outline: grow each final instance by this many px (annotators' outlines
    # are slightly wider than the dark core). 0 = off.
    grow: int = 0

    def filter_on(self) -> bool:
        return self.min_dark > 0 or self.min_elong > 0 or self.max_rho < 9.0


def remove_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Drop 8-connected components smaller than `min_area` (version-independent)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[lab]


def clean(mask: np.ndarray, p: PostParams) -> np.ndarray:
    mask = remove_small(mask, p.min_fragment)
    holes = ~remove_small(~mask, p.hole_size + 1)  # small background islands = holes
    return mask | holes


def instance_features(m: np.ndarray, flat: np.ndarray, disk) -> tuple[float, float, float]:
    """(darkness, elongation, centre distance / radius) of one instance mask."""
    ys, xs = np.nonzero(m)
    dark = float(1.0 - np.median(flat[ys, xs]))
    cov = np.cov(np.vstack([xs, ys])) if len(ys) > 2 else np.eye(2)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    elong = float(np.sqrt(ev[0] / max(ev[1], 1e-6)))
    rho = float(np.hypot(xs.mean() - disk.cx, ys.mean() - disk.cy) / disk.r)
    return dark, elong, rho


def keep_instance(feat, p: PostParams) -> bool:
    dark, elong, rho = feat
    return dark >= p.min_dark and elong >= p.min_elong and rho <= p.max_rho


def grow_instance(m: np.ndarray, g: int) -> np.ndarray:
    """Dilate one instance by g px (computed on its bounding box for speed)."""
    ys, xs = np.nonzero(m)
    y0, y1 = max(ys.min() - g - 1, 0), min(ys.max() + g + 2, m.shape[0])
    x0, x1 = max(xs.min() - g - 1, 0), min(xs.max() + g + 2, m.shape[1])
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))
    out = np.zeros_like(m)
    out[y0:y1, x0:x1] = cv2.dilate(m[y0:y1, x0:x1].astype(np.uint8), k).astype(bool)
    return out


def finalize(inst: list[np.ndarray], p: PostParams, flat=None, disk=None) -> list[np.ndarray]:
    """Apply the optional false-positive filter and outline growth (no-ops with defaults)."""
    if p.filter_on():
        if flat is None or disk is None:
            raise ValueError("the false-positive filter needs the flattened image and the disk")
        inst = [m for m in inst if keep_instance(instance_features(m, flat, disk), p)]
    if p.grow > 0:
        inst = [grow_instance(m, p.grow) for m in inst]
    return inst


def to_instances(mask: np.ndarray, p: PostParams = PostParams(), flat=None, disk=None) -> list[np.ndarray]:
    """Binary mask (2048x2048) -> list of boolean instance masks (2048x2048).
    `flat`/`disk` are only needed when the false-positive filter is switched on."""
    return finalize(_link(mask, p), p, flat, disk)


def _link(mask: np.ndarray, p: PostParams) -> list[np.ndarray]:
    mask = clean(mask, p)
    if not mask.any():
        return []
    full = mask.shape[0]
    s = p.work_size / full
    small = cv2.resize(mask.astype(np.uint8), (p.work_size, p.work_size), interpolation=cv2.INTER_NEAREST)
    small = np.maximum(small, cv2.resize(mask.astype(np.uint8), (p.work_size, p.work_size),
                                         interpolation=cv2.INTER_AREA) > 0)
    r = max(1, int(round(p.link_radius * s)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    grown = cv2.dilate(small.astype(np.uint8), k)
    n, lab = cv2.connectedComponents(grown, connectivity=8)
    lab_full = cv2.resize(lab.astype(np.int32).astype(np.float32), (full, full),
                          interpolation=cv2.INTER_NEAREST).astype(np.int32)
    lab_full[~mask] = 0

    areas = np.bincount(lab_full.ravel(), minlength=n)
    out = []
    for i in np.argsort(-areas):
        if i == 0 or areas[i] < p.min_instance:
            continue
        out.append(lab_full == i)
    return out


def instances_from_prob(prob: np.ndarray, disk_mask: np.ndarray, p: PostParams = PostParams(),
                        flat=None, disk=None) -> list[np.ndarray]:
    return to_instances((prob >= p.prob_threshold) & disk_mask, p, flat, disk)
