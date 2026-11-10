"""Local evaluation that mirrors the competition: Panoptic Quality (Kirillov et al., CVPR 2019),
plus Dice / IoU distributions and fragmentation (one-to-many) / over-merging (many-to-one) counts.

PQ = sum_{TP} IoU(y, y_hat) / (|TP| + 0.5|FP| + 0.5|FN|), with a match iff IoU > 0.5
(at IoU > 0.5 matches are guaranteed unique).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def to_idx(masks):
    """Boolean masks -> sorted flat pixel indices (compact; also accepted by score_image)."""
    return [m if m.ndim == 1 else np.flatnonzero(m.ravel()) for m in masks]


def overlap_matrix(gt: list[np.ndarray], pr: list[np.ndarray]):
    """Returns (intersection [G,P], gt_areas [G], pred_areas [P])."""
    gi, pi = to_idx(gt), to_idx(pr)
    inter = np.zeros((len(gi), len(pi)), np.int64)
    for a, g in enumerate(gi):
        for b, q in enumerate(pi):
            if len(g) and len(q) and g[0] <= q[-1] and q[0] <= g[-1]:
                inter[a, b] = np.intersect1d(g, q, assume_unique=True).size
    return inter, np.array([len(g) for g in gi]), np.array([len(q) for q in pi])


@dataclass
class ImageScore:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    iou_sum: float = 0.0
    ious: list = field(default_factory=list)
    dices: list = field(default_factory=list)
    one_to_many: int = 0  # GT filaments covered by >1 prediction (fragmentation)
    many_to_one: int = 0  # predictions covering >1 GT filament (over-merging)

    @property
    def pq(self) -> float:
        d = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.iou_sum / d if d > 0 else float("nan")


def score_image(gt: list[np.ndarray], pr: list[np.ndarray], cover: float = 0.5) -> ImageScore:
    """gt / pr: lists of boolean masks or of flat index arrays (see to_idx)."""
    s = ImageScore()
    if not gt and not pr:
        return s
    inter, ga, pa = overlap_matrix(gt, pr)
    union = ga[:, None] + pa[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    matched_g, matched_p = set(), set()
    for g, p in zip(*np.nonzero(iou > 0.5)):
        matched_g.add(g)
        matched_p.add(p)
        s.iou_sum += iou[g, p]
        s.ious.append(float(iou[g, p]))
        s.dices.append(float(2 * inter[g, p] / (ga[g] + pa[p])))
    s.tp = len(matched_g)
    s.fn = len(gt) - s.tp
    s.fp = len(pr) - len(matched_p)
    if len(gt) and len(pr):
        frac_of_pred = inter / np.maximum(pa[None, :], 1)  # pred mostly inside this GT
        frac_of_gt = inter / np.maximum(ga[:, None], 1)    # GT mostly inside this pred
        s.one_to_many = int(((frac_of_pred > cover).sum(1) > 1).sum())
        s.many_to_one = int(((frac_of_gt > cover).sum(0) > 1).sum())
    return s


def summarize(scores: list[ImageScore]) -> dict:
    tp = sum(s.tp for s in scores)
    fp = sum(s.fp for s in scores)
    fn = sum(s.fn for s in scores)
    iou_sum = sum(s.iou_sum for s in scores)
    per_img = [s.pq for s in scores if not np.isnan(s.pq)]
    ious = [v for s in scores for v in s.ious]
    dices = [v for s in scores for v in s.dices]
    d = tp + 0.5 * fp + 0.5 * fn
    return {
        "PQ_dataset": iou_sum / d if d else float("nan"),
        "PQ_mean_per_image": float(np.mean(per_img)) if per_img else float("nan"),
        "SQ": iou_sum / tp if tp else 0.0,          # segmentation quality
        "RQ": tp / d if d else 0.0,                 # recognition quality
        "TP": tp, "FP": fp, "FN": fn,
        "mean_IoU_matched": float(np.mean(ious)) if ious else 0.0,
        "mean_Dice_matched": float(np.mean(dices)) if dices else 0.0,
        "one_to_many": sum(s.one_to_many for s in scores),
        "many_to_one": sum(s.many_to_one for s in scores),
    }


def semantic_dice(gt_union: np.ndarray, pred_union: np.ndarray) -> float:
    inter = np.logical_and(gt_union, pred_union).sum()
    tot = gt_union.sum() + pred_union.sum()
    return 1.0 if tot == 0 else 2 * inter / tot
