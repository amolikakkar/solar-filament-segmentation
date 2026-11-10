"""Method A: classical computer vision (no training).

Idea: after flattening, filaments are thin, elongated dark structures on the disk.
  1. Dark-feature map = max(black-hat transform, plain darkness) on a smoothed image.
     Black-hat (closing - image) responds to dark features narrower than the kernel,
     which is exactly what filaments and their barbs are.
  2. Hysteresis threshold, with thresholds set from robust statistics of the disk
     (median + k * MAD), so it adapts to each image's seeing and contrast.
  3. Boundary refinement at a fraction of each filament's peak contrast (FWHM-like).
  4. Shape filter: drop compact, very dark blobs (sunspots) and tiny specks.
The output is a single binary mask; `postprocess.to_instances` splits it into filaments.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage import measure
from skimage.filters import apply_hysteresis_threshold

from .preprocess import Disk


@dataclass
class ClassicalParams:
    work_size: int = 1024      # process at half resolution for speed
    smooth_sigma: float = 1.2  # pixels at work_size
    blackhat_size: int = 31    # pixels at work_size; > typical filament width
    k_low: float = 3.0         # hysteresis thresholds in robust-sigma units
    k_high: float = 5.5
    limb_shrink: float = 0.04  # ignore the outer 4% of the radius (limb noise)
    sunspot_max_intensity: float = 0.55  # umbra much darker than any filament
    sunspot_max_ecc: float = 0.85
    min_area: int = 40         # pixels at work_size
    edge_frac: float = 0.5     # boundary at this fraction of each filament's peak contrast


def segment_classical(flat: np.ndarray, disk: Disk, p: ClassicalParams = ClassicalParams()) -> np.ndarray:
    full = flat.shape[0]
    s = p.work_size / full
    img = cv2.resize(flat, (p.work_size, p.work_size), interpolation=cv2.INTER_AREA)
    dmask = disk.scaled(s).mask(p.work_size, shrink=p.limb_shrink)

    sm = cv2.GaussianBlur(img, (0, 0), p.smooth_sigma)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.blackhat_size, p.blackhat_size))
    blackhat = cv2.morphologyEx(sm, cv2.MORPH_BLACKHAT, k)
    darkness = np.clip(1.0 - sm, 0, None)
    score = np.maximum(blackhat, darkness)
    score[~dmask] = 0

    vals = score[dmask]
    med = np.median(vals)
    mad = 1.4826 * np.median(np.abs(vals - med)) + 1e-6
    mask = apply_hysteresis_threshold(score, med + p.k_low * mad, med + p.k_high * mad)
    mask &= dmask

    # shape / intensity filter
    lab = measure.label(mask, connectivity=2)
    keep = np.zeros(lab.max() + 1, bool)
    peak = np.zeros(lab.max() + 1, np.float32)
    for r in measure.regionprops(lab, intensity_image=sm):
        if r.area < p.min_area:
            continue
        is_spot = r.intensity_min < p.sunspot_max_intensity and r.eccentricity < p.sunspot_max_ecc
        keep[r.label] = not is_spot
        peak[r.label] = np.percentile(score[r.slice][r.image], 95)
    # "half-maximum" boundary: smoothing and hysteresis widen features, so trim each
    # filament to pixels whose contrast is at least edge_frac of that filament's peak.
    mask = keep[lab] & (score >= p.edge_frac * peak[lab])

    return cv2.resize(mask.astype(np.uint8), (full, full), interpolation=cv2.INTER_NEAREST).astype(bool)
