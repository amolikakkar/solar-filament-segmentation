"""Pre-processing shared by both methods.

1. Find the solar disk (centre + radius) so everything off-disk can be ignored.
2. Flatten limb darkening and slow intensity gradients by dividing by a smooth
   background estimate. After this, quiet Sun ~= 1.0 and filaments are < 1.
3. Optional CLAHE for the neural network input.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Disk:
    cx: float
    cy: float
    r: float

    def mask(self, size: int = 2048, shrink: float = 0.0) -> np.ndarray:
        """Boolean disk mask; `shrink` (fraction of r) trims the noisy limb."""
        yy, xx = np.ogrid[:size, :size]
        rr = self.r * (1.0 - shrink)
        return (xx - self.cx) ** 2 + (yy - self.cy) ** 2 <= rr ** 2

    def scaled(self, factor: float) -> "Disk":
        return Disk(self.cx * factor, self.cy * factor, self.r * factor)


def find_disk(img: np.ndarray) -> Disk:
    """Locate the solar disk: Otsu threshold -> largest blob -> fill holes ->
    least-squares circle fit to the blob's outer contour (robust to bright plage
    and dark features, unlike an enclosing circle)."""
    small = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    u8 = np.clip(small / max(small.max(), 1e-6) * 255, 0, 255).astype(np.uint8)
    u8 = cv2.GaussianBlur(u8, (5, 5), 0)
    _, th = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(th)
    if n <= 1:
        return Disk(1024, 1024, 900)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    blob = (lab == k).astype(np.uint8)
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    # algebraic (Kasa) circle fit, then one round of outlier rejection
    for _ in range(2):
        A = np.c_[2 * pts, np.ones(len(pts))]
        b = (pts ** 2).sum(1)
        (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
        r = np.sqrt(c + cx ** 2 + cy ** 2)
        d = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
        pts = pts[d < max(2.0, 2.5 * np.median(d))]
    s = img.shape[0] / 512.0
    return Disk((cx + 0.5) * s - 0.5, (cy + 0.5) * s - 0.5, r * s)


def flatten(img: np.ndarray, disk: Disk, n_bins: int = 256, grad_sigma: float = 24.0) -> np.ndarray:
    """Divide by a background model so quiet Sun ~ 1. Off-disk pixels set to 1.

    Background = (radial limb-darkening profile) x (smooth large-scale gradient):
      * radial profile: median intensity in thin rings around the disk centre
        (medians ignore filaments, which cover a tiny fraction of each ring);
      * gradient: heavily smoothed residual, computed with masked (normalised)
        convolution so off-disk pixels do not leak in at the limb.
    """
    size = img.shape[0]
    f = 512 / size
    small = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    d = disk.scaled(f)
    yy, xx = np.mgrid[:512, :512]
    rho = np.sqrt((xx - d.cx) ** 2 + (yy - d.cy) ** 2) / d.r
    on = rho <= 0.995
    bins = np.minimum((rho * n_bins).astype(int), n_bins)
    prof = np.full(n_bins + 1, np.nan, np.float32)
    for b in np.unique(bins[on]):
        prof[b] = np.median(small[on & (bins == b)])
    ok = ~np.isnan(prof)
    prof = np.interp(np.arange(n_bins + 1), np.flatnonzero(ok), prof[ok]).astype(np.float32)

    resid = np.where(on, small / np.maximum(prof[bins], 1e-4), 0).astype(np.float32)
    resid = np.where(on, np.clip(resid, 0.85, 1.15), 0).astype(np.float32)  # ignore strong features
    w = on.astype(np.float32)
    num = cv2.GaussianBlur(resid, (0, 0), grad_sigma)
    den = cv2.GaussianBlur(w, (0, 0), grad_sigma)
    grad = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 1.0).astype(np.float32)

    yy, xx = np.mgrid[:size, :size]
    rho_f = np.sqrt((xx - disk.cx) ** 2 + (yy - disk.cy) ** 2) / disk.r
    radial = np.interp(rho_f * n_bins, np.arange(n_bins + 1), prof).astype(np.float32)
    bg = radial * cv2.resize(grad, (size, size), interpolation=cv2.INTER_LINEAR)
    out = img / np.maximum(bg, 1e-4)
    out[rho_f > 0.995] = 1.0
    return out.astype(np.float32)


def to_network_input(flat: np.ndarray, disk_mask: np.ndarray) -> np.ndarray:
    """Map flattened image (quiet Sun ~1) to a well-scaled float input for the CNN."""
    x = np.clip((flat - 0.6) / 0.6, 0, 1)  # 0.6 -> 0, 1.2 -> 1
    u8 = (x * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    x = clahe.apply(u8).astype(np.float32) / 255.0
    x[~disk_mask] = 0.0
    return x


def preprocess(img: np.ndarray) -> tuple[np.ndarray, Disk]:
    """Full preprocessing: returns (flattened image, disk)."""
    disk = find_disk(img)
    return flatten(img, disk), disk
