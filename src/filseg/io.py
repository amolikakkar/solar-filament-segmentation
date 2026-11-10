"""Data loading for MAGFiLO (COCO-style annotations + GONG H-alpha images).

The Kaggle folder layout is not hard-coded: `discover()` searches a root folder for
image files and COCO json files, so the same code works on Kaggle, Colab or a laptop.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from .rle import decode_rle

IMG_EXTS = (".jpg", ".jpeg", ".png", ".fts", ".fits", ".tif", ".tiff")
IMG_SIZE = 2048


def image_id_from_path(path: str) -> str:
    """'.../20150125172714Mh.jpg' -> '20150125172714Mh' (the id used in submissions)."""
    return os.path.splitext(os.path.basename(path))[0]


def read_image(path: str) -> np.ndarray:
    """Read an H-alpha image as float32 grayscale in [0, 1]."""
    if path.lower().endswith((".fts", ".fits")):
        from astropy.io import fits  # optional dependency

        data = fits.getdata(path).astype(np.float32)
        data = np.nan_to_num(data)
        lo, hi = np.percentile(data, [0.5, 99.5])
        img = np.clip((data - lo) / max(hi - lo, 1e-6), 0, 1)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        maxv = 65535.0 if img.dtype == np.uint16 else 255.0
        img = img.astype(np.float32) / maxv
    if img.shape != (IMG_SIZE, IMG_SIZE):
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return img


def list_images(folder: str) -> list[str]:
    files = []
    for ext in IMG_EXTS:
        files += glob.glob(os.path.join(folder, "**", f"*{ext}"), recursive=True)
        files += glob.glob(os.path.join(folder, "**", f"*{ext.upper()}"), recursive=True)
    return sorted(set(files))


def discover(root: str) -> dict:
    """Print and return what is inside the competition data folder."""
    out = {"json": sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)),
           "csv": sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)),
           "dirs": {}}
    for d in sorted({os.path.dirname(p) for p in list_images(root)}):
        out["dirs"][d] = len(list_images(d))
    for k, v in out.items():
        print(k, ":", v)
    return out


# --------------------------------------------------------------------------- COCO
@dataclass
class CocoSet:
    """Minimal COCO reader: image paths + per-image list of instance annotations."""

    json_path: str
    image_dir: str
    images: dict = field(default_factory=dict)   # coco image id -> record
    anns: dict = field(default_factory=dict)     # coco image id -> [ann, ...]
    path_of: dict = field(default_factory=dict)  # coco image id -> file path

    def __post_init__(self):
        with open(self.json_path) as f:
            coco = json.load(f)
        on_disk = {os.path.basename(p): p for p in list_images(self.image_dir)}
        for im in coco["images"]:
            name = os.path.basename(im["file_name"])
            if name in on_disk:
                self.images[im["id"]] = im
                self.path_of[im["id"]] = on_disk[name]
                self.anns[im["id"]] = []
        for a in coco.get("annotations", []):
            if a["image_id"] in self.anns:
                self.anns[a["image_id"]].append(a)
        print(f"{os.path.basename(self.json_path)}: {len(self.images)} images on disk, "
              f"{sum(len(v) for v in self.anns.values())} filament annotations")

    def ids(self) -> list:
        return sorted(self.images)

    def instance_masks(self, img_id) -> list[np.ndarray]:
        """List of boolean 2048x2048 masks, one per annotated filament."""
        im = self.images[img_id]
        h, w = im.get("height", IMG_SIZE), im.get("width", IMG_SIZE)
        masks = []
        for a in self.anns[img_id]:
            m = ann_to_mask(a["segmentation"], h, w)
            if (h, w) != (IMG_SIZE, IMG_SIZE):
                m = cv2.resize(m.astype(np.uint8), (IMG_SIZE, IMG_SIZE),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            if m.any():
                masks.append(m)
        return masks

    def union_mask(self, img_id) -> np.ndarray:
        masks = self.instance_masks(img_id)
        if not masks:
            return np.zeros((IMG_SIZE, IMG_SIZE), bool)
        return np.logical_or.reduce(masks)


def ann_to_mask(seg, h: int, w: int) -> np.ndarray:
    """COCO segmentation (polygons, uncompressed RLE or compressed RLE) -> bool mask."""
    if isinstance(seg, list):  # polygons
        m = np.zeros((h, w), np.uint8)
        for poly in seg:
            pts = np.asarray(poly, np.float32).reshape(-1, 2)
            if len(pts) >= 3:
                cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
        return m.astype(bool)
    if isinstance(seg, dict):
        counts = seg["counts"]
        hh, ww = seg.get("size", [h, w])
        if isinstance(counts, list):  # uncompressed RLE
            flat = np.zeros(hh * ww, np.uint8)
            pos, val = 0, 0
            for c in counts:
                flat[pos:pos + c] = val
                pos += c
                val ^= 1
            return flat.reshape((ww, hh)).T.astype(bool)
        return decode_rle(counts, hh, ww)
    raise ValueError(f"Unknown segmentation format: {type(seg)}")
