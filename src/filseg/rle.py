"""COCO compressed-RLE encoding (the `segmentation_rle` column of the submission).

Uses pycocotools when installed; otherwise falls back to a pure-Python port of
pycocotools' rleToString / rleFrString so the code runs anywhere. The two paths
produce identical strings (checked in tests/test_rle.py when pycocotools is present).
"""
from __future__ import annotations

import numpy as np

try:
    from pycocotools import mask as _cocomask
except ImportError:  # pragma: no cover
    _cocomask = None


def _counts(mask: np.ndarray) -> list[int]:
    """Run lengths over the column-major (Fortran) flattened mask, starting with zeros."""
    flat = np.asarray(mask, dtype=np.uint8).flatten(order="F")
    change = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    runs = np.diff(bounds).tolist()
    if flat[0] == 1:
        runs = [0] + runs
    return runs


def _counts_to_string(cnts: list[int]) -> str:
    out = []
    for i, x in enumerate(cnts):
        if i > 2:
            x -= cnts[i - 2]
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            out.append(chr(c + 48))
    return "".join(out)


def _string_to_counts(s: str) -> list[int]:
    cnts, p = [], 0
    while p < len(s):
        x, k, more = 0, 0, True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(cnts) > 2:
            x += cnts[-2]
        cnts.append(x)
    return cnts


def encode_rle(mask: np.ndarray) -> str:
    """bool/uint8 HxW mask -> COCO compressed RLE counts string."""
    if _cocomask is not None:
        rle = _cocomask.encode(np.asfortranarray(mask.astype(np.uint8)))
        return rle["counts"].decode("ascii")
    return _counts_to_string(_counts(mask))


def decode_rle(counts: str | bytes, h: int = 2048, w: int = 2048) -> np.ndarray:
    """COCO compressed RLE counts string -> bool HxW mask."""
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    if _cocomask is not None:
        return _cocomask.decode({"size": [h, w], "counts": counts.encode("ascii")}).astype(bool)
    flat = np.zeros(h * w, np.uint8)
    pos, val = 0, 0
    for c in _string_to_counts(counts):
        flat[pos:pos + c] = val
        pos += c
        val ^= 1
    return flat.reshape((w, h)).T.astype(bool)
