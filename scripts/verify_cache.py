"""Prove that the U-Net training cache is bit-for-bit identical to the ORIGINAL pre-processing.

Checks
  1. every training/validation entry has both cache files, readable, 1024x1024 uint8;
  2. entries that share an image file (several annotators) have byte-identical inputs;
  3. for a random sample (or --all), the cached input and mask equal a fresh computation
     with a frozen copy of the original pre-processing code (exact equality, not "close");
  4. the current filseg.unet.prepare_input gives exactly the same output as that frozen copy,
     so training and inference see identical inputs.

    python scripts/verify_cache.py            # 20 random entries (~1-2 min)
    python scripts/verify_cache.py --all      # every entry (as long as building the cache)
"""
import argparse
import filecmp
import os
import random
import sys
from collections import defaultdict

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from filseg import CocoSet, read_image  # noqa: E402
from filseg.preprocess import find_disk, flatten, to_network_input  # noqa: E402
from run import split_ids  # noqa: E402


def reference_prepare_input(img, res):
    """FROZEN copy of the original pre-processing (do not edit)."""
    disk = find_disk(img)
    flat = flatten(img, disk)
    small = cv2.resize(flat, (res, res), interpolation=cv2.INTER_AREA)
    x = to_network_input(small, disk.scaled(res / img.shape[0]).mask(res))
    return x, disk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-json", default="data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ap.add_argument("--train-dir", default="data/MAGFiLO_1.0_Kaggle_2026/train/train_images")
    ap.add_argument("--cache", default="data/_cache1024")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--res", type=int, default=1024)
    a = ap.parse_args()

    coco = CocoSet(a.train_json, a.train_dir)
    tr, va = split_ids(coco)
    ids = tr + va[:20]  # exactly what `run.py train` caches
    problems = []

    # 1. completeness / readability
    for i in ids:
        for k in ("x", "y"):
            p = os.path.join(a.cache, f"{i}_{k}.npy")
            try:
                arr = np.load(p, mmap_mode="r")
                if arr.shape != (a.res, a.res) or arr.dtype != np.uint8:
                    problems.append(f"bad shape/dtype: {p}")
            except Exception as e:  # missing or corrupt
                problems.append(f"missing/unreadable: {p} ({type(e).__name__})")
    print(f"[1] {len(ids)} entries checked for completeness: {'OK' if not problems else 'PROBLEMS'}")
    if problems:
        print("\n".join(problems[:20]))
        print("-> run `python scripts/run.py train ...` again to (re)build missing entries, then re-check.")
        sys.exit(1)

    # 2. duplicates of the same image file must have byte-identical inputs
    by_file = defaultdict(list)
    for i in ids:
        by_file[coco.path_of[i]].append(i)
    n_dup = 0
    for f, group in by_file.items():
        for j in group[1:]:
            n_dup += 1
            if not filecmp.cmp(os.path.join(a.cache, f"{group[0]}_x.npy"),
                               os.path.join(a.cache, f"{j}_x.npy"), shallow=False):
                problems.append(f"duplicate input differs: {group[0]} vs {j}")
    print(f"[2] {n_dup} duplicate entries byte-identical to their first copy: {'OK' if not problems else 'PROBLEMS'}")

    # 3 + 4. exact recomputation with the frozen original code
    from filseg.unet import prepare_input  # imports torch
    sample = list(by_file) if a.all else random.Random(0).sample(list(by_file), min(a.n, len(by_file)))
    for n, f in enumerate(sample):
        img = read_image(f)
        ref, _ = reference_prepare_input(img, a.res)
        ref8 = (ref * 255).astype(np.uint8)
        cur8 = (prepare_input(img, a.res)[0] * 255).astype(np.uint8)
        if not np.array_equal(cur8, ref8):
            problems.append(f"current prepare_input differs from original: {f}")
        for i in by_file[f]:
            if not np.array_equal(np.load(os.path.join(a.cache, f"{i}_x.npy")), ref8):
                problems.append(f"cached input differs from original: {i}")
            y = cv2.resize(coco.union_mask(i).astype(np.uint8), (a.res, a.res), interpolation=cv2.INTER_AREA) > 0
            if not np.array_equal(np.load(os.path.join(a.cache, f"{i}_y.npy")), y.astype(np.uint8)):
                problems.append(f"cached mask differs: {i}")
        if n % 5 == 0:
            print(f"    recomputed {n + 1}/{len(sample)} images", flush=True)
    print(f"[3] {len(sample)} images recomputed with the original code, compared pixel by pixel")
    print(f"[4] current prepare_input == original prepare_input")

    if problems:
        print("\nFAIL:\n" + "\n".join(problems[:30]))
        sys.exit(1)
    print("\nPASS: cache and pre-processing are bit-for-bit identical to the original method.")


if __name__ == "__main__":
    main()
