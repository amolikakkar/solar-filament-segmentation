"""Save a U-Net's validation probability maps, so post-processing experiments (outline shape,
combining models) can be scored in seconds instead of re-running the network each time.

The saved map is exactly what `pipeline.predict_instances` thresholds (float32, flip TTA, zero
outside 99% of the disk radius), stored losslessly (np.savez_compressed).

  python scripts/cache_probs.py --weights weights/unet_r34.pt --tag gpu        # tags used: gpu = ResNet-34,
  python scripts/cache_probs.py --weights weights/unet_r50.pt --tag r50        # r50 = ResNet-50, cpu = small U-Net

Writes data/_probs_<tag>/<image stem>.npz, one per validation image FILE (106 files).
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from filseg import CocoSet, pipeline, unet  # noqa: E402
from filseg.io import read_image  # noqa: E402
from run import split_ids  # noqa: E402

JSON = "data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMGS = "data/MAGFiLO_1.0_Kaggle_2026/train/train_images"


def _replace_retry(tmp, dst, tries=20):
    """os.replace, retried: on Windows a virus scanner can hold a just-written file for a moment
    (WinError 32). Only the file move is retried; the saved data are unchanged."""
    for k in range(tries):
        try:
            os.replace(tmp, dst)
            return
        except PermissionError:
            if k == tries - 1:
                raise
            time.sleep(0.5)


def val_files(coco):
    _, va = split_ids(coco)
    return list(dict.fromkeys(coco.path_of[i] for i in va))   # unique files, split order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tta", default="flip", choices=["flip", "rot"],
                    help="flip = 3 views (default, as in the pipeline); rot = 6 views incl. 90-degree turns")
    a = ap.parse_args()
    tta = True if a.tta == "flip" else "rot"
    coco = CocoSet(JSON, IMGS)
    files = val_files(coco)
    out = os.path.join("data", f"_probs_{a.tag}")
    os.makedirs(out, exist_ok=True)
    model = unet.load_model(a.weights)
    t0 = time.time()
    for n, f in enumerate(files, 1):
        dst = os.path.join(out, os.path.splitext(os.path.basename(f))[0] + ".npz")
        if not os.path.exists(dst):
            prob = pipeline.semantic_unet(read_image(f), model, tta=tta)
            tmp = dst[:-4] + ".tmp.npz"
            np.savez_compressed(tmp, prob=prob.astype(np.float32))
            _replace_retry(tmp, dst)
        if n % 10 == 0 or n == len(files):
            print(f"{n}/{len(files)} files  ({(time.time() - t0) / n:.1f} s/file)", flush=True)
    print("done ->", out)


if __name__ == "__main__":
    main()
