"""Prove that the new post-processing with DEFAULT settings gives exactly the same instances as the
backed-up original code (backups/2026-11-10_before-fp-filter), for the classical method.

    python scripts/check_identical.py --n 5
"""
import argparse
import importlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


def load(src_dir):
    for k in [k for k in sys.modules if k == "filseg" or k.startswith("filseg.")]:
        del sys.modules[k]
    sys.path.insert(0, src_dir)
    mod = importlib.import_module("filseg.pipeline")
    sys.path.remove(src_dir)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    sys.path.insert(0, HERE)
    import json

    from run import split_ids

    new = load(os.path.join(ROOT, "src"))
    from filseg import CocoSet, read_image
    from filseg.classical import ClassicalParams
    from filseg.postprocess import PostParams

    coco = CocoSet("data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
                   "data/MAGFiLO_1.0_Kaggle_2026/train/train_images")
    _, va = split_ids(coco)
    files = sorted({coco.path_of[i] for i in va})[: a.n]
    cp = ClassicalParams(**json.load(open("weights/classical_params.json")))
    pp_kw = {k: v for k, v in json.load(open("backups/2026-11-10_before-fp-filter/weights/postprocess_classical.json")).items()}
    new_out = [new.predict_instances(read_image(f), "classical", pp=PostParams(**pp_kw), cp=cp) for f in files]

    old = load(os.path.join(ROOT, "backups", "2026-11-10_before-fp-filter", "src"))
    from filseg.classical import ClassicalParams as OCP
    from filseg.postprocess import PostParams as OPP
    old_out = [old.predict_instances(read_image(f), "classical", pp=OPP(**pp_kw), cp=OCP(**cp.__dict__)) for f in files]

    ok = all(len(x) == len(y) and all(np.array_equal(p, q) for p, q in zip(x, y)) for x, y in zip(new_out, old_out))
    print(f"{len(files)} images, {sum(len(x) for x in new_out)} instances compared:",
          "PASS (bit-for-bit identical)" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
