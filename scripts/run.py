"""Command-line entry point. Run from the repo root, e.g.

  python scripts/run.py inspect  --data data/
  python scripts/run.py classical --data data/ --train-json data/train/annotations.json --train-dir data/train/images
  python scripts/run.py submit   --method classical --test-dir data/test/images --out submission_classical.csv
  python scripts/run.py train    --data data/ --train-json ... --train-dir ... --minutes 180
  python scripts/run.py tune     --method unet --train-json ... --train-dir ...
  python scripts/run.py submit   --method unet --test-dir data/test/images --out submission.csv
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from filseg import CocoSet, ClassicalParams, PostParams, discover, list_images  # noqa: E402
from dataclasses import asdict  # noqa: E402
from filseg import pipeline  # noqa: E402

CP_FILE = "weights/classical_params.json"
FINAL_WEIGHTS = "weights/unet_r50.pt"   # final model (stored as .part1/.part2, joined on loading)


def model_tag(method, weights):
    """Name that links a model to its settings files: 'classical', or the checkpoint name without '.pt'
    (weights/unet_r50.pt -> 'unet_r50', whose settings are weights/postprocess_unet_r50.json)."""
    if method == "classical":
        return "classical"
    return "+".join(os.path.basename(w.strip())[: -len(".pt")] for w in weights.split(","))


def pp_file(method, weights):
    return f"weights/postprocess_{model_tag(method, weights)}.json"


def split_ids(coco, val_frac=0.15, seed=0):
    """Train/validation split by image FILE: MAGFiLO lists some images several times
    (one entry per annotator), and all copies of an image must land on the same side."""
    files = np.array(sorted({coco.path_of[i] for i in coco.ids()}))
    rng = np.random.default_rng(seed)
    rng.shuffle(files)
    val_files = set(files[: max(1, int(len(files) * val_frac))])
    tr = [i for i in coco.ids() if coco.path_of[i] not in val_files]
    va = [i for i in coco.ids() if coco.path_of[i] in val_files]
    return tr, va


def load_pp(path):
    if os.path.exists(path):
        return PostParams(**json.load(open(path)))
    print(f"note: {path} not found, using default post-processing settings")
    return PostParams()


def load_cp():
    return ClassicalParams(**json.load(open(CP_FILE))) if os.path.exists(CP_FILE) else ClassicalParams()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["inspect", "classical", "train", "tune", "tunefilter", "submit"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--train-json", default="data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    ap.add_argument("--train-dir", default="data/MAGFiLO_1.0_Kaggle_2026/train/train_images")
    ap.add_argument("--test-dir", default="data/MAGFiLO_1.0_Kaggle_2026/test/test_images")
    ap.add_argument("--method", default="unet", choices=["classical", "unet"])
    ap.add_argument("--minutes", type=float, default=120)
    ap.add_argument("--max-train", type=int, default=0, help="use at most N training images (0 = all)")
    ap.add_argument("--n-val", type=int, default=40, help="validation images used for tuning")
    ap.add_argument("--weights", default=FINAL_WEIGHTS,
                    help="checkpoint; several comma-separated checkpoints are averaged (ensemble)")
    ap.add_argument("--pp", default=None,
                    help="post-processing settings json (default: weights/postprocess_<checkpoint name>.json)")
    ap.add_argument("--out", default="submission.csv")
    # training options (left unset = the original defaults in unet.TrainConfig)
    ap.add_argument("--arch", choices=["small", "smp"], default=None)
    ap.add_argument("--encoder", default=None, help="smp encoder, e.g. resnet34, efficientnet-b4")
    ap.add_argument("--crop", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    a = ap.parse_args()
    os.makedirs("weights", exist_ok=True)
    tag = model_tag(a.method, a.weights)
    pp_path = a.pp or pp_file(a.method, a.weights)

    if a.cmd == "inspect":
        discover(a.data)
        return

    if a.cmd in ("classical", "train", "tune", "tunefilter"):
        coco = CocoSet(a.train_json, a.train_dir)
        tr, va = split_ids(coco)
        if a.max_train:
            tr = tr[: a.max_train]
        va_small = va[: a.n_val]

    if a.cmd == "classical":
        # 1) detector thresholds on a few images, 2) instance grouping, 3) report validation scores
        cp, df = pipeline.tune_classical(coco, va_small[:25])
        df.to_csv("weights/tune_classical_detector.csv", index=False)
        json.dump(asdict(cp), open(CP_FILE, "w"), indent=2)
        best, df = pipeline.tune_postprocess(coco, va_small, "classical", cp=cp)
        df.to_csv("weights/tune_classical.csv", index=False)
        json.dump(asdict(best), open(pp_file("classical", a.weights), "w"), indent=2)
        pipeline.evaluate(coco, va_small, "classical", pp=best, cp=cp)

    elif a.cmd == "train":
        from filseg import unet

        cache = os.path.join(a.data, "_cache1024")
        tr_files = unet.build_cache(coco, tr, cache)
        va_files = unet.build_cache(coco, va[:20], cache)
        extra = {k: getattr(a, k) for k in ("arch", "encoder", "crop", "batch", "lr") if getattr(a, k) is not None}
        cfg = unet.TrainConfig(time_budget_min=a.minutes, **extra)
        unet.train(tr_files, va_files, cfg, a.weights)

    elif a.cmd == "tune":
        from filseg import unet

        model = unet.load_model(a.weights) if a.method == "unet" else None
        best, df = pipeline.tune_postprocess(coco, va_small, a.method, model=model)
        df.to_csv(f"weights/tune_{tag}.csv", index=False)
        json.dump(asdict(best), open(pp_path, "w"), indent=2)
        pipeline.evaluate(coco, va_small, a.method, model=model, pp=best)

    elif a.cmd == "tunefilter":
        # stage 2: false-positive filter + outline growth, on ALL validation files
        model = None
        if a.method == "unet":
            from filseg import unet

            model = unet.load_model(a.weights)
        pp0 = load_pp(pp_path)
        best, df = pipeline.tune_filter(coco, va, a.method, model=model, pp=pp0, cp=load_cp())
        df.to_csv(f"weights/tune_filter_{tag}.csv", index=False)
        json.dump(asdict(best), open(pp_path, "w"), indent=2)
        print("filter OFF :", df[(df.min_dark == 0) & (df.min_elong == 0) & (df.max_rho >= 9) & (df.grow == 0)]
              .iloc[0][["PQ_dataset", "TP", "FP", "FN"]].to_dict())
        print("best       :", df.iloc[0].to_dict())

    elif a.cmd == "submit":
        model = None
        if a.method == "unet":
            from filseg import unet

            model = unet.load_model(a.weights)
        paths = list_images(a.test_dir)
        pp = load_pp(pp_path)
        print(f"{a.method}: {a.weights if a.method == 'unet' else ''} with {pp_path}: {pp}")
        pipeline.make_submission(paths, a.method, a.out, model=model, pp=pp, cp=load_cp())


if __name__ == "__main__":
    main()
