"""Report figures (PNG, 300 dpi) from saved evaluation/tuning files; no model is run except for overlays.

  python scripts/make_figures.py hist   --tags unet_gpu:U-Net_ResNet-34 unet_cpu:U-Net_small classical:Classical
  python scripts/make_figures.py link   --tune weights/tune_unet.csv:U-Net weights/tune_classical.csv:Classical
  python scripts/make_figures.py overlay --weights weights/unet.pt --pp weights/postprocess_unet.json --n 2

Output: figures/*.png
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Validated categorical palette (fixed order; colour follows the model, never its rank)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})
os.makedirs("figures", exist_ok=True)


def _pairs(tag):
    return pd.read_csv(f"weights/eval_{tag}_pairs.csv")


def hist(tags):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3), sharey=False)
    bins = np.linspace(0, 1, 21)
    for k, (tag, name) in enumerate(tags):
        df = _pairs(tag)
        for ax, col in zip(axes, ("iou", "dice")):
            ax.hist(df[col], bins=bins, histtype="step", linewidth=2, color=SERIES[k],
                    label=f"{name} (n={len(df)})")
    for ax, t in zip(axes, ("IoU of overlapping pairs", "Dice of overlapping pairs")):
        ax.set_title(t, loc="left", color=INK)
        ax.set_xlabel(t.split()[0])
        ax.axvline(0.5, color=INK2, linewidth=0.8, linestyle="--") if t.startswith("IoU") else None
    axes[0].set_ylabel("pairs")
    axes[0].text(0.49, axes[0].get_ylim()[1] * 0.92, "match threshold", fontsize=6.5, color=INK2, ha="right")
    h, lab = axes[1].get_legend_handles_labels()   # legend below the panels so it never covers a curve
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=len(lab), frameon=False)
    fig.savefig("figures/fig_iou_dice_hist.png")
    print("figures/fig_iou_dice_hist.png")


def link(tunes):
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for k, (path, name) in enumerate(tunes):
        df = pd.read_csv(path)
        best = df.groupby("link_radius")["PQ_dataset"].max()
        ax.plot(best.index, best.values, color=SERIES[k], linewidth=2, marker="o", markersize=4, label=name)
        i = best.idxmax()
        ax.annotate(f"{best[i]:.3f}", (i, best[i]), textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=6.5, color=INK)
    ax.set_xlabel("link radius (px, 1024-px working image)")
    ax.set_ylabel("validation PQ (best over other settings)")
    ax.legend(frameon=False)
    fig.savefig("figures/fig_pq_vs_link.png")
    print("figures/fig_pq_vs_link.png")


def overlay(weights, pp_path, n, method):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    sys.path.insert(0, os.path.dirname(__file__))
    import cv2
    from filseg import CocoSet, PostParams, pipeline
    from filseg.io import read_image
    from filseg.metrics import score_image
    from run import split_ids

    coco = CocoSet("data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
                   "data/MAGFiLO_1.0_Kaggle_2026/train/train_images")
    _, va = split_ids(coco)
    pp = PostParams(**json.load(open(pp_path)))
    model = None
    if method == "unet":
        from filseg import unet
        model = unet.load_model(weights)
    # pick evenly spaced validation entries (deterministic), score each, show best and worst of them
    cand = va[:: max(1, len(va) // 8)][:8]
    rows = []
    for e in cand:
        img = read_image(coco.path_of[e])
        pred = pipeline.predict_instances(img, method, model, pp)
        gt = coco.instance_masks(e)
        rows.append((score_image(gt, pred).pq, e, img, gt, pred))
    rows.sort(key=lambda r: -r[0])
    pick = [rows[0], rows[-1]][:n]
    fig, axes = plt.subplots(1, len(pick), figsize=(3.4 * len(pick), 3.5))
    for ax, (pq, e, img, gt, pred) in zip(np.atleast_1d(axes), pick):
        small = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
        lo, hi = np.percentile(small, [1, 99.5])   # auto contrast (raw images are not 0-255)
        ax.imshow(small, cmap="gray", vmin=lo, vmax=hi)
        for masks, col, lw in ((gt, "#86b6ef", 1.6), (pred, "#eb6834", 1.0)):
            for m in masks:
                cs, _ = cv2.findContours(cv2.resize(m.astype(np.uint8), (1024, 1024),
                                                    interpolation=cv2.INTER_NEAREST),
                                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for c in cs:
                    c = c[:, 0, :]
                    ax.plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]], color=col, linewidth=lw)
        ax.set_title(f"{e}  (image PQ {pq:.2f})", fontsize=7, color=INK)
        ax.axis("off")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color="#86b6ef", lw=2, label="annotator"),
                        Line2D([], [], color="#eb6834", lw=2, label="prediction")],
               loc="lower center", ncol=2, frameon=False)
    fig.savefig("figures/fig_overlays.png")
    print("figures/fig_overlays.png", [(r[1], round(r[0], 3)) for r in pick])


def pipeline_fig(weights, pp_path, entry):
    """Four panels from one validation image: raw -> flattened -> U-Net probability -> instances (+ annotator)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    sys.path.insert(0, os.path.dirname(__file__))
    import cv2
    from filseg import CocoSet, PostParams, pipeline, preprocess, unet
    from filseg.io import read_image
    from run import split_ids

    coco = CocoSet("data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
                   "data/MAGFiLO_1.0_Kaggle_2026/train/train_images")
    _, va = split_ids(coco)
    e = entry or va[0]
    pp = PostParams(**json.load(open(pp_path)))
    model = unet.load_model(weights)
    img = read_image(coco.path_of[e])
    flat, _ = preprocess(img)
    prob = pipeline.semantic_unet(img, model)
    pred = pipeline.predict_instances(img, "unet", model, pp)
    r = lambda a, interp=cv2.INTER_AREA: cv2.resize(a, (1024, 1024), interpolation=interp)  # noqa: E731
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.0))
    raw = r(img.astype(np.float32))
    lo, hi = np.percentile(raw, [1, 99.5])   # auto contrast (raw images are not 0-255)
    axes[0].imshow(raw, cmap="gray", vmin=lo, vmax=hi)
    axes[1].imshow(r(flat.astype(np.float32)), cmap="gray", vmin=0.6, vmax=1.2)
    axes[2].imshow(r(prob.astype(np.float32)), cmap="Blues_r", vmin=0, vmax=1)
    axes[3].imshow(r(flat.astype(np.float32)), cmap="gray", vmin=0.6, vmax=1.2)
    for masks, col, lw in ((coco.instance_masks(e), "#86b6ef", 1.0), (pred, "#eb6834", 0.7)):
        for m in masks:
            cs, _ = cv2.findContours(r(m.astype(np.uint8), cv2.INTER_NEAREST), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_NONE)
            for c in cs:
                c = c[:, 0, :]
                axes[3].plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]], color=col, linewidth=lw)
    titles = ["(a) GONG H-alpha input", "(b) disk fit + flattening", "(c) U-Net probability",
              f"(d) {len(pred)} instances (orange)\nvs annotator (blue)"]
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=7, color=INK)
        ax.axis("off")
    fig.savefig("figures/fig_pipeline.png")
    print("figures/fig_pipeline.png", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["hist", "link", "overlay", "pipeline"])
    ap.add_argument("--entry", default=None, help="validation entry id for the pipeline figure")
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--tune", nargs="*", default=[])
    ap.add_argument("--weights", default="weights/unet.pt")
    ap.add_argument("--pp", default="weights/postprocess_unet.json")
    ap.add_argument("--method", default="unet")
    ap.add_argument("--n", type=int, default=2)
    a = ap.parse_args()
    split = lambda xs: [tuple(x.split(":", 1)) if ":" in x else (x, x) for x in xs]  # noqa: E731
    if a.what == "hist":
        hist([(t, n.replace("_", " ")) for t, n in split(a.tags)])
    elif a.what == "link":
        link([(p, n.replace("_", " ")) for p, n in split(a.tune)])
    elif a.what == "pipeline":
        pipeline_fig(a.weights, a.pp, a.entry)
    else:
        overlay(a.weights, a.pp, a.n, a.method)


if __name__ == "__main__":
    main()
