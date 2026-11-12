# Solar Filament Segmentation (BigData Cup 2026)

Instance segmentation of solar filaments in GONG H-alpha full-disk images (MAGFiLO ground truth),
scored with **Panoptic Quality (PQ)**. Two methods share one pre-processing and post-processing pipeline.
The **final model** is a U-Net with an ImageNet-pretrained ResNet-50 encoder (the earlier ResNet-34 version is kept for comparison).
A plain-language explanation of every step is in [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md).

## Results
Validation = 106 held-out image files (168 annotator entries), scored the organisers' way (dataset-level PQ, match at IoU > 0.5);
95% intervals from 1,000 bootstrap resamples of files. Speed = prediction time per image on a laptop CPU.

| Model | Validation PQ | SQ | RQ | TP / FP / FN | Kaggle public PQ | CPU s / image |
|---|---|---|---|---|---|---|
| **U-Net, ImageNet ResNet-50 encoder (final)** | **0.419** (0.399–0.439) | 0.676 | 0.620 | 763 / 493 / 444 | to be added | 15.1 |
| U-Net, ImageNet ResNet-34 encoder (previous) | 0.411 (0.387–0.433) | 0.678 | 0.606 | 763 / 549 / 444 | 0.33 | 6.0 |
| Small U-Net (1.9M parameters, trained 6 h on a laptop CPU) | 0.338 (0.313–0.364) | 0.669 | 0.506 | 549 / 415 / 658 | 0.27 | 8.5 |
| Classical (no training) | 0.262 (0.236–0.288) | 0.650 | 0.403 | 420 / 460 / 787 | 0.21 | 2.1 |

Tried and not used (all on the same 106 files): averaging the ResNet-50 and ResNet-34 maps (0.413), six test-time
views with 90° turns (0.415), "human-like" outline shaping (+0.0002 at best), ResNet-34 on 768 px crops (0.396 vs 0.407
on 60 entries). Details: `docs/HOW_IT_WORKS.md` and `weights/lab_*.csv`.

## Method
| Stage | What it does |
|---|---|
| Pre-processing (`preprocess.py`) | Finds the solar disk (Otsu threshold, then a least-squares circle fit to the disk edge with outlier rejection), then divides by a radial limb-darkening profile × smooth gradient, so the quiet Sun ≈ 1 everywhere |
| **A. Classical** (`classical.py`) | Dark-feature map = max(black-hat, darkness). Robust hysteresis threshold (median + k·MAD), a half-contrast boundary for each filament, and a sunspot filter. No training |
| **B. U-Net, final** (`unet.py`, `--arch smp --encoder resnet50`) | U-Net with an ImageNet-pretrained **ResNet-50** encoder (32.5M parameters, `segmentation-models-pytorch`), trained for 240 min on a Kaggle T4 GPU on 512 px crops of 1024 px flattened images (batch 8). BCE + Dice loss, AdamW, time-budgeted cosine learning rate; prediction averages 3 views (original + 2 flips). Runs on a CPU for prediction |
| B'. U-Net, ResNet-34 (previous submission) | Same recipe with a ResNet-34 encoder (24.4M parameters, batch 16, 150 min). Kept for comparison (`weights/unet_r34.pt`) |
| B''. Small U-Net (`unet.py`, default) | Compact U-Net (about 1.9M parameters) trained on 256 px crops on a laptop CPU; same loss, schedule and views. Kept as a CPU-only baseline (`weights/unet_cpu.pt`) |
| Instances (`postprocess.py`) | Removes specks and fills holes, links fragments closer than `link_radius`, and drops tiny instances |
| Tuning (`pipeline.py`, `scripts/postproc_lab.py`) | Grid search of the threshold, `link_radius` and `min_instance` on validation files, choosing the values with the best **dataset-level** PQ (the way the organisers score). For the final model the probability maps of all 106 validation files are saved once (`scripts/cache_probs.py`) so the whole grid is scored on all of them, with a two-half check against luck |
| Metrics (`metrics.py`) | PQ / SQ / RQ, matched IoU & Dice distributions, one-to-many (fragmentation), many-to-one (over-merging) |
| Submission | One row per filament: `filament_id = <image_id>_<k>` with the COCO compressed RLE string (`rle.py`) |

## Setup
```bash
conda create -n filseg python=3.11 -y && conda activate filseg
pip install -r requirements.txt
```
`requirements.txt` pins the exact versions used; `requirements-lock.txt` is the full `pip freeze` of the environment.
Unzip the competition data into `data/`, then check where the images and annotation json actually are:
```bash
python scripts/run.py inspect --data data
```
The default paths match the Kaggle zip extracted into `data/` (`data/MAGFiLO_1.0_Kaggle_2026/...`), so `--train-json`, `--train-dir` and `--test-dir` can be left out.
Note: some images are annotated more than once (by different annotators), so the train/validation split is done by image file.

## Reproduce
```bash
# Method A: classical (about 10-20 min tuning on a laptop)
python scripts/run.py classical
python scripts/run.py submit --method classical --out submission_classical.csv

# Method B (final): pretrained ResNet-50 U-Net; needs a GPU for training (on Kaggle: kaggle_train.ipynb, Version 4)
python scripts/run.py train --data data --arch smp --encoder resnet50 --crop 512 --batch 8 --lr 5e-4 --minutes 240 --weights weights/unet_r50.pt
python scripts/cache_probs.py --weights weights/unet_r50.pt --tag r50          # save the 106 validation maps once
python scripts/postproc_lab.py combo --tags r50                                 # score the settings grid on all of them
python scripts/run.py submit --method unet --out submission.csv                 # default: weights/unet_r50.pt + postprocess_unet_r50.json

# Previous model (ResNet-34) and the small CPU-trained U-Net
python scripts/run.py train --data data --arch smp --encoder resnet34 --crop 512 --batch 16 --lr 5e-4 --minutes 150 --weights weights/unet_r34.pt
python scripts/run.py train --data data --minutes 360 --weights weights/unet_cpu.pt

# Full validation (106 files) and report figures
python scripts/eval_val.py --method unet --weights weights/unet_r50.pt --pp weights/postprocess_unet_r50.json --tag unet_r50
python scripts/make_figures.py hist --tags unet_r50:U-Net_ResNet-50 unet_r34:U-Net_ResNet-34 classical_v2:Classical
```
Each checkpoint has its own settings file, `weights/postprocess_<checkpoint name>.json`; `run.py` picks it
automatically (or pass `--pp`). Check that the U-Net training cache is bit-for-bit identical to the reference
pre-processing: `python scripts/verify_cache.py` (add `--all` to check every image).

Everything the notebook needs is saved in `weights/`: the final model **`unet_r50.pt.part1` + `unet_r50.pt.part2`**
(130 MB, split because GitHub accepts at most 100 MB per file; `unet.load_model("weights/unet_r50.pt")` joins the
parts byte for byte, `scripts/split_weights.py` made and verified them) with `postprocess_unet_r50.json`;
the previous `unet_r34.pt` / `postprocess_unet_r34.json`; the small `unet_cpu.pt` / `postprocess_unet_cpu.json`;
`classical_params.json` / `postprocess_classical.json`; and the evaluation results `eval_*.json`, `tune_*.csv`, `lab_*.csv`.
**`pipeline.ipynb`** runs the whole pipeline from those saved files on a CPU, with no retraining.

Tests: `python tests/test_features.py` (instances, split weights, views, averaging; ~20 s) and
`python tests/test_pipeline.py` (RLE, PQ, classical end to end on generated images).
After any code change, `scripts/check_same_results.py` compares a fresh `eval_val.py` run with the saved one
entry by entry (it must print PASS).

## Repository layout
```
docs/HOW_IT_WORKS.md  plain-language walk-through of every step, results, rejected ideas, FAQ
src/filseg/   io.py  preprocess.py  classical.py  unet.py  postprocess.py  metrics.py  rle.py  pipeline.py
scripts/run.py        command-line entry point (inspect, classical, train, tune, tunefilter, submit)
scripts/eval_val.py   full-validation scoring (organisers' counting, bootstrap interval)
scripts/cache_probs.py + postproc_lab.py   save validation probability maps once, then score settings / outline
                      shaping / model averaging on all 106 files with a two-half check
scripts/split_weights.py   store a checkpoint as parts under GitHub's 100 MB limit and verify them
scripts/check_same_results.py   prove a code change left every validation score identical
scripts/make_figures.py  report figures -> figures/
scripts/verify_cache.py, fix_cache.py, check_identical.py   checks of the pre-processing cache
scripts/fp_filter_study.py, smooth_study.py   earlier studies for the classical method
pipeline.ipynb        end-to-end notebook (required by the organisers)
kaggle_train.ipynb    GPU training of the final model on Kaggle
weights/              trained U-Nets + tuned parameters + evaluation results (committed)
report/               LaTeX source of the 4-page report (organisers' ACM template)
tests/                synthetic-data smoke tests
```

## References
- Ahmadzadeh et al., *MAGFiLO*, Scientific Data (2024), doi:10.1038/s41597-024-03876-y
- Kirillov et al., *Panoptic Segmentation*, CVPR (2019), doi:10.1109/CVPR.2019.00963
- Ronneberger et al., *U-Net*, MICCAI (2015)
- He et al., *Deep Residual Learning for Image Recognition*, CVPR (2016)
- Iakubovskii, *Segmentation Models PyTorch*, GitHub (2019)

## License
Code: MIT (see `LICENSE`). The MAGFiLO/GONG data is not included and is licensed separately (CC BY-NC 4.0).

This work uses GONG data from the NSO Integrated Synoptic Program (NSO/AURA/NSF, with NOAA contribution).
