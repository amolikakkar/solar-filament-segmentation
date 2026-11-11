# Solar Filament Segmentation (BigData Cup 2026)

Instance segmentation of solar filaments in GONG H-alpha full-disk images (MAGFiLO ground truth),
scored with **Panoptic Quality (PQ)**. Two methods share one pre-processing and post-processing pipeline.
The **final model** is a U-Net with an ImageNet-pretrained ResNet-34 encoder.

## Results
Validation = 106 held-out image files (168 annotator entries), scored the organisers' way (dataset-level PQ, match at IoU > 0.5);
95% intervals from 1,000 bootstrap resamples of files. Speed = prediction time per image on a laptop CPU.

| Model | Validation PQ | SQ | RQ | Kaggle public PQ | CPU s / image |
|---|---|---|---|---|---|
| **U-Net, ImageNet ResNet-34 encoder (final)** | **0.411** (0.387–0.433) | 0.678 | 0.606 | **0.33** | 6.0 |
| Small U-Net (1.9M parameters, trained 6 h on a laptop CPU) | 0.338 | 0.669 | 0.506 | 0.27 | 8.5 |
| Classical (no training) | 0.262 (0.236–0.288) | 0.650 | 0.403 | 0.21 | 2.1 |

## Method
| Stage | What it does |
|---|---|
| Pre-processing (`preprocess.py`) | Finds the solar disk (Otsu threshold, then a least-squares circle fit to the disk edge with outlier rejection), then divides by a radial limb-darkening profile × smooth gradient, so the quiet Sun ≈ 1 everywhere |
| **A. Classical** (`classical.py`) | Dark-feature map = max(black-hat, darkness). Robust hysteresis threshold (median + k·MAD), a half-contrast boundary for each filament, and a sunspot filter. No training |
| **B. U-Net, final** (`unet.py`, `--arch smp`) | U-Net with an ImageNet-pretrained ResNet-34 encoder (24.4M parameters, `segmentation-models-pytorch`), trained for 150 min on a Kaggle T4 GPU on 512 px crops of 1024 px flattened images. BCE + Dice loss, AdamW, time-budgeted cosine learning rate; flip TTA at inference. Runs on a CPU for prediction |
| B'. Small U-Net (`unet.py`, default) | Compact U-Net (about 1.9M parameters) trained on 256 px crops on a laptop CPU; same loss, schedule and TTA. Kept as a CPU-only baseline (`weights/unet_cpu.pt`) |
| Instances (`postprocess.py`) | Removes specks and fills holes, links fragments closer than `link_radius`, and drops tiny instances |
| Tuning (`pipeline.py`) | Grid search of the threshold, `link_radius` and `min_instance` on a validation split, choosing the values with the best **dataset-level** PQ (the way the organisers score) |
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

# Method B (final): pretrained ResNet-34 U-Net; needs a GPU for training (on Kaggle: kaggle_train.ipynb)
python scripts/run.py train --data data --arch smp --encoder resnet34 --crop 512 --batch 16 --lr 5e-4 --minutes 150
python scripts/run.py tune  --method unet --n-val 60
python scripts/run.py submit --method unet --out submission.csv

# Small U-Net baseline (trains on a CPU; time-boxed)
python scripts/run.py train --data data --minutes 360

# Full validation (106 files) and report figures
python scripts/eval_val.py --method unet --weights weights/unet.pt --pp weights/postprocess_unet.json --tag unet_gpu
python scripts/make_figures.py hist --tags unet_gpu:U-Net_ResNet-34 unet_cpu:U-Net_small classical_v2:Classical
```
Check that the U-Net training cache is bit-for-bit identical to the reference pre-processing:
`python scripts/verify_cache.py` (add `--all` to check every image).

Everything the notebook needs is saved in `weights/`: `unet.pt` (final model), `postprocess_unet.json`,
`classical_params.json`, `postprocess_classical.json`, plus the small baseline `unet_cpu.pt` / `postprocess_unet_cpu.json`.
**`pipeline.ipynb`** runs the whole pipeline from those saved files on a CPU, with no retraining.

Smoke tests on generated data: `python tests/test_pipeline.py`

## Repository layout
```
src/filseg/   io.py  preprocess.py  classical.py  unet.py  postprocess.py  metrics.py  rle.py  pipeline.py
scripts/run.py        command-line entry point
scripts/eval_val.py   full-validation scoring (organisers' counting, bootstrap interval)
scripts/make_figures.py  report figures -> figures/
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
