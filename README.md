# Solar Filament Segmentation (BigData Cup 2026)

Instance segmentation of solar filaments in GONG H-alpha full-disk images (MAGFiLO ground truth),
scored with **Panoptic Quality (PQ)**. Two methods share one pre-processing and post-processing pipeline:

| Stage | What it does |
|---|---|
| Pre-processing (`preprocess.py`) | Finds the solar disk (Otsu + enclosing circle), then divides by a radial limb-darkening profile × smooth gradient, so the quiet Sun ≈ 1 everywhere |
| **A. Classical** (`classical.py`) | Dark-feature map = max(black-hat, darkness). Robust hysteresis threshold (median + k·MAD), a half-contrast boundary for each filament, and a sunspot filter. No training |
| **B. U-Net** (`unet.py`) | Compact U-Net (about 1.9M parameters) trained on 256 px crops of 1024 px images. Uses BCE + Dice loss and a time-budgeted cosine learning-rate schedule, with flip TTA at inference |
| Instances (`postprocess.py`) | Removes specks and fills holes, links fragments closer than `link_radius`, and drops tiny instances |
| Tuning (`pipeline.py`) | Grid search of the threshold, `link_radius` and `min_instance` on a validation split, choosing the values with the best **dataset-level** PQ (the way the organisers score) |
| Metrics (`metrics.py`) | PQ / SQ / RQ, matched IoU & Dice distributions, one-to-many (fragmentation), many-to-one (over-merging) |
| Submission | One row per filament: `filament_id = <image_id>_<k>` with the COCO compressed RLE string (`rle.py`) |

## Setup
```bash
conda create -n filseg python=3.11 -y && conda activate filseg
pip install -r requirements.txt
```
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

# Method B: U-Net (training is time-boxed; GPU used automatically if present)
python scripts/run.py train --data data --minutes 240
python scripts/run.py tune  --method unet
python scripts/run.py submit --method unet --out submission.csv
```
Check that the U-Net training cache is bit-for-bit identical to the reference pre-processing:
`python scripts/verify_cache.py` (add `--all` to check every image).

Everything the notebook needs is saved in `weights/`: `unet.pt`, `classical_params.json` and `postprocess_*.json`.
**`pipeline.ipynb`** runs the whole pipeline from those saved files, with no retraining.

Smoke tests on generated data: `python tests/test_pipeline.py`

## Repository layout
```
src/filseg/   io.py  preprocess.py  classical.py  unet.py  postprocess.py  metrics.py  rle.py  pipeline.py
scripts/run.py        command-line entry point
pipeline.ipynb        end-to-end notebook (required by the organisers)
weights/              trained U-Net + tuned parameters (committed)
tests/                synthetic-data smoke tests
```

## References
- Ahmadzadeh et al., *MAGFiLO*, Scientific Data (2024), doi:10.1038/s41597-024-03876-y
- Kirillov et al., *Panoptic Segmentation*, CVPR (2019), doi:10.1109/CVPR.2019.00963
- Ronneberger et al., *U-Net*, MICCAI (2015)

## License
Code: MIT (see `LICENSE`). The MAGFiLO/GONG data is not included and is licensed separately (CC BY-NC 4.0).

This work uses GONG data from the NSO Integrated Synoptic Program (NSO/AURA/NSF, with NOAA contribution).
