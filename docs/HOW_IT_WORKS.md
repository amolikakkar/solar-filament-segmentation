# How the solution works, in plain words

This page explains every step of the solution without assuming a machine-learning background:
what each step does, **why** it is there, and where to find it in the code. The last sections list
what we tried that did not help, the honest limitations, and answers to questions a reviewer may ask.

---

## 1. The task

The GONG telescopes take full-disk pictures of the Sun in H-alpha light (2048 × 2048 pixels, grayscale).
Solar **filaments** show up as long, dark, thread-like clouds on the disk. They matter for space weather
because filament eruptions can launch coronal mass ejections.

We must draw an outline (a "mask") around **each filament separately**. This is *instance* segmentation:
two filaments next to each other must come out as two separate masks, not one big blob.

## 2. The data

* **707 training images.** Several of them were outlined by 2 or 3 different human experts, so the
  annotation file has **1,154 image entries** (one per expert per image) with 8,199 filament outlines.
* **180 test images**, with no outlines given. Kaggle scores our predictions on them.
* The experts often disagree: scoring one expert against another gives only about **PQ 0.34**
  (from a public analysis notebook by another participant). So a perfect score is not possible, even in principle.

**Our validation split.** We hold back 15% of the image *files* (106 files = 168 expert entries) and never
train on them. We split **by file**, not by entry: otherwise the same picture, outlined by a second expert,
could be in both the training and the validation set, and the validation score would be too optimistic.
(`scripts/run.py: split_ids`, fixed random seed 0.)

## 3. How the score works (Panoptic Quality, PQ)

1. Each predicted filament is paired with an expert's filament if they overlap by **more than half**
   (IoU > 0.5; IoU = shared area ÷ combined area).
2. Paired = **true positive (TP)**. Unpaired prediction = **false positive (FP)**. Missed expert filament = **false negative (FN)**.
3. PQ = (sum of IoU of all pairs) ÷ (TP + ½ FP + ½ FN).

It rewards finding filaments (recognition quality, **RQ**) and drawing them accurately (segmentation quality,
**SQ**); PQ = SQ × RQ. The organisers add up all images and all experts together ("dataset-level"),
and we choose all our settings the same way (`pipeline.py: TUNE_KEY = "PQ_dataset"`).

## 4. The pipeline, step by step

```
image ─► find the disk ─► flatten brightness ─► U-Net ─► probability map ─► masks ─► separate filaments ─► CSV
```

| # | Step | What it does | Why | Code |
|---|---|---|---|---|
| 1 | Read the image | Loads the JPEG as one grayscale channel | The images are grayscale; the organisers say not to treat them as colour | `io.py` |
| 2 | Find the solar disk | Otsu threshold separates Sun from sky; a least-squares circle is fitted to the disk edge, once more after dropping outlier edge points | Everything outside the disk must be ignored, and the disk size and position change between telescopes | `preprocess.py: find_disk` |
| 3 | Flatten the brightness | Divides the image by the Sun's average brightness at each distance from the centre (the disk is darker near its edge, "limb darkening"), and by a smooth left-right/up-down gradient | Afterwards the quiet Sun is about 1.0 everywhere, so "dark" means the same thing at the centre and at the edge | `preprocess.py: flatten` |
| 4 | Prepare the network input | Keeps brightness between 0.6 and 1.2, boosts local contrast (CLAHE), shrinks to 1024 × 1024 | Filaments are subtle; CLAHE makes them clearer. 1024 px keeps the detail while making the network 4× faster | `preprocess.py: to_network_input`, `unet.py: prepare_input` |
| 5 | U-Net | A neural network that looks at the image and gives, for every pixel, a probability (0–1) that it belongs to a filament | U-Nets are the standard tool for drawing outlines; they combine "what is it" (deep layers) with "exactly where" (skip connections) | `unet.py` |
| 6 | Three views (TTA) | The image is also given flipped left-right and flipped up-down; the three answers are averaged | Averaging removes some random mistakes; the Sun has no preferred direction | `unet.py: predict_prob` |
| 7 | Disk edge | Probabilities outside 99% of the disk radius are set to 0 | The very edge of the disk is noisy and almost never annotated | `pipeline.py: semantic_unet` |
| 8 | Probability → mask | Pixels with probability ≥ **0.7** become filament | The threshold was chosen to give the best PQ on validation | `postprocess.py` |
| 9 | Clean-up | Removes specks under 60 pixels and fills holes up to 200 pixels | Tiny specks are noise; human outlines have no small holes | `postprocess.py: clean` |
| 10 | Separate the filaments | Pieces closer than about **12 px** are joined into one filament; each joined group is one filament; groups under 400 pixels are dropped | PQ counts *objects*: splitting one filament in two, or keeping tiny bits, costs points. Joining is done by temporarily widening the pieces, so the final outlines are not widened | `postprocess.py: _link` |
| 11 | Write the answer | Each filament mask is compressed with the COCO run-length code and written as one CSV row `<image>_<k>` | This is the format Kaggle asks for | `rle.py`, `pipeline.py: make_submission` |

### How the U-Net was trained (`kaggle_train.ipynb`, `unet.py: train`)

* **Network.** U-Net whose "encoder" (the half that reads the image) is a **ResNet-50** already trained on
  ImageNet, a large public photo collection (32.5 million numbers to learn in total). Starting from a network
  that already recognises edges, lines and textures learns much faster than starting from zero. The first layer
  is adapted to one (grayscale) input channel. ImageNet weights are public, as the rules require.
  (Our previous model used the smaller ResNet-34 encoder, 24.4 million; it is kept for comparison.)
* **Training images.** The flattened 1024 px images. The network is shown random 512 × 512 pieces, 70% of them
  centred on a filament (filaments cover under 1% of the disk, so random pieces would be almost all empty).
  Pieces are randomly flipped, rotated by 90° and made slightly brighter or darker, so the network does not
  learn one fixed orientation or exposure.
* **What it learns from (the loss).** Binary cross-entropy (is each pixel right?) plus Dice loss (does the
  overall shape overlap?). Dice copes with filaments being so rare.
* **How it learns.** AdamW optimiser, learning rate 0.0005 lowered smoothly (cosine schedule) over a fixed
  **240-minute** budget on a Kaggle T4 GPU, batch of 8 (77 rounds of 200 steps). After every 200 steps it is
  checked on 20 validation entries, and the best version is kept (validation Dice 0.707). The weights file is
  130 MB, more than GitHub accepts in one file, so it is stored as two parts (`weights/unet_r50.pt.part1`,
  `.part2`) that the code joins back byte for byte when loading.
* **Tuning after training.** Threshold, fragment joining and minimum size were searched on **all 106 validation
  files** (24 combinations) and saved in `weights/postprocess_unet_r50.json`. To check that this choice is not
  just luck on these files, the search was repeated on one half and scored on the other half: both halves chose
  the same setting.

## 5. Results

| Model | Validation PQ (106 files) | Kaggle public PQ | Seconds per image (laptop CPU) |
|---|---|---|---|
| **U-Net + ResNet-50 (final)** | **0.419** (95% range 0.399–0.439) | to be added | 15.1 |
| U-Net + ResNet-34 (previous submission) | 0.411 (95% range 0.387–0.433) | 0.33 | 6.0 |
| Small U-Net trained 6 h on the laptop CPU | 0.338 | 0.27 | 8.5 |
| Classical image processing (no learning) | 0.262 | 0.21 | 2.1 |

* For the three earlier methods the public score is about **0.8 × the validation score**. So the validation set
  ranks methods correctly; the test images are a bit harder.
* The final model finds 763 of 1,207 expert filaments with IoU > 0.5 (the same number as the ResNet-34) but makes
  10% fewer false detections (493 vs 549); matched outlines overlap with Dice 0.80 on average.
* For the ResNet-34, `pipeline.ipynb` re-created the Kaggle submission on the laptop **identical to the pixel**
  (1,475 of 1,475 filament masks). After every code change since, the validation scores were re-computed and
  checked to be identical, entry by entry.

## 6. What we tried that did not help (and why we left it out)

| Idea | Result on validation | Decision |
|---|---|---|
| Classical method (black-hat filter + robust threshold + sunspot filter) | PQ 0.262 | Kept as a no-training baseline |
| Small U-Net trained only on a CPU | PQ 0.338 | Kept as a baseline; the pretrained encoder is clearly better |
| Joining nearby fragments (link radius 6–32 px) | Helps the small U-Net (+0.02), slightly **hurts** the final model | Final model uses no extra joining |
| "Human-like" outlines (widen, smooth, fill, close notches; 48 variants) | Best +0.0002; lower when checked on unseen halves | Not used |
| Same ResNet-34 with larger 768 px training pieces | PQ 0.396 vs 0.407 on the same 60 entries | Not used |
| Averaging the ResNet-34 and ResNet-50 maps | PQ 0.413 vs 0.419 for the ResNet-50 alone | Not used |
| Six views instead of three (adding 90°, 180°, 270° turns) | PQ 0.415 vs 0.419, and twice as slow | Not used |
| False-positive filter (darkness, elongation, distance from centre) | Helped the classical method; no gain for the U-Net (0.3384 → 0.3385), which already ignores non-filament dark features | Not used for the U-Nets |

## 7. Honest limitations

* About 4 in 10 predicted filaments are not matched to an expert filament (FP 493 vs TP 763), often faint
  ones that some experts drew and others did not.
* Long filaments are sometimes split into pieces (45 cases on validation where one expert filament touches
  several predictions), and close neighbours are sometimes merged (39 cases the other way round).
* The ResNet-50 is about 2.5× slower than the ResNet-34 on a laptop CPU (15 vs 6 seconds per image) for a small gain.
* Fine "barbs" (thin side threads) are partly lost because the network works at 1024 px, half the original size.
* Experts themselves agree only to about PQ 0.34, which limits any method.

## 8. Questions a reviewer may ask

**Why a U-Net?** It is the standard, well-understood architecture for pixel-accurate outlines, and it trains
well on a few hundred images.

**Why use a network pre-trained on everyday photos for solar images?** Its first layers detect general
patterns (edges, lines, blobs) that also describe filaments. Fine-tuning a pre-trained ResNet-34 gave PQ 0.411
vs 0.338 for a small U-Net trained from scratch.

**Why ResNet-50 and not ResNet-34?** We trained both on the Kaggle GPU and compared them on the same 106
validation files with the same search for settings: ResNet-50 0.419, ResNet-34 0.414. The gain is small, but
the same setting won on both halves of the validation files. The price is speed (15 instead of 6 seconds per image on a laptop CPU).

**Why flatten the image first?** The Sun is darker near its edge. Without flattening, the network would
confuse the darker edge with filaments.

**How do you stop the model fitting the validation set?** The 106 validation files are never used for training.
Settings were chosen on 60 of them, and the final numbers are reported on all 106 with a bootstrap 95% range.
New ideas were also checked by choosing settings on one half and scoring on the other half.

**Why is the public score lower than validation?** The same 0.8 ratio appears for all three methods, so the
test images are harder rather than the model being over-tuned. Only half of the test set is public, so it is also noisy.

**What about the several experts per image?** Each expert's outlines are scored separately against the same
prediction, as the organisers do. In training, each expert's outline is a separate example.

**How fast is it?** About 15 seconds per image on a laptop CPU with the ResNet-50 (6 s with the ResNet-34); no GPU
is needed for prediction. The 180 test images take about 45 minutes on the laptop.

**Can it be reproduced?** Yes. `pipeline.ipynb` loads `weights/unet_r50.pt` (joined from its two parts; no
retraining, no internet download) and writes the submission file. `requirements.txt` pins every package version.
When the code was changed, we re-scored the earlier model with the new code and checked that every number was
identical (`scripts/check_same_results.py`).

**Did you use AI tools?** Yes, as a coding assistant. Every step above was run, checked and explained; each
design choice was kept only when the validation score showed it helped, and the rejected ideas are listed in section 6.

## 9. Words used

* **Mask**: a black-and-white image, white where the filament is.
* **IoU / Dice**: two ways to measure how much two masks overlap (1 = identical, 0 = no overlap).
* **Epoch / step**: one round of training on a batch of examples.
* **Encoder / decoder**: the half of the U-Net that understands the picture / the half that draws the outline.
* **TTA (test-time augmentation)**: predicting on flipped copies too and averaging.
* **Threshold**: the probability above which a pixel is called "filament".
* **Bootstrap 95% range**: re-scoring on many random re-samples of the validation files to see how much the score could vary.
