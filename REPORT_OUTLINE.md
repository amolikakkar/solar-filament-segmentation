# 4-page report outline (adapt to the section names in the organisers' Overleaf template)

**Title:** Physics-informed pre-processing and PQ-tuned instance grouping for solar filament segmentation

1. **Introduction** (~0.4 page). Why filaments matter: they erupt as CMEs and drive space weather. The task, the PQ metric, and the main contributions:
   (i) limb-darkening flattening shared by both methods; (ii) a training-free classical baseline;
   (iii) a CPU-trainable U-Net; (iv) fragment linking tuned directly for PQ.
2. **Data** (~0.3 page). MAGFiLO/GONG, 2048² H-alpha images, train/validation split (15%, seed 0), and the distribution of filaments per image.
3. **Method** (~1.4 pages, include one pipeline figure)
   - 3.1 Pre-processing: disk fit; background = radial median profile × masked-Gaussian gradient.
   - 3.2 Classical detector: black-hat plus darkness score, MAD-based hysteresis, half-contrast boundary, sunspot filter.
   - 3.3 U-Net: architecture table, loss, augmentation, 1024 px input, 256 px crops, time-budgeted cosine LR, flip TTA.
   - 3.4 Instance formation: dilation-based linking and size filter. Explain the fragmentation vs merging trade-off.
   - 3.5 Parameter selection: grid search that maximises validation PQ.
4. **Experiments** (~1.3 pages)
   - Table: PQ, SQ, RQ, mean Dice, mean IoU, one-to-many, many-to-one, and seconds per image for classical vs U-Net (validation and public leaderboard).
   - Figure: histograms of IoU and per-image PQ.
   - Ablation: PQ vs `link_radius` (from `weights/tune_*.csv`) and the effect of flattening.
   - Figure: 2–3 qualitative overlays, one good and one failure (barbs, limb, sunspots).
5. **Discussion & limitations** (~0.3 page). Seeing and cloud artefacts, the limb, and faint barbs lost at 1024 px. Future work: full-resolution tiles, a boundary-aware loss, spine-guided grouping.
6. **Conclusion and reproducibility** (~0.2 page). Repo link, `pipeline.ipynb`, weights included, hardware used.

References: MAGFiLO (Ahmadzadeh et al. 2024), Panoptic Segmentation (Kirillov et al. 2019), U-Net (Ronneberger et al. 2015).
Acknowledge the GONG/NSO data (see the competition's Acknowledgment text).
