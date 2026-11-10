# Plan to 15 Nov (deadline 11:59 pm Central US time = Sunday 16 Nov, ~10:30-11:30 am IST — aim to finish by the evening of the 15th)

| Day | Do this | Done when |
|---|---|---|
| **Mon 9** | Download data → `data/`. `pip install -r requirements.txt`. Run `inspect` and fix the paths. Run `classical`, then `submit --method classical`, and upload to Kaggle | First leaderboard score |
| **Tue 10** | Create a GitHub repo (keep it private until the close) and push. Start `train --minutes 300` overnight (or on a Kaggle GPU, see below) | `weights/unet.pt` exists |
| **Wed 11** | `tune --method unet`, then `submit --method unet` and upload. Open `pipeline.ipynb` and look at the overlays: is the problem missed barbs, merged filaments or false dark spots? | Two scores to compare |
| **Thu 12** | Fix the biggest failure. Ideas: train longer, `prob_threshold`, `link_radius`, more classical grid values. Submit the best | Best score so far |
| **Fri 13** | Write the report (outline in `REPORT_OUTLINE.md`). Make figures from the notebook | Draft PDF |
| **Sat 14** | Pin versions (`pip freeze > requirements.txt`). Restart the notebook and run all from scratch. Commit the weights. Polish the README | Clean run |
| **Sun 15** | Make the repo **public**, submit the Google form (report PDF + repo link), choose final Kaggle submissions | Form confirmation |

## Training on a laptop vs Kaggle GPU
A laptop CPU manages roughly 1–3 training steps per second at batch 8, so 4–5 hours gives a usable model.
Before you sleep, plug the laptop in and stop it from going to sleep.
A Kaggle notebook with a free **GPU T4** is around 10× faster and already has the data attached.
To use it, turn on Internet in the notebook settings, `!git clone <your repo>`, `%cd` into it, and run
the same `scripts/run.py` commands with the `/kaggle/input/...` paths. Then download `weights/unet.pt` back into the repo.
Either way the code is the same.

## Things that move PQ the most
1. **Instance grouping.** One prediction per filament. Check `one_to_many` and `many_to_one` in the evaluation output.
2. **Boundaries.** A match needs IoU > 0.5, so thin filaments with sloppy edges fail even when they are found.
3. **False positives.** Each one costs 0.5 in the denominator. `min_instance` removes small spurious blobs.
