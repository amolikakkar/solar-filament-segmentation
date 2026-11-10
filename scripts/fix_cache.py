"""One-off clean-up of the U-Net training cache (run once, with training STOPPED).

On 10 Nov 2026 (02:07 IST) a faster but slightly different pre-processing was
briefly used. Any cache file written from that moment on MAY have been produced by
it, so all of them are moved (not deleted) to data/_cache1024_quarantine/.
The next `run.py train` recreates them with the original pre-processing.
Files written before that moment were made by the original code and are kept.

    python scripts/fix_cache.py            # move the files
    python scripts/fix_cache.py --dry-run  # only show what would be moved
"""
import argparse
import os
import sys
import time

CUTOFF = 1794256620  # 2026-11-09 20:37:00 UTC = 2026-11-10 02:07 IST (1 min before the change)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/_cache1024")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    q = a.cache.rstrip("/\\") + "_quarantine"
    names = sorted(os.listdir(a.cache))
    # an entry = <id>_x.npy + <id>_y.npy; move BOTH if either is recent (or a leftover .tmp)
    recent = {n.rsplit("_", 1)[0] for n in names
              if n.endswith(".tmp") or os.path.getmtime(os.path.join(a.cache, n)) >= CUTOFF}
    move = [n for n in names if n.rsplit("_", 1)[0] in recent or n.endswith(".tmp")]
    print(f"cache files: {len(names)}  kept (original method): {len(names) - len(move)}  "
          f"to quarantine: {len(move)}   (cutoff {time.ctime(CUTOFF)})")
    if a.dry_run:
        return
    os.makedirs(q, exist_ok=True)
    for k, n in enumerate(move):
        try:
            os.replace(os.path.join(a.cache, n), os.path.join(q, n))  # overwrites an older copy
        except PermissionError:
            print(f"\nSTOPPED after moving {k} of {len(move)} files: '{n}' is in use by another program.\n"
                  "A training run (or verify_cache) is probably still running in another window.\n"
                  "Close EVERY Miniforge window, open a new one, and run this script again\n"
                  "(it is safe to re-run: it continues where it stopped).")
            sys.exit(1)
    print(f"moved {len(move)} files to {q}. You can delete that folder once training has finished.")


if __name__ == "__main__":
    main()
