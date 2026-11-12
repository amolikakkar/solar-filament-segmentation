"""Prove that a code change did not change results: compare two eval_val.py outputs entry by entry.

  python scripts/check_same_results.py weights/eval_unet_r34_entries.csv weights/eval_recheck_entries.csv

Every annotator entry present in both files must have the same TP, FP, FN, number of predictions and
sum of matched IoUs (to 1e-12). Prints PASS or FAIL with the differing entries.
"""
import sys

import pandas as pd

a, b = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
m = a.merge(b, on="entry", suffixes=("_ref", "_new"))
cols = ["TP", "FP", "FN", "n_pred", "n_gt"]
bad = m[(m[[c + "_ref" for c in cols]].to_numpy() != m[[c + "_new" for c in cols]].to_numpy()).any(axis=1)
        | ((m.iou_sum_ref - m.iou_sum_new).abs() > 1e-12)]
print(f"{len(m)} entries compared ({len(a)} in reference, {len(b)} in new)")
if len(m) and bad.empty:
    print("PASS: identical TP / FP / FN / predictions / matched IoU for every entry")
else:
    print("FAIL:", len(bad), "entries differ")
    print(bad[["entry"] + [c + s for c in cols + ["iou_sum"] for s in ("_ref", "_new")]].head(20).to_string())
    sys.exit(1)
