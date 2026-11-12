"""Store a large checkpoint as parts below GitHub's 100 MB file limit, and prove nothing changed.

  python scripts/split_weights.py weights/v4/unet_r50.pt weights/unet_r50.pt

writes weights/unet_r50.pt.part1, .part2, ... (each at most 90 MB). `filseg.unet.load_model("weights/unet_r50.pt")`
joins them back automatically. The script checks that the joined bytes have the same SHA-256 as the source file
and that every tensor of the loaded model is identical.
"""
import hashlib
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from filseg import unet  # noqa: E402

PART = 90 * 1024 * 1024


def main(src: str, dst: str) -> None:
    data = open(src, "rb").read()
    n = 0
    for n, start in enumerate(range(0, len(data), PART), 1):
        with open(f"{dst}.part{n}", "wb") as f:
            f.write(data[start:start + PART])
        print(f"{dst}.part{n}: {min(PART, len(data) - start) / 1e6:.1f} MB")
    joined = b"".join(open(f"{dst}.part{k}", "rb").read() for k in range(1, n + 1))
    same_bytes = hashlib.sha256(joined).hexdigest() == hashlib.sha256(data).hexdigest()
    a = torch.load(src, map_location="cpu", weights_only=False)["state"]
    b = unet.read_checkpoint(dst)["state"]
    same_tensors = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    print(f"{n} parts; joined bytes identical (SHA-256): {same_bytes}; all {len(a)} tensors identical: {same_tensors}")
    if not (same_bytes and same_tensors):
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
