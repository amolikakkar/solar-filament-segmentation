"""Smoke tests: `python -m pytest tests -q` (or `python tests/test_pipeline.py`)."""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from filseg import CocoSet, decode_rle, encode_rle, score_image, summarize  # noqa: E402
from filseg import pipeline, rle  # noqa: E402
from make_synthetic import make  # noqa: E402


def test_rle_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(5):
        m = np.zeros((2048, 2048), bool)
        for _ in range(4):
            r, c = rng.integers(0, 1900, 2)
            m[r:r + rng.integers(1, 150), c:c + rng.integers(1, 150)] = True
        m[0, 0] = rng.random() < 0.5  # mask starting with a 1
        s = encode_rle(m)
        assert np.array_equal(decode_rle(s), m)
        # pure-python path must agree with whatever encode_rle used
        assert rle._counts_to_string(rle._counts(m)) == s
        assert rle._string_to_counts(s) == rle._counts(m)


def test_pq_perfect_and_empty():
    a = np.zeros((64, 64), bool); a[5:20, 5:20] = True
    b = np.zeros((64, 64), bool); b[40:60, 40:50] = True
    s = summarize([score_image([a, b], [a, b])])
    assert abs(s["PQ_dataset"] - 1) < 1e-9
    s = summarize([score_image([a, b], [])])
    assert s["PQ_dataset"] == 0 and s["FN"] == 2
    half = a.copy(); half[5:20, 12:20] = False  # fragment
    other = a & ~half
    s = score_image([a], [half, other])
    assert s.one_to_many == 1


def test_end_to_end_classical():
    d = tempfile.mkdtemp()
    make(d, n=3)
    coco = CocoSet(os.path.join(d, "annotations.json"), os.path.join(d, "images"))
    res, _ = pipeline.evaluate(coco, coco.ids(), "classical")
    assert res["PQ_dataset"] > 0.3, res
    df = pipeline.make_submission(sorted(coco.path_of.values()), "classical", os.path.join(d, "sub.csv"))
    assert len(df) > 0 and df.filament_id.is_unique
    m = decode_rle(df.segmentation_rle.iloc[0])
    assert m.shape == (2048, 2048) and m.any()


if __name__ == "__main__":
    test_rle_roundtrip(); test_pq_perfect_and_empty(); test_end_to_end_classical()
    print("all tests passed")
