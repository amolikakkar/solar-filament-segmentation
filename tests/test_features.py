"""Fast checks of the instance step, model loading, test-time views and averaging (no data needed, ~20 s).
Run: python tests/test_features.py   (or python -m pytest tests -q)"""
import os
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
sys.path.insert(0, HERE)

from filseg import PostParams, pipeline, unet  # noqa: E402
from filseg.postprocess import to_instances  # noqa: E402
from make_synthetic import make  # noqa: E402


def _blob(m, r0, c0, h, w):
    m[r0:r0 + h, c0:c0 + w] = True


def test_instances_split_join_and_size():
    """Two pieces 6 px apart join only when the link radius allows it; specks and small objects are dropped."""
    m = np.zeros((2048, 2048), bool)
    _blob(m, 100, 100, 30, 300)          # piece A
    _blob(m, 100, 406, 30, 300)          # piece B, 6 px gap
    _blob(m, 900, 900, 5, 5)             # 25 px speck -> removed (min_fragment 60)
    _blob(m, 1500, 1500, 15, 20)         # 300 px object -> dropped (min_instance 400)
    apart = to_instances(m, PostParams(link_radius=0))
    joined = to_instances(m, PostParams(link_radius=12))
    assert len(apart) == 2 and len(joined) == 1, (len(apart), len(joined))
    assert joined[0].sum() == apart[0].sum() + apart[1].sum()      # linking never widens outlines
    small = to_instances(m, PostParams(link_radius=0, min_instance=100))
    assert len(small) == 3                                           # 300 px object kept when allowed


def test_checkpoint_parts_load_identically():
    """A checkpoint stored as .part1/.part2 (GitHub size limit) loads to exactly the same tensors."""
    d = tempfile.mkdtemp()
    model = unet.build_model({"arch": "small", "base": 4})
    torch.save({"cfg": {"arch": "small", "base": 4, "res": 256}, "state": model.state_dict()}, os.path.join(d, "m.pt"))
    data = open(os.path.join(d, "m.pt"), "rb").read()
    half = len(data) // 2
    open(os.path.join(d, "p.pt.part1"), "wb").write(data[:half])
    open(os.path.join(d, "p.pt.part2"), "wb").write(data[half:])
    a = unet.load_model(os.path.join(d, "m.pt")).state_dict()
    b = unet.load_model(os.path.join(d, "p.pt")).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_views_and_averaging():
    """tta=True is the 3-view average; 'rot' adds turned views; a list of models averages their maps."""
    d = tempfile.mkdtemp()
    make(d, n=1)
    img = pipeline.read_image(os.path.join(d, "images", sorted(os.listdir(os.path.join(d, "images")))[0]))
    torch.manual_seed(0)
    m1 = unet.build_model({"arch": "small", "base": 4}).eval()
    m2 = unet.build_model({"arch": "small", "base": 4}).eval()
    m1.res = m2.res = 256
    one, _ = unet.predict_prob(m1, img, tta=False)
    three, _ = unet.predict_prob(m1, img, tta=True)
    six, _ = unet.predict_prob(m1, img, tta="rot")
    assert one.shape == three.shape == six.shape == (2048, 2048)
    assert not np.array_equal(one, three) and not np.array_equal(three, six)
    avg = pipeline.semantic_unet(img, [m1, m2])
    manual = np.float32(0.5) * pipeline.semantic_unet(img, m1) + np.float32(0.5) * pipeline.semantic_unet(img, m2)
    assert np.array_equal(avg, manual)


def test_settings_file_names():
    """Each checkpoint has its own settings file: weights/unet_r50.pt -> weights/postprocess_unet_r50.json."""
    import run
    assert run.pp_file("unet", "weights/unet_r50.pt") == "weights/postprocess_unet_r50.json"
    assert run.pp_file("unet", "weights/unet_cpu.pt") == "weights/postprocess_unet_cpu.json"
    assert run.pp_file("classical", "weights/unet_r50.pt") == "weights/postprocess_classical.json"


if __name__ == "__main__":
    test_instances_split_join_and_size()
    test_checkpoint_parts_load_identically()
    test_views_and_averaging()
    test_settings_file_names()
    print("all feature tests passed")
