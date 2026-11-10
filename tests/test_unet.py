"""~1-minute U-Net smoke test on generated images: cache -> train -> load -> predict -> submission.
Run: python tests/test_unet.py"""
import os
import sys
import tempfile

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from filseg import CocoSet, decode_rle, pipeline  # noqa: E402
from filseg import unet  # noqa: E402
from make_synthetic import make  # noqa: E402


def test_unet_smoke():
    d = tempfile.mkdtemp()
    make(d, n=3)
    coco = CocoSet(os.path.join(d, "annotations.json"), os.path.join(d, "images"))
    ids = coco.ids()
    files = unet.build_cache(coco, ids, os.path.join(d, "cache"), res=256)
    cfg = unet.TrainConfig(res=256, crop=128, batch=2, steps_per_epoch=5, time_budget_min=0.3)
    wpath = os.path.join(d, "unet.pt")
    model = unet.train(files[:2], files[2:], cfg, wpath)
    model = unet.load_model(wpath)
    prob, disk = unet.predict_prob(model, pipeline.read_image(coco.path_of[ids[0]]))
    assert prob.shape == (2048, 2048) and 0 <= prob.min() and prob.max() <= 1
    df = pipeline.make_submission(sorted(coco.path_of.values()), "unet", os.path.join(d, "sub.csv"), model=model)
    if len(df):
        assert decode_rle(df.segmentation_rle.iloc[0]).shape == (2048, 2048)
    res, _ = pipeline.evaluate(coco, ids[:1], "unet", model=model)
    print("U-Net pipeline runs end to end (scores are meaningless after 20 s of training)")


if __name__ == "__main__":
    test_unet_smoke()
    print("unet test passed")
