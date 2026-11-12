"""Method B: a compact U-Net trained on the MAGFiLO masks (CPU friendly).

Design choices for a laptop CPU:
  * Images are pre-processed once and cached at 1024x1024 (half resolution keeps
    barbs ~2-5 px wide, which a U-Net can still see).
  * Training uses random 256x256 crops (cost does not depend on image size),
    with flips / 90-degree rotations (the Sun has no preferred orientation).
  * ~1.9M parameters (base width 16). Training is bounded by a time budget so it
    always finishes and saves the best checkpoint.
  * Loss = BCE + soft Dice (filaments cover <1% of pixels, Dice handles imbalance).
The network outputs a filament probability map; instances come from postprocess.py.
"""
from __future__ import annotations

import io
import math
import os
import shutil
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .io import read_image
from .preprocess import find_disk, flatten, to_network_input


# ----------------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, cin: int = 1, base: int = 16, depth: int = 4):
        super().__init__()
        ch = [base * 2 ** i for i in range(depth + 1)]
        self.downs = nn.ModuleList([Block(cin, ch[0])] + [Block(ch[i], ch[i + 1]) for i in range(depth)])
        self.ups = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([Block(ch[i] * 2, ch[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.downs):
            x = blk(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.ups, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], 1))
        return self.head(x)


# ----------------------------------------------------------------------------- data
@dataclass
class TrainConfig:
    res: int = 1024
    crop: int = 256
    batch: int = 8
    lr: float = 2e-3
    base: int = 16
    time_budget_min: float = 120.0  # stop after this many minutes
    steps_per_epoch: int = 200
    pos_crop_frac: float = 0.7      # fraction of crops centred on a filament
    seed: int = 0
    threads: int = 0                # 0 -> all cores
    arch: str = "small"             # "small" = the compact U-Net above; "smp" = U-Net with an
    encoder: str = "resnet34"       #   ImageNet-pretrained encoder (segmentation_models_pytorch, GPU)


def build_model(cfg: dict, pretrained: bool = False) -> nn.Module:
    """Create the network described by a TrainConfig dict (also stored inside each checkpoint)."""
    if cfg.get("arch", "small") == "small":
        return UNet(base=cfg.get("base", 16))
    import segmentation_models_pytorch as smp  # only needed for the pretrained-encoder model

    return smp.Unet(encoder_name=cfg.get("encoder", "resnet34"),
                    encoder_weights="imagenet" if pretrained else None, in_channels=1, classes=1)


def prepare_input(img: np.ndarray, res: int):
    """Raw 2048 image -> (network input at `res`, disk at 2048).

    Reference pre-processing: disk fit and flattening at full 2048 resolution,
    then resize to `res`. Training cache and inference both use this function, so
    the network always sees exactly the same kind of input.
    (scripts/verify_cache.py checks this bit-for-bit against a frozen copy.)
    """
    disk = find_disk(img)
    flat = flatten(img, disk)
    small = cv2.resize(flat, (res, res), interpolation=cv2.INTER_AREA)
    x = to_network_input(small, disk.scaled(res / img.shape[0]).mask(res))
    return x, disk


def _replace(tmp: str, path: str, tries: int = 40) -> None:
    """os.replace with retries: on Windows, antivirus / indexing can lock a freshly
    written file for a moment ("being used by another process")."""
    for k in range(tries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if k == tries - 1:
                raise PermissionError(
                    f"{path} is locked by another program. Close every other Miniforge window "
                    "(an earlier training run may still be running) and run the command again.")
            time.sleep(0.25)


def _save_atomic(path: str, arr: np.ndarray) -> None:
    """Write to a temporary file, then rename: an interrupted run (Ctrl+C) can never
    leave a half-written cache file behind."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    _replace(tmp, path)


def build_cache(coco, ids, cache_dir: str, res: int = 1024) -> list[str]:
    """Pre-process each training image once; store input + union mask as uint8 .npy pairs.

    Plain .npy files are memory-mapped during training, so RAM use stays low even with
    thousands of images (each pair is 2 MB on disk at 1024 px).

    Speed-up with NO effect on results: MAGFiLO lists some images several times (one
    entry per annotator). The network input depends only on the image file, so for the
    2nd/3rd entry of the same file the already-computed input is copied byte-for-byte
    instead of being recomputed. Each entry's own mask is always computed.
    """
    os.makedirs(cache_dir, exist_ok=True)
    paths, x_of_file = [], {}
    for n, i in enumerate(ids):
        out = os.path.join(cache_dir, f"{i}")
        f = coco.path_of[i]
        if not os.path.exists(out + "_x.npy"):
            if f in x_of_file:   # same image file already prepared -> exact copy
                shutil.copyfile(x_of_file[f] + "_x.npy", out + "_x.npy.tmp")
                _replace(out + "_x.npy.tmp", out + "_x.npy")
            else:
                x, _ = prepare_input(read_image(f), res)
                _save_atomic(out + "_x.npy", (x * 255).astype(np.uint8))
        if not os.path.exists(out + "_y.npy"):
            y = cv2.resize(coco.union_mask(i).astype(np.uint8), (res, res), interpolation=cv2.INTER_AREA)
            _save_atomic(out + "_y.npy", (y > 0).astype(np.uint8))
        x_of_file.setdefault(f, out)
        paths.append(out)
        if n % 50 == 0:
            print(f"cache {n}/{len(ids)}", flush=True)
    return paths


def load_pair(prefix: str, mmap: bool = False):
    mode = "r" if mmap else None
    return np.load(prefix + "_x.npy", mmap_mode=mode), np.load(prefix + "_y.npy", mmap_mode=mode)


class CropDataset(torch.utils.data.Dataset):
    def __init__(self, files, cfg: TrainConfig, length: int):
        self.data = [load_pair(f, mmap=True) for f in files]
        self.cfg, self.length = cfg, length
        self.rng = np.random.default_rng(cfg.seed)
        self._fg_cache = {}

    def __len__(self):
        return self.length

    def _fg(self, x, y):
        key = id(y)
        if key not in self._fg_cache:  # foreground coordinates, subsampled to save RAM
            ys, xs = np.nonzero(np.asarray(y))
            self._fg_cache[key] = (ys[::7].astype(np.int16), xs[::7].astype(np.int16))
        return self._fg_cache[key]

    def __getitem__(self, _):
        x, y = self.data[self.rng.integers(len(self.data))]
        c, H = self.cfg.crop, x.shape[0]
        ys, xs = self._fg(x, y) if self.rng.random() < self.cfg.pos_crop_frac else ((), ())
        if len(ys):
            k = self.rng.integers(len(ys))
            r0 = int(np.clip(ys[k] - c // 2 + self.rng.integers(-c // 4, c // 4), 0, H - c))
            c0 = int(np.clip(xs[k] - c // 2 + self.rng.integers(-c // 4, c // 4), 0, H - c))
        else:
            r0, c0 = self.rng.integers(0, H - c, size=2)
        xc = np.asarray(x[r0:r0 + c, c0:c0 + c], dtype=np.float32) / 255
        yc = np.asarray(y[r0:r0 + c, c0:c0 + c], dtype=np.float32)
        k = self.rng.integers(4)
        xc, yc = np.rot90(xc, k), np.rot90(yc, k)
        if self.rng.random() < 0.5:
            xc, yc = xc[:, ::-1], yc[:, ::-1]
        # brightness / contrast jitter for seeing & instrument differences
        xc = np.clip(xc * self.rng.uniform(0.85, 1.15) + self.rng.uniform(-0.05, 0.05), 0, 1)
        return torch.from_numpy(xc.copy())[None], torch.from_numpy(yc.copy())[None]


def loss_fn(logits, y):
    bce = F.binary_cross_entropy_with_logits(logits, y)
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


# ----------------------------------------------------------------------------- train / predict
def train(train_files, val_files, cfg: TrainConfig, out_path: str) -> UNet:
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads or os.cpu_count())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg.__dict__, pretrained=True).to(dev)
    print(f"device={dev}  params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    ds = CropDataset(train_files, cfg, cfg.steps_per_epoch * cfg.batch)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    val = [load_pair(f) for f in val_files]
    t0, best, epoch = time.time(), -1.0, 0
    budget = cfg.time_budget_min * 60
    while time.time() - t0 < budget:
        model.train()
        tot = 0.0
        for xb, yb in dl:
            # learning rate follows a cosine over the *time* budget (short warm-up first)
            frac = min((time.time() - t0) / budget, 1.0)
            lr = cfg.lr * min(1.0, frac / 0.03 + 0.05) * 0.5 * (1 + math.cos(math.pi * frac))
            for g in opt.param_groups:
                g["lr"] = lr
            xb, yb = xb.to(dev), yb.to(dev)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            if time.time() - t0 > budget:
                break
        epoch += 1
        d = evaluate_dice(model, val, dev)
        print(f"epoch {epoch}  loss {tot / len(dl):.4f}  val_dice {d:.4f}  "
              f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
        if d > best:
            best = d
            torch.save({"state": model.state_dict(), "cfg": cfg.__dict__}, out_path)
    print(f"best val dice {best:.4f} -> {out_path}")
    return load_model(out_path)


@torch.no_grad()
def evaluate_dice(model, val, dev) -> float:
    model.eval()
    inter = tot = 0.0
    for xv, yv in val:
        x = torch.from_numpy(xv.astype(np.float32) / 255)[None, None].to(dev)
        p = (torch.sigmoid(model(x))[0, 0].cpu().numpy() > 0.5)
        y = yv > 0
        inter += (p & y).sum()
        tot += p.sum() + y.sum()
    return 2 * inter / max(tot, 1)


def read_checkpoint(path: str) -> dict:
    """torch.load a checkpoint file. GitHub refuses files over 100 MB, so a large checkpoint is stored
    as `<path>.part1`, `<path>.part2`, ... (see scripts/split_weights.py); if `<path>` itself is
    missing, the parts are joined back in memory, byte for byte, before loading."""
    if not os.path.exists(path) and os.path.exists(path + ".part1"):
        data, k = bytearray(), 1
        while os.path.exists(f"{path}.part{k}"):
            with open(f"{path}.part{k}", "rb") as f:
                data += f.read()
            k += 1
        return torch.load(io.BytesIO(bytes(data)), map_location="cpu", weights_only=False)
    return torch.load(path, map_location="cpu", weights_only=False)


def load_model(path: str) -> UNet:
    """Load a checkpoint. A comma-separated list of paths loads several models (a list), whose
    probability maps pipeline.semantic_unet averages with equal weights."""
    if "," in path:
        return [load_model(p.strip()) for p in path.split(",")]
    ck = read_checkpoint(path)
    model = build_model(ck["cfg"], pretrained=False)  # weights come from the checkpoint
    model.load_state_dict(ck["state"])
    model.eval()
    model.res = ck["cfg"].get("res", 1024)
    return model


@torch.no_grad()
def predict_prob(model: UNet, img: np.ndarray, tta: bool | str = True):
    """Raw 2048 image -> (probability map 2048x2048, disk).

    tta=True   : average of 3 views (original, left-right flip, up-down flip) - the default.
    tta="rot"  : the same 3 views plus the image turned by 90, 180 and 270 degrees (6 views).
    tta=False  : the original view only.
    """
    res = getattr(model, "res", 1024)
    x, disk = prepare_input(img, res)
    t = torch.from_numpy(x)[None, None]
    dev = next(model.parameters()).device
    t = t.to(dev)
    prob = torch.sigmoid(model(t))
    if tta:
        prob = prob + torch.sigmoid(model(t.flip(-1))).flip(-1)
        prob = prob + torch.sigmoid(model(t.flip(-2))).flip(-2)
        if tta == "rot":
            for k in (1, 2, 3):
                prob = prob + torch.rot90(torch.sigmoid(model(torch.rot90(t, k, (-2, -1)))), -k, (-2, -1))
            prob = prob / 6
        else:
            prob = prob / 3
    p = prob[0, 0].cpu().numpy()
    return cv2.resize(p, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR), disk
