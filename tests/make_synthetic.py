"""Generate a tiny fake MAGFiLO-like dataset (limb-darkened disk + dark filaments + sunspots)
with COCO polygon annotations, so the whole pipeline can be smoke-tested without real data."""
import json
import os
import sys

import cv2
import numpy as np


def make(out_dir, n=6, seed=0):
    rng = np.random.default_rng(seed)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "filament"}]}
    aid = 1
    for k in range(n):
        S = 2048
        cx, cy, r = 1024 + rng.uniform(-20, 20), 1024 + rng.uniform(-20, 20), rng.uniform(880, 940)
        yy, xx = np.mgrid[:S, :S]
        rho = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
        mu = np.sqrt(np.clip(1 - rho ** 2, 0, 1))
        img = np.where(rho <= 1, 0.55 * (0.4 + 0.6 * mu), 0.02).astype(np.float32)
        img *= 1 + 0.08 * (xx / S - 0.5)  # instrument gradient
        name = f"2015012517{k:04d}Mh.jpg"
        coco["images"].append({"id": k + 1, "file_name": name, "height": S, "width": S})
        for _ in range(rng.integers(3, 7)):
            while True:
                x0, y0 = rng.uniform(cx - 0.75 * r, cx + 0.75 * r), rng.uniform(cy - 0.75 * r, cy + 0.75 * r)
                if (x0 - cx) ** 2 + (y0 - cy) ** 2 < (0.75 * r) ** 2:
                    break
            ang, L = rng.uniform(0, np.pi), rng.uniform(80, 300)
            pts = []
            for t in np.linspace(-0.5, 0.5, 8):
                pts.append([x0 + t * L * np.cos(ang) + rng.normal(0, 6), y0 + t * L * np.sin(ang) + rng.normal(0, 6)])
            m = np.zeros((S, S), np.uint8)
            cv2.polylines(m, [np.int32(pts)], False, 1, thickness=int(rng.integers(6, 16)))
            for _ in range(3):  # barbs
                p = pts[rng.integers(1, 7)]
                q = [p[0] + rng.normal(0, 25), p[1] + rng.normal(0, 25)]
                cv2.line(m, np.int32(p), np.int32(q), 1, thickness=3)
            img[m > 0] *= rng.uniform(0.7, 0.82)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            seg = [c.reshape(-1).astype(float).tolist() for c in cnts if len(c) >= 3]
            x, y, w, h = cv2.boundingRect(m)
            coco["annotations"].append({"id": aid, "image_id": k + 1, "category_id": 1, "segmentation": seg,
                                        "area": int(m.sum()), "bbox": [x, y, w, h], "iscrowd": 0})
            aid += 1
        for _ in range(2):  # sunspots (should NOT be segmented)
            sx, sy = cx + rng.uniform(-0.5, 0.5) * r, cy + rng.uniform(-0.5, 0.5) * r
            cv2.circle(img, (int(sx), int(sy)), int(rng.uniform(8, 18)), 0.12, -1)
        img = cv2.GaussianBlur(img, (0, 0), 1.5) + rng.normal(0, 0.012, img.shape).astype(np.float32)
        cv2.imwrite(os.path.join(img_dir, name), np.clip(img * 255, 0, 255).astype(np.uint8))
    with open(os.path.join(out_dir, "annotations.json"), "w") as f:
        json.dump(coco, f)
    print(f"wrote {n} images to {out_dir}")


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "synthetic")
