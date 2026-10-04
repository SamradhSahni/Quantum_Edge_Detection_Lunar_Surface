"""
Procedural lunar crater fallback dataset.

Used when the SomBench HuggingFace download is unavailable.  Generates
physically-motivated grayscale crater tiles (512x512) with EXACT
analytic rim masks, so the quantitative benchmark keeps an independent
ground truth even in fallback mode.  Every result table produced from
this data is explicitly labelled "procedural fallback".

Crater model per tile:
  * background terrain: fractal-ish value noise + illumination ramp
  * N craters: bowl profile (rim raise, inner depression, ejecta skirt)
    with radius from a power law; rim = analytic circle at r = R
  * shadows: directional darkening on the far side from the "sun"
  * noise: mild Gaussian terrain texture
"""

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

RNG_SEED = config.SEED


def _value_noise(n: int, cell: int, rng) -> np.ndarray:
    grid = rng.random((n // cell + 2, n // cell + 2))
    img = cv2.resize(grid, (n, n), interpolation=cv2.INTER_CUBIC)
    return (img - img.min()) / (img.max() - img.min() + 1e-12)


def _crater_field(n: int, craters, yy, xx):
    """Sum of bowl-profile craters; returns (elevation, rim_mask)."""
    elev = np.zeros((n, n))
    rim = np.zeros((n, n), dtype=np.uint8)
    for (cy, cx, R) in craters:
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        # rim crest at r=R: raised ring; inner bowl; outer ejecta skirt
        rim_ring = np.exp(-((dist - R) ** 2) / (2 * (0.06 * R) ** 2))
        bowl = -0.6 * np.clip((R * 0.75 - dist) / (R * 0.75), 0, 1) ** 2 * (dist < R * 0.75)
        skirt = 0.25 * np.exp(-(dist - R) ** 2 / (2 * (0.8 * R) ** 2)) * (dist > R)
        elev += 0.5 * rim_ring + bowl + skirt
        ring = np.abs(dist - R) <= max(1.2, 0.045 * R)
        rim |= ring.astype(np.uint8)
    return elev, rim


def generate_tile(idx: int, n: int = 512, rng=None):
    rng = rng or np.random.default_rng(RNG_SEED + idx)
    yy, xx = np.mgrid[0:n, 0:n]

    base = 0.35 + 0.25 * _value_noise(n, 64, rng) + 0.12 * _value_noise(n, 16, rng)

    n_craters = rng.integers(3, 8)
    craters = []
    for _ in range(n_craters):
        R = rng.uniform(n * 0.05, n * 0.18)
        cy = rng.uniform(R, n - R)
        cx = rng.uniform(R, n - R)
        craters.append((cy, cx, R))

    elev, rim = _crater_field(n, craters, yy, xx)
    img = base + 0.35 * elev

    # directional illumination (sun from the left-top)
    sun = np.array([-0.7, -0.7])
    grad = np.zeros(2)
    gy, gx = np.gradient(-elev)
    shade = 1.0 + 0.5 * np.clip(-(gx * sun[0] + gy * sun[1]), -1, 1)
    img = img * shade

    # slight linear illumination ramp across the tile
    ramp = np.linspace(1.0, 0.85, n)[None, :]
    img = img * ramp

    img = img + 0.01 * rng.standard_normal((n, n))
    img = np.clip(img, 0, 1)

    return img, rim, craters


def generate_fallback_dataset(n_tiles: int = 12, size: int = 512) -> dict:
    """Generate tiles + COCO annotations + rim masks under data/lunar_fallback
    AND mirror them into data/sombench/ so all downstream code is agnostic."""
    import shutil

    coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "crater"}]}
    ann_id = 1
    for i in range(n_tiles):
        img, rim, craters = generate_tile(i, size)
        fname = f"lunarfb_{i:03d}.png"
        cv2.imwrite(os.path.join(config.LUNAR_FALLBACK_DIR, "images", fname),
                    np.round(img * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(config.LUNAR_FALLBACK_DIR, "masks",
                                 os.path.splitext(fname)[0] + "_mask.png"), rim * 255)
        # mirror into sombench dirs (fallback source is recorded in metadata)
        shutil.copy(os.path.join(config.LUNAR_FALLBACK_DIR, "images", fname),
                    os.path.join(config.SOMBENCH_IMG_DIR, fname))
        shutil.copy(os.path.join(config.LUNAR_FALLBACK_DIR, "masks",
                                 os.path.splitext(fname)[0] + "_mask.png"),
                    os.path.join(config.SOMBENCH_MASK_DIR,
                                 os.path.splitext(fname)[0] + "_mask.png"))
        # approximate polygon annotation from the rim mask
        cnts, _ = cv2.findContours(rim, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < 40 or c.shape[0] < 4:
                continue
            pts = c.reshape(-1, 2)
            coco["annotations"].append({
                "id": ann_id, "image_id": i, "category_id": 1,
                "segmentation": [pts.astype(float).flatten().tolist()],
                "bbox": [float(pts[:, 0].min()), float(pts[:, 1].min()),
                         float(pts[:, 0].max() - pts[:, 0].min()),
                         float(pts[:, 1].max() - pts[:, 1].min())],
                "attributes": {"source": "procedural_fallback"}})
            ann_id += 1
        coco["images"].append({"id": i, "file_name": fname, "width": size,
                               "height": size,
                               "metadata": {"source": "procedural_fallback"}})

    os.makedirs(config.SOMBENCH_ANN_DIR, exist_ok=True)
    with open(os.path.join(config.SOMBENCH_ANN_DIR, "sombench_coco.json"), "w") as f:
        json.dump(coco, f)
    with open(os.path.join(config.LUNAR_FALLBACK_DIR, "metadata.json"), "w") as f:
        json.dump({"source": "procedural_fallback", "n_tiles": n_tiles,
                   "seed": RNG_SEED, "note":
                   "Analytic crater tiles generated offline; used because the "
                   "SomBench HuggingFace download was unavailable."}, f)
    print(f"generated {n_tiles} procedural lunar tiles ({size}x{size})")
    return coco


if __name__ == "__main__":
    generate_fallback_dataset()
