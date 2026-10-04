"""
Crater-patch extraction from lunar tiles (SomBench or fallback).

For every tile with a rim mask, extract image patches centred on crater
rims at 8x8 and 16x16 (the resolutions the quantum pipelines use) with
matching rim ground-truth masks downsampled to patch resolution.
"""

import os
import re

import cv2
import numpy as np

import config
from preprocessing.image_loader import list_images, load_grayscale


def extract_crater_patches(sizes=(8, 16), max_patches_per_size: int = 30) -> dict:
    """Extract rim-centred patches + masks into data/sombench/patches|masks.

    Patch naming:  patch_{size}px_{tile}_{iy}_{ix}.png
    Returns a summary dict {size: n_patches}.
    """
    summary = {}
    tiles = list_images(config.SOMBENCH_IMG_DIR)
    masks = {os.path.splitext(f)[0]: f for f in list_images(config.SOMBENCH_MASK_DIR)}

    count = {s: 0 for s in sizes}
    for tile_name in tiles:
        if count[min(sizes)] >= max_patches_per_size:
            break
        stem = os.path.splitext(tile_name)[0]
        mask_name = None
        for key in masks:
            if stem.startswith(key.replace("_mask", "")) or stem == key.replace("_mask", ""):
                mask_name = masks[key]
                break
        if mask_name is None:
            continue
        img = load_grayscale(os.path.join(config.SOMBENCH_IMG_DIR, tile_name))
        mask = cv2.imread(os.path.join(config.SOMBENCH_MASK_DIR, mask_name),
                          cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = (mask > 0).astype(np.uint8)

        # find rim pixels and take patch windows centred on them
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            continue
        step = max(1, len(ys) // 12)
        for size in sizes:
            half = size // 2
            for i in range(0, len(ys), step):
                if count[size] >= max_patches_per_size:
                    break
                cy, cx = int(ys[i]), int(xs[i])
                y0, x0 = cy - half, cx - half
                if y0 < 0 or x0 < 0 or y0 + size > img.shape[0] or x0 + size > img.shape[1]:
                    continue
                patch = img[y0:y0 + size, x0:x0 + size]
                pmask = mask[y0:y0 + size, x0:x0 + size]
                if patch.shape != (size, size):
                    continue
                name = f"patch_{size}px_{stem}_{y0}_{x0}.png"
                cv2.imwrite(os.path.join(config.SOMBENCH_PATCH_DIR, name),
                            np.round(patch * 255).astype(np.uint8))
                cv2.imwrite(os.path.join(config.SOMBENCH_MASK_DIR, name),
                            pmask * 255)
                count[size] += 1
    summary = dict(count)
    print(f"extracted patches: {summary}")
    return summary


def load_patches(size: int):
    """All patches of one size as [(name, image, mask)]."""
    out = []
    for f in list_images(config.SOMBENCH_PATCH_DIR):
        if not f.startswith(f"patch_{size}px_"):
            continue
        img = load_grayscale(os.path.join(config.SOMBENCH_PATCH_DIR, f))
        mask_path = os.path.join(config.SOMBENCH_MASK_DIR, f)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8) if mask is not None else np.zeros((size, size))
        out.append((f, img, mask))
    return out
