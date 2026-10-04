"""
COCO polygon annotation -> pixel-level binary edge mask rasteriser.

SomBench crater annotations are polygons in COCO format.  This module
converts them to per-tile binary masks and then to RIM EDGE masks
(the crater-rim boundary ring, not the filled disc) via a
filled-disc minus eroded-filled-disc construction -- the ground truth
for the quantitative benchmark (PROJECT_SUMMARY.md Section 4).
"""

import json
import os

import cv2
import numpy as np


def load_coco(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rasterise_filled_masks(coco: dict, image_dir: str, out_dir: str,
                           image_id: int = None) -> dict:
    """Rasterise every annotation's polygon segmentation to a filled
    binary mask (per image) and save as PNG.  Returns {filename: mask}."""
    os.makedirs(out_dir, exist_ok=True)
    images = {im["id"]: im for im in coco["images"]}
    anns_by_img = {}
    for ann in coco.get("annotations", []):
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    out = {}
    ids = [image_id] if image_id is not None else sorted(images)
    for iid in ids:
        im = images[iid]
        w, h = im["width"], im["height"]
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns_by_img.get(iid, []):
            segs = ann.get("segmentation", [])
            if segs and isinstance(segs[0], list):  # polygon format
                for poly in segs:
                    pts = np.array(poly, dtype=np.float64).reshape(-1, 2)
                    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
            elif isinstance(segs, dict) and "counts" in segs:  # RLE
                try:
                    from pycocotools import mask as mask_utils
                    rle = segs if isinstance(segs["counts"], list) else \
                        mask_utils.frPyObjects(segs, h, w)
                    m = mask_utils.decode(rle if isinstance(segs["counts"], list) else segs)
                    mask |= m.astype(np.uint8)
                except ImportError:
                    continue
        fname = os.path.splitext(im["file_name"])[0] + "_mask.png"
        cv2.imwrite(os.path.join(out_dir, fname), mask * 255)
        out[im["file_name"]] = mask
    return out


def filled_to_rim_edge(filled_mask: np.ndarray, rim_width: int = 2) -> np.ndarray:
    """Filled crater disc -> rim edge ring.

    rim = filled minus eroded(filled, rim_width); a 1-2 px ring that
    marks the boundary the detectors must recover.  Falls back to the
    morphological gradient for tiny craters.
    """
    filled = (np.asarray(filled_mask) > 0).astype(np.uint8)
    if filled.sum() == 0:
        return np.zeros_like(filled)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(filled, kernel, iterations=rim_width)
    rim = filled - eroded
    if rim.sum() == 0:  # crater smaller than the ring width
        rim = cv2.morphologyEx(filled, cv2.MORPH_GRADIENT, kernel)
    return rim


def group_by_incidence(coco: dict) -> dict:
    """Group image ids by incidence-angle band (illumination metadata).

    Bands: <45 deg (high sun), 45-60, >60 (low sun / long shadows).
    Returns {band: [image filenames]}.
    """
    bands = {"high_sun_lt45": [], "mid_45_60": [], "low_sun_gt60": []}
    for im in coco.get("images", []):
        inc = im.get("incidence_angle") or im.get("sun_elevation")
        if inc is None:
            # try metadata fields commonly present in geospatial COCO
            inc = (im.get("metadata") or {}).get("incidence_angle")
        if inc is None:
            bands.setdefault("unknown", []).append(im["file_name"])
            continue
        inc = float(inc)
        if inc < 45:
            bands["high_sun_lt45"].append(im["file_name"])
        elif inc <= 60:
            bands["mid_45_60"].append(im["file_name"])
        else:
            bands["low_sun_gt60"].append(im["file_name"])
    return bands
