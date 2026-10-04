"""
SomBench WAC lunar crater dataset acquisition (primary dataset).

Strategy:
  1. Try the HuggingFace hub dataset `ibm-nasa-geospatial/sombench`
     (crater-detection configuration) via the `datasets` library.
  2. If the hub is unreachable or the schema differs, fall back to the
     PROCEDURAL lunar dataset (preprocessing/lunar_fallback.py):
     physically-motivated synthetic crater tiles with analytic rim
     masks, explicitly marked as fallback in every result table.

Both paths land in data/sombench/{images,annotations,masks} so all
downstream code is source-agnostic.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

SOMBENCH_HF_ID = "ibm-nasa-geospatial/sombench"


def try_download_sombench(max_images: int = 40) -> bool:
    """Attempt HuggingFace download; returns True on success."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets library not available")
        return False
    try:
        print(f"attempting HuggingFace download: {SOMBENCH_HF_ID}")
        ds = load_dataset(SOMBENCH_HF_ID, split="train", streaming=True)
        from preprocessing.annotation_rasteriser import filled_to_rim_edge
        import cv2

        n_saved = 0
        coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "crater"}]}
        ann_id = 1
        for i, example in enumerate(ds):
            if n_saved >= max_images:
                break
            img = None
            if "image" in example:
                img = np.array(example["image"].convert("L")) if hasattr(example["image"], "convert") else np.array(example["image"])
            elif "pixel_values" in example:
                img = np.array(example["pixel_values"])
            if img is None or img.ndim != 2:
                continue
            h, w = img.shape[:2]
            fname = f"sombench_{i:04d}.png"
            cv2.imwrite(os.path.join(config.SOMBENCH_IMG_DIR, fname), img)

            # annotations: object-level polygons/bboxes if present
            anns = example.get("objects") or example.get("annotations") or {}
            filled = np.zeros((h, w), dtype=np.uint8)
            bboxes = anns.get("bbox", []) if isinstance(anns, dict) else []
            for (x0, y0, bw, bh) in bboxes:
                cv2.rectangle(filled, (int(x0), int(y0)),
                              (int(x0 + bw), int(y0 + bh)), 1, -1)
            coco["images"].append({"id": i, "file_name": fname,
                                   "width": w, "height": h})
            if bboxes:
                coco["annotations"].append({
                    "id": ann_id, "image_id": i, "category_id": 1,
                    "segmentation": [], "bbox": list(map(float, bboxes[0]))})
                ann_id += 1
                rim = filled_to_rim_edge(filled)
            else:
                rim = np.zeros((h, w), dtype=np.uint8)
            cv2.imwrite(os.path.join(config.SOMBENCH_MASK_DIR,
                                     os.path.splitext(fname)[0] + "_mask.png"), rim * 255)
            n_saved += 1

        with open(os.path.join(config.SOMBENCH_ANN_DIR, "sombench_coco.json"), "w") as f:
            json.dump(coco, f)
        print(f"saved {n_saved} SomBench tiles")
        return n_saved > 0
    except Exception as e:
        print(f"HuggingFace download failed: {type(e).__name__}: {e}")
        return False


def ensure_sombench(max_images: int = 40) -> dict:
    """Guarantee a usable lunar dataset; report which source was used."""
    have_imgs = any(f.endswith(".png") for f in os.listdir(config.SOMBENCH_IMG_DIR)) \
        if os.path.isdir(config.SOMBENCH_IMG_DIR) else False
    if not have_imgs:
        ok = try_download_sombench(max_images)
        if not ok:
            print("falling back to procedural lunar crater dataset")
            from preprocessing.lunar_fallback import generate_fallback_dataset
            generate_fallback_dataset(n_tiles=max_images)
    else:
        print("SomBench tiles already present on disk")

    # Patch extraction (8x8/16x16) from whatever tiles exist
    from preprocessing.patch_extractor import extract_crater_patches
    summary = extract_crater_patches()
    return summary


if __name__ == "__main__":
    print(json.dumps(ensure_sombench(), indent=2))
