"""
Image loaders: grayscale conversion, normalisation to [0,1], resizing to
the power-of-two patches the quantum pipelines require.
"""

import os

import cv2
import numpy as np


def load_grayscale(path: str) -> np.ndarray:
    """Load an image as [0,1] float grayscale (any input depth/colour)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float64)
    if img.max() > 1.5:  # 8-bit or 16-bit input
        img = img / 255.0 if img.max() <= 255.5 else img / 65535.0
    return img


def load_grayscale_from_bytes(data: bytes) -> np.ndarray:
    """Decode raw bytes (uploaded file) to [0,1] float grayscale."""
    import io
    from PIL import Image
    arr = np.array(Image.open(io.BytesIO(data)).convert("L"), dtype=np.float64)
    if arr.max() > 1.5:
        arr /= 255.0
    return arr


def to_power_of_two(img: np.ndarray, size: int) -> np.ndarray:
    """Resize to size x size (area interpolation preserves flux)."""
    if img.shape[0] == size and img.shape[1] == size:
        return img
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def normalise_unit(img: np.ndarray) -> np.ndarray:
    """Clip to [0,1]."""
    return np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0)


def list_images(directory: str, exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
    out = []
    for e in exts:
        out.extend(f for f in os.listdir(directory) if f.lower().endswith(e))
    return sorted(out)
