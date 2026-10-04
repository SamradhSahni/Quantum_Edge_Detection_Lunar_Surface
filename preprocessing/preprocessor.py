"""
Preprocessing strategies (Experiment 7): None | Gaussian | Median |
CLAHE | Histogram stretch | Gamma (0.5, 1.5).

All operate on [0,1] float images and return [0,1] float images.
"""

import cv2
import numpy as np

STRATEGIES = ["none", "gaussian", "median", "clahe", "hist_stretch",
              "gamma_0.5", "gamma_1.5"]


def apply_preprocessing(image: np.ndarray, strategy: str) -> np.ndarray:
    img = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)
    if strategy == "none":
        return img
    if strategy == "gaussian":  # sigma=1 per experiment spec
        k = int(2 * round(3 * 1.0) + 1)
        return cv2.GaussianBlur(img, (k, k), sigmaX=1.0, borderType=cv2.BORDER_REFLECT)
    if strategy == "median":
        return cv2.medianBlur(np.round(img * 255).astype(np.uint8), 3).astype(np.float64) / 255.0
    if strategy == "clahe":  # clip=2, tile 4x4 per experiment spec
        u8 = np.round(img * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        return clahe.apply(u8).astype(np.float64) / 255.0
    if strategy == "hist_stretch":
        lo, hi = img.min(), img.max()
        return (img - lo) / (hi - lo) if hi > lo else img.copy()
    if strategy == "gamma_0.5":
        return np.power(img, 0.5)
    if strategy == "gamma_1.5":
        return np.power(img, 1.5)
    raise ValueError(f"unknown preprocessing strategy {strategy}")
