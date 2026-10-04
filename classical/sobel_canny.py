"""Sobel and Canny baseline wrappers (self-contained, no re-export loops)."""

import cv2
import numpy as np


def sobel_magnitude(image: np.ndarray) -> np.ndarray:
    img = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    smax = mag.max()
    return mag / smax if smax > 0 else np.zeros_like(mag)


def canny_edges(image: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    img = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)
    img8 = np.round(img * 255.0).astype(np.uint8)
    return (cv2.Canny(img8, low, high) > 0).astype(np.uint8)


def canny_soft_response(image: np.ndarray) -> np.ndarray:
    img = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)
    img8 = np.round(img * 255.0).astype(np.uint8)
    mag = sobel_magnitude(img)
    nms_mask = cv2.Canny(img8, 20, 60) > 0
    out = mag * nms_mask
    om = out.max()
    return out / om if om > 0 else out
