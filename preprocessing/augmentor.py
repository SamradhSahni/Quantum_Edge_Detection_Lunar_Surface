"""
Image degradations (Experiment 4): Gaussian noise, Gaussian blur,
contrast scaling, and lunar-specific illumination conditions.
"""

import cv2
import numpy as np


def gaussian_noise(image: np.ndarray, sigma: float, seed: int = 42) -> np.ndarray:
    """Additive zero-mean Gaussian noise with std sigma (in [0,1] units)."""
    if sigma <= 0:
        return image.copy()
    rng = np.random.default_rng(seed)
    noisy = np.asarray(image, dtype=np.float64) + rng.normal(0.0, sigma, image.shape)
    return np.clip(noisy, 0.0, 1.0)


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with kernel auto-sized from sigma."""
    if sigma <= 0:
        return image.copy()
    k = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(np.asarray(image, dtype=np.float64), (k, k), sigmaX=sigma,
                            borderType=cv2.BORDER_REFLECT)


def contrast_scale(image: np.ndarray, factor: float) -> np.ndarray:
    """Scale intensity about the mid-grey point (keeps [0,1] range)."""
    img = np.asarray(image, dtype=np.float64)
    return np.clip(0.5 + (img - 0.5) * factor, 0.0, 1.0)


def low_illumination(image: np.ndarray, factor: float = 0.3) -> np.ndarray:
    """Uniform darkening (lunar night side / terminator)."""
    return np.clip(np.asarray(image, dtype=np.float64) * factor, 0.0, 1.0)


def directional_gradient_illumination(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Linear horizontal illumination ramp (low sun angle across terrain)."""
    img = np.asarray(image, dtype=np.float64)
    n = img.shape[1]
    ramp = np.linspace(1.0 - strength, 1.0, n)[None, :]
    return np.clip(img * ramp, 0.0, 1.0)


def shadow_region(image: np.ndarray, factor: float = 0.7) -> np.ndarray:
    """Half-image darkening (crater shadow)."""
    img = np.asarray(image, dtype=np.float64).copy()
    img[:, img.shape[1] // 2:] *= factor
    return np.clip(img, 0.0, 1.0)


def apply_degradation(image: np.ndarray, family: str, level, seed: int = 42) -> np.ndarray:
    """Dispatch by family name used across Experiment 4 and the dashboard."""
    if family == "gaussian_noise":
        return gaussian_noise(image, level, seed)
    if family == "gaussian_blur":
        return gaussian_blur(image, level)
    if family == "contrast":
        return contrast_scale(image, level)
    if family == "low_illumination":
        return low_illumination(image, level)
    if family == "directional_illumination":
        return directional_gradient_illumination(image, level)
    if family == "shadow":
        return shadow_region(image, level)
    raise ValueError(f"unknown degradation family {family}")
