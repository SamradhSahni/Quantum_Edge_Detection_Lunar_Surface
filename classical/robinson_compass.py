"""
Classical edge-detection baselines: Sobel, Canny, Robinson compass.

All functions map an n x n grayscale float image (values in [0, 1]) to a
CONTINUOUS edge-strength map in the same [0, 1] scale so that every
method -- classical and quantum -- is thresholded by the identical
relative-threshold protocol (quantum/threshold_selector.py) and scored
by the identical metrics.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Sobel
# ---------------------------------------------------------------------------
def sobel_magnitude(image: np.ndarray) -> np.ndarray:
    """Gradient magnitude from 3x3 Sobel kernels, normalised to [0, 1]."""
    img = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    smax = mag.max()
    return mag / smax if smax > 0 else np.zeros_like(mag)


# ---------------------------------------------------------------------------
# Canny
# ---------------------------------------------------------------------------
def canny_edges(image: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """Canny edge map as a binary [0, 1] array.

    Canny is inherently binary (with hysteresis); its "strength map" for
    PR-curve purposes is the binary map itself, and its threshold-free
    PR point is the single operating point selected by its own internal
    double-threshold.  For fairness the same relative-threshold sweep is
    additionally applied to its raw Sobel-magnitude NMS response.
    """
    img = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)
    img8 = np.round(img * 255.0).astype(np.uint8)
    edges = cv2.Canny(img8, low, high)
    return (edges > 0).astype(np.uint8)


def canny_soft_response(image: np.ndarray) -> np.ndarray:
    """Continuous Canny-style response: Sobel magnitude restricted to
    pixels surviving non-maximum suppression.  Gives Canny a proper
    continuous map for PR-AUC and threshold sweeps."""
    img = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)
    img8 = np.round(img * 255.0).astype(np.uint8)
    mag = sobel_magnitude(img)
    nms_mask = cv2.Canny(img8, 20, 60) > 0
    out = mag * nms_mask
    om = out.max()
    return out / om if om > 0 else out


# ---------------------------------------------------------------------------
# Robinson compass operator (8 directions)
# ---------------------------------------------------------------------------
def _robinson_masks():
    """The eight 3x3 Robinson compass masks (Robinson, 1977).

    K0 (east, 0 deg) is the classic Prewitt-style edge mask; the other
    seven are 45-degree rotations of K0.
    """
    K0 = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float64)
    masks = [K0]
    for _ in range(7):
        masks.append(np.rot90(masks[-1], k=-1))  # rotate 45 deg clockwise
    return masks


ROBINSON_MASKS = _robinson_masks()
ROBINSON_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]


def robinson_compass(image: np.ndarray):
    """8-direction Robinson operator.

    Returns
    -------
    magnitude : n x n array   E(x,y) = max_k |R_k(x,y)|, normalised to [0,1]
    direction : n x n int     D(x,y) = argmax_k |R_k(x,y)| in degrees
    responses : list of the 8 raw response maps (unnormalised)
    """
    img = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)
    responses = [cv2.filter2D(img, -1, K, borderType=cv2.BORDER_REFLECT)
                 for K in ROBINSON_MASKS]
    absresp = [np.abs(r) for r in responses]
    stack = np.stack(absresp, axis=0)
    didx = np.argmax(stack, axis=0)
    magnitude = np.take_along_axis(stack, didx[None], axis=0)[0]
    mmax = magnitude.max()
    magnitude = magnitude / mmax if mmax > 0 else np.zeros_like(magnitude)
    direction = np.array(ROBINSON_ANGLES)[didx]
    return magnitude, direction, responses


def robinson_direction_strength(responses, angle: int) -> np.ndarray:
    """Normalised |response| for one compass angle (Experiment 6).

    `responses` is the list returned by robinson_compass(); the value at
    each pixel is |R_k| scaled by the global max over all directions.
    """
    k = ROBINSON_ANGLES.index(angle)
    a = np.abs(responses[k])
    amax = max(float(np.abs(r).max()) for r in responses)
    return a / amax if amax > 0 else a
