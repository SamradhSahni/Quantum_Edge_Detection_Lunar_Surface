"""
Unified classical pipeline: runs Sobel, Canny, and Robinson together and
returns continuous strength maps + binary maps under the common
relative-threshold protocol.
"""

import numpy as np

from classical.robinson_compass import robinson_compass
from classical.sobel_canny import canny_soft_response, canny_edges, sobel_magnitude


def run_classical(image: np.ndarray, rel_threshold: float = 0.3):
    """All three classical methods on one image.

    Returns dict:
        method -> {'strength': continuous map, 'binary': binary map,
                   'extra': method-specific outputs (Robinson direction map)}
    Thresholds are RELATIVE (fraction of each map's max), identical in
    form to the quantum pipeline's validation-selected threshold.
    """
    image = np.asarray(image, dtype=np.float64)
    out = {}

    s = sobel_magnitude(image)
    out["sobel"] = {"strength": s, "binary": _bin(s, rel_threshold), "extra": {}}

    c_soft = canny_soft_response(image)
    c_bin = canny_edges(image)
    out["canny"] = {"strength": c_soft, "binary": c_bin, "extra": {}}

    r_mag, r_dir, r_resp = robinson_compass(image)
    out["robinson"] = {"strength": r_mag, "binary": _bin(r_mag, rel_threshold),
                       "extra": {"direction": r_dir, "responses": r_resp}}
    return out


def _bin(strength: np.ndarray, rel: float) -> np.ndarray:
    smax = float(strength.max())
    if smax <= 0:
        return np.zeros(strength.shape, dtype=np.uint8)
    return (strength >= rel * smax).astype(np.uint8)
