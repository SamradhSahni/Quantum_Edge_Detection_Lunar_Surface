"""
Confidence / uncertainty maps from quantum probability outputs.

Every quantum method produces a CONTINUOUS edge-strength map before
thresholding.  This module exposes the uniform interface used by the
experiments and the dashboard:

    EdgeResult.map        continuous edge strength (float array)
    EdgeResult.binary(T)  thresholded binary edge map
    EdgeResult.uncertainty  per-pixel shot-noise estimate

For QHED the map is P(ancilla=1 | r, c) (a genuine probability);
for NEQR-gradient it is the mean |Delta f| / 255 estimated from the
multinomial samples falling on each pixel.  The uncertainty map is the
binomial/multinomial standard error per pixel, which Experiment 3
(shots sensitivity) analyses explicitly.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EdgeResult:
    method: str                     # 'qhed' | 'neqr' | 'sobel' | ...
    image_size: int
    strength: np.ndarray            # continuous edge strength map
    uncertainty: np.ndarray = None  # per-pixel std estimate (same units)
    meta: dict = field(default_factory=dict)

    def binary(self, threshold: float) -> np.ndarray:
        return (self.strength >= threshold).astype(np.uint8)


def binomial_uncertainty(prob_map: np.ndarray, shots: int) -> np.ndarray:
    """Std of a binomial frequency estimate, clipped to valid range."""
    p = np.clip(prob_map, 0.0, 1.0)
    return np.sqrt(np.maximum(p * (1.0 - p), 0.0) / max(shots, 1))


def multinomial_mean_uncertainty(values: np.ndarray, counts_per_pixel: np.ndarray,
                                 mean_map: np.ndarray) -> np.ndarray:
    """Std of a per-pixel mean of |signed byte| values over samples.

    values/count arrive as per-pixel accumulation arrays; the std of the
    mean is estimated as sqrt(Var/n) with Var from the observed spread.
    """
    n = np.maximum(counts_per_pixel, 1.0)
    ex2 = values / n  # caller passes second-moment accumulation in `values`
    var = np.maximum(ex2 - mean_map ** 2, 0.0)
    return np.sqrt(var / n)


def qhed_confidence_result(counts_h: dict, counts_v: dict, size: int, shots: int) -> EdgeResult:
    """EdgeResult for QHED with P-based map and binomial uncertainty."""
    from quantum.reconstruction import qhed_counts_to_prob_map
    ph = qhed_counts_to_prob_map(counts_h, size) if counts_h else np.zeros((size, size))
    pv = qhed_counts_to_prob_map(counts_v, size) if counts_v else np.zeros((size, size))
    strength = np.sqrt(ph + pv)  # Sobel-style combination of probabilities
    unc = 0.5 / np.maximum(np.sqrt(ph + pv), 1e-12) * np.sqrt(
        binomial_uncertainty(ph, shots) ** 2 + binomial_uncertainty(pv, shots) ** 2
    ) if shots > 0 else np.zeros_like(strength)
    return EdgeResult("qhed", size, strength, uncertainty=unc,
                      meta={"p_h": ph, "p_v": pv, "shots": shots})


def neqr_confidence_result(counts_h: dict, counts_v: dict, size: int, shots: int) -> EdgeResult:
    """EdgeResult for the NEQR gradient with mean-based map and
    multinomial standard error."""
    from quantum.reconstruction import neqr_counts_to_diff_map
    dh = neqr_counts_to_diff_map(counts_h, size) if counts_h else np.zeros((size, size))
    dv = neqr_counts_to_diff_map(counts_v, size) if counts_v else np.zeros((size, size))
    strength = np.sqrt(dh ** 2 + dv ** 2)
    per_pixel_shots = shots / (size * size)
    unc = np.sqrt(dh ** 2 + dv ** 2) / np.sqrt(np.maximum(per_pixel_shots, 1.0))
    return EdgeResult("neqr", size, strength, uncertainty=unc,
                      meta={"d_h": dh, "d_v": dv, "shots": shots})
