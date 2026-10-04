"""
Edge-map reconstruction from quantum measurement counts.

Both quantum pipelines end with shots-based counts over a single
classical register; this module converts counts into continuous
per-pixel edge-strength maps (the "confidence maps" of the study) and
into Sobel-style combined magnitudes.

QHED (QPIE) counts:  bitstring v over [col(k), row(k), ancilla(1)]
    v & 1                -> ancilla outcome
    (v >> 1) & (n-1)     -> row
    (v >> (k+1)) & (n-1) -> col
    P(a=1, r, c) = |a(r,c) - a(r,c+1)|^2 / 4   (horizontal run)

NEQR-gradient counts: bitstring v over [G(8), F(8), col(k), row(k)]
    v & (n-1)        -> row
    (v >> k) & (n-1) -> col
    (v >> 2k)        -> F byte (pixel value at (r,c))
    (v >> (2k+8))    -> G byte (f(r,c) - f(nb(r,c)) mod 256)
    edge strength    -> |signed(G)| / 255
"""

import numpy as np

from quantum.quantum_gradient import signed_byte


# --------------------------------------------------------------------------
# QHED reconstruction
# --------------------------------------------------------------------------
def qhed_counts_to_prob_map(counts: dict, size: int, direction: str = "horizontal") -> np.ndarray:
    """P(ancilla=1, r, c) map (values in [0, 1/2], unnormalised by shots)."""
    k = int(np.log2(size))
    total = sum(counts.values())
    if total == 0:
        return np.zeros((size, size))
    edge = np.zeros((size, size))
    for key, cnt in counts.items():
        v = int(key.replace(" ", ""), 2)
        if (v & 1) == 0:
            continue
        pos = v >> 1
        r, c = pos & (size - 1), (pos >> k) & (size - 1)
        edge[r, c] += cnt / total
    return edge


def qhed_prob_map_to_magnitude(p_map: np.ndarray, image_norm: float) -> np.ndarray:
    """Convert P(a=1) to |Delta I| units:  |dI| = 2*C*sqrt(P)."""
    return 2.0 * image_norm * np.sqrt(np.clip(p_map, 0.0, None))


def reconstruct_qhed_edge_map(counts_h: dict, counts_v: dict, image,
                              shots_h: int, shots_v: int) -> np.ndarray:
    """Combined Sobel-style QHED edge magnitude map in |Delta I| units."""
    from quantum.qpie_encoder import image_to_amplitudes
    _, norm = image_to_amplitudes(np.asarray(image, dtype=np.float64))
    size = np.asarray(image).shape[0]
    if norm == 0.0:
        return np.zeros((size, size))
    ph = qhed_counts_to_prob_map(counts_h, size) if counts_h else np.zeros((size, size))
    pv = qhed_counts_to_prob_map(counts_v, size) if counts_v else np.zeros((size, size))
    mag_h = qhed_prob_map_to_magnitude(ph, norm)
    mag_v = qhed_prob_map_to_magnitude(pv, norm)
    return np.sqrt(mag_h ** 2 + mag_v ** 2)


# --------------------------------------------------------------------------
# NEQR-gradient reconstruction
# --------------------------------------------------------------------------
def neqr_counts_to_diff_map(counts: dict, size: int,
                            direction: str = "horizontal",
                            intensity_bits: int = 8) -> np.ndarray:
    """Mean TRUE difference magnitude |f - nb| / max-level per pixel.

    Count keys carry [carry(1), G(w), F(w), col(k), row(k)]: the true
    magnitude is G if carry==0 else 2^w - G (carry = subtractor
    borrow = sign of f - nb).  This recovers contrasts above half the
    register range that a bare modular byte would wrap.
    """
    k = int(np.log2(size))
    w = intensity_bits
    acc = np.zeros((size, size))
    cnt = np.zeros((size, size))
    for key, c in counts.items():
        v = int(key.replace(" ", ""), 2)
        pos = v & (2 ** (2 * k) - 1)
        r, cix = pos & (size - 1), (pos >> k) & (size - 1)
        d = (v >> (2 * k + w)) & ((1 << w) - 1)
        carry = (v >> (2 * k + 2 * w)) & 1
        mag = d if carry == 0 else (1 << w) - d
        acc[r, cix] += mag * c
        cnt[r, cix] += c
    out = np.zeros((size, size))
    mask = cnt > 0
    out[mask] = acc[mask] / cnt[mask] / (2 ** w - 1)
    return out


def reconstruct_neqr_edge_map(counts_h: dict, counts_v: dict, image,
                              intensity_bits: int = 8) -> np.ndarray:
    """Combined Sobel-style NEQR-gradient edge magnitude map in [0, sqrt2]."""
    size = np.asarray(image).shape[0]
    dh = neqr_counts_to_diff_map(counts_h, size, intensity_bits=intensity_bits) \
        if counts_h else np.zeros((size, size))
    dv = neqr_counts_to_diff_map(counts_v, size, intensity_bits=intensity_bits) \
        if counts_v else np.zeros((size, size))
    return np.sqrt(dh ** 2 + dv ** 2)
