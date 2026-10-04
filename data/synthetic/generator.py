"""
Synthetic ground-truth dataset: eight geometry families with EXACT
analytic pixel-level edge masks (PROJECT_SUMMARY.md Phase 1, Experiment 6).

Geometries (8x8 canonical, 4x4 and 16x16 for scaling):
    horizontal_line | vertical_line | diagonal_45 | corner
    circle | ellipse | multi_circles | irregular_curve

Each image is a two-level grayscale patch (dark background, bright
shape -- lunar-rim-like step edges) and each mask marks the boundary
pixels analytically.  Masks are INDEPENDENT of every detector: they are
the geometric definition of the boundary, not the output of any filter.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config

GEOMETRIES = ["horizontal_line", "vertical_line", "diagonal_45", "corner",
              "circle", "ellipse", "multi_circles", "irregular_curve"]


# ---------------------------------------------------------------------------
# geometry builders: return (image, mask) float arrays in [0,1]
# ---------------------------------------------------------------------------
def _canvas(n: int, background: float = 0.15, foreground: float = 0.95):
    img = np.full((n, n), background, dtype=np.float64)
    return img, background, foreground


def horizontal_line(n):
    img, bg, fg = _canvas(n)
    r = n // 2
    img[r, :] = fg
    mask = np.zeros((n, n))
    mask[r, :] = 1
    return img, mask


def vertical_line(n):
    img, bg, fg = _canvas(n)
    c = n // 2
    img[:, c] = fg
    mask = np.zeros((n, n))
    mask[:, c] = 1
    return img, mask


def diagonal_45(n):
    """Anti-aliased 45-degree line through the centre."""
    img, bg, fg = _canvas(n)
    mask = np.zeros((n, n))
    for r in range(n):
        x = n - 1 - r  # main diagonal
        img[r, x] = fg
        mask[r, x] = 1
    return img, mask


def corner(n):
    """Bright quadrant -> L-shaped boundary (corner junction)."""
    img, bg, fg = _canvas(n)
    r0, c0 = n // 2, n // 2
    img[:r0, c0:] = fg
    mask = np.zeros((n, n))
    mask[:r0, c0 - 1] = 1   # vertical arm
    mask[r0 - 1, c0:] = 1   # horizontal arm
    mask[r0 - 1, c0 - 1] = 1  # corner pixel
    return img, mask


def circle(n, radius=None):
    img, bg, fg = _canvas(n)
    if radius is None:
        radius = n * 0.32
    yy, xx = np.mgrid[0:n, 0:n]
    cy = cx = (n - 1) / 2.0
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    img[dist <= radius] = fg
    mask = np.zeros((n, n))
    mask[(dist <= radius) & (dist > radius - 1)] = 1
    # guarantee at least the analytic ring on coarse grids
    if mask.sum() == 0:
        ring = np.abs(dist - radius)
        mask[ring == ring.min()] = 1
    return img, mask


def ellipse(n, a=None, b=None):
    img, bg, fg = _canvas(n)
    if a is None:
        a = n * 0.38
    if b is None:
        b = n * 0.22
    yy, xx = np.mgrid[0:n, 0:n]
    cy = cx = (n - 1) / 2.0
    val = np.sqrt(((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2)
    img[val <= 1.0] = fg
    mask = np.zeros((n, n))
    mask[(val <= 1.0) & (val > 1.0 - 1.0 / max(a, b))] = 1
    if mask.sum() == 0:
        mask[np.abs(val - 1.0) == np.min(np.abs(val - 1.0))] = 1
    return img, mask


def multi_circles(n):
    img, bg, fg = _canvas(n)
    mask = np.zeros((n, n))
    centres = [(0.30, 0.35, 0.16), (0.72, 0.65, 0.20)]
    for cyf, cxf, rf in centres:
        cy, cx, rad = cyf * n, cxf * n, rf * n
        yy, xx = np.mgrid[0:n, 0:n]
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        img[dist <= rad] = fg
        mask[(dist <= rad) & (dist > rad - 1)] = 1
    return img, mask


def irregular_curve(n):
    """Blobby boundary via low-frequency sinusoidal perturbation of a circle."""
    img, bg, fg = _canvas(n)
    yy, xx = np.mgrid[0:n, 0:n]
    cy = cx = (n - 1) / 2.0
    theta = np.arctan2(yy - cy, xx - cx)
    rad0 = n * 0.32
    rad = rad0 * (1 + 0.25 * np.sin(3 * theta) + 0.12 * np.sin(5 * theta + 1.0))
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    img[dist <= rad] = fg
    mask = np.zeros((n, n))
    mask[(dist <= rad) & (dist > rad - 1)] = 1
    if mask.sum() == 0:
        ring = np.abs(dist - rad)
        mask[ring <= ring.min() + 1e-9] = 1
    return img, mask


BUILDERS = {
    "horizontal_line": horizontal_line,
    "vertical_line": vertical_line,
    "diagonal_45": diagonal_45,
    "corner": corner,
    "circle": circle,
    "ellipse": ellipse,
    "multi_circles": multi_circles,
    "irregular_curve": irregular_curve,
}


def generate(size: int = 8):
    """All geometries at one resolution -> {name: (image, mask)}."""
    return {name: builder(size) for name, builder in BUILDERS.items()}


def generate_dataset(sizes=(4, 8, 16), save=True):
    """Full synthetic dataset across sizes; optionally save to disk."""
    dataset = {}
    for n in sizes:
        for name, (img, mask) in generate(n).items():
            key = f"{name}_{n}x{n}"
            dataset[key] = (img, mask)
            if save:
                np.save(os.path.join(config.SYNTH_IMG_DIR, key + ".npy"), img)
                np.save(os.path.join(config.SYNTH_MASK_DIR, key + ".npy"), mask)
    return dataset


def validation_split(dataset: dict, n_val: int = 3, seed: int = config.SEED):
    """Split synthetic keys into validation / test (stratified by geometry).

    Used for threshold selection: thresholds are tuned on the validation
    geometries and frozen for the test geometries (Section 5.5).
    """
    rng = np.random.default_rng(seed)
    by_geometry = {}
    for key in dataset:
        geo = key.rsplit("_", 1)[0] if key[0].isdigit() is False else key
        geo = "_".join(key.split("_")[:-1])  # strip trailing NxN
        by_geometry.setdefault(geo, []).append(key)
    val_keys, test_keys = [], []
    for geo, keys in sorted(by_geometry.items()):
        keys = sorted(keys)
        rng.shuffle(keys)
        n_v = min(n_val, max(1, len(keys) // 3))
        val_keys.extend(keys[:n_v])
        test_keys.extend(keys[n_v:])
    return val_keys, test_keys


if __name__ == "__main__":
    ds = generate_dataset()
    print(f"generated {len(ds)} synthetic patches:")
    for k in sorted(ds):
        img, mask = ds[k]
        print(f"  {k:28s} edge pixels in mask: {int(mask.sum())}")
