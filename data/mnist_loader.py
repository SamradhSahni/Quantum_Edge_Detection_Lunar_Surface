"""
MNIST utility -- CIRCUIT DEBUGGING ONLY (not used in any scientific
comparison, per PROJECT_SUMMARY.md Section 4).

Loads a digit, downsamples it to a power-of-two patch, and provides a
quick smoke test that both quantum pipelines run end-to-end on real
image content.
"""

import numpy as np


def load_mnist_patch(size: int = 8, digit: int = 1, seed: int = 42):
    """Return one MNIST digit as a [0,1] size x size patch."""
    try:
        from torchvision import datasets
        ds = datasets.MNIST(root=config.MNIST_DIR, train=True, download=True)
        idx = next(i for i, (_, y) in enumerate(ds) if y == digit)
        img = np.array(ds[idx][0], dtype=np.float64) / 255.0
    except Exception as e:  # offline fallback: draw a synthetic stroke
        print(f"MNIST unavailable ({e}); using a synthetic stroke digit")
        img = np.zeros((28, 28))
        cv2 = __import__("cv2")
        cv2.line(img, (14, 4), (14, 23), 1.0, 3)
        cv2.line(img, (14, 4), (8, 10), 1.0, 3)
    from preprocessing.image_loader import to_power_of_two
    return to_power_of_two(img, size)


import config  # noqa: E402  (placed late to allow the fallback import above)
