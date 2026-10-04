"""
Master experiment runner: executes Experiments 1-10 in sequence.

Usage:
    python experiments/run_all.py            # everything
    python experiments/run_all.py 1 2 3      # subset by number
"""

import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

EXPERIMENTS = {
    "1": "exp1_representation",
    "2": "exp2_resolution",
    "3": "exp3_shots",
    "4": "exp4_image_noise",
    "5": "exp5_quantum_noise",
    "6": "exp6_geometry",
    "7": "exp7_preprocessing",
    "8": "exp8_sombench",
    "9": "exp9_resources",
    "10": "exp10_failure_analysis",
}


def ensure_datasets():
    """Synthetic + lunar datasets present before running."""
    if not os.listdir(config.SYNTH_IMG_DIR):
        from data.synthetic.generator import generate_dataset
        generate_dataset()
        print("synthetic dataset generated")
    if not os.listdir(config.SOMBENCH_IMG_DIR):
        from preprocessing.sombench_data import ensure_sombench
        ensure_sombench(max_images=12)
        print("lunar dataset generated")
    from preprocessing.patch_extractor import extract_crater_patches
    if not os.listdir(config.SOMBENCH_PATCH_DIR):
        extract_crater_patches(max_patches_per_size=24)


def main():
    wanted = sys.argv[1:] or list(EXPERIMENTS)
    ensure_datasets()
    for num in wanted:
        name = EXPERIMENTS[num]
        print(f"\n{'#' * 70}\n# Running experiment {num}: {name}\n{'#' * 70}")
        t0 = time.time()
        try:
            mod = __import__(f"experiments.{name}", fromlist=["main"])
            mod.main()
            print(f"[experiment {num}] finished in {time.time() - t0:.0f}s")
        except Exception:
            print(f"[experiment {num}] FAILED after {time.time() - t0:.0f}s:")
            traceback.print_exc()
            if len(wanted) == 1:
                raise
    print("\nALL REQUESTED EXPERIMENTS DONE")


if __name__ == "__main__":
    main()
