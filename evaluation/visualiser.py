"""
Visualisation: side-by-side grids, robustness curves, confidence maps,
failure galleries, and circuit diagrams.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 10,
})


def show_patch(ax, img, title="", cmap="gray", vmin=0, vmax=1):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def comparison_grid(image, gt=None, method_outputs: dict = None, title: str = "",
                    save_path: str = None, confidence=None):
    """Input | GT | per-method edge maps in one row.

    method_outputs: {label: binary or strength map}
    confidence: optional continuous map shown last.
    """
    panels = [("Input", image, "gray")]
    if gt is not None:
        panels.append(("Ground truth", gt, "gray"))
    if method_outputs:
        for label, m in method_outputs.items():
            panels.append((label, m, "gray"))
    if confidence is not None:
        panels.append(("Quantum confidence", confidence, "magma"))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 2.4))
    for ax, (t, img, cmap) in zip(axes, panels):
        vmax = 1.0 if np.asarray(img).dtype.kind == "f" and img.max() <= 1.0001 else None
        show_patch(ax, img, t, cmap=cmap, vmax=vmax)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig


def robustness_curves(levels, results_by_method: dict, xlabel: str, title: str,
                      save_path: str = None):
    """F1 vs degradation level with one line per method."""
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for method, vals in results_by_method.items():
        color = config.METHOD_COLORS.get(method, None)
        label = config.METHOD_LABELS.get(method, method)
        ax.plot(levels, vals, marker="o", ms=3, lw=1.4, color=color, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Boundary F1 (1-px tol)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig


def direction_map_plot(direction_map, magnitude, title="", save_path=None):
    """Robinson dominant-direction map (angle hue)."""
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.6))
    show_patch(axes[0], magnitude, "Robinson |response|", cmap="gray")
    im = axes[1].imshow(direction_map, cmap="hsv", vmin=0, vmax=360)
    axes[1].set_title("Dominant direction (deg)")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig


def shots_curve(shots_list, mean_f1, std_f1, method: str, title="", save_path=None):
    """F1 vs shots with error bars (Experiment 3)."""
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    color = config.METHOD_COLORS.get(method)
    ax.errorbar(shots_list, mean_f1, yerr=std_f1, marker="o", ms=3,
                capsize=3, lw=1.4, color=color, label=config.METHOD_LABELS.get(method, method))
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Shots")
    ax.set_ylabel("Boundary F1 (1-px tol)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig


def failure_grid(rows, save_path):
    """Failure-gallery grid.

    rows: list of (caption, {label: image}) tuples rendered as grid rows.
    """
    n_rows = len(rows)
    n_cols = max(len(r[1]) for r in rows) if rows else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.9 * n_cols, 2.2 * n_rows))
    axes = np.atleast_2d(axes)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    for i, (caption, panels) in enumerate(rows):
        for j, (label, img) in enumerate(panels.items()):
            show_patch(axes[i, j], img, label)
        axes[i, 0].set_ylabel(caption, fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig


def save_circuit_diagram(circuit, save_path, fold: int = -1):
    """Render a circuit diagram via Qiskit's text drawer (no GUI deps)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        text = circuit.draw(output="text", fold=fold)
        with open(os.path.splitext(save_path)[0] + ".txt", "w", encoding="utf-8") as f:
            f.write(str(text))
    except Exception as e:  # pragma: no cover
        with open(os.path.splitext(save_path)[0] + ".txt", "w", encoding="utf-8") as f:
            f.write(f"circuit diagram failed: {e}\n{circuit}")
