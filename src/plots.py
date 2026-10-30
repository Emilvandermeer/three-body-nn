from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = ["tab:red", "tab:blue", "tab:green"]



def plot_rollout_overlay(truth, pred, filename, titles=None):
    """truth, pred: (n, T, 12). Solid thick = ground truth (Verlet), dashed = network."""
    n = len(truth)
    cols = min(n, 2)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        if idx >= n:
            ax.axis("off")
            continue
        for b in range(3):
            ax.plot(truth[idx, :, 2 * b], truth[idx, :, 2 * b + 1], color=COLORS[b], lw=3, alpha=0.35)
            ax.plot(pred[idx, :, 2 * b], pred[idx, :, 2 * b + 1], color=COLORS[b], lw=1, ls="--")
            ax.plot(truth[idx, 0, 2 * b], truth[idx, 0, 2 * b + 1], "o", color=COLORS[b])
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        if titles:
            ax.set_title(titles[idx], fontsize=9)
    fig.suptitle("Ground truth (thick, faint) vs network rollout (dashed); dots = start")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_error_vs_time(times, err, filename, chaos=None, thresholds=()):
    """err: (B, T). chaos: optional dict {eps: (n, T') error array} of Verlet-vs-perturbed-Verlet."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    med = np.median(err, axis=0)
    lo, hi = np.percentile(err, [25, 75], axis=0)
    ax.plot(times, med, color="tab:blue", label="network (median)")
    ax.fill_between(times, lo, hi, color="tab:blue", alpha=0.25, label="network (IQR)")
    for eps, e in (chaos or {}).items():
        ax.plot(times[: e.shape[1]], np.median(e, axis=0), ls="--", label=f"Verlet, start perturbed by {eps:g}")
    for thr in thresholds:
        ax.axhline(thr, color="grey", ls=":", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("simulated time")
    ax.set_ylabel("position error (RMS over bodies)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_energy_drift(times, drift_pred, drift_truth, filename):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(times, np.median(drift_pred, axis=0), color="tab:blue", label="network (median)")
    ax.plot(times, np.max(drift_pred, axis=0), color="tab:blue", ls=":", label="network (max)")
    ax.plot(times, np.median(drift_truth, axis=0), color="k", label="Verlet ground truth (median)")
    ax.set_yscale("log")
    ax.set_xlabel("simulated time")
    ax.set_ylabel("relative energy drift |E - E0| / |E0|")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_training_curves(csv_path, filename, baselines=None):
    rows = list(csv.DictReader(open(csv_path)))
    step = [int(r["step"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(step, [float(r["train_loss"]) for r in rows], label="train loss (augmented)")
    ax.plot(step, [float(r["val_loss"]) for r in rows], label="val loss (MSE, normalised)")
    for name, val in (baselines or {}).items():
        ax.axhline(val, ls=":", color="grey")
        ax.text(step[0], val, f" baseline: {name}", va="bottom", fontsize=7, color="grey")
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
