"""Figures for the single-shot approach. Same look as src/plots.py (which we reuse for the
training curves), all saved as png at 150 dpi; set FORMAT = "pdf" for vector graphics in the report."""

from __future__ import annotations

import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = ["tab:red", "tab:blue", "tab:green"]
FORMAT = "png"


def _save(fig, filename):
    if FORMAT != "png":
        filename = filename.rsplit(".", 1)[0] + "." + FORMAT
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_prediction_overlay(truth, pred, filename, titles=None, label="single-shot network"):
    """truth, pred: (n, T, 12). Solid thick = ground truth (Verlet), dashed = network.
    For the single-shot network every point of the dashed curve is a separate prediction from
    the same start state, the curve is NOT produced by iterating."""
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
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if titles:
            ax.set_title(titles[idx], fontsize=9)
    fig.suptitle(f"Ground truth (thick, faint) vs {label} (dashed); dots = start")
    _save(fig, filename)


def plot_error_vs_tau(curves: dict, filename, main: str, chaos=None, thresholds=(), log_x=False):
    """curves: {name: (tau (n,), err (B, n))}. The `main` curve gets median + IQR band, all others
    only their median. chaos: optional {eps: (tau, err)} of Verlet vs perturbed Verlet."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    # tab:blue is reserved for the main curve; distinct colour AND line style for the others
    # so the figure stays readable in a black and white printout
    others = itertools.cycle(zip(["tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:olive"],
                                 ["--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), "-"]))
    for name, (tau, err) in curves.items():
        med = np.nanmedian(err, axis=0)
        if name == main:
            lo, hi = np.nanpercentile(err, [25, 75], axis=0)
            ax.plot(tau, med, color="tab:blue", lw=2, label=f"{name} (median)")
            ax.fill_between(tau, lo, hi, color="tab:blue", alpha=0.25, label=f"{name} (IQR)")
        else:
            col, ls = next(others)
            ax.plot(tau, med, color=col, ls=ls, lw=1.3, label=f"{name} (median)")
    for (eps, (tau, err)), ls in zip((chaos or {}).items(), itertools.cycle(["--", ":"])):
        ax.plot(tau, np.median(err, axis=0), color="grey", ls=ls, lw=1,
                label=f"Verlet, start perturbed by {eps:g}")
    for thr in thresholds:
        ax.axhline(thr, color="grey", ls=":", lw=0.8)
    ax.set_yscale("log")
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(r"lead time $\tau$ (simulated time units)")
    ax.set_ylabel("position error (RMS over bodies)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    _save(fig, filename)


def plot_energy_error(tau, rel_err, filename, label="single-shot network", truth=None):
    """rel_err: (B, n) relative energy error |E(pred) - E0| / |E0| of every prediction."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tau, np.nanmedian(rel_err, axis=0), color="tab:blue", label=f"{label} (median)")
    lo, hi = np.nanpercentile(rel_err, [25, 75], axis=0)
    ax.fill_between(tau, lo, hi, color="tab:blue", alpha=0.25, label=f"{label} (IQR)")
    if truth is not None:
        ax.plot(tau, np.median(truth, axis=0), color="k", label="Verlet ground truth (median)")
    ax.set_yscale("log")
    ax.set_xlabel(r"lead time $\tau$")
    ax.set_ylabel("relative energy error |E - E0| / |E0|")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, filename)


def plot_cost_vs_tau(tau, costs: dict, filename):
    """costs: {method: seconds needed to obtain the state at lead time tau, per trajectory (n,)}."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    styles = itertools.cycle(["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 3))])
    colors = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    for name, c in costs.items():
        ax.plot(tau, c, ls=next(styles), color=next(colors), lw=1.5, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"lead time $\tau$")
    ax.set_ylabel("wall-clock seconds per trajectory")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=7)
    _save(fig, filename)


def plot_sweep_bars(rows: list, filename, title=""):
    """Best validation loss of every run (mean +- std over seeds) with the constant velocity
    baseline of the same lead-time range as a grey tick."""
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(rows)), 4.5))
    x = np.arange(len(rows))
    m = [r["best_val_mse_norm"] for r in rows]
    s = [0 if np.isnan(r["best_val_mse_norm_std"]) else r["best_val_mse_norm_std"] for r in rows]
    ax.bar(x, m, yerr=s, capsize=4, color="tab:blue", alpha=0.8, label="network")
    ax.scatter(x, [r["cv_val_mse_norm"] for r in rows], marker="_", s=600, color="grey", label="constant velocity")
    ax.set_xticks(x)
    ax.set_xticklabels([r["run"] for r in rows], rotation=20, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("validation loss (MSE, normalised)")
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    _save(fig, filename)


def plot_sweep_curves(curves: dict, filename, log_x=False):
    """curves: {run: (tau, median network error, median constant velocity error)} on validation."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.get_cmap("viridis")
    for j, (name, (tau, med, cv)) in enumerate(curves.items()):
        col = cmap(j / max(1, len(curves) - 1))
        ax.plot(tau, med, color=col, lw=1.6, label=name)
    # constant velocity does not depend on the run, so draw it once on the longest grid
    longest = max(curves.values(), key=lambda c: c[0][-1])
    ax.plot(longest[0], longest[2], color="grey", ls=":", label="constant velocity")
    ax.set_yscale("log")
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(r"lead time $\tau$")
    ax.set_ylabel("median position error (validation)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    _save(fig, filename)


def plot_horizon_test(curves: dict, others: dict, filename):
    """Test error vs tau of the single-shot networks trained for different T_max (one curve each,
    `curves` {name: (tau, median)}), together with reference curves `others` {name: (tau, median)}
    such as the single-step rollout and constant velocity."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.get_cmap("viridis")
    for j, (name, (tau, med)) in enumerate(curves.items()):
        ax.plot(tau, med, color=cmap(j / max(1, len(curves) - 1)), lw=1.8, label=name)
    for (name, (tau, med)), (col, ls) in zip(others.items(), itertools.cycle([("tab:red", "--"), ("grey", ":")])):
        ax.plot(tau, med, color=col, ls=ls, lw=1.3, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"lead time $\tau$")
    ax.set_ylabel("median position error (test, start at t = 0)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=7)
    _save(fig, filename)
