"""Compare the two approaches on the SAME test trajectories and the SAME metrics:

  * Approach 2 (single-step): outputs/single_step_k1/best.pt  -- iterated one step at a time
  * Approach 1 (single-shot): outputs/single_shot/best.pt     -- predicts any lead time directly

Both are evaluated against the Verlet ground truth on a shared time grid, up to the
largest horizon the single-shot model was trained for (its T_max). Metrics are printed
to the terminal; comparison figures are written to outputs/comparison/.

Run from the repo root (so `src` and `DataGeneration` are importable):

    uv run src/compare_models.py        # or: PYTHONPATH=src:. python src/compare_models.py
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import metrics
import plots
from data import load_trajectories, split_indices
from models import load_predictor
from models_single_shot import load_single_shot
from rollout import rollout
from baselines_single_shot import verlet_batched


@dataclass
class CompareConfig:
    step_ckpt: str = "outputs/single_step_k1/best.pt"
    shot_ckpt: str = "outputs/single_shot/tmax2/best.pt"
    data_dir: str = "data/traj_chunks"
    split: str = "test"
    n_traj: int = 200
    horizon: float = 20.0                 # capped at the single-shot T_max
    thresholds: tuple = (0.1, 0.5, 1.0)
    report_times: tuple = (0.5, 1, 2, 5, 10)
    n_overlay: int = 3
    out_dir: str = "outputs/comparison"
    device: str = "auto"


CONFIG = CompareConfig()


def get_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def run(cfg: CompareConfig):
    device = get_device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)

    step = load_predictor(cfg.step_ckpt, str(device))     # single-step predictor
    shot = load_single_shot(cfg.shot_ckpt, str(device))   # single-shot predictor
    meta = step.meta
    dt_save = meta["dt_save"]

    # --- shared test set (same split_seed guarantees the same trajectories) ---
    assert step.config["split_seed"] == shot.config["split_seed"], \
        "the two models were trained with different split_seed: test sets differ!"
    data, _ = load_trajectories(cfg.data_dir)
    idx = split_indices(len(data), step.config["split_seed"])[cfg.split][: cfg.n_traj]
    traj = data[idx]

    # --- shared time grid: single-step step size, capped at the single-shot horizon ---
    horizon = min(cfg.horizon, shot.t_max, (traj.shape[1] - 1) * dt_save)
    k = step.k
    n_calls = min(int(round(horizon / step.dt_step)), (traj.shape[1] - 1) // k)
    times = np.arange(n_calls + 1) * step.dt_step
    truth = traj[:, ::k][:, : n_calls + 1]
    print(f"comparing on {len(traj)} {cfg.split} trajectories up to t = {times[-1]:g} "
          f"(single-shot T_max = {shot.t_max:g}); single-step k = {k}, step = {step.dt_step:g}; device = {device}")

    s0 = torch.from_numpy(traj[:, 0]).to(device)

    # --- produce both trajectories on the shared grid ---
    path_step = rollout(step, s0, n_calls).cpu().numpy()                      # iterate
    taus = torch.from_numpy(times[1:]).to(device)
    pred = shot.predict_grid(s0, taus).cpu().numpy()                          # direct, all lead times at once
    path_shot = np.concatenate([traj[:, :1], pred], axis=1)                   # prepend t=0 = s0

    results = {}
    for name, path in (("single-step", path_step), ("single-shot", path_shot)):
        err = metrics.position_error(path, truth)
        finite = metrics.finite_trajectories(path)
        drift = metrics.relative_energy_drift(path[finite], meta["E"], meta["M"], meta["G"]) \
            if finite.any() else np.array([[np.nan]])
        results[name] = {
            "path": path, "err": err, "finite": finite,
            "div": metrics.divergence_summary(err, times, cfg.thresholds),
            "err_at": {t: float(np.nanmedian(err[:, int(round(t / step.dt_step))]))
                       for t in cfg.report_times if t <= times[-1]},
            "drift_end": float(np.nanmedian(drift[:, -1])),
        }

    cost = cost_report(step, shot, s0, n_calls, times, meta, traj[:, 0], device)
    print_table(cfg, results, times, cost)
    make_plots(cfg, results, times, truth, meta, step.dt_step, cost)
    print(f"\nfigures written to {cfg.out_dir}/")


def cost_report(step, shot, s0, n_calls, times, meta, starts, device):
    """Seconds to generate one full-horizon trajectory (batched over all starts)."""
    def timed(fn, sync=True):
        if sync and device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync and device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    B = s0.shape[0]
    taus = torch.from_numpy(times[1:]).to(device)
    rollout(step, s0, min(n_calls, 20))                       # warm-up
    shot.predict_grid(s0[:8], taus)
    t_step = timed(lambda: rollout(step, s0, n_calls)) / B
    t_shot = timed(lambda: shot.predict_grid(s0, taus)) / B
    M = np.asarray(meta["M"])
    n_steps = int(round(times[-1] / meta["dt_solver"]))
    t_verlet = timed(lambda: verlet_batched(starts, M, meta["G"], meta["dt_solver"], n_steps,
                                            meta["save_every"]), sync=False) / len(starts)
    # random access: cost to get ONE state at the final time
    one = timed(lambda: shot.predict(s0, torch.full((B,), float(times[-1]), device=device))) / B
    return {"verlet": t_verlet, "single-step": t_step, "single-shot": t_shot, "shot_random_access": one}


def print_table(cfg, results, times, cost):
    w = 26
    P = print
    P("\n" + "=" * 64)
    P(f" MODEL COMPARISON  (horizon t = {times[-1]:g}, same test trajectories)")
    P("=" * 64)
    names = ["single-step", "single-shot"]
    P(f"\n{'':<32}" + "".join(f"{n:>16}" for n in names))
    P("-" * 64)

    def row(label, fn):
        P(f"{label:<32}" + "".join(f"{fn(n):>16}" for n in names))

    P("Divergence time (median, higher = better)")
    for thr in cfg.thresholds:
        row(f"  error > {thr:g}", lambda n, thr=thr: f"{results[n]['div'][thr][1]:.2f}")
    P("Median position error at t")
    for t in cfg.report_times:
        if t <= times[-1]:
            row(f"  t = {t:g}", lambda n, t=t: f"{results[n]['err_at'][t]:.3e}")
    P("Stability / physics")
    row("  finite trajectories", lambda n: f"{results[n]['finite'].mean()*100:.0f}%")
    row("  energy drift at end", lambda n: f"{results[n]['drift_end']:.2e}")
    P("Cost (s per trajectory, batched)")
    row("  full horizon", lambda n: f"{cost[n]:.2e}")
    P(f"\n  Verlet (batched) reference      : {cost['verlet']:.2e} s/traj")
    P(f"  single-shot random-access (1 t) : {cost['shot_random_access']:.2e} s/traj  "
      f"(one far state in a single forward pass)")
    P("=" * 64 + "\n")


COLORS = {"single-step": "tab:blue", "single-shot": "tab:orange"}


def make_plots(cfg, results, times, truth, meta, dt_step, cost):
    out = cfg.out_dir
    n_ov = min(cfg.n_overlay, truth.shape[0])
    ref_t = min(5.0, float(times[-1]))                 # reference time for the per-trajectory figures
    ref_i = int(round(ref_t / dt_step))

    # 1) ground truth vs each model, in SEPARATE files (rollout.py style overlay)
    for name, key in (("single-step", "trajectories_single_step.png"),
                      ("single-shot", "trajectories_single_shot.png")):
        plots.plot_rollout_overlay(truth[:n_ov], results[name]["path"][:n_ov],
                                   os.path.join(out, key), titles=[f"{name}: traj {i}" for i in range(n_ov)])

    # 2) error vs time, both models on one axes
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for n in results:
        err = np.where(np.isfinite(results[n]["err"]), results[n]["err"], 1e3)
        ax.plot(times, np.median(err, axis=0), color=COLORS[n], label=n)
        lo, hi = np.percentile(err, [25, 75], axis=0)
        ax.fill_between(times, lo, hi, color=COLORS[n], alpha=0.2)
    for thr in cfg.thresholds:
        ax.axhline(thr, color="grey", ls=":", lw=0.8)
    ax.set(xlabel="simulated time", ylabel="position error (RMS over bodies)", yscale="log",
           title="Rollout error vs time")
    ax.grid(alpha=0.3); ax.legend()
    _save(fig, out, "error_vs_time_compare.png")

    # 3) divergence time: grouped bars per threshold
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    x = np.arange(len(cfg.thresholds)); wd = 0.38
    for off, n in zip((-wd/2, wd/2), results):
        ax.bar(x + off, [results[n]["div"][thr][1] for thr in cfg.thresholds], wd, label=n, color=COLORS[n])
    ax.set_xticks(x); ax.set_xticklabels([f"error > {thr:g}" for thr in cfg.thresholds])
    ax.set(ylabel="median divergence time", title="How long each model tracks the truth")
    ax.grid(alpha=0.3, axis="y"); ax.legend()
    _save(fig, out, "divergence_bars.png")

    # 4) energy drift vs time (physical consistency)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for n in results:
        fin = results[n]["finite"]
        if fin.any():
            d = metrics.relative_energy_drift(results[n]["path"][fin], meta["E"], meta["M"], meta["G"])
            ax.plot(times, np.nanmedian(d, axis=0), color=COLORS[n], label=n)
    dt_truth = metrics.relative_energy_drift(truth, meta["E"], meta["M"], meta["G"])
    ax.plot(times, np.median(dt_truth, axis=0), color="k", lw=1, ls="--", label="Verlet truth")
    ax.set(xlabel="simulated time", ylabel="relative energy drift $|E-E_0|/|E_0|$", yscale="log",
           title="Energy conservation along the rollout")
    ax.grid(alpha=0.3); ax.legend()
    _save(fig, out, "energy_drift_compare.png")

    # 5) cost comparison (the accuracy-vs-compute story)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    labels = ["Verlet\n(batched)", "single-step\n(full rollout)", "single-shot\n(full grid)",
              "single-shot\n(one far state)"]
    vals = [cost["verlet"], cost["single-step"], cost["single-shot"], cost["shot_random_access"]]
    bars = ax.bar(labels, vals, color=["grey", COLORS["single-step"], COLORS["single-shot"], "#f0a35e"])
    ax.set(ylabel="seconds per trajectory", yscale="log", title="Compute cost per trajectory")
    ax.grid(alpha=0.3, axis="y")
    ax.bar_label(bars, fmt="%.1e", padding=3, fontsize=8)
    _save(fig, out, "cost_bars.png")

    # 6) error vs closest approach: do close encounters drive the error?
    min_d = metrics.min_pair_distance(truth).min(axis=1)       # per trajectory, from the truth
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for n in results:
        e = results[n]["err"][:, ref_i]
        m = np.isfinite(e)
        ax.scatter(min_d[m], e[m], s=18, alpha=0.6, color=COLORS[n], label=n)
    ax.set(xlabel="closest approach in the trajectory (min inter-body distance)",
           ylabel=f"position error at t = {ref_t:g}", yscale="log",
           title="Error vs close encounters")
    ax.grid(alpha=0.3); ax.legend()
    _save(fig, out, "error_vs_closest_approach.png")

    # 7) is a trajectory's difficulty intrinsic? single-step vs single-shot error per trajectory
    es = results["single-step"]["err"][:, ref_i]
    eh = results["single-shot"]["err"][:, ref_i]
    m = np.isfinite(es) & np.isfinite(eh)
    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    ax.scatter(es[m], eh[m], s=18, alpha=0.6, color="tab:purple")
    lim = [min(es[m].min(), eh[m].min()) * 0.8, max(es[m].max(), eh[m].max()) * 1.2]
    ax.plot(lim, lim, "k--", lw=1, label="y = x")
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
           xlabel=f"single-step error at t = {ref_t:g}", ylabel=f"single-shot error at t = {ref_t:g}",
           title="Per-trajectory difficulty: are the hard cases shared?")
    ax.set_aspect("equal"); ax.grid(alpha=0.3, which="both"); ax.legend()
    _save(fig, out, "error_correlation.png")


def _save(fig, out_dir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, name), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run(CONFIG)