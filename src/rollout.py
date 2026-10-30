from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC_DIR)
for _p in (_ROOT, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

import metrics
import plots
from data import fixed_pairs, load_trajectories, split_indices
from models import load_predictor


@dataclass
class RolloutConfig:
    checkpoint: str = "outputs/single_step_k1/best.pt"
    data_dir: str = "data/traj_chunks"
    split: str = "test"                      # "train" | "val" | "test"
    n_traj: int = 200                        # trajectories to roll out
    horizon: float = 20.0                    # simulated time (capped at trajectory length)
    thresholds: tuple = (0.1, 0.5, 1.0)      # position-error levels for divergence time
    chaos_eps: tuple = (1e-6, 1e-3)          # Verlet-vs-Verlet perturbations; () to skip
    n_chaos: int = 10                        # trajectories used for the chaos reference
    onestep_stride: int = 10                 # sub-sampling for one-step (teacher-forced) scores
    n_plot: int = 4                          # trajectories in the static overlay figure
    out_dir: str | None = None               # default: <checkpoint dir>/rollout_<split>
    device: str = "auto"                     # "auto", "cuda" or "cpu"


CONFIG = RolloutConfig()


# ---------------------------------------------------------------------------
# rollout + timing
# ---------------------------------------------------------------------------
def get_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


@torch.no_grad()
def rollout(pred, s0: torch.Tensor, n_calls: int) -> torch.Tensor:
    """Apply the network n_calls times. s0 (B,12) float64 -> (B,n_calls+1,12)."""
    out = torch.empty(s0.shape[0], n_calls + 1, 12, dtype=s0.dtype, device=s0.device)
    out[:, 0] = s0
    s = s0
    for i in range(1, n_calls + 1):
        s = pred.predict_next(s)
        out[:, i] = s
    return out


@torch.no_grad()
def one_step_scores(pred, traj: torch.Tensor, k: int, stride: int) -> dict:
    """Teacher-forced one-step regression scores on the test trajectories."""
    s0, s1 = fixed_pairs(traj, k, stride)
    nxt = pred.predict_next(s0)
    return metrics.regression_scores((nxt - s0).cpu().numpy(), (s1 - s0).cpu().numpy())


def _time(fn, device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def cost_report(pred, s0, n_calls, meta, traj_starts) -> dict:
    """Seconds per simulated time unit: Verlet vs the network (single + batched)."""
    import DataGeneration.VerletSolver as verlet

    calls = min(n_calls, 500)
    horizon = calls * pred.dt_step

    def net(batch):
        s = s0[:1].repeat(batch, 1)
        rollout(pred, s, 20)  # warm-up
        return _time(lambda: rollout(pred, s, calls), pred.device) / horizon / batch

    v_h = min(horizon, 10.0)
    v_steps = int(round(v_h / meta["dt_solver"]))
    M = np.asarray(meta["M"])
    v = _time(lambda: [verlet.simulate(traj_starts[j], M, meta["G"], meta["dt_solver"], v_steps, meta["save_every"])
                       for j in range(3)], pred.device) / 3 / v_h
    return {"verlet": v, "net_single": net(1), "net_batched": net(1024)}


def print_report(cfg, pred, times, err, finite, onestep, div, chaos_div, drift_net, drift_true,
                 mind_net, mind_true, cost):
    k, dt = pred.k, pred.dt_step
    hz = float(times[-1])
    line = "-" * 62
    P = print
    P("\n" + "=" * 62)
    P(f" ROLLOUT  k={k}  step={dt:g}  |  {len(err)} {cfg.split} trajectories  |  horizon {hz:g}")
    P("=" * 62)

    P("\nOne-step prediction (teacher-forced, change Δ per step)")
    P(line)
    for name in ("position", "velocity"):
        s = onestep[name]
        P(f"  {name:8s}  RMSE {s['rmse']:.4e}   MAE {s['mae']:.4e}   R2 {s['r2']:.5f}")

    P("\nRollout accuracy (iterated)")
    P(line)
    for t in (1, 2, 5, 10, 20):
        if t <= hz:
            e = err[:, int(round(t / dt))]
            P(f"  median position error @ t={t:<3g} : {np.nanmedian(e):.4e}")
    P(f"  trajectories finite over full rollout : {finite.mean() * 100:.1f}%  "
      f"({(~finite).sum()} blew up)")

    P("\nDivergence time  (median time to exceed a position-error threshold)")
    P(line)
    for thr, (frac, t) in div.items():
        P(f"  error > {thr:<4g} : t = {t:6.2f}   ({frac * 100:.0f}% of trajectories diverge)")
    for eps, summ in chaos_div.items():
        frac, t = summ[cfg.thresholds[0]]
        P(f"  [chaos ref] Verlet perturbed by {eps:g}: error > {cfg.thresholds[0]:g} at t = {t:6.2f} "
          f"({frac * 100:.0f}%)")

    P("\nPhysical consistency  (finite rollouts only)")
    P(line)
    P(f"  relative energy drift at end : network {np.nanmedian(drift_net[:, -1]):.3e}   "
      f"truth {np.nanmedian(drift_true[:, -1]):.3e}")
    P(f"  median min inter-body distance : network {np.nanmedian(mind_net):.3e}   "
      f"truth {np.nanmedian(mind_true):.3e}")

    P("\nCost  (seconds per simulated time unit)")
    P(line)
    P(f"  Verlet solver            : {cost['verlet']:.3e}")
    P(f"  network, single traj     : {cost['net_single']:.3e}   ({cost['verlet'] / cost['net_single']:.0f}x vs Verlet)")
    P(f"  network, batched (1024)  : {cost['net_batched']:.3e}   ({cost['verlet'] / cost['net_batched']:.0f}x vs Verlet)")
    P("=" * 62 + "\n")
def run(cfg: RolloutConfig):
    device = get_device(cfg.device)
    out_dir = cfg.out_dir or os.path.join(os.path.dirname(cfg.checkpoint), f"rollout_{cfg.split}")
    os.makedirs(out_dir, exist_ok=True)
 
    pred = load_predictor(cfg.checkpoint, str(device))
    meta, k = pred.meta, pred.k
 
    data, _ = load_trajectories(cfg.data_dir)
    idx = split_indices(len(data), pred.config["split_seed"])[cfg.split][: cfg.n_traj]
    traj = data[idx]
 
    n_calls = min(int(round(cfg.horizon / pred.dt_step)), (traj.shape[1] - 1) // k)
    times = np.arange(n_calls + 1) * pred.dt_step
    truth = traj[:, ::k][:, : n_calls + 1]
 
    # iterate the network
    s0 = torch.from_numpy(traj[:, 0]).to(device)
    path = rollout(pred, s0, n_calls).cpu().numpy()
    err = metrics.position_error(path, truth)
    finite = metrics.finite_trajectories(path)
 
    # metrics
    onestep = one_step_scores(pred, torch.from_numpy(traj).to(device), k, cfg.onestep_stride)
    div = metrics.divergence_summary(err, times, cfg.thresholds)
 
    chaos_err, chaos_div = {}, {}
    n_c = min(cfg.n_chaos, len(traj))
    for eps in cfg.chaos_eps:
        ce = metrics.chaos_reference(traj[:n_c, 0], meta, times[-1], eps, k, n_calls, truth[:n_c])
        chaos_err[eps] = ce
        chaos_div[eps] = metrics.divergence_summary(ce, times[: ce.shape[1]], cfg.thresholds)
 
    drift_net = metrics.relative_energy_drift(path[finite], meta["E"], meta["M"], meta["G"])
    drift_true = metrics.relative_energy_drift(truth, meta["E"], meta["M"], meta["G"])
    mind_net = metrics.min_pair_distance(path[finite]).min(axis=1) if finite.any() else np.array([np.nan])
    mind_true = metrics.min_pair_distance(truth).min(axis=1)
 
    cost = cost_report(pred, s0, n_calls, meta, traj[:, 0])
 
    print_report(cfg, pred, times, err, finite, onestep, div, chaos_div,
                 drift_net, drift_true, mind_net, mind_true, cost)
 
    # ---- figures 
    err_plot = np.where(np.isfinite(err), err, 1e3)  # blown-up rows count as hopelessly diverged
    plots.plot_error_vs_time(times, err_plot, os.path.join(out_dir, "error_vs_time.png"),
                             chaos=chaos_err, thresholds=cfg.thresholds)
    plots.plot_energy_drift(times, drift_net, drift_true, os.path.join(out_dir, "energy_drift.png"))
    n_plot = min(cfg.n_plot, len(traj))
    plots.plot_rollout_overlay(truth[:n_plot], path[:n_plot], os.path.join(out_dir, "trajectories.png"),
                               titles=[f"traj {i}" for i in range(n_plot)])
    print(f"figures  written to {out_dir}/")


if __name__ == "__main__":
    run(CONFIG)