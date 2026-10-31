"""Test-set evaluation of a trained single-shot network, and the comparison with the
single-step network (Approach 2) and the Verlet solver.

Run from the repository root (after training, e.g. with src/train_single_shot.py):
    python src/evaluate_single_shot.py

What it does, on the TEST trajectories (same split as rollout.py, since both read split_seed
from the checkpoint config and train_single_step / train_single_shot both use split_seed=0):

  * predicts s(tau) from the start state s(0) of every test trajectory for all lead times on a
    grid up to T_max (one forward pass per (start, tau), nothing is iterated)
  * accuracy vs tau: position / velocity error, R^2 of the predicted change, divergence times,
    compared with
      - constant velocity ("x += v tau") and mean state baselines,
      - k-nearest-neighbour regression (k chosen on the VALIDATION trajectories),
      - the single-step network iterated with rollout.rollout (if its checkpoint exists),
      - the chaos reference (Verlet vs Verlet with a perturbed start, metrics.chaos_reference)
  * physical consistency: relative energy error of every prediction
  * cost: wall-clock seconds needed to obtain the state at lead time tau, for one trajectory
    and per trajectory in a batch, for the single-shot network, the single-step rollout, the
    Verlet solver from DataGeneration (pure Python loop) and a numpy-vectorised Verlet
    (baselines_single_shot.verlet_batched). Also: how many predictions are needed before
    training (and generating the data) pays off.

Outputs (printed report + files) in <checkpoint dir>/eval_<split>/:
    eval.json, error_vs_tau.png, trajectories.png, energy_error.png, cost_vs_tau.png
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC_DIR)
for _p in (_ROOT, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

import metrics
import plots_single_shot as plots
from baselines_single_shot import KNNBaseline, verlet_batched
from data import load_trajectories, split_indices
from data_single_shot import lead_time_grid
from models_single_shot import load_single_shot


@dataclass
class EvalConfig:
    checkpoint: str = "outputs/single_shot/final/best.pt"
    single_step_checkpoint: str = "outputs/single_step_k1/best.pt"  # skipped if the file does not exist
    data_dir: str = "data/traj_chunks"
    split: str = "test"                      # "val" | "test"
    n_traj: int = 200                        # trajectories to evaluate (200 = the whole test split)
    t0: int = 0                              # start index on the trajectory (0 = generated initial condition)
    tau_stride: int = 5                      # lead time grid spacing in saved samples (5 -> 0.05 time units)
    report_taus: tuple = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
    thresholds: tuple = (0.1, 0.5, 1.0)      # position-error levels for divergence time
    knn_ks: tuple = (1, 5, 10, 20, 50)       # k values tried for the kNN baseline (chosen on validation)
    knn_anchor_stride: int = 5
    chaos_eps: tuple = (1e-6, 1e-3)          # Verlet-vs-Verlet perturbations; () to skip
    n_chaos: int = 10
    n_plot: int = 4                          # trajectories in the overlay figure
    cost_batch: int = 1024                   # batch size for the "batched" cost numbers
    out_dir: str | None = None               # default: <checkpoint dir>/eval_<split>
    device: str = "auto"


CONFIG = EvalConfig()


def get_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def velocity_error(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """RMS over the 3 bodies of the Euclidean velocity error. (...,T,12) -> (...,T)."""
    d = pred[..., metrics.VEL] - truth[..., metrics.VEL]
    return np.sqrt(np.sum(d * d, axis=-1) / 3.0)


def _time(fn, device, repeats: int = 1) -> float:
    """Average wall-clock seconds of fn() (synchronising the GPU if there is one)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


def choose_knn_k(knn: KNNBaseline, val_traj: torch.Tensor, ks: torch.Tensor, candidates, t0: int) -> tuple:
    """Pick k for the kNN baseline by the mean (over lead times) median position error on the
    VALIDATION trajectories, using the same protocol as the test evaluation."""
    s0 = val_traj[:, t0]
    truth = val_traj[:, t0 + ks].cpu().numpy()
    idx = knn.neighbours(s0, max(candidates))
    scores = {}
    for k in candidates:
        pred = knn.predict_grid(s0, ks, k=k, idx=idx).cpu().numpy()
        scores[k] = float(np.mean(np.median(metrics.position_error(pred, truth), axis=0)))
    best = min(scores, key=scores.get)
    return best, scores


def cost_report(pred, pred_ss, meta, s0: torch.Tensor, starts_np: np.ndarray, batch: int) -> dict:
    """Seconds needed per trajectory, measured once and expressed as
         single-shot : constant (one forward pass, independent of tau)
         rollout     : per network call  -> times tau / dt_step
         Verlet      : per time unit     -> times tau
    so cost(tau) = a + b * tau with the numbers below (we checked the scaling instead of assuming
    it for the network by timing two lead times)."""
    import DataGeneration.VerletSolver as verlet
    from rollout import rollout

    dev = pred.device
    out = {}

    # single-shot network: one call, for one start state and for a batch of start states
    for name, B in (("single_shot_single", 1), ("single_shot_batched", batch)):
        s = s0[:1].repeat(B, 1)
        res = {}
        for tau_val in (0.01, pred.t_max):
            tau = torch.full((B,), tau_val, dtype=s.dtype, device=dev)
            pred.predict(s, tau)  # warm-up
            res[tau_val] = _time(lambda: pred.predict(s, tau), dev, repeats=50) / B
        out[name] = res[pred.t_max]
        out[name + "_short_tau"] = res[0.01]

    # single-step rollout, cost per network call
    if pred_ss is not None:
        for name, B in (("rollout_single", 1), ("rollout_batched", batch)):
            s = s0[:1].repeat(B, 1)
            rollout(pred_ss, s, 10)  # warm-up
            out[name + "_per_call"] = _time(lambda: rollout(pred_ss, s, 100), dev) / 100 / B
        out["rollout_dt_step"] = pred_ss.dt_step

    # Verlet as used for data generation (pure Python loop over time steps), per time unit
    M, G, dt = np.asarray(meta["M"]), meta["G"], meta["dt_solver"]
    h = 1.0
    n = int(round(h / dt))
    out["verlet_python_per_time"] = _time(
        lambda: [verlet.simulate(starts_np[j], M, G, dt, n, meta["save_every"]) for j in range(3)], torch.device("cpu")
    ) / 3 / h
    # numpy-vectorised Verlet: one trajectory, and a batch of trajectories at once
    h_b = 0.5
    n_b = int(round(h_b / dt))
    one = starts_np[:1]
    many = np.repeat(starts_np[:1], batch, axis=0)
    out["verlet_numpy_single_per_time"] = _time(lambda: verlet_batched(one, M, G, dt, n_b, n_b), torch.device("cpu")) / h_b
    out["verlet_numpy_batched_per_time"] = _time(
        lambda: verlet_batched(many, M, G, dt, n_b, n_b), torch.device("cpu")) / h_b / batch
    return out


def cost_curves(cost: dict, taus: np.ndarray) -> dict:
    """Seconds per trajectory to reach each lead time, from the measured unit costs."""
    c = {
        "Verlet (Python loop, as in DataGeneration)": cost["verlet_python_per_time"] * taus,
        "Verlet (numpy, 1 trajectory)": cost["verlet_numpy_single_per_time"] * taus,
        "Verlet (numpy, batch)": cost["verlet_numpy_batched_per_time"] * taus,
        "single-shot net (1 trajectory)": np.full_like(taus, cost["single_shot_single"]),
        "single-shot net (batch)": np.full_like(taus, cost["single_shot_batched"]),
    }
    if "rollout_single_per_call" in cost:
        calls = np.ceil(taus / cost["rollout_dt_step"] - 1e-9)
        c["single-step rollout (1 trajectory)"] = cost["rollout_single_per_call"] * calls
        c["single-step rollout (batch)"] = cost["rollout_batched_per_call"] * calls
    return c


def _at(taus: np.ndarray, err: np.ndarray, t: float) -> float:
    """Median error at lead time t (nearest grid point), nan if t is outside the grid."""
    if t > taus[-1] + 1e-9:
        return float("nan")
    return float(np.nanmedian(err[:, int(np.argmin(np.abs(taus - t)))]))


def print_report(cfg, pred, res: dict):
    line = "-" * 70
    P = print
    P("\n" + "=" * 70)
    P(f" SINGLE-SHOT  T_max={pred.t_max:g}  output={pred.output}  |  {res['n_traj']} {cfg.split} trajectories, "
      f"start index {cfg.t0}")
    P("=" * 70)

    P("\nChange s(tau) - s0 over all lead times (R^2 = 1 - MSE / variance of the true change)")
    P(line)
    for name in ("position", "velocity"):
        s = res["regression"][name]
        P(f"  {name:8s}  RMSE {s['rmse']:.4e}   MAE {s['mae']:.4e}   R2 {s['r2']:.5f}")

    P("\nMedian position error at lead time tau")
    P(line)
    names = list(res["error_at"].keys())
    P("  tau   " + "".join(f"{n[:21]:>22s}" for n in names))
    for t in cfg.report_taus:
        if t <= pred.t_max + 1e-9:
            P(f"  {t:<5g} " + "".join(f"{res['error_at'][n][str(t)]:22.4e}" for n in names))
    P(f"  (kNN uses k = {res['knn_k']}, chosen on validation: {res['knn_scores']})")

    P("\nDivergence time  (median time to exceed a position-error threshold)")
    P(line)
    for name, div in res["divergence"].items():
        txt = "  ".join(f">{float(thr):g}: t={t:5.2f} ({frac * 100:3.0f}%)" for thr, (frac, t) in div.items())
        P(f"  {name:28s} {txt}")
    P("  (capped at the horizon; % = fraction of trajectories that cross the threshold within it)")

    P("\nPhysical consistency")
    P(line)
    P(f"  median relative energy error, all predictions : network {res['energy']['median_rel_err']:.3e}   "
      f"truth {res['energy']['median_rel_err_truth']:.3e}")
    P(f"  ... at tau = T_max                            : network {res['energy']['median_rel_err_tmax']:.3e}")

    c = res["cost"]
    P("\nCost to obtain the state at tau = T_max  (seconds per trajectory)")
    P(line)
    for name, v in res["cost_at_tmax"].items():
        P(f"  {name:45s}: {v:.3e}")
    P(f"  single-shot net is tau-independent: {c['single_shot_single_short_tau']:.2e} s at tau=0.01 vs "
      f"{c['single_shot_single']:.2e} s at tau=T_max")
    a = res["amortisation"]
    P(f"  training took {a['train_seconds']:.0f} s; vs batched numpy Verlet the net saves "
      f"{a['saving_per_query']:.2e} s per prediction at tau=T_max -> pays off after ~{a['break_even_queries']:.2e} predictions")
    P("=" * 70 + "\n")


def run(cfg: EvalConfig):
    device = get_device(cfg.device)
    out_dir = cfg.out_dir or os.path.join(os.path.dirname(cfg.checkpoint), f"eval_{cfg.split}")
    os.makedirs(out_dir, exist_ok=True)

    pred = load_single_shot(cfg.checkpoint, str(device))
    meta, k_max = pred.meta, pred.norm.k_max
    dt_save = meta["dt_save"]

    data, _ = load_trajectories(cfg.data_dir)
    split = split_indices(len(data), pred.config["split_seed"])
    traj_np = data[split[cfg.split]][: cfg.n_traj]
    if cfg.t0 + k_max >= traj_np.shape[1]:
        raise ValueError("start index + T_max does not fit in the trajectories")
    traj = torch.from_numpy(traj_np).to(device)

    ks_np = lead_time_grid(k_max, cfg.tau_stride)
    ks = torch.as_tensor(ks_np, device=device)
    taus = ks_np * dt_save
    s0 = traj[:, cfg.t0]
    truth = traj_np[:, cfg.t0 + ks_np]                                    # (B, n, 12)
    s0_np = traj_np[:, cfg.t0]

    # ---- the single-shot network: all (start, tau) combinations in one go
    net_pred = pred.predict_grid(s0, torch.as_tensor(taus, device=device)).cpu().numpy()

    # ---- baselines
    mean_state = data[split["train"]].reshape(-1, 12).mean(axis=0)
    cv_pred = s0_np[:, None, :].repeat(len(taus), axis=1).copy()
    cv_pred[..., :6] += s0_np[:, None, 6:] * taus[None, :, None]
    mean_pred = np.broadcast_to(mean_state, truth.shape)

    train_t = torch.from_numpy(data[split["train"]]).to(device)
    knn = KNNBaseline(train_t, pred.norm, k_max, cfg.knn_anchor_stride)
    knn_k, knn_scores = choose_knn_k(knn, torch.from_numpy(data[split["val"]]).to(device), ks, cfg.knn_ks, cfg.t0)
    knn_pred = knn.predict_grid(s0, ks, k=knn_k).cpu().numpy()
    del train_t, knn

    errs = {
        "single-shot net": metrics.position_error(net_pred, truth),
        "constant velocity": metrics.position_error(cv_pred, truth),
        "mean state": metrics.position_error(mean_pred, truth),
        f"kNN (k={knn_k})": metrics.position_error(knn_pred, truth),
    }
    curves = {name: (taus, e) for name, e in errs.items()}

    # ---- single-step network, iterated (Approach 2), if a checkpoint exists
    pred_ss = None
    if cfg.single_step_checkpoint and os.path.exists(cfg.single_step_checkpoint):
        from models import load_predictor
        from rollout import rollout

        pred_ss = load_predictor(cfg.single_step_checkpoint, str(device))
        k_ss = pred_ss.k
        n_calls = int(np.floor(pred.t_max / pred_ss.dt_step + 1e-9))
        path = rollout(pred_ss, s0, n_calls).cpu().numpy()
        t_ss = np.arange(n_calls + 1) * pred_ss.dt_step
        truth_ss = traj_np[:, cfg.t0 :: k_ss][:, : n_calls + 1]
        e_ss = metrics.position_error(path, truth_ss)
        e_ss = np.where(np.isfinite(e_ss), e_ss, 1e3)  # blown-up rollouts count as hopelessly diverged
        curves[f"single-step rollout (k={k_ss})"] = (t_ss[1:], e_ss[:, 1:])
    else:
        print(f"no single-step checkpoint at {cfg.single_step_checkpoint!r}, skipping that comparison")

    # ---- chaos reference on a uniform grid (metrics.chaos_reference needs equal spacing)
    chaos = {}
    n_c = min(cfg.n_chaos, len(traj_np))
    n_uni = k_max // cfg.tau_stride
    truth_uni = traj_np[:n_c, cfg.t0 :: cfg.tau_stride][:, : n_uni + 1]
    t_uni = np.arange(n_uni + 1) * cfg.tau_stride * dt_save
    for eps in cfg.chaos_eps:
        ce = metrics.chaos_reference(s0_np[:n_c], meta, t_uni[-1], eps, cfg.tau_stride, n_uni, truth_uni)
        chaos[eps] = (t_uni[1:], ce[:, 1:])

    # ---- metrics
    res = {"config": asdict(cfg), "n_traj": len(traj_np), "t_max": pred.t_max, "output": pred.output,
           "knn_k": knn_k, "knn_scores": knn_scores}
    res["regression"] = metrics.regression_scores(net_pred - s0_np[:, None], truth - s0_np[:, None])
    res["velocity_error_median_tmax"] = float(np.median(velocity_error(net_pred, truth)[:, -1]))
    res["error_at"] = {name: {str(t): _at(tt, e, t) for t in cfg.report_taus} for name, (tt, e) in curves.items()}
    res["divergence"] = {name: {str(k_): v for k_, v in metrics.divergence_summary(e, tt, cfg.thresholds).items()}
                         for name, (tt, e) in curves.items()}
    for eps, (tt, e) in chaos.items():
        res["divergence"][f"Verlet perturbed {eps:g}"] = {
            str(k_): v for k_, v in metrics.divergence_summary(e, tt, cfg.thresholds).items()}
    E_err = metrics.relative_energy_drift(net_pred, meta["E"], meta["M"], meta["G"])
    E_true = metrics.relative_energy_drift(truth, meta["E"], meta["M"], meta["G"])
    res["energy"] = {"median_rel_err": float(np.median(E_err)), "median_rel_err_truth": float(np.median(E_true)),
                     "median_rel_err_tmax": float(np.median(E_err[:, -1]))}
    res["median_error_curves"] = {name: {"tau": tt.tolist(), "median": np.median(e, axis=0).tolist(),
                                         "q25": np.percentile(e, 25, axis=0).tolist(),
                                         "q75": np.percentile(e, 75, axis=0).tolist()}
                                  for name, (tt, e) in curves.items()}

    # ---- cost
    cost = cost_report(pred, pred_ss, meta, s0, s0_np, cfg.cost_batch)
    res["cost"] = cost
    cc = cost_curves(cost, taus)
    res["cost_at_tmax"] = {name: float(v[-1]) for name, v in cc.items()}
    m_path = os.path.join(os.path.dirname(cfg.checkpoint), "metrics.json")
    train_s = json.load(open(m_path))["train_seconds"] if os.path.exists(m_path) else float("nan")
    saving = cost["verlet_numpy_batched_per_time"] * pred.t_max - cost["single_shot_batched"]
    res["amortisation"] = {"train_seconds": train_s, "saving_per_query": saving,
                           "break_even_queries": train_s / saving if saving > 0 else float("inf")}

    print_report(cfg, pred, res)
    with open(os.path.join(out_dir, "eval.json"), "w") as f:
        json.dump(res, f, indent=2)

    # ---- figures
    plots.plot_error_vs_tau(curves, os.path.join(out_dir, "error_vs_tau.png"), main="single-shot net",
                            chaos=chaos, thresholds=cfg.thresholds)
    plots.plot_energy_error(taus, E_err, os.path.join(out_dir, "energy_error.png"), truth=E_true)
    plots.plot_cost_vs_tau(taus, cc, os.path.join(out_dir, "cost_vs_tau.png"))
    n_plot = min(cfg.n_plot, len(traj_np))
    ks_dense = np.arange(1, k_max + 1)
    dense = pred.predict_grid(s0[:n_plot], torch.as_tensor(ks_dense * dt_save, device=device)).cpu().numpy()
    dense = np.concatenate([s0_np[:n_plot, None], dense], axis=1)
    plots.plot_prediction_overlay(traj_np[:n_plot, cfg.t0 : cfg.t0 + k_max + 1], dense,
                                  os.path.join(out_dir, "trajectories.png"),
                                  titles=[f"test traj {i}, tau up to {pred.t_max:g}" for i in range(n_plot)])
    print(f"figures written to {out_dir}/")
    return res


if __name__ == "__main__":
    run(CONFIG)
