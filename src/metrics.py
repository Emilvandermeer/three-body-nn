"""Metrics for comparing an iterated network rollout against the Verlet ground
truth. Pure numpy, raw units, and NaN-aware: trajectories that blow up during a
rollout do not poison the aggregates."""

from __future__ import annotations

import numpy as np
import os
import sys
POS = slice(0, 6)   # the 6 position entries of a 12-state
VEL = slice(6, 12)  # the 6 velocity entries

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC_DIR)
for _p in (_ROOT, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def position_error(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """RMS over the 3 bodies of the Euclidean position error. (...,T,12) -> (...,T)."""
    d = pred[..., POS] - truth[..., POS]
    return np.sqrt(np.sum(d * d, axis=-1) / 3.0)


def finite_trajectories(pred: np.ndarray) -> np.ndarray:
    """Mask of trajectories whose whole rollout stayed finite. (B,T,12) -> (B,)."""
    return np.isfinite(pred).all(axis=(1, 2))


def regression_scores(delta_pred: np.ndarray, delta_true: np.ndarray) -> dict:
    """Standard regression metrics (RMSE, MAE, R^2) of the predicted one-step
    change, split into position and velocity, over all finite entries."""
    scores = {}
    for name, sel in (("position", POS), ("velocity", VEL)):
        p = np.asarray(delta_pred)[..., sel].ravel()
        t = np.asarray(delta_true)[..., sel].ravel()
        m = np.isfinite(p) & np.isfinite(t)
        err = p[m] - t[m]
        ss_res = float(np.mean(err ** 2))
        var = float(np.var(t[m]))
        scores[name] = {
            "rmse": float(np.sqrt(ss_res)),
            "mae": float(np.mean(np.abs(err))),
            "r2": 1.0 - ss_res / var if var > 0 else float("nan"),
        }
    return scores


def divergence_time(err: np.ndarray, times: np.ndarray, threshold: float) -> np.ndarray:
    """First time err (B,T) exceeds threshold; inf if it never does."""
    hit = err > threshold
    idx = np.where(hit.any(1), hit.argmax(1), -1)
    return np.where(idx >= 0, times[np.maximum(idx, 0)], np.inf)


def divergence_summary(err: np.ndarray, times: np.ndarray, thresholds) -> dict:
    """{threshold: (fraction_diverged, median_crossing_time)}. Non-diverging
    trajectories are capped at the horizon, so the median is a lower bound when
    fewer than half diverge."""
    horizon = float(times[-1])
    out = {}
    for thr in thresholds:
        t = divergence_time(err, times, thr)
        out[thr] = (float(np.mean(np.isfinite(t))), float(np.median(np.minimum(t, horizon))))
    return out


def total_energy(states: np.ndarray, M, G: float) -> np.ndarray:
    """Total energy (kinetic + potential) for states of shape (...,12)."""
    M = np.asarray(M, float)
    s = np.asarray(states)
    pos = s[..., POS].reshape(*s.shape[:-1], 3, 2)
    vel = s[..., VEL].reshape(*s.shape[:-1], 3, 2)
    kinetic = 0.5 * np.sum(M[:, None] * vel ** 2, axis=(-2, -1))
    potential = np.zeros(s.shape[:-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        for i, j in ((0, 1), (0, 2), (1, 2)):
            r = np.linalg.norm(pos[..., i, :] - pos[..., j, :], axis=-1)
            potential = potential - G * M[i] * M[j] / r
    return kinetic + potential


def relative_energy_drift(states: np.ndarray, E0: float, M, G: float) -> np.ndarray:
    return np.abs(total_energy(states, M, G) - E0) / abs(E0)


def min_pair_distance(states: np.ndarray) -> np.ndarray:
    """Smallest of the 3 inter-body distances. (...,T,12) -> (...,T)."""
    s = np.asarray(states)
    pos = s[..., POS].reshape(*s.shape[:-1], 3, 2)
    pairs = [np.linalg.norm(pos[..., i, :] - pos[..., j, :], axis=-1) for i, j in ((0, 1), (0, 2), (1, 2))]
    return np.min(np.stack(pairs, axis=-1), axis=-1)


def chaos_reference(start_states, meta, horizon, eps, k, n_calls, truth, seed=0):
    """How fast the Verlet solver itself diverges when the start is perturbed by
    eps. Returns position errors of shape (n, n_calls+1). Needs the repo root on
    sys.path (for DataGeneration)."""
    import DataGeneration.VerletSolver as verlet

    rng = np.random.default_rng(seed)
    M, G = np.asarray(meta["M"]), meta["G"]
    n_steps = int(round(horizon / meta["dt_solver"]))
    errs = []
    for j in range(len(start_states)):
        pert = start_states[j] + eps * rng.standard_normal(12)
        sim = verlet.simulate(pert, M, G, meta["dt_solver"], n_steps, meta["save_every"])
        sim = sim[::k][: n_calls + 1]
        errs.append(position_error(sim, truth[j, : len(sim)]))
    return np.stack(errs)