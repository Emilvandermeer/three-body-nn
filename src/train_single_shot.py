"""Training of the single-shot network (Approach 1): f(s0, tau) ~ s(t0 + tau), 0 < tau <= T_max.

Run from the repository root:
    python src/train_single_shot.py

Settings are in CONFIG below (same style as train_single_step.py). For the hyperparameter
search and the horizon sweep use src/sweep_single_shot.py, which calls run() with modified
copies of CONFIG.

Outputs in cfg.out_dir:
    config.json, train_log.csv, best.pt (lowest validation loss), last.pt,
    metrics.json (baselines, best/final validation loss, validation error per lead time),
    training_curves.png
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass

# lets us import from src/ and from the repo root (DataGeneration/), same trick as rollout.py
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC_DIR)
for _p in (_ROOT, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn.functional as F

from augment import Augmenter
from data import load_trajectories, split_indices
from data_single_shot import (OUTPUT_MODES, SingleShotNormaliser, TripleSampler, constant_velocity_delta,
                              fixed_triples)
from models_single_shot import SingleShotMLP, save_checkpoint
from plots import plot_training_curves


@dataclass
class SingleShotConfig:
    data_dir: str          # folder with chunk_c*.npz
    out_dir: str           # where checkpoints / logs / figures go
    t_max: float           # largest lead time (time units); the network is trained on tau in (0, t_max]
    output: str            # "residual" or "direct", see SingleShotNormaliser
    start_mode: str        # "any" (every saved state can be a start state) or "initial" (only t0 = 0)
    hidden: int            # width of each hidden layer
    depth: int             # number of hidden layers
    activation: str        # "silu", "relu" or "tanh"
    batch_size: int
    steps: int             # total optimiser steps
    lr: float              # peak learning rate
    lr_schedule: str       # "cosine" (linear warm-up, then cosine decay to 0) or "constant"
    warmup_steps: int      # length of the linear warm-up (cosine schedule only)
    weight_decay: float
    grad_clip: float       # max gradient norm
    loss: str              # "mse" or "l1" (Breen et al. used the mean absolute error)
    seed: int              # seeds the model init and the triple sampler
    split_seed: int        # seeds the train/val/test split (keep fixed, and equal to the single-step runs!)

    # All augments from src/augment.py. They work unchanged on (s0, s1) with any lead time:
    # time reversal turns (s0, s1) into (reversed s1, reversed s0) with the SAME tau.
    rotate: bool
    reflect: bool
    permute: bool
    time_reverse: bool

    eval_every: int        # steps between validation evaluations / checkpoints
    val_start_stride: int  # spacing (samples) of validation start times
    val_tau_stride: int    # spacing (samples) of validation lead times
    device: str            # "auto", "cuda" or "cpu"
    max_trajectories: int  # 0 = use all; can change if you just want to train on a subset of the data for speed


CONFIG = SingleShotConfig(
    data_dir="data/traj_chunks",
    out_dir="outputs/single_shot/tmax2",
    t_max=2.0,
    output="residual",
    start_mode="any",
    hidden=256,
    depth=4,
    activation="silu",
    batch_size=1024,
    steps=30000,
    lr=1e-3,
    lr_schedule="cosine",
    warmup_steps=500,
    weight_decay=1e-4,
    grad_clip=1.0,
    loss="mse",
    seed=42,
    split_seed=0,           # same as train_single_step.CONFIG, so both approaches see the same test trajectories
    rotate=True,
    reflect=True,
    permute=True,
    time_reverse=True,
    eval_every=1000,
    val_start_stride=100,
    val_tau_stride=10,
    device="auto",
    max_trajectories=0,
)


def get_device(name: str) -> torch.device:
    'Function to use gpu if available, otherwise cpu'
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def lr_at(step: int, cfg: SingleShotConfig) -> float:
    """Learning rate for a given optimiser step. Warm-up avoids large, badly directed updates
    while Adam's moment estimates are still poor; the cosine decay lets the last part of
    training settle into a minimum instead of bouncing around at the peak learning rate."""
    if cfg.lr_schedule == "constant":
        return cfg.lr
    if step <= cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return 0.5 * cfg.lr * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(net, norm: SingleShotNormaliser, output: str, s0, s1, k, pred_delta_fn=None, chunk=32768):
    """Errors on a fixed set of triples. Loss in the normalised space (what we train on),
    RMSEs in raw units.

    Returns (mse_normalised, rmse_position, rmse_velocity, per_row_position_error) where the last
    one is the RMS-over-bodies position error of every triple (used for the error vs tau curves).
    `pred_delta_fn(s0, tau)` lets us score the trivial baselines through the same code (net is None then).
    """
    if net is not None:
        net.eval()
    se_norm, n = 0.0, 0
    se_pos, se_vel = 0.0, 0.0
    pos_err = []
    for a in range(0, len(s0), chunk):
        a0, a1 = s0[a : a + chunk], s1[a : a + chunk]
        tau = k[a : a + chunk].to(a0.dtype) * norm.dt_save
        if pred_delta_fn is None:
            z = net(norm.encode_input(a0, tau).float()).double()
            delta = norm.decode_delta(z, a0, tau, output)
        else:
            delta = pred_delta_fn(a0, tau)
        err = delta - (a1 - a0)  # raw error of the predicted state, float64
        se_norm += torch.sum(norm.encode_delta(err) ** 2).item()
        n += err.numel()
        se_pos += torch.sum(err[:, :6] ** 2).item()
        se_vel += torch.sum(err[:, 6:] ** 2).item()
        pos_err.append(torch.sqrt(torch.sum(err[:, :6] ** 2, dim=1) / 3.0))
    n_rows = len(s0)
    if net is not None:
        net.train()
    return se_norm / n, math.sqrt(se_pos / (6 * n_rows)), math.sqrt(se_vel / (6 * n_rows)), torch.cat(pos_err)


def error_by_tau(pos_err: torch.Tensor, k: torch.Tensor, dt_save: float) -> dict:
    """Median / quartiles of the position error for every lead time on the validation grid."""
    ks = torch.unique(k)
    out = {"tau": [], "median": [], "q25": [], "q75": []}
    for kk in ks.tolist():
        e = pos_err[k == kk].cpu().numpy()
        out["tau"].append(kk * dt_save)
        out["median"].append(float(np.median(e)))
        out["q25"].append(float(np.percentile(e, 25)))
        out["q75"].append(float(np.percentile(e, 75)))
    return out


def run(cfg: SingleShotConfig) -> dict:
    if cfg.output not in OUTPUT_MODES:
        raise ValueError(f"unknown output mode {cfg.output!r}")
    device = get_device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # data loading stuff (identical split to the single-step approach)
    data, meta = load_trajectories(cfg.data_dir)
    if cfg.max_trajectories:
        data = data[: cfg.max_trajectories]
    split = split_indices(len(data), cfg.split_seed)
    train_np, val_np = data[split["train"]], data[split["val"]]
    k_max = int(round(cfg.t_max / meta["dt_save"]))
    print(f"trajectories: train {len(train_np)}, val {len(val_np)}, test {len(split['test'])}; "
          f"samples per trajectory {data.shape[1]}; dt_save = {meta['dt_save']}; "
          f"T_max = {cfg.t_max:g} ({k_max} samples); output = {cfg.output}; start = {cfg.start_mode}; device = {device}")

    norm = SingleShotNormaliser.fit(train_np, k_max, meta["dt_save"], cfg.start_mode, seed=cfg.seed)  # TRAIN only
    print("normaliser:", norm.to_dict())

    train = torch.from_numpy(train_np).to(device)
    val = torch.from_numpy(val_np).to(device)
    sampler = TripleSampler(train, k_max, cfg.start_mode, seed=cfg.seed)
    augment = Augmenter(cfg.rotate, cfg.reflect, cfg.permute, cfg.time_reverse)
    val_s0, val_s1, val_k = fixed_triples(val, k_max, cfg.val_start_stride, cfg.val_tau_stride)
    print(f"validation triples: {len(val_s0)}")

    # trivial baselines, scored in exactly the same way as the network.
    # "mean state" = always predict the average training state, i.e. the best you can do
    # without looking at s0 at all. Errors of anything useful must stay below this.
    mean_state = train.reshape(-1, train.shape[-1]).mean(dim=0)
    baselines_fns = {
        "no change": lambda s, tau: torch.zeros_like(s),
        "constant velocity": constant_velocity_delta,
        "mean state": lambda s, tau: mean_state[None, :] - s,
    }
    baselines = {}
    for name, fn in baselines_fns.items():
        mse, rp, rv, _ = evaluate(None, norm, cfg.output, val_s0, val_s1, val_k, pred_delta_fn=fn)
        baselines[name] = {"val_mse_norm": mse, "rmse_pos": rp, "rmse_vel": rv}
        print(f"baseline {name:18s} val MSE (normalised) {mse:.4e} | RMSE pos {rp:.3e} vel {rv:.3e}")

    # Model, optimiser, loss function
    net = SingleShotMLP(cfg.hidden, cfg.depth, cfg.activation).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"parameters: {n_params:,}")
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = F.mse_loss if cfg.loss == "mse" else F.l1_loss

    log_path = os.path.join(cfg.out_dir, "train_log.csv")
    best_val, best_step = float("inf"), -1
    run_loss, run_n = torch.zeros((), device=device), 0
    t_start = time.time()
    meta_ck = dict(meta)
    val_mse = float("nan")

    with open(log_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "lr", "train_loss", "val_loss", "val_rmse_pos", "val_rmse_vel", "seconds"])

        # training loop
        net.train()
        for step in range(1, cfg.steps + 1):
            lr = lr_at(step, cfg)
            for g in opt.param_groups:
                g["lr"] = lr

            s0, s1, k = sampler.sample(cfg.batch_size)
            s0, s1 = augment(s0, s1, sampler.gen)  # same tau for the augmented pair, see SingleShotConfig
            tau = k.to(s0.dtype) * meta["dt_save"]
            # loss in the normalised change space, in float32. For "residual" the constant
            # velocity part is added inside decode_delta, so the gradient flows only through z.
            x = norm.encode_input(s0, tau).float()
            y = norm.encode_delta(s1 - s0).float()
            pred = norm.encode_delta(norm.decode_delta(net(x), s0, tau, cfg.output))

            loss = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()

            run_loss += loss.detach()
            run_n += 1

            if step % cfg.eval_every == 0 or step == cfg.steps:
                val_mse, rp, rv, _ = evaluate(net, norm, cfg.output, val_s0, val_s1, val_k)
                tr = run_loss.item() / max(1, run_n)
                run_loss, run_n = torch.zeros((), device=device), 0

                # this block just for logging stuff
                secs = time.time() - t_start
                writer.writerow([step, f"{lr:.3e}", f"{tr:.6e}", f"{val_mse:.6e}",
                                 f"{rp:.6e}", f"{rv:.6e}", f"{secs:.1f}"])
                fh.flush()
                print(f"step {step:>7d} | lr {lr:.2e} | train {tr:.3e} | val {val_mse:.3e} | "
                      f"RMSE pos {rp:.2e} vel {rv:.2e} | {secs:.0f}s")

                # checkpoint for saving best model (early stopping on the validation loss)
                if val_mse < best_val:
                    best_val, best_step = val_mse, step
                    save_checkpoint(os.path.join(cfg.out_dir, "best.pt"), net, norm, asdict(cfg), meta_ck,
                                    extra={"step": step, "val_mse": val_mse})

    save_checkpoint(os.path.join(cfg.out_dir, "last.pt"), net, norm, asdict(cfg), meta_ck,
                    extra={"step": cfg.steps, "val_mse": val_mse})

    # validation error per lead time of the BEST checkpoint (for the horizon sweep figure)
    best = torch.load(os.path.join(cfg.out_dir, "best.pt"), map_location=device, weights_only=True)
    net.load_state_dict(best["net"])
    _, rp_best, rv_best, pos_err = evaluate(net, norm, cfg.output, val_s0, val_s1, val_k)
    cv_err = evaluate(None, norm, cfg.output, val_s0, val_s1, val_k, pred_delta_fn=constant_velocity_delta)[3]

    summary = {
        "best_val_mse_norm": best_val,
        "best_step": best_step,
        "final_val_mse_norm": val_mse,
        "best_val_rmse_pos": rp_best,
        "best_val_rmse_vel": rv_best,
        "baselines": baselines,
        "n_params": n_params,
        "train_seconds": time.time() - t_start,
        "val_error_by_tau": error_by_tau(pos_err, val_k, meta["dt_save"]),
        "val_error_by_tau_const_vel": error_by_tau(cv_err, val_k, meta["dt_save"]),
    }
    with open(os.path.join(cfg.out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    plot_training_curves(
        log_path,
        os.path.join(cfg.out_dir, "training_curves.png"),
        baselines={k_: v["val_mse_norm"] for k_, v in baselines.items() if k_ != "no change"},
    )
    print(f"done. best val MSE {best_val:.3e} at step {best_step}. Outputs in {cfg.out_dir}")
    return summary


if __name__ == "__main__":
    run(CONFIG)
