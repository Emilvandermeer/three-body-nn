from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass

# this just makes it so that I can do imports from src/ and DataGeneration/
#? Maybe can refactor to use relative imports instead of messing with sys.path, but just quick fix for now
_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (_ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn.functional as F

from augment import Augmenter
from data import Normaliser, PairSampler, fixed_pairs, load_trajectories, split_indices
from models import SingleStepMLP, save_checkpoint
from plots import plot_training_curves


@dataclass
class TrainConfig:
    data_dir: str          # folder with chunk_c*.npz
    out_dir: str           # where checkpoints / logs / figures go
    k: int                 # network step = k * dt_save (dt_save = 0.01 time units)
    hidden: int            # width of each hidden layer
    depth: int             # number of hidden layers
    batch_size: int
    steps: int             # total optimiser steps
    lr: float              # peak learning rate
    weight_decay: float
    grad_clip: float       # max gradient norm
    loss: str              
    seed: int              # seeds the model init and the pair sampler
    split_seed: int        # seeds the train/val/test split (keep fixed across runs

    #All augments from src/augment.py
    rotate: bool           
    reflect: bool          
    permute: bool          
    time_reverse: bool     

    eval_every: int        # steps between validation evaluations / checkpoints
    val_stride: int        # sub-sampling of validation start times
    device: str            # "auto", "cuda" or "cpu"
    max_trajectories: int  # 0 = use all; can change if you just want to train on a subset of the data for speed



CONFIG = TrainConfig(
    data_dir="data/traj_chunks",
    out_dir="outputs/single_step_k1",
    k=1,                    
    hidden=256,
    depth=4,
    batch_size=2048,
    steps=100000,
    lr=1e-3,
    weight_decay=1e-4,
    grad_clip=1.0,
    loss="mse",             
    seed=42,
    split_seed=0,
    rotate=True,            # set any of these to False for the augmentation ablation
    reflect=True,
    permute=True,
    time_reverse=True,
    eval_every=2000,
    val_stride=10, #
    device="auto",
    max_trajectories=0,     
)



def get_device(name: str) -> torch.device:
    'Function to use gpu if available, otherwise cpu'
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


@torch.no_grad()
def evaluate(net, norm: Normaliser, s0: torch.Tensor, s1: torch.Tensor, pred_delta_fn=None, chunk=16384):
    """One-step errors on a fixed set of pairs. the loss is computed in the normalised space, 
    but the RMSEs are reported in raw units.

    Returns (mse_normalised, rmse_position, rmse_velocity); the RMSEs are in raw
    units, for the change over one network step. `pred_delta_fn` lets us score
    the trivial baselines through the same code path (then net is None).
    """
    if net is not None:
        net.eval()
    se_norm, n = 0.0, 0
    se_pos, se_vel = 0.0, 0.0
    for a in range(0, len(s0), chunk):
        #a0 and a1 are the start and end states for a chunk of pairs. 
        #y is the normalised change, pred is the predicted normalised change
        a0, a1 = s0[a : a + chunk], s1[a : a + chunk]
        y = norm.encode_delta(a1 - a0).float()
        if pred_delta_fn is None:
            pred = net(norm.encode_state(a0).float())
        else:
            pred = norm.encode_delta(pred_delta_fn(a0)).float()
        se_norm += torch.sum((pred - y) ** 2).item()
        n += y.numel()

        #convert the normalised change back to raw units to compute RMSE in raw units
        raw_err = norm.decode_delta((pred - y).double())
        se_pos += torch.sum(raw_err[:, :6] ** 2).item()
        se_vel += torch.sum(raw_err[:, 6:] ** 2).item()
    n_rows = len(s0)
    if net is not None:
        net.train()
    return se_norm / n, math.sqrt(se_pos / (6 * n_rows)), math.sqrt(se_vel / (6 * n_rows))


def run(cfg: TrainConfig) -> dict:
    device = get_device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # data loading stuff
    data, meta = load_trajectories(cfg.data_dir)
    if cfg.max_trajectories:
        data = data[: cfg.max_trajectories]
    split = split_indices(len(data), cfg.split_seed)
    train_np, val_np = data[split["train"]], data[split["val"]]
    print(f"trajectories: train {len(train_np)}, val {len(val_np)}, test {len(split['test'])}; "
          f"samples per trajectory {data.shape[1]}; dt_save = {meta['dt_save']}; "
          f"network step = {cfg.k * meta['dt_save']:g} time units; device = {device}")

    norm = Normaliser.fit(train_np, cfg.k)  # TRAIN statistics only
    print("normaliser:", norm.to_dict())

    train = torch.from_numpy(train_np).to(device)  
    val = torch.from_numpy(val_np).to(device)
    sampler = PairSampler(train, cfg.k, seed=cfg.seed)
    augment = Augmenter(cfg.rotate, cfg.reflect, cfg.permute, cfg.time_reverse)
    val_s0, val_s1 = fixed_pairs(val, cfg.k, cfg.val_stride)
    print(f"validation pairs: {len(val_s0)}")

    # baselines: no-change and constant-velocity. 
    # our model should beat these at least
    dt_step = cfg.k * meta["dt_save"]
    zero_delta = lambda s: torch.zeros_like(s)  

    def const_vel_delta(s):  # "x += v * dt, v unchanged"
        d = torch.zeros_like(s)
        d[:, :6] = s[:, 6:] * dt_step
        return d

    baselines = {}
    for name, fn in (("no change", zero_delta), ("constant velocity", const_vel_delta)):
        mse, rp, rv = evaluate(None, norm, val_s0, val_s1, pred_delta_fn=fn)
        baselines[name] = {"val_mse_norm": mse, "rmse_pos": rp, "rmse_vel": rv}
        print(f"baseline {name:18s} val MSE (normalised) {mse:.4e} | RMSE pos {rp:.3e} vel {rv:.3e}")

    #Model, optimiser, loss function
    net = SingleStepMLP(cfg.hidden, cfg.depth).to(device)
    print(f"parameters: {sum(p.numel() for p in net.parameters()):,}")
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = F.mse_loss if cfg.loss == "mse" else (lambda a, b: F.smooth_l1_loss(a, b, beta=1.0))

    log_path = os.path.join(cfg.out_dir, "train_log.csv")
    best_val, best_step = float("inf"), -1
    run_loss, run_n = torch.zeros((), device=device), 0
    t_start = time.time()
    meta_ck = dict(meta)

    with open(log_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "lr", "train_loss", "val_loss", "val_rmse_pos", "val_rmse_vel", "seconds"])


        #training loop
        net.train()
        for step in range(1, cfg.steps + 1):
            s0, s1 = sampler.sample(cfg.batch_size)
            s0, s1 = augment(s0, s1, sampler.gen)  
            x = norm.encode_state(s0).float()
            y = norm.encode_delta(s1 - s0).float()

            loss = loss_fn(net(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()

            run_loss += loss.detach()  
            run_n += 1

            if step % cfg.eval_every == 0 or step == cfg.steps:
                val_mse, rp, rv = evaluate(net, norm, val_s0, val_s1)
                tr = run_loss.item() / max(1, run_n)
                run_loss, run_n = torch.zeros((), device=device), 0

                #this block just for logging stuff
                secs = time.time() - t_start
                writer.writerow([step, f"{cfg.lr:.3e}", f"{tr:.6e}", f"{val_mse:.6e}",
                                 f"{rp:.6e}", f"{rv:.6e}", f"{secs:.1f}"])
                fh.flush()
                print(f"step {step:>7d} | train {tr:.3e} | val {val_mse:.3e} | "
                      f"RMSE pos {rp:.2e} vel {rv:.2e} | {secs:.0f}s")

                #checkpoint for saving best model
                if val_mse < best_val:
                    best_val, best_step = val_mse, step
                    save_checkpoint(os.path.join(cfg.out_dir, "best.pt"), net, norm, asdict(cfg), meta_ck,
                                    extra={"step": step, "val_mse": val_mse})

    save_checkpoint(os.path.join(cfg.out_dir, "last.pt"), net, norm, asdict(cfg), meta_ck,
                    extra={"step": cfg.steps, "val_mse": val_mse})

    summary = {
        "best_val_mse_norm": best_val,
        "best_step": best_step,
        "final_val_mse_norm": val_mse,
        "baselines": baselines,
        "train_seconds": time.time() - t_start,
    }
    with open(os.path.join(cfg.out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    plot_training_curves(
        log_path,
        os.path.join(cfg.out_dir, "training_curves.png"),
        baselines={k_: v["val_mse_norm"] for k_, v in baselines.items()},
    )
    print(f"done. best val MSE {best_val:.3e} at step {best_step}. Outputs in {cfg.out_dir}")
    return summary


if __name__ == "__main__":
    run(CONFIG)
