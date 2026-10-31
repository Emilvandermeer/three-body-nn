"""Hyperparameter search, ablations and horizon sweep for the single-shot network.

Run from the repository root, one experiment at a time:
    python src/sweep_single_shot.py arch       # architecture search at T_max = 2
    python src/sweep_single_shot.py ablation   # design choices around the default network
    python src/sweep_single_shot.py horizon    # accuracy vs T_max (the Breen et al. 2019 style split)
    python src/sweep_single_shot.py summary    # only (re)write the summary tables / figures
    python src/sweep_single_shot.py evaluate   # test-set evaluation of the horizon models (seed 0)

Every configuration is trained with several seeds (model init + sampling order), so the
summaries report mean +- standard deviation over seeds. All selection is done on the
VALIDATION trajectories; the test trajectories are only touched by evaluate_single_shot.py.

Runs that already have a metrics.json are skipped, so an interrupted sweep can just be
restarted (same idea as generate_dataset.py skipping existing chunks). A run that is in progress
has a running.lock file, so two sweep processes can work through the same experiment side by
side (e.g. one per CPU core: `python src/sweep_single_shot.py horizon 1` twice). If a run
crashes, delete its running.lock before restarting.

Outputs: outputs/single_shot/sweeps/<experiment>/<run>/seed<k>/ (one training run each),
plus summary.csv, summary.json and a figure per experiment in outputs/single_shot/sweeps/<experiment>/.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import replace

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC_DIR)
for _p in (_ROOT, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

import plots_single_shot
import train_single_shot
from train_single_shot import CONFIG

SWEEP_DIR = "outputs/single_shot/sweeps"
SEEDS = (0, 1, 2)

# Searches use a reduced budget (20k steps) so that a full grid x 3 seeds fits on a CPU in a
# few hours; the cosine schedule is stretched to whatever `steps` is, so every run still
# finishes its learning rate decay.
SEARCH_STEPS = 20000

# name -> overrides of CONFIG. "default" is CONFIG itself (residual output, any start, all augments).
EXPERIMENTS = {
    # width x depth (and Breen et al.'s 10 x 128 ReLU network) at T_max = 2
    "arch": {
        "w128_d4": dict(hidden=128, depth=4),
        "w256_d4": dict(hidden=256, depth=4),
        "w512_d4": dict(hidden=512, depth=4),
        "w128_d8": dict(hidden=128, depth=8),
        "w128_d10_silu": dict(hidden=128, depth=10),
        "w128_d10_relu_breen": dict(hidden=128, depth=10, activation="relu"),
    },
    # one design choice changed at a time, everything else as in the default network
    "ablation": {
        "default": dict(),
        "direct_output": dict(output="direct"),
        "initial_starts_only": dict(start_mode="initial"),
        "no_augmentation": dict(rotate=False, reflect=False, permute=False, time_reverse=False),
    },
    # how hard does the task get when the network has to cover longer lead times?
    "horizon": {
        "tmax0.5": dict(t_max=0.5),
        "tmax1": dict(t_max=1.0),
        "tmax2": dict(t_max=2.0),
        "tmax5": dict(t_max=5.0),
        "tmax10": dict(t_max=10.0),
    },
}

# architecture of the ablations: the default network (= CONFIG, same size as the single-step net)
ABLATION_ARCH = dict(hidden=256, depth=4)
# architecture of the horizon sweep: the winner of the "arch" search (see its summary.csv)
HORIZON_ARCH = dict(hidden=256, depth=4)


def run_dir(experiment: str, name: str, seed: int) -> str:
    return os.path.join(SWEEP_DIR, experiment, name, f"seed{seed}")


def run_experiment(experiment: str, seeds=SEEDS, steps: int = SEARCH_STEPS):
    base = replace(CONFIG, steps=steps)
    if experiment == "ablation":
        base = replace(base, **ABLATION_ARCH)
    elif experiment == "horizon":
        base = replace(base, **HORIZON_ARCH)
    for name, overrides in EXPERIMENTS[experiment].items():
        for seed in seeds:
            out = run_dir(experiment, name, seed)
            lock = os.path.join(out, "running.lock")
            if os.path.exists(os.path.join(out, "metrics.json")) or os.path.exists(lock):
                print(f"Skipping {out} (already done or running)")
                continue
            os.makedirs(out, exist_ok=True)
            open(lock, "w").close()
            print(f"\n######## {experiment} / {name} / seed {seed}")
            train_single_shot.run(replace(base, **overrides, seed=seed, out_dir=out))
            os.remove(lock)


def _interp(tau_grid, tau, values):
    return np.interp(tau_grid, np.asarray(tau), np.asarray(values))


def summarise(experiment: str) -> list:
    """Collect metrics.json of all finished runs of an experiment -> mean / std over seeds."""
    rows, curves = [], {}
    for name in EXPERIMENTS[experiment]:
        ms = []
        for seed in SEEDS:
            p = os.path.join(run_dir(experiment, name, seed), "metrics.json")
            if os.path.exists(p):
                ms.append(json.load(open(p)))
        if not ms:
            continue
        cfg = json.load(open(os.path.join(run_dir(experiment, name, SEEDS[0]), "config.json")))

        def stat(key):
            v = np.array([m[key] for m in ms], float)
            return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan")

        # median validation position error at a few lead times, averaged over seeds
        t_max = cfg["t_max"]
        at = {}
        for frac in (0.25, 0.5, 1.0):
            vals = [_interp([frac * t_max], m["val_error_by_tau"]["tau"], m["val_error_by_tau"]["median"])[0] for m in ms]
            at[f"med_pos_err@{frac:g}Tmax"] = (float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"))

        row = {"run": name, "t_max": t_max, "hidden": cfg["hidden"], "depth": cfg["depth"],
               "activation": cfg["activation"], "output": cfg["output"], "start_mode": cfg["start_mode"],
               "augment": all(cfg[a] for a in ("rotate", "reflect", "permute", "time_reverse")),
               "n_seeds": len(ms), "n_params": ms[0]["n_params"]}
        for key in ("best_val_mse_norm", "best_val_rmse_pos", "best_val_rmse_vel", "train_seconds"):
            row[key], row[key + "_std"] = stat(key)
        for key, (m_, s_) in at.items():
            row[key], row[key + "_std"] = m_, s_
        b = ms[0]["baselines"]
        row["cv_val_mse_norm"] = b["constant velocity"]["val_mse_norm"]
        row["nochange_val_mse_norm"] = b["no change"]["val_mse_norm"]
        rows.append(row)

        # error vs tau curves (median over validation triples), mean over seeds
        tau = np.asarray(ms[0]["val_error_by_tau"]["tau"])
        med = np.mean([m["val_error_by_tau"]["median"] for m in ms], axis=0)
        cv = np.asarray(ms[0]["val_error_by_tau_const_vel"]["median"])
        curves[name] = (tau, med, cv)

    if not rows:
        print(f"no finished runs for {experiment}")
        return rows
    d = os.path.join(SWEEP_DIR, experiment)
    with open(os.path.join(d, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump(rows, open(os.path.join(d, "summary.json"), "w"), indent=2)

    # print a compact table
    print(f"\n{experiment}: best validation loss (normalised MSE), mean +- std over seeds")
    print("-" * 78)
    for r in rows:
        print(f"  {r['run']:22s} params {r['n_params']:>8,d} | val MSE {r['best_val_mse_norm']:.4e} +- "
              f"{r['best_val_mse_norm_std']:.1e} | pos RMSE {r['best_val_rmse_pos']:.3e} | "
              f"(const. vel. {r['cv_val_mse_norm']:.3e})")

    plots_single_shot.plot_sweep_bars(rows, os.path.join(d, "val_loss.png"),
                                      title=f"{experiment}: best validation loss (mean ± std over seeds)")
    plots_single_shot.plot_sweep_curves(curves, os.path.join(d, "val_error_vs_tau.png"),
                                        log_x=(experiment == "horizon"))
    return rows


def evaluate_horizon(seed: int = 0, single_step_checkpoint: str = "outputs/single_step_k1/best.pt"):
    """Test-set evaluation (evaluate_single_shot.run) of the horizon-sweep models of one seed,
    plus one figure with the test error vs tau of all of them and of the single-step rollout."""
    import evaluate_single_shot

    curves = {}
    longest = {}  # reference curves (single-step rollout, constant velocity): keep the longest tau range
    for name in EXPERIMENTS["horizon"]:
        ck = os.path.join(run_dir("horizon", name, seed), "best.pt")
        if not os.path.exists(ck):
            continue
        res = evaluate_single_shot.run(replace(evaluate_single_shot.CONFIG, checkpoint=ck,
                                               single_step_checkpoint=single_step_checkpoint))
        for key, c in res["median_error_curves"].items():
            tau, med = np.array(c["tau"]), np.array(c["median"])
            if key == "single-shot net":
                curves[f"single-shot, T_max={res['t_max']:g}"] = (tau, med)
            elif key.startswith("single-step") or key == "constant velocity":
                if key not in longest or tau[-1] > longest[key][0][-1]:
                    longest[key] = (tau, med)
    if not curves:
        print("no horizon models found")
        return
    plots_single_shot.plot_horizon_test(curves, longest,
                                        os.path.join(SWEEP_DIR, "horizon", f"test_error_vs_tau_seed{seed}.png"))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "summary"
    if len(sys.argv) > 2:  # optional: number of CPU threads, handy to run two sweeps side by side
        torch.set_num_threads(int(sys.argv[2]))
    if what in EXPERIMENTS:
        run_experiment(what)
        summarise(what)
    elif what == "evaluate":
        evaluate_horizon()
    elif what == "summary":
        for e in EXPERIMENTS:
            if os.path.isdir(os.path.join(SWEEP_DIR, e)):
                summarise(e)
    else:
        raise SystemExit(f"unknown experiment {what!r}; use one of {list(EXPERIMENTS)} or 'summary'")
