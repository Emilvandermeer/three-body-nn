# Single-shot MLP (Approach 1): results

All numbers below come from the files in this folder. They were produced on a 2-core CPU, with no GPU.

## Setup (identical for all runs unless stated)

- **Data:** the partners' dataset, 2000 trajectories × 2001 samples, Δt_save = 0.01, E = −0.8987, equal masses.
  - Split by trajectory, 80/10/10 (`split_seed = 0`), the same split as the single-step code.
  - Model selection uses only the validation trajectories. The test trajectories are used only by `evaluate_single_shot.py`.
- **Task:** f(s0, τ) ≈ s(t0 + τ) for 0 < τ ≤ T_max, in one forward pass.
  - Input: 12 normalised state values plus τ/T_max.
  - Output, "residual" mode: s0 + cv(s0, τ) + (τ/T_max)·z·rscale, where cv is constant-velocity motion.
- **Training pairs:** start state s0 = any saved state of a training trajectory; τ uniform on the grid of (0, T_max].
- **Augmentation:** rotation, reflection, relabelling the bodies, time reversal (the partners' `augment.py`). Checked physically valid for any τ: Verlet applied to an augmented start reproduces the augmented target to 1e-13.
- **Optimisation:**
  - AdamW, peak learning rate 1e-3, weight decay 1e-4.
  - 500 warm-up steps, then cosine decay to 0.
  - Gradient-norm clipping at 1, batch size 1024, 20,000 steps.
  - MSE loss on the normalised change.
  - Early stopping: best validation loss, evaluated every 1000 steps.
- **Seeds:** 3 seeds (0, 1, 2) per configuration; results are mean ± std over seeds.
- **Validation grid:** all validation trajectories, start times every 1.0 time unit, lead times every 0.1 (plus 0.01–0.05).

Note on budget: 43 of the 45 runs reached their best validation loss at the last or second-to-last evaluation (steps 19k–20k). They are budget-limited rather than converged, so longer training would lower all errors. The two exceptions are "initial starts only" runs (best at step 3000–4000): with only 1600 different start states they overfit early.

## 1. Architecture search (T_max = 2, validation position RMSE)

| Network | Parameters | Position RMSE | Velocity RMSE |
|---|---|---|---|
| 128 × 4, SiLU | 52,876 | 0.0763 ± 0.0007 | 0.151 |
| 256 × 4, SiLU | 204,044 | 0.0577 ± 0.0001 | 0.115 |
| **512 × 4, SiLU** | 801,292 | **0.0437 ± 0.0001** | **0.091** |
| 128 × 8, SiLU | 118,924 | 0.0600 ± 0.0003 | 0.113 |
| 128 × 10, SiLU | 151,948 | 0.0563 ± 0.0002 | 0.105 |
| 128 × 10, ReLU (Breen et al. 2019) | 151,948 | 0.0747 ± 0.0008 | 0.142 |
| constant velocity baseline | – | 0.531 | – |

- **Width:** the error drops monotonically with width, and wider was not tried (CPU time).
- **Activation:** SiLU beats ReLU by 25% at the same 10 × 128 size.
- **Choice:** 512 × 4 is used for the horizon sweep.

Figures: `sweeps/arch/val_rmse_pos.png`, `sweeps/arch/val_error_vs_tau.png`

## 2. Ablations (256 × 4, T_max = 2, one change at a time)

| Variant | Validation position RMSE | Test median position error at τ = 0.5 / 1 / 2 (start at t = 0, seed 0) |
|---|---|---|
| default (residual output, any start, all augmentations) | 0.0577 ± 0.0001 | 0.032 / 0.076 / 0.284 |
| direct output (no constant-velocity term) | 0.0601 ± 0.0010 | – |
| no augmentation | 0.0679 ± 0.0003 | – |
| initial starts only (t0 = 0, as Breen et al.) | 0.2138 ± 0.0070 | 0.042 / 0.096 / 0.273 |

- **Augmentation:** lowers the error by 15%.
- **Residual output:** lowers the error by 4%.
- **Start states:** "any start" is 3.7× better on validation, which uses start states from anywhere on a trajectory. On the test protocol, which starts from the generated initial conditions, the two are similar. The initial-only model is even slightly better at τ = 2. So using every state as a start mainly buys generalisation to arbitrary states, not accuracy from the generated initial conditions.
- **Units:** compare raw RMSE across ablations, not the normalised loss. The normaliser is fitted per start mode.

Figures: `sweeps/ablation/val_rmse_pos.png`, `sweeps/ablation/{default,initial_starts_only}/seed0/eval_test/`

## 3. Horizon sweep (512 × 4, validation position RMSE over all τ ≤ T_max)

| T_max | Position RMSE | Constant velocity |
|---|---|---|
| 0.5 | 0.0043 ± 0.0002 | 0.064 |
| 1 | 0.0146 ± 0.0001 | 0.195 |
| 2 | 0.0437 ± 0.0001 | 0.531 |
| 5 | 0.1695 ± 0.0008 | 1.47 |
| 10 | 0.3873 ± 0.0017 | 3.12 |

The T_max = 2 run is the same configuration and seeds as the 512 × 4 run of the architecture search, and it gave identical numbers. That confirms the runs are deterministic.

## 4. Test set (200 test trajectories, start at t = 0, seed-0 models)

Median position error at lead time τ:

| Method | τ = 0.1 | 0.5 | 1 | 2 | 5 | 10 |
|---|---|---|---|---|---|---|
| single-shot, T_max = 0.5 | 0.0019 | 0.0147 | | | | |
| single-shot, T_max = 1 | 0.0031 | 0.0129 | 0.051 | | | |
| single-shot, T_max = 2 | 0.0060 | 0.0193 | 0.051 | 0.214 | | |
| single-shot, T_max = 5 | 0.0141 | 0.0534 | 0.095 | 0.208 | 0.573 | |
| single-shot, T_max = 10 | 0.0369 | 0.0934 | 0.159 | 0.348 | 0.619 | 1.09 |
| single-step rollout (k = 1)* | 0.0021 | 0.0192 | 0.065 | 0.220 | 0.756 | 1.92 |
| kNN (k chosen on validation) | 0.012–0.014 | 0.15–0.17 | 0.46–0.48 | 0.92–0.93 | 1.60–1.65 | 2.11 |
| constant velocity | 0.0090 | 0.221 | 0.754 | 1.93 | 5.38 | 11.5 |
| mean state | 0.745 | 0.845 | 1.06 | 1.41 | 2.18 | 2.69 |

\*The single-step model is the partners' code and configuration, trained by me for 30k instead of 100k steps (`outputs/single_step_k1_cpu30k`), only so the comparison could be run. Re-run `evaluate_single_shot.py` / `sweep_single_shot.py evaluate` with the partners' properly trained checkpoint for the final numbers.

**What the table shows:**
- **Short horizons:** the single-shot network competes with the single-step rollout only when its training range is short. The T_max = 0.5 net matches or beats it at τ = 0.1–0.5. The T_max = 1 and 2 nets beat it at τ = 0.5–1 but are 1.5–3× worse at τ = 0.1.
- **Cost of a wide range:** a network that covers a long range is worse at short τ, because its error is spread over the whole range. At τ = 0.1 the T_max = 10 network is 19× worse than the T_max = 0.5 network.
- **Long horizons:** single-shot beats the iterated single-step model. At τ = 5 the errors are 0.57 (T_max = 5) against 0.76, and at τ = 10 they are 1.09 against 1.92. The rollout's errors compound with every step.
- **Overall:** the network beats kNN, constant velocity and the mean-state baseline at every τ ≥ 0.5.
- **R² of the predicted change** (position / velocity):

| T_max | Position R² | Velocity R² |
|---|---|---|
| 0.5 | 0.995 | 0.951 |
| 1 | 0.982 | 0.944 |
| 2 | 0.958 | 0.926 |
| 5 | 0.890 | 0.855 |
| 10 | 0.808 | 0.807 |

- **Chaos reference:** Verlet started from a state perturbed by 1e-6 stays within about 1e-4 of the reference over 10 time units, on 10 test trajectories. With a 1e-3 perturbation, 40% exceed 0.1 by t = 10. So on these horizons the networks' errors come from approximation, not from chaotic amplification of tiny input errors.
- **Energy:** the median relative energy error of the predictions grows with T_max: 0.024, 0.059, 0.14, 0.33, 0.60. The single-shot net has no built-in energy conservation; Breen et al. added a projection layer onto the energy surface to reduce this error.

Figures, per model in `sweeps/horizon/tmax*/seed0/eval_test/`:
- `error_vs_tau.png`
- `trajectories.png`
- `energy_error.png`
- `cost_vs_tau.png`

Combined figure: `sweeps/horizon/test_error_vs_tau_seed0.png`.

## 5. Cost (seconds per trajectory to obtain the state at τ = T_max; batch = 1024)

| Method | T_max = 0.5 | 2 | 10 |
|---|---|---|---|
| Verlet, Python loop (as in `DataGeneration`) | 3.8e-2 | 2.2e-1 | 7.9e-1 |
| Verlet, numpy, 1 trajectory | 9.0e-3 | 3.5e-2 | 2.3e-1 |
| Verlet, numpy, batched | 3.0e-4 | 1.3e-3 | 6.6e-3 |
| single-step rollout, 1 trajectory / batched | 4.4e-3 / 2.0e-4 | 2.5e-2 / 8.4e-4 | 8.2e-2 / 3.8e-3 |
| single-shot net, 1 trajectory / batched | 2.6e-4 / 1.3e-5 | 2.7e-4 / 1.2e-5 | 3.4e-4 / 1.4e-5 |

- **Scaling:** single-shot cost is independent of τ. Measured: 4.1e-4 s at τ = 0.01 against 3.4e-4 s at τ = 10.
- **Batched speed-up against batched numpy Verlet:** about 23× at τ = 0.5, 110× at τ = 2 and 470× at τ = 10.
- **Batched speed-up against the batched single-step rollout:** about 15× at τ = 0.5, 70× at τ = 2 and 270× at τ = 10.
- **Fair baseline:** the Python-loop Verlet used for data generation is 3–6× slower than numpy for a single trajectory. Timing it overstates the network's advantage; `baselines_single_shot.verlet_batched` gives identical results to machine precision.
- **Amortisation:** training took about 15 minutes per network on CPU, not counting data generation. Against batched Verlet this pays off after about 1.5e5 predictions at T_max = 10, or 7e5 at T_max = 2.

## Reproduce

From the repo root:

```bash
python src/check_single_shot.py
python src/sweep_single_shot.py arch 1 & python src/sweep_single_shot.py ablation 1
python src/sweep_single_shot.py horizon 1
python src/sweep_single_shot.py summary
python src/sweep_single_shot.py evaluate
```

`evaluate` expects the single-step checkpoint at `outputs/single_step_k1/best.pt`.
