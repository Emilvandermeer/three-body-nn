"""
Sampling and normalisation for the single-shot approach (Approach 1).

The single-shot network gets a start state s0 and a lead time tau and has to output the
state tau time units later in ONE forward pass (no iterating):

    f(s0, tau) ~ s(t0 + tau),      0 < tau <= T_max

So instead of the (state, next state) pairs of the single-step approach we need
(s0, tau, s1) triples, with s0 = s(t0) and s1 = s(t0 + tau).

Which states can be start states ("start_mode"):
  * "any"     : every saved state of a training trajectory. The equations of motion do not
                depend on t explicitly (autonomous ODE) and every state on a trajectory has
                the same energy E, so s(t0) is just as valid an initial condition as s(0).
                This gives ~1600 * 2001 different start states instead of 1600.
                Caveat: states later on a trajectory are not distributed exactly like the
                generated initial conditions (the bodies spread out further than the [-1, 1]
                box of Arc_Generater), but they are on the same energy surface.
  * "initial" : only t0 = 0, i.e. only the generated initial conditions (what Breen et al.
                2019 did). Used as an ablation to see how much the "any" trick buys us.

State layout (same as everywhere else in the project):
    [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3]

Everything here reuses the conventions of src/data.py: raw float64 states, normalisation
with one scalar per group (positions / velocities), split by trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from data import N_POS, STATE_DIM

START_MODES = ("any", "initial")
OUTPUT_MODES = ("direct", "residual")


def constant_velocity_delta(s0: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Change predicted by "x += v * tau, v unchanged" (first order Taylor expansion of the
    true solution around t0). s0 (B,12), tau (B,) -> (B,12)."""
    d = torch.zeros_like(s0)
    d[:, :N_POS] = s0[:, N_POS:] * tau[:, None]
    return d


@dataclass
class SingleShotNormaliser:
    """Scalar rescaling of the network input and output.

    Input  : x = [s0 / state scales (12 numbers), tau / T_max (1 number)]  -> 13 numbers
    Output : the network output z (12 numbers) is turned into the predicted change
             delta = s(t0 + tau) - s0 in one of two ways ("output mode"):

      * "direct"   : delta = z * dscale
      * "residual" : delta = cv(s0, tau) + (tau / T_max) * z * rscale
                     where cv is the constant velocity guess (see constant_velocity_delta).
                     For tau -> 0 this goes to the right answer (delta -> 0) no matter what
                     the network does, and the network only has to learn the correction on
                     top of straight-line motion, which is zero at tau = 0 and grows with tau.

    dscale = RMS of the true change, rscale = RMS of (true change - cv) / (tau / T_max),
    both with one value for positions and one for velocities (like data.Normaliser),
    estimated on training triples only, with tau drawn uniformly from (0, T_max].
    """

    pos_scale: float   # RMS of start-state positions
    vel_scale: float   # RMS of start-state velocities
    dpos_scale: float  # RMS of the position change
    dvel_scale: float  # RMS of the velocity change
    rpos_scale: float  # RMS of the scaled position residual (for "residual" output)
    rvel_scale: float  # RMS of the scaled velocity residual
    k_max: int         # largest lead time in saved samples, T_max = k_max * dt_save
    dt_save: float     # time between two saved samples (0.01)

    # same caching trick as in data.Normaliser, so we dont rebuild the scale vectors every call
    _cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    @property
    def t_max(self) -> float:
        return self.k_max * self.dt_save

    @classmethod
    def fit(cls, train_traj: np.ndarray, k_max: int, dt_save: float, start_mode: str = "any",
            n_samples: int = 200_000, seed: int = 0):
        """Monte-Carlo estimate of all scales from n_samples random training triples drawn
        exactly like TripleSampler draws them (but with numpy, so it is independent of the
        torch generator used for training)."""
        n, T, _ = train_traj.shape
        if k_max >= T:
            raise ValueError(f"trajectories have {T} samples, need more than k_max={k_max}")
        rng = np.random.default_rng(seed)
        i = rng.integers(0, n, n_samples)
        k = rng.integers(1, k_max + 1, n_samples)
        if start_mode == "any":
            t0 = (rng.random(n_samples) * (T - k)).astype(np.int64)  # uniform in 0..T-1-k
        else:
            t0 = np.zeros(n_samples, dtype=np.int64)
        s0, s1 = train_traj[i, t0].astype(np.float64), train_traj[i, t0 + k].astype(np.float64)
        tau = k * dt_save
        frac = (k / k_max)[:, None]

        delta = s1 - s0
        cv = np.zeros_like(s0)
        cv[:, :N_POS] = s0[:, N_POS:] * tau[:, None]
        resid = (delta - cv) / frac

        def rms(a):
            return float(np.sqrt(np.mean(a**2)))

        return cls(
            rms(s0[:, :N_POS]), rms(s0[:, N_POS:]),
            rms(delta[:, :N_POS]), rms(delta[:, N_POS:]),
            rms(resid[:, :N_POS]), rms(resid[:, N_POS:]),
            int(k_max), float(dt_save),
        )

    def _vecs(self, ref: torch.Tensor):
        key = (ref.device, ref.dtype)
        if key not in self._cache:
            s = ref.new_tensor([self.pos_scale] * N_POS + [self.vel_scale] * N_POS)
            d = ref.new_tensor([self.dpos_scale] * N_POS + [self.dvel_scale] * N_POS)
            r = ref.new_tensor([self.rpos_scale] * N_POS + [self.rvel_scale] * N_POS)
            self._cache[key] = (s, d, r)
        return self._cache[key]

    def encode_input(self, s0: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """(B,12) raw start states + (B,) lead times -> (B,13) network input."""
        frac = (tau / self.t_max).to(s0.dtype)[:, None]
        return torch.cat([s0 / self._vecs(s0)[0], frac], dim=1)

    def decode_delta(self, z: torch.Tensor, s0: torch.Tensor, tau: torch.Tensor, output: str) -> torch.Tensor:
        """Network output z (B,12) -> predicted raw change delta (B,12). Works in whatever
        dtype z has (float32 during training, float64 at prediction time)."""
        _, d, r = self._vecs(z)
        if output == "direct":
            return z * d
        if output == "residual":
            s0, tau = s0.to(z.dtype), tau.to(z.dtype)
            return constant_velocity_delta(s0, tau) + (tau / self.t_max)[:, None] * z * r
        raise ValueError(f"unknown output mode {output!r}, use one of {OUTPUT_MODES}")

    def encode_delta(self, delta: torch.Tensor) -> torch.Tensor:
        """Raw change -> normalised change; the loss is measured in this space for both
        output modes, so that their validation losses are directly comparable."""
        return delta / self._vecs(delta)[1]

    def to_dict(self) -> dict:
        return {
            "pos_scale": self.pos_scale, "vel_scale": self.vel_scale,
            "dpos_scale": self.dpos_scale, "dvel_scale": self.dvel_scale,
            "rpos_scale": self.rpos_scale, "rvel_scale": self.rvel_scale,
            "k_max": self.k_max, "dt_save": self.dt_save,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SingleShotNormaliser":
        return cls(**d)


class TripleSampler:
    """Draws random (s0, s1, k) triples from a trajectory tensor, s1 = state k saved
    samples after s0, with k uniform in 1..k_max (so tau = k * dt_save uniform on the grid of
    (0, T_max]). Like data.PairSampler the trajectories live on the compute device and
    triples are produced on the fly by indexing."""

    def __init__(self, traj: torch.Tensor, k_max: int, start_mode: str = "any", seed: int = 0):
        if start_mode not in START_MODES:
            raise ValueError(f"unknown start mode {start_mode!r}, use one of {START_MODES}")
        if k_max >= traj.shape[1]:
            raise ValueError(f"trajectories have {traj.shape[1]} samples, need more than k_max={k_max}")
        self.traj = traj
        self.k_max = k_max
        self.start_mode = start_mode
        self.gen = torch.Generator(device=traj.device)
        self.gen.manual_seed(seed)

    def sample(self, batch_size: int):
        n, T, _ = self.traj.shape
        dev = self.traj.device
        i = torch.randint(0, n, (batch_size,), generator=self.gen, device=dev)
        k = torch.randint(1, self.k_max + 1, (batch_size,), generator=self.gen, device=dev)
        if self.start_mode == "any":
            # uniform integer in 0..T-1-k (k differs per row, so we cant use randint directly)
            u = torch.rand(batch_size, generator=self.gen, device=dev, dtype=torch.float64)
            t0 = (u * (T - k)).long()
        else:
            t0 = torch.zeros_like(k)
        return self.traj[i, t0], self.traj[i, t0 + k], k


def lead_time_grid(k_max: int, stride: int) -> np.ndarray:
    """Lead times (in saved samples) used for evaluation: every `stride`-th sample up to k_max,
    plus the first few samples (so the very short lead times are always included) and k_max."""
    ks = set(range(1, min(5, k_max) + 1)) | set(range(stride, k_max + 1, stride)) | {k_max}
    return np.array(sorted(ks), dtype=np.int64)


def fixed_triples(traj: torch.Tensor, k_max: int, start_stride: int = 100, tau_stride: int = 10):
    """Deterministic triples for validation: start times every `start_stride` samples (as long
    as the whole lead-time range still fits in the trajectory), and for each start all lead
    times of lead_time_grid(k_max, tau_stride).

    Returns s0, s1 (N,12) and k (N,), ordered trajectory-major, then start, then lead time.
    """
    n, T, _ = traj.shape
    ks = torch.as_tensor(lead_time_grid(k_max, tau_stride), device=traj.device)
    t0s = torch.arange(0, T - k_max, start_stride, device=traj.device)
    t0g, kg = torch.meshgrid(t0s, ks, indexing="ij")
    t0g, kg = t0g.reshape(-1), kg.reshape(-1)
    s0 = traj[:, t0g].reshape(-1, STATE_DIM)
    s1 = traj[:, t0g + kg].reshape(-1, STATE_DIM)
    k = kg.repeat(n)
    return s0, s1, k
