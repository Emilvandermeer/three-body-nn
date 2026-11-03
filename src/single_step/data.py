"""
Loading, splitting, normalising and sampling the three-body trajectories.

State layout:
    [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3]
the first 6 entries are positions and the last 6 are velocities.
"""

from __future__ import annotations

import glob
import re
from dataclasses import dataclass, field

import numpy as np
import torch

#global constants so we dont have to hardcode them in multiple different places
STATE_DIM = 12
N_POS = 6


def _chunk_id(path: str) -> int:
    """Extract the chunk number from a chunk_cNNN.npz path."""
    return int(re.search(r"chunk_c(\d+)\.npz$", path).group(1))


def load_trajectories(data_dir: str = "data/traj_chunks"):
    """Load all chunk files.

    Returns tuple (data, meta) where
    data : float64 array of shape (n_traj, n_samples, 12)
    meta : dict with M - masses (list), G - gravitational constant, E - total energy, dt_solver - solver time step, 
    save_every - save frequency, dt_save - time between two consecutive saved samples
    """
    files = sorted(glob.glob(f"{data_dir}/chunk_c*.npz"), key=_chunk_id)
    if not files:
        raise FileNotFoundError(f"no chunk_c*.npz files found in {data_dir!r}")

    trajs, meta = [], None
    for f in files:
        data = np.load(f)
        trajs.append(data["trajectories"])
        if meta is None:
            meta = {
                "M": [float(m) for m in data["M"]],
                "G": float(data["G"]),
                "E": float(data["E"]),
                "dt_solver": float(data["dt"]),
                "save_every": int(data["save_every"]),
            }
    meta["dt_save"] = meta["dt_solver"] * meta["save_every"]
    return np.concatenate(trajs), meta


def split_indices(n: int, seed: int = 42, fractions=(0.8, 0.1, 0.1)) -> dict:
    """Random split BY TRAJECTORY (never by state pair: neighbouring states are
    almost identical, so a pair-level split would leak validation data)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    return {
        "train": np.sort(perm[:n_train]),
        "val": np.sort(perm[n_train : n_train + n_val]),
        "test": np.sort(perm[n_train + n_val :]),
    }



@dataclass
class Normaliser:
    """Scalar rescaling of states and of one-step changes.

    * One scale for all positions and one for all velocities
    * The network predicts delta = s(t + k*dt) - s(t); positions and velocities
      of delta get their own scales because they differ by orders of magnitude.
    """

    pos_scale: float
    vel_scale: float
    dpos_scale: float
    dvel_scale: float
    k: int

    #the cache is used to store the scaling vectors for a given device and dtype, so that we don't have to recompute them every time we normalise or denormalise a
    #state or delta, its not super neccesary but it saves a bit of time and memory when training the neural network and
    # makes the code cleaner, other way the code looked much messier

    _cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    @classmethod
    def fit(cls, train_traj: np.ndarray, k: int):
        n, T, _ = train_traj.shape
        if T <= k:
            raise ValueError(f"trajectories have {T} samples, need more than k={k}")
        sums, counts = np.zeros(4), np.zeros(4)
        for a in range(0, n, 100):  # blocks keep the temporary arrays small
            blk = train_traj[a : a + 100].astype(np.float64)
            pos, vel = blk[..., :N_POS], blk[..., N_POS:]
            dpos = pos[:, k:] - pos[:, :-k]
            dvel = vel[:, k:] - vel[:, :-k]
            for j, arr in enumerate((pos, vel, dpos, dvel)):
                sums[j] += np.sum(arr**2)
                counts[j] += arr.size
        rms = np.sqrt(sums / counts)
        return cls(float(rms[0]), float(rms[1]), float(rms[2]), float(rms[3]), k)

    # this function caches the scaling vectors for a given device and dtype, 
    #so that we don't have to recompute them every time we normalise or denormalise a state or delta.
    def _vecs(self, ref: torch.Tensor):
        key = (ref.device, ref.dtype)
        if key not in self._cache:
            s = ref.new_tensor([self.pos_scale] * N_POS + [self.vel_scale] * N_POS)
            d = ref.new_tensor([self.dpos_scale] * N_POS + [self.dvel_scale] * N_POS)
            self._cache[key] = (s, d)
        return self._cache[key]

    #The encode/decode methods are used to normalise and denormalise the states and deltas. 
    # we need to normalise the states and deltas before feeding them into the neural network, 
    # and denormalise the outputs of the network to get back to the original scale.

    def encode_state(self, s: torch.Tensor) -> torch.Tensor:
        return s / self._vecs(s)[0]

    def decode_state(self, z: torch.Tensor) -> torch.Tensor:
        return z * self._vecs(z)[0]

    def encode_delta(self, d: torch.Tensor) -> torch.Tensor:
        return d / self._vecs(d)[1]

    def decode_delta(self, z: torch.Tensor) -> torch.Tensor:
        return z * self._vecs(z)[1]

    def to_dict(self) -> dict:
        return {
            "pos_scale": self.pos_scale,
            "vel_scale": self.vel_scale,
            "dpos_scale": self.dpos_scale,
            "dvel_scale": self.dvel_scale,
            "k": self.k,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Normaliser":
        return cls(**d)


class PairSampler:
    """Draws random (state at t, state at t + k) pairs from a trajectory tensor.

    The whole trajectory tensor lives on the compute device (float64, ~300 MB for
    the training split), and pairs are produced on the fly by indexing.
    """

    def __init__(self, traj: torch.Tensor, k: int, seed: int = 0):
        self.traj = traj
        self.k = k
        self.gen = torch.Generator(device=traj.device)
        self.gen.manual_seed(seed)

    def sample(self, batch_size: int):
        n, T, _ = self.traj.shape
        dev = self.traj.device
        i = torch.randint(0, n, (batch_size,), generator=self.gen, device=dev)
        t = torch.randint(0, T - self.k, (batch_size,), generator=self.gen, device=dev)
        return self.traj[i, t], self.traj[i, t + self.k]


def fixed_pairs(traj: torch.Tensor, k: int, stride: int = 10):
    """Deterministic pairs (every `stride`-th start time of every trajectory),
    used for validation."""
    T = traj.shape[1]
    t = torch.arange(0, T - k, stride, device=traj.device)
    s0 = traj[:, t].reshape(-1, STATE_DIM)
    s1 = traj[:, t + k].reshape(-1, STATE_DIM)
    return s0, s1
