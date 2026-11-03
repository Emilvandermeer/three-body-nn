"""Reference methods the single-shot network is compared against.

  * KNNBaseline   : k-nearest-neighbour regression, the "very elementary baseline method"
                    the course instructions ask for. No training, just a lookup.
  * verlet_batched: the same velocity Verlet scheme as DataGeneration/VerletSolver.py, but
                    vectorised over many trajectories at once with numpy. Gives exactly the
                    same numbers as VerletSolver.simulate (checked in the tests), and is used
                    for a FAIR cost comparison: VerletSolver.simulate is a pure Python loop
                    over time steps, so timing it mostly measures Python overhead, while the
                    network gets to process a whole batch in one call.

The trivial baselines (no change, constant velocity, mean state) are one-liners and live in
train_single_shot.py / evaluate_single_shot.py.
"""

from __future__ import annotations

import numpy as np
import torch

from data import N_POS
from data_single_shot import SingleShotNormaliser


class KNNBaseline:
    """k-nearest-neighbour regression on the start state.

    Anchors are training states s(t0) (start times every `anchor_stride` samples, only those
    whose whole future up to T_max is inside the trajectory). For a query (s0, tau) we find the
    k anchors closest to s0 (Euclidean distance between NORMALISED states, so positions and
    velocities count equally) and predict

        s(t0 + tau) ~ s0 + mean over the k anchors of [ s_a(t_a + tau) - s_a(t_a) ],

    i.e. we copy the average change that the neighbouring training states went through.
    Note that the lookup is not rotation/relabelling invariant, unlike the augmented network.
    """

    def __init__(self, train_traj: torch.Tensor, norm: SingleShotNormaliser, k_max: int,
                 anchor_stride: int = 5, k: int = 10):
        n, T, _ = train_traj.shape
        self.traj = train_traj
        self.k = k
        t0s = torch.arange(0, T - k_max, anchor_stride, device=train_traj.device)
        ti, t0 = torch.meshgrid(torch.arange(n, device=train_traj.device), t0s, indexing="ij")
        self.anchor_traj, self.anchor_t0 = ti.reshape(-1), t0.reshape(-1)
        scale = train_traj.new_tensor([norm.pos_scale] * N_POS + [norm.vel_scale] * N_POS)
        self.scale = scale
        self.anchors = (train_traj[self.anchor_traj, self.anchor_t0] / scale).float()

    @torch.no_grad()
    def neighbours(self, s0: torch.Tensor, k: int | None = None, chunk: int = 32) -> torch.Tensor:
        """Indices (into the anchor list) of the k nearest anchors of every row of s0. (B,12) -> (B,k)."""
        k = k or self.k
        q = (s0 / self.scale).float()
        out = []
        for a in range(0, len(q), chunk):
            dist = torch.cdist(q[a : a + chunk], self.anchors)
            out.append(torch.topk(dist, k, dim=1, largest=False).indices)
        return torch.cat(out)

    @torch.no_grad()
    def predict_grid(self, s0: torch.Tensor, ks: torch.Tensor, k: int | None = None,
                     idx: torch.Tensor | None = None, chunk: int = 64) -> torch.Tensor:
        """s0 (B,12), lead times ks (n,) in saved samples -> predicted states (B,n,12).
        `idx` can pass precomputed neighbours (from self.neighbours with a larger k) so we
        can try several k without redoing the expensive distance computation."""
        k = k or self.k
        if idx is None:
            idx = self.neighbours(s0, k)
        idx = idx[:, :k]  # topk returns sorted indices, so the first k are the k nearest
        out = []
        for a in range(0, len(s0), chunk):
            ti, t0 = self.anchor_traj[idx[a : a + chunk]], self.anchor_t0[idx[a : a + chunk]]  # (b,k)
            fut = self.traj[ti[..., None], t0[..., None] + ks[None, None, :]]                   # (b,k,n,12)
            start = self.traj[ti, t0]                                                           # (b,k,12)
            delta = (fut - start[:, :, None, :]).mean(dim=1)                                    # (b,n,12)
            out.append(s0[a : a + chunk, None, :] + delta)
        return torch.cat(out)


def _acceleration(pos: np.ndarray, M: np.ndarray, G: float) -> np.ndarray:
    """pos (B,3,2) -> acceleration (B,3,2): a_i = -G sum_j m_j (x_i - x_j) / |x_i - x_j|^3."""
    diff = pos[:, :, None, :] - pos[:, None, :, :]          # (B,3,3,2), diff[b,i,j] = x_i - x_j
    r3 = np.sum(diff**2, axis=-1) ** 1.5                     # (B,3,3)
    r3[:, np.arange(3), np.arange(3)] = 1.0                  # self-interaction: diff is 0 there anyway
    return -G * np.sum(M[None, None, :, None] * diff / r3[..., None], axis=2)


def verlet_batched(states: np.ndarray, M, G: float, dt: float, n_steps: int, save_every: int = 1) -> np.ndarray:
    """Velocity Verlet for B trajectories at once (same update rule as VerletSolver.simulate).

    states (B,12) -> (B, n_steps // save_every + 1, 12), state layout as in VerletSolver.
    """
    M = np.asarray(M, float)
    B = states.shape[0]
    pos = states[:, :N_POS].reshape(B, 3, 2).copy()
    vel = states[:, N_POS:].reshape(B, 3, 2).copy()
    out = [np.concatenate([pos.reshape(B, -1), vel.reshape(B, -1)], axis=1)]
    acc = _acceleration(pos, M, G)
    for i in range(1, n_steps + 1):
        pos = pos + dt * vel + 0.5 * acc * dt**2
        acc_next = _acceleration(pos, M, G)
        vel = vel + 0.5 * (acc + acc_next) * dt
        acc = acc_next
        if i % save_every == 0:
            out.append(np.concatenate([pos.reshape(B, -1), vel.reshape(B, -1)], axis=1))
    return np.stack(out, axis=1)
