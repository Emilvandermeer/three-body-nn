"""Augmentations of training data pairs:
 (s0, s1) are the states at times t and t + k, respectively, for some trajectory.
 Augments allow us to generate more training data from the same trajectories

For equal masses the planar three-body equations are invariant under:

  * rotation of the plane, # this one probably doesnt affect the results much but implemented anyways
  * relabelling of the bodies, # this augment ensures that the neural network does not learn to associate a specific body with a specific index in the 
                                state vector (since its this long 12-vector), good for generalisation

    # the straightforward augments, just changing the sign
  * reflection (y -> -y), 
  * time reversal (v -> -v together with swapping input and target).
So a transformed pair is again a valid pair from the true dynamics.

Everything works on raw (un-normalised) float64 states of shape (B, 12), BEFORE
the normaliser is applied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .data import N_POS


def _split(s: torch.Tensor):
    b = s.shape[0]
    return s[:, :N_POS].reshape(b, 3, 2), s[:, N_POS:].reshape(b, 3, 2)


def _join(pos: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
    b = pos.shape[0]
    return torch.cat([pos.reshape(b, N_POS), vel.reshape(b, N_POS)], dim=1)


def apply_rotation(s: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Rotate every position and velocity vector by theta (shape (B,))."""
    pos, vel = _split(s)
    c, sn = torch.cos(theta)[:, None], torch.sin(theta)[:, None]

    def rot(v):
        return torch.stack(
            [c * v[..., 0] - sn * v[..., 1], sn * v[..., 0] + c * v[..., 1]], dim=-1
        )

    return _join(rot(pos), rot(vel))


def apply_reflection(s: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Flip y -> -y for the rows where mask (shape (B,), bool) is True."""
    pos, vel = _split(s)
    sign = torch.where(mask, -1.0, 1.0).to(s.dtype)[:, None]

    def flip(v):
        return torch.stack([v[..., 0], sign * v[..., 1]], dim=-1)

    return _join(flip(pos), flip(vel))


def apply_permutation(s: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """New body j is old body perm[b, j]. perm has shape (B, 3). Only valid
    because all three masses are equal."""
    pos, vel = _split(s)
    idx = perm[:, :, None].expand(-1, -1, 2)
    return _join(torch.gather(pos, 1, idx), torch.gather(vel, 1, idx))


def apply_time_reversal(s: torch.Tensor) -> torch.Tensor:
    """Flip all velocities."""
    pos, vel = _split(s)
    return _join(pos, -vel)


@dataclass
class Augmenter:
    rotate: bool = True
    reflect: bool = True
    permute: bool = True
    time_reverse: bool = True

    def __call__(self, s0: torch.Tensor, s1: torch.Tensor, gen: torch.Generator):
        b, dev, dt = s0.shape[0], s0.device, s0.dtype

        if self.time_reverse:
            m = (torch.rand(b, generator=gen, device=dev) < 0.5)[:, None]
            r0, r1 = apply_time_reversal(s1), apply_time_reversal(s0)
            s0, s1 = torch.where(m, r0, s0), torch.where(m, r1, s1)

        if self.permute:
            perm = torch.argsort(torch.rand(b, 3, generator=gen, device=dev), dim=1)
            s0, s1 = apply_permutation(s0, perm), apply_permutation(s1, perm)

        if self.reflect:
            m = torch.rand(b, generator=gen, device=dev) < 0.5
            s0, s1 = apply_reflection(s0, m), apply_reflection(s1, m)

        if self.rotate:
            th = torch.rand(b, generator=gen, device=dev, dtype=dt) * (2 * math.pi)
            s0, s1 = apply_rotation(s0, th), apply_rotation(s1, th)

        return s0, s1
