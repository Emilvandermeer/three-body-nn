from __future__ import annotations

import torch
from torch import nn

from data import STATE_DIM, Normaliser


class SingleStepMLP(nn.Module):
    """12 -> [hidden, SiLU] * depth -> 12.
    SiLU used because it is smooth and has non-zero gradient everywhere, unlike ReLU.

    Works in normalised units: input is the normalised state, output is the
    normalised change delta = s(t+k) - s(t).(This gets adjusted through the normaliser encoder/decoders)
    The output layer is initialised to zero so that the network starts out predicting no change
    """

    def __init__(self, hidden: int = 256, depth: int = 4, dim: int = STATE_DIM):
        super().__init__()
        layers, d = [], dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


class SingleStepPredictor:
    """Trained net + normaliser: maps a raw state to the raw state one network step later (k * dt_save time units)"""

    def __init__(self, net: SingleStepMLP, norm: Normaliser, k: int, meta: dict, config: dict):
        self.net = net.eval()
        self.norm = norm
        self.k = k
        self.meta = meta
        self.config = config

    @property
    def dt_step(self) -> float:
        return self.k * self.meta["dt_save"]

    @property
    def device(self):
        return next(self.net.parameters()).device

    @torch.no_grad()
    def predict_next(self, s: torch.Tensor) -> torch.Tensor:
        dtype = next(self.net.parameters()).dtype
        x = self.norm.encode_state(s).to(dtype)
        delta = self.norm.decode_delta(self.net(x).to(s.dtype))
        return s + delta


def save_checkpoint(path, net: SingleStepMLP, norm: Normaliser, config: dict, meta: dict, extra=None):
    torch.save(
        {"net": net.state_dict(), "norm": norm.to_dict(), "config": config, "meta": meta, **(extra or {})},
        path,
    )


def load_predictor(path, device: str = "cpu") -> SingleStepPredictor:
    ck = torch.load(path, map_location=device, weights_only=True)
    cfg = ck["config"]
    net = SingleStepMLP(hidden=cfg["hidden"], depth=cfg["depth"]).to(device)
    net.load_state_dict(ck["net"])
    norm = Normaliser.from_dict(ck["norm"])
    return SingleStepPredictor(net, norm, norm.k, ck["meta"], cfg)
