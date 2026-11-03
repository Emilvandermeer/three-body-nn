from __future__ import annotations

import torch
from torch import nn

from data import STATE_DIM
from data_single_shot import SingleShotNormaliser

ACTIVATIONS = {"silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh}


class SingleShotMLP(nn.Module):
    """13 -> [hidden, activation] * depth -> 12.

    Input is [normalised start state (12), tau / T_max (1)], output is the normalised
    prediction z, which the normaliser turns into the change s(t0 + tau) - s0 (see
    SingleShotNormaliser.decode_delta for the "direct" and "residual" output modes).

    Same building blocks as models.SingleStepMLP so the two approaches only differ in what
    they are asked to predict: SiLU by default (smooth, non-zero gradient everywhere),
    "relu" is there so we can also run the Breen et al. 2019 setting (10 x 128, ReLU).
    The output layer is initialised to zero, so at the start of training the network predicts
    "no change" in direct mode and exactly the constant velocity baseline in residual mode.
    """

    def __init__(self, hidden: int = 256, depth: int = 4, activation: str = "silu", dim: int = STATE_DIM):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}, use one of {list(ACTIVATIONS)}")
        layers, d = [], dim + 1  # +1 for the lead time
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), ACTIVATIONS[activation]()]
            d = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


class SingleShotPredictor:
    """Trained net + normaliser: maps a raw start state and a lead time tau directly to the raw
    state tau time units later. Only meant for 0 < tau <= T_max (the range it was trained on)."""

    def __init__(self, net: SingleShotMLP, norm: SingleShotNormaliser, output: str, meta: dict, config: dict):
        self.net = net.eval()
        self.norm = norm
        self.output = output
        self.meta = meta
        self.config = config

    @property
    def t_max(self) -> float:
        return self.norm.t_max

    @property
    def device(self):
        return next(self.net.parameters()).device

    @torch.no_grad()
    def predict(self, s0: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """s0 (B,12) float64, tau (B,) in time units -> s(t0 + tau) (B,12) float64."""
        dtype = next(self.net.parameters()).dtype
        tau = tau.to(s0.dtype)
        z = self.net(self.norm.encode_input(s0, tau).to(dtype)).to(s0.dtype)
        return s0 + self.norm.decode_delta(z, s0, tau, self.output)

    @torch.no_grad()
    def predict_grid(self, s0: torch.Tensor, taus: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
        """All lead times for all start states at once: s0 (B,12), taus (n,) -> (B,n,12).
        Every (start, tau) combination is an independent forward pass, so they are simply
        stacked into one big batch (processed in chunks to keep memory bounded)."""
        B, n = s0.shape[0], taus.shape[0]
        s0_flat = s0[:, None, :].expand(B, n, s0.shape[1]).reshape(B * n, -1)
        tau_flat = taus.to(s0.dtype)[None, :].expand(B, n).reshape(-1)
        out = torch.cat([self.predict(s0_flat[a : a + chunk], tau_flat[a : a + chunk])
                         for a in range(0, B * n, chunk)])
        return out.reshape(B, n, -1)


def save_checkpoint(path, net: SingleShotMLP, norm: SingleShotNormaliser, config: dict, meta: dict, extra=None):
    torch.save(
        {"net": net.state_dict(), "norm": norm.to_dict(), "config": config, "meta": meta, **(extra or {})},
        path,
    )


def load_single_shot(path, device: str = "cpu") -> SingleShotPredictor:
    ck = torch.load(path, map_location=device, weights_only=True)
    cfg = ck["config"]
    net = SingleShotMLP(hidden=cfg["hidden"], depth=cfg["depth"], activation=cfg["activation"]).to(device)
    net.load_state_dict(ck["net"])
    norm = SingleShotNormaliser.from_dict(ck["norm"])
    return SingleShotPredictor(net, norm, cfg["output"], ck["meta"], cfg)
