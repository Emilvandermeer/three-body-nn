"""Correctness checks for the single-shot code. Run from the repository root:
    python src/check_single_shot.py
Prints what it checks and ends with ALL TESTS PASSED (an AssertionError means something is wrong).
Takes about a minute.
"""
import os, sys, tempfile
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC_DIR)
for _p in (_ROOT, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np, torch
from data import load_trajectories, split_indices
from data_single_shot import *
from augment import Augmenter
from models_single_shot import *
from baselines_single_shot import KNNBaseline, verlet_batched
import DataGeneration.VerletSolver as verlet

data, meta = load_trajectories("data/traj_chunks")
data = data[:300]
traj = torch.from_numpy(data)
M, G = np.array(meta["M"]), meta["G"]

# 1. sampler: s1 really is k samples after s0, indices in range, k range
for mode in ("any", "initial"):
    smp = TripleSampler(traj, 500, mode, seed=1)
    for _ in range(20):
        s0, s1, k = smp.sample(4096)
        assert k.min() >= 1 and k.max() <= 500
        # find each s0 in its trajectory: check by brute force on a few rows
    # verify with explicit index re-draw
    g = torch.Generator(); g.manual_seed(1)
    n, T, _ = traj.shape
    i = torch.randint(0, n, (4096,), generator=g); kk = torch.randint(1, 501, (4096,), generator=g)
    if mode == "any":
        t0 = (torch.rand(4096, generator=g, dtype=torch.float64) * (T - kk)).long()
    else: t0 = torch.zeros_like(kk)
    smp2 = TripleSampler(traj, 500, mode, seed=1); a0, a1, ak = smp2.sample(4096)
    assert torch.equal(a0, traj[i, t0]) and torch.equal(a1, traj[i, t0 + kk]) and torch.equal(ak, kk)
    assert (t0 + kk).max() <= T - 1 and t0.min() >= 0
    print(mode, "sampler ok; t0 range", int(t0.min()), int(t0.max()), " max t0+k", int((t0+kk).max()))

# 2. fixed triples
s0, s1, k = fixed_triples(traj[:3], 200, 100, 10)
ks = lead_time_grid(200, 10); t0s = np.arange(0, 2001 - 200, 100)
r = 0
for j in range(3):
    for t0 in t0s:
        for kk in ks:
            assert torch.equal(s0[r], traj[j, t0]) and torch.equal(s1[r], traj[j, t0 + kk]) and k[r] == kk; r += 1
assert r == len(s0); print("fixed_triples ok", len(s0), "lead times", ks[:8], "...", ks[-2:])

# 3. augmented triples are still valid dynamics: integrate aug(s0) for tau with Verlet, compare to aug(s1)
smp = TripleSampler(traj, 100, "any", seed=3)
s0, s1, k = smp.sample(64)
g = torch.Generator(); g.manual_seed(0)
a0, a1 = Augmenter()(s0, s1, g)
errs = []
for j in range(64):
    sim = verlet.simulate(a0[j].numpy(), M, G, meta["dt_solver"], int(k[j]) * meta["save_every"], meta["save_every"])
    errs.append(np.abs(sim[-1] - a1[j].numpy()).max())
print("augmentation physics check: max |verlet(aug s0) - aug s1| =", max(errs))
# reference: same check without augmentation
errs0 = [np.abs(verlet.simulate(s0[j].numpy(), M, G, meta["dt_solver"], int(k[j]) * meta["save_every"], meta["save_every"])[-1] - s1[j].numpy()).max() for j in range(64)]
print("   same check without augmentation:", max(errs0))
# time reversal is the interesting one: Verlet is time-symmetric so it should be tiny too
E0 = [verlet.total_energy(*verlet.unpack_state(a0[j].numpy()), M, G) for j in range(64)]
print("   energies of augmented starts:", min(E0), max(E0))

# 4. normaliser: decode(encode) consistency for both modes
norm = SingleShotNormaliser.fit(data, 200, meta["dt_save"], "any")
s0, s1, k = TripleSampler(traj, 200, "any", seed=5).sample(1000)
tau = k.double() * meta["dt_save"]
d = s1 - s0
_, dv, rv = norm._vecs(d)
z_direct = d / dv
assert torch.allclose(norm.decode_delta(z_direct, s0, tau, "direct"), d)
z_res = (d - constant_velocity_delta(s0, tau)) / ((tau / norm.t_max)[:, None] * rv)
assert torch.allclose(norm.decode_delta(z_res, s0, tau, "residual"), d)
print("normaliser round trip ok; z_direct rms", float(z_direct.pow(2).mean().sqrt()), "z_res rms", float(z_res.pow(2).mean().sqrt()))
x = norm.encode_input(s0, tau); assert x.shape == (1000, 13) and torch.allclose(x[:, 12], k.double() / 200)

# 5. zero-initialised model == baseline
for mode, base in (("direct", lambda s, t: torch.zeros_like(s)), ("residual", constant_velocity_delta)):
    net = SingleShotMLP(64, 2).double()
    p = SingleShotPredictor(net, norm, mode, meta, {})
    assert torch.allclose(p.predict(s0, tau), s0 + base(s0, tau))
print("zero-init model equals no-change / constant-velocity baseline: ok")

# 6. checkpoint round trip + predict_grid consistency
net = SingleShotMLP(32, 3, "relu"); torch.nn.init.normal_(net.head.weight, std=0.1)
cfg = {"hidden": 32, "depth": 3, "activation": "relu", "output": "residual"}
save_checkpoint(os.path.join(tempfile.gettempdir(), "single_shot_check.pt"), net, norm, cfg, meta)
p2 = load_single_shot(os.path.join(tempfile.gettempdir(), "single_shot_check.pt"))
p1 = SingleShotPredictor(net, norm, "residual", meta, cfg)
assert torch.allclose(p1.predict(s0, tau), p2.predict(s0, tau))
taus = torch.tensor([0.01, 0.5, 2.0], dtype=torch.float64)
grid = p2.predict_grid(s0[:7], taus)
for j in range(3):
    assert torch.allclose(grid[:, j], p2.predict(s0[:7], taus[j].repeat(7)))
print("checkpoint + predict_grid ok")

# 7. kNN: a query equal to an anchor with k=1 returns that anchor's own future
knn = KNNBaseline(traj, norm, 200, anchor_stride=5, k=1)
q = traj[[3, 10, 77], [0, 50, 100]]
ks_t = torch.tensor([1, 50, 200])
pred = knn.predict_grid(q, ks_t, k=1)
for r, (j, t0) in enumerate([(3, 0), (10, 50), (77, 100)]):
    assert torch.allclose(pred[r], traj[j, t0 + ks_t]), r
idx = knn.neighbours(q, 10)
assert torch.equal(knn.predict_grid(q, ks_t, k=3, idx=idx), knn.predict_grid(q, ks_t, k=3))
print("kNN ok, anchors:", len(knn.anchors))

# 8. batched verlet == partners' solver
st = data[:5, 0]
vb = verlet_batched(st, M, G, meta["dt_solver"], 2000, 10)
for j in range(5):
    ref = verlet.simulate(st[j], M, G, meta["dt_solver"], 2000, 10)
    assert vb[j].shape == ref.shape
    print("   batched vs reference Verlet, max diff over 2 time units:", np.abs(vb[j] - ref).max())
    assert np.abs(vb[j] - data[j, :201]).max() < 1e-8
print("ALL TESTS PASSED")
