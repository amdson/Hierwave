"""Numbers behind tests/kernels/test_colour_gibbs.py (z-scores, AIS estimates) plus negative controls."""
import json, sys
sys.path.insert(0, "tests/kernels")
import numpy as np, jax
import test_colour_gibbs as tc
from test_colour_gibbs import *
_lay, _random_state, _zscores = tc._lay, tc._random_state, tc._zscores
out = {}
lam = 2.0
for name in ["toy_modes", "toy_support"]:
    p, lay = _lay(name); ex = toy_exact(name, lam)
    for level in (0, 1, 2):
        st = exact_samples(ex, np.random.default_rng(level), 256); key = jax.random.PRNGKey(level)
        for i in range(6):
            key, sub = jax.random.split(key); st = cg.sweep_level(lay, st, level, 1.0, lam, sub)
        out[f"invariance/{name}/L{level}"] = float(np.abs(_zscores(ex, st)).max())
    # negative control: sweep at the wrong temperature (tau = 1.5) from exact samples
    st = exact_samples(ex, np.random.default_rng(9), 256); key = jax.random.PRNGKey(9)
    for i in range(6):
        key, sub = jax.random.split(key); st = cg.sweep_all(lay, st, 1.5, lam, sub)
    out[f"control_wrong_tau/{name}"] = float(np.abs(_zscores(ex, st)).max())
    st = _random_state(p, lay, 256, 3); key = jax.random.PRNGKey(0); traj = []
    for i in range(200):
        key, sub = jax.random.split(key); st = cg.sweep_all(lay, st, 1.0, lam, sub)
        if i >= 100 and i % 4 == 0: traj.append(block_stats(st, ex).mean(1))
    s = np.mean(traj, 0); est, se = s.mean(0), s.std(0, ddof=1) / 16
    z = (est - np.array([ex["exact"][k] for k in ex["keys"]])) / np.maximum(se, 2e-3)
    out[f"mixing/{name}"] = dict(max_abs_z=float(np.abs(z).max()), z=dict(zip(ex["keys"], z.round(2).tolist())))
    # AIS
    tau_hi, B = 4.0, 4000
    ex0 = toy_exact(name, lam, 1 / tau_hi); ex1 = toy_exact(name, lam)
    true = ex1["logZ"] - ex0["logZ"]
    for n_steps in (8, 64):
        rng = np.random.default_rng(0)
        st = exact_samples(ex0, rng, B); mask = cg.subtree_mask(lay, 2, np.zeros((B, 2), np.int32))
        _, logw, info = cg.anneal_block(lay, st, mask, jax.random.PRNGKey(0), tau_hi, n_steps, lam)
        lw = np.asarray(logw, np.float64); m = lw.max()
        lme = lambda a: np.log(np.mean(np.exp(a - m))) + m
        se = np.std([lme(lw[i]) for i in rng.integers(0, B, (200, B))])
        out[f"ais/{name}/steps{n_steps}"] = dict(est=float(lme(lw)), se=float(se), exact=true,
                                                 mean_logw=float(lw.mean()), ess_frac=float(np.exp(2 * lme(lw) - lme(2 * lw)) if False else (np.exp(lw - m).sum() ** 2 / (np.exp(lw - m) ** 2).sum()) / B))
json.dump(out, open("reportables/2026-10-14_speed2/runs/test_numbers.json", "w"), indent=1)
print(json.dumps(out, indent=1))
