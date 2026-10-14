"""Diagnostic 2 (ideas #21, last bullet): binary vs graded penalty landscape on an enumerable case
(_gradcheck.tiny_problem, T = 1, H = 2, W = 8; every reachable outcome of the pass enumerated by _gradcheck.Exact).
(a) State space: over all reachable states x and all single-cell changes x -> x', the fraction of moves with
    H(x') == H(x) but D(x') != D(x) (graded slope where the binary landscape is flat), split by whether x is valid.
(b) Parameter space: exact Adam descent (lr 0.01, 300 steps) on J_pen = sum_x q lam.H and J_gpen = sum_x q lam.D from
    the same random inits (cfast tanh actor); P_valid, E[H], E[D] along the way; gradient norms at the inits.
CPU.  Output: runs/diag2_landscape.json and a printed table."""
import json, sys, time
import jax, jax.numpy as jnp, numpy as np
from castlegen import graded as gr
from castlegen.bench.problem import Region
from castlegen.variants import _gradcheck as gc
from castlegen.variants.cfast import CFast
from castlegen.variants.cpass import adam_init, adam_step

LAM = np.array([10.0, 10.0])
p = gc.tiny_problem(); region = Region(p.W_train)
var = CFast(p, 0, "/tmp/graded_diag2", pen_only=True)
ex = gc.Exact(var, LAM, reference=False, ref_check=50)
X = ex.X; n = len(next(iter(X.values())))
names = [c.name for c in p.sampled()]
def HD(st):
    G = gr.graded(p, st, region)
    return np.array([G.viol[l] for l in range(p.T + 1)]), np.array([G.mass[l] for l in range(p.T + 1)])
H = np.zeros((n, 2)); D = np.zeros((n, 2))
for i in range(n):
    H[i], D[i] = HD({k: np.asarray(a[i]) for k, a in X.items()})
assert np.allclose(H, ex.H)
out = {"n_states": n, "n_valid": int((H.sum(1) == 0).sum()), "frac_D_gt_H": float((D.sum(1) > H.sum(1)).mean())}
# (a) single-cell moves
cnt = {"valid": [0, 0, 0], "invalid": [0, 0, 0]}        # [moves, H flat & D moves, H flat & D flat (both change-free)]
hist = {}
for i in range(n):
    x = {k: np.asarray(a[i]) for k, a in X.items()}
    h0, d0 = H[i].sum(), D[i].sum()
    key = "valid" if h0 == 0 else "invalid"
    for c in p.sampled():
        arr = x[c.name]
        for idx in np.ndindex(arr.shape):
            for v in range(c.D):
                if v == arr[idx]: continue
                y = {k: a.copy() for k, a in x.items()}; y[c.name][idx] = v
                h1, d1 = HD(y); h1, d1 = h1.sum(), d1.sum()
                cnt[key][0] += 1
                if h1 == h0 and d1 != d0: cnt[key][1] += 1
                if h1 == h0 and d1 == d0: cnt[key][2] += 1
                hist[(int(np.sign(h1 - h0)), int(np.sign(d1 - d0)))] = hist.get((int(np.sign(h1 - h0)), int(np.sign(d1 - d0))), 0) + 1
out["moves"] = {k: {"n": v[0], "H_flat_D_slope": v[1] / v[0], "both_flat": v[2] / v[0]} for k, v in cnt.items()}
out["sign_hist"] = {f"dH{a} dD{b}": c for (a, b), c in sorted(hist.items())}
print("states", n, "valid", out["n_valid"], "D>H frac", round(out["frac_D_gt_H"], 3))
print("moves", json.dumps(out["moves"], indent=1)); print("sign hist", out["sign_hist"])
# (b) exact descent
Xj = {k: jnp.asarray(a) for k, a in X.items()}
Hl, Dl = jnp.asarray(H @ LAM, jnp.float32), jnp.asarray(D @ LAM, jnp.float32)
valid = jnp.asarray(H.sum(1) == 0)
def J(params, w):
    lq = ex._lq(params, Xj, 1.0); q = jnp.exp(lq)
    return jnp.sum(q * w), q
@jax.jit
def step(params, opt, w, lr):
    (val, q), g = jax.value_and_grad(lambda pr: J(pr, w), has_aux=True)(params)
    gn = jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(g)))
    params, opt = adam_step(params, g, opt, lr)
    return params, opt, val, gn, jnp.sum(q * valid), jnp.sum(q * jnp.asarray(H.sum(1), jnp.float32)), jnp.sum(q * jnp.asarray(D.sum(1), jnp.float32))
runs = []
for seed in range(4):
    p0 = gc.randomize(var.params, jax.random.PRNGKey(100 + seed), scale=1.0)
    rec = {"seed": seed}
    for name, w in (("binary", Hl), ("graded", Dl)):
        params, opt = p0, adam_init(p0); tr = []
        for t in range(300):
            params, opt, val, gn, pv, eh, ed = step(params, opt, w, 0.01)
            if t in (0, 9, 29, 99, 199, 299):
                tr.append({"t": t + 1, "J": float(val), "gnorm": float(gn), "P_valid": float(pv), "EH": float(eh), "ED": float(ed)})
        rec[name] = tr
    runs.append(rec)
    print(f"seed {seed}: binary P_valid {[round(r['P_valid'], 3) for r in rec['binary']]} EH {[round(r['EH'], 3) for r in rec['binary']]}")
    print(f"         graded P_valid {[round(r['P_valid'], 3) for r in rec['graded']]} EH {[round(r['EH'], 3) for r in rec['graded']]} gnorm0 b/g {rec['binary'][0]['gnorm']:.3f}/{rec['graded'][0]['gnorm']:.3f}", flush=True)
out["descent"] = runs
json.dump(out, open("reportables/2026-10-14_graded/runs/diag2_landscape.json", "w"), indent=1)
print("DIAG2_DONE")
