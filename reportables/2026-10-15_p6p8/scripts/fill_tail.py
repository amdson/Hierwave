"""P8: fill_len.py at larger W. usage: fill_tail.py W NF [SNAP]. Arms: bin8, bin20, grad."""
import json, sys, time
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "reportables/2026-10-14_cgraded"); sys.path.insert(0, "reportables/2026-10-14_cmix")
import mix
Wn, NF = int(sys.argv[1]), int(sys.argv[2]); SNAP = int(sys.argv[3]) if len(sys.argv) > 3 else 300
mix.W = Wn
from mix import starts, stripped
from run import graded_problem
from castlegen.kernels import colour_gibbs as cg
from castlegen.variants._cpass_core import Layout
MAXF = 400
p, st0, fam, fseed = starts(NF); B = fam.size
ps = stripped(p)
lay, lays, layg = Layout(p, Wn, True), Layout(ps, Wn, True), Layout(graded_problem(p, 20.0), Wn, True)
certs = [c for c in p.sampled() if c.name.startswith("cert")]
cmask = {c.name: jnp.full((B,) + st0[c.name].shape[1:], c.name.startswith("cert"), bool) for c in p.sampled() if c.level >= 1}
clev = list(range(p.T, 0, -1))
rng = np.random.default_rng(0); key = jax.random.PRNGKey(0)
st = {k: v for k, v in st0.items() if k in ps.chan}
t0 = time.time()
for s in range(SNAP):
    key, sub = jax.random.split(key); st = cg.sweep_all(lays, st, 1.0, None, sub)
print("terrain done", time.time() - t0, flush=True)
init = dict(st)
for c in certs:
    init[c.name] = jnp.asarray(rng.integers(0, c.D, size=(B,) + st0[c.name].shape[1:]).astype(np.int32))
out = []
for pen, L, lam in (("grad", layg, 0.0), ("bin20", lay, 20.0), ("bin8", lay, 8.0)):
    full = dict(init); first = np.full(B, -1)
    for j in range(1, MAXF + 1):
        key, sub = jax.random.split(key)
        full = cg.sweep_all(L, full, 1.0, lam, sub, levels=clev, mask=cmask)
        _, N = cg.state_energy(lay, full)
        first = np.where((first < 0) & (np.asarray(N) == 0), j, first)
        if (first >= 0).all(): break
    ok = first >= 0
    surv = {k: float((~ok | (first > k)).mean()) for k in (20, 40, 60, 80, 120)}
    r = {"W": Wn, "pen": pen, "B": B, "n_valid": int(ok.sum()), "p50": float(np.median(first[ok])) if ok.any() else None,
         "max": int(first.max()) if ok.all() else ">%d" % MAXF, "surv": surv, "first": first.tolist(), "t": time.time() - t0}
    out.append(r); print(json.dumps(r), flush=True)
json.dump(out, open(f"reportables/2026-10-15_p6p8/runs/fill_tail_{Wn}_n{NF}.json", "w"))
