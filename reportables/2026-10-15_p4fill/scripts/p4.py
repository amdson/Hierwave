"""P4: fill from tight abstraction vs random.  usage: p4.py W NF [SNAP].
Arms: rand_grad (random certs, graded lam20), tight_grad (lift certs, graded lam20), tight_hard (lift certs, hard sweeps).
Record at k in KS: valid fraction, mean A_tight - A per level (cert_stats col 3), slack B-A (col 2)."""
import json, sys, time
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "reportables/2026-10-14_cgraded"); sys.path.insert(0, "reportables/2026-10-14_cmix")
import mix
Wn, NF = int(sys.argv[1]), int(sys.argv[2]); SNAP = int(sys.argv[3]) if len(sys.argv) > 3 else 300
mix.W = Wn
from mix import starts, stripped
from run import graded_problem
from certs import cert_stats
from castlegen.bench.problems import ground as G
from castlegen.kernels import colour_gibbs as cg
from castlegen.variants._cpass_core import Layout
KS = (0, 5, 10, 20, 40, 80)
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
tn = np.asarray(st["t"])
lifted = [G.lift(tn[b]) for b in range(B)]
init_r, init_t = dict(st), dict(st)
for c in certs:
    init_r[c.name] = jnp.asarray(rng.integers(0, c.D, size=(B,) + st0[c.name].shape[1:]).astype(np.int32))
    init_t[c.name] = jnp.asarray(np.stack([l[c.name] for l in lifted]).astype(np.int32))
out = {}
def rec(full):
    _, N = cg.state_energy(lay, full); ok = np.asarray(N) == 0
    fn = {k: np.asarray(v) for k, v in full.items()}
    cs = cert_stats(p, fn)   # over all worlds
    r = {"valid": float(ok.mean()), "tight_gap": [cs[l][3] for l in range(1, p.T + 1)], "slack": [cs[l][2] for l in range(1, p.T + 1)]}
    if ok.any() and not ok.all():
        cv = cert_stats(p, {k: v[ok] for k, v in fn.items()}); r["tight_gap_valid"] = [cv[l][3] for l in range(1, p.T + 1)]
    return r
for arm, init, L, lam in (("rand_grad", init_r, layg, 0.0), ("tight_grad", init_t, layg, 0.0), ("tight_hard", init_t, lay, None)):
    full = dict(init); rows = {}
    rows[0] = rec(full)
    for j in range(1, max(KS) + 1):
        key, sub = jax.random.split(key)
        full = cg.sweep_all(L, full, 1.0, lam, sub, levels=clev, mask=cmask)
        if j in KS: rows[j] = rec(full)
    out[arm] = rows; print(arm, json.dumps(rows), flush=True)
json.dump(out, open(f"reportables/2026-10-15_p4fill/runs/p4_{Wn}_n{NF}.json", "w"), indent=1)
