"""smoke + support probe: cstage with the saved hard_pen params (no training)."""
import pickle, sys, time
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.variant import load_variants
from castlegen.bench.problems import get
from castlegen.bench.problem import Region
from castlegen.bench.run import EVAL_SEED0, _with_designed
from castlegen.bench.energy import energy
V = load_variants(); p = get("ground")
v = V["ckpop_hard_pen_stage"](p, 0, "/tmp/stage_smoke")
tr = pickle.load(open("reportables/2026-10-14_cr2_penalty/runs/arm_ckpop_hard_pen_s0/params.pkl", "rb")); tr = tr.get("trained", tr)
v.params = jax.tree_util.tree_map(jnp.asarray, tr["params"]); v.lam = np.asarray(tr["lam"], float)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
reg = Region(128)
for i in range(n):
    t0 = time.time(); g = v.generate(EVAL_SEED0 + i, reg); dt = time.time() - t0
    vr = energy(p, _with_designed(p, g["raw"], EVAL_SEED0 + i, reg), reg).viol
    vf = energy(p, _with_designed(p, g["final"], EVAL_SEED0 + i, reg), reg).viol
    print(i, f"{dt:.2f}s raw {[vr[l] for l in range(5)]} final {[vf[l] for l in range(5)]}", flush=True)
W, T = 128, p.T; half = (W // 2) >> T << T; seed = EVAL_SEED0 + 999
t0 = time.time()
ga = v.generate(seed, Region(W, periodic=False, x0=0)); gb = v.generate(seed, Region(W, periodic=False, x0=half))
print(f"open windows {time.time()-t0:.1f}s", flush=True)
for kind in ("raw", "final"):
    mm = {}
    for c in p.sampled():
        s = half >> c.level
        a = np.asarray(ga[kind][c.name])[:, s:]; b = np.asarray(gb[kind][c.name])[:, :a.shape[1]]
        mm[c.name] = float((a != b).mean())
    print("support mismatch", kind, mm, flush=True)
print("SMOKE_DONE")
