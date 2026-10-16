"""Smoke: cstage with keyed noise -- periodic batched generation + the bench's finite-support check.
usage: smoke.py R F [round_reach]"""
import sys, time, pickle
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.variant import load_variants
from castlegen.bench.problems import get
from castlegen.bench.problem import Region
from castlegen.bench.run import EVAL_SEED0
from castlegen.bench.energy import energy
from castlegen.bench.run import _with_designed
R, F = int(sys.argv[1]), int(sys.argv[2])
p = get("ground"); V = load_variants()
v = V["ckpop_hard_pen_stage"](p, 0, "/tmp/stage2_smoke")
v.cfg["stage_R"], v.cfg["stage_F"] = R, F
if len(sys.argv) > 3:
    v.REACH_ROUND = int(sys.argv[3])
ck = pickle.load(open("reportables/2026-10-14_cr2_penalty/runs/arm_ckpop_hard_pen_s0/params.pkl", "rb")); tr = ck.get("trained", ck)
v.params = jax.tree_util.tree_map(jnp.asarray, tr["params"]); v.lam = np.asarray(tr["lam"], float)
reg = Region(128)
NREP = int(sys.argv[4]) if len(sys.argv) > 4 else 2
for rep in range(NREP):
    t0 = time.time(); viol = np.zeros(5, int); chg = []
    for i in range(32):
        s = EVAL_SEED0 + 7000 * rep + i
        g = v.generate(s, reg)
        x = _with_designed(p, g["final"], s, reg)
        viol += np.array([energy(p, x, reg).viol[l] for l in range(5)])
        chg.append((g["raw"]["t"] != g["final"]["t"]).mean())
    print(f"periodic 32 worlds rep {rep}: {time.time()-t0:.1f}s viol {viol.tolist()} tiles changed {np.mean(chg):.4f}", flush=True)
W = p.W_eval; T = p.T; half = (W // 2) >> T << T
seed = EVAL_SEED0 + 999
t0 = time.time()
ga = v.generate(seed, Region(W, periodic=False, x0=0)); gb = v.generate(seed, Region(W, periodic=False, x0=half))
mism = {}
for kind in ("raw", "final"):
    for c in p.sampled():
        s = half >> c.level
        a = np.asarray(ga[kind][c.name])[:, s:]; b = np.asarray(gb[kind][c.name])[:, :a.shape[1]]
        mism[(kind, c.name)] = int((a != b).sum())
print(f"support R{R} F{F} reach {v._reach(R, F)}: {time.time()-t0:.1f}s mismatches", mism, "TOTAL", sum(mism.values()), flush=True)
