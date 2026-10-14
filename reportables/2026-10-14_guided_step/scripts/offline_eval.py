"""Offline (no training) evaluation of generation variants on saved parameters.

  python offline_eval.py <base variant> <params.pkl> <fmt: pen|w4> <k> <n worlds> <R max> <out.json>

Held-out worlds: the bench's eval seeds for training seed k (EVAL_SEED0 + 1000 k + i), the periodic
W_eval strip.  For each world: raw pass draw; cpass one-round repair; jitter repair R = 1, 2 (baseline,
_jitter_repair.py); guided steps 1..R (states recorded after every step) and each of those followed by
cpass's repair.  Per state: violated hard instances per level (bench/energy.py), graded mass per level
(castlegen/graded.py), the problem's stats, cells changed vs raw per level; times per world."""
import json, pickle, sys, time
import numpy as np
import jax, jax.numpy as jnp
from castlegen.bench import energy as en
from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.bench.run import EVAL_SEED0
from castlegen.bench.variant import load_variants
from castlegen import graded
from castlegen.variants import _cpass_core as core
from castlegen.variants._guided_step import HardMaps, guided_revise
from castlegen.variants._jitter_repair import jitter_repair

base, ppath, fmt, k, n, R, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
V = load_variants(); prob = get("ground"); T = prob.T
v = V[base](prob, seed=k, workdir="/tmp/guided_offline")
d = pickle.load(open(ppath, "rb"))
pr, lam = (d["params"], d["lam"]) if fmt == "pen" else (d["trained"]["params"], d["trained"]["lam"])
assert set(pr) == set(v.params)
v.params = jax.tree_util.tree_map(jnp.asarray, pr); v.lam = np.asarray(lam, float)
region = Region(prob.W_eval); lay = v.layout(region)
lamj = jnp.asarray(v.lam, jnp.float32)
hm = HardMaps(lay)
rep = lay.jit_repair
jr = {r: jax.jit(lambda st, s, r=r: jitter_repair(lay, st, s, r)) for r in (1, 2)}
rev = jax.jit(lambda st, s: guided_revise(lay, v.params, st, s, R, record=True))
rep_many = jax.jit(lambda sts, s: [core.repair(lay, st, s) for st in sts])

kinds = ["raw", "repair", "jitter1", "jitter2"] + [f"guided{r}" for r in range(1, R + 1)] + [f"guided{r}+repair" for r in range(1, R + 1)]
acc = {kd: {"viol": np.zeros(T + 1), "mass": np.zeros(T + 1), "stats": {}, "changed": np.zeros(T + 1), "worlds_valid": 0} for kd in kinds}
nhard = np.zeros(T + 1); times = {kd: 0.0 for kd in ("gen", "repair", "jitter1", "jitter2", "guided")}
step_changed = np.zeros(R); step_logq = np.zeros(R)

def measure(kd, st, raw_np, des):
    x = {**des, **{c.name: np.asarray(st[c.name]) for c in prob.sampled()}}
    E = en.energy(prob, x, region)
    D = graded.level_mass(prob, x, region)
    a = acc[kd]
    for l in range(T + 1):
        a["viol"][l] += E.viol[l]; a["mass"][l] += D[l]
    a["worlds_valid"] += int(sum(E.viol.values()) == 0)
    for name, val in prob.stats(x).items():
        a["stats"].setdefault(name, []).append(float(val))
    for c in prob.sampled():
        a["changed"][c.level] += int((np.asarray(st[c.name]) != raw_np[c.name]).sum())
    return E

for i in range(-1, n):                       # world -1: warm-up (compiles), not counted
    seed = EVAL_SEED0 + 1000 * k + max(i, 0)
    des = prob.designed_values(seed, region)
    desj = {kk: jnp.asarray(vv, jnp.int32) for kk, vv in des.items()}
    t0 = time.time(); raw = lay.jit_gen(v.params, seed, desj, lamj); jax.block_until_ready(raw["t"]); t1 = time.time()
    fr = rep(raw, seed); jax.block_until_ready(fr["t"]); t2 = time.time()
    j1 = jr[1](raw, seed); jax.block_until_ready(j1["t"]); t3 = time.time()
    j2 = jr[2](raw, seed); jax.block_until_ready(j2["t"]); t4 = time.time()
    g, info = rev(raw, seed); jax.block_until_ready(g["t"]); t5 = time.time()
    gr = rep_many(info["states"], seed); jax.block_until_ready(gr[-1]["t"])
    if i < 0:
        print(f"warm-up: gen {t1-t0:.1f}s repair {t2-t1:.1f}s jitter1 {t3-t2:.1f}s jitter2 {t4-t3:.1f}s guided{R} {t5-t4:.1f}s", flush=True)
        continue
    for kd, dt in (("gen", t1 - t0), ("repair", t2 - t1), ("jitter1", t3 - t2), ("jitter2", t4 - t3), ("guided", t5 - t4)):
        times[kd] += dt
    raw_np = {c.name: np.asarray(raw[c.name]) for c in prob.sampled()}
    E = measure("raw", raw, raw_np, des)
    for l in range(T + 1):
        nhard[l] += E.nhard[l]
    measure("repair", fr, raw_np, des); measure("jitter1", j1, raw_np, des); measure("jitter2", j2, raw_np, des)
    for r in range(1, R + 1):
        measure(f"guided{r}", info["states"][r - 1], raw_np, des)
        measure(f"guided{r}+repair", gr[r - 1], raw_np, des)
    step_changed += np.asarray(info["changed"]); step_logq += np.asarray(info["log_q"])
    print(f"world {i}: raw {[int(acc['raw']['viol'][l]) for l in range(T+1)]} rep {[int(acc['repair']['viol'][l]) for l in range(T+1)]} "
          f"g{R} {[int(acc[f'guided{R}']['viol'][l]) for l in range(T+1)]} changed/step {np.asarray(info['changed']).tolist()}", flush=True)

res = {"base": base, "params": ppath, "k": k, "n": n, "R": R, "nhard": (nhard / n).tolist(), "lam": v.lam.tolist(),
       "time_per_world": {kd: t / n for kd, t in times.items()}, "step_changed": (step_changed / n).tolist(),
       "step_logq": (step_logq / n).tolist(), "kinds": {}}
for kd in kinds:
    a = acc[kd]
    res["kinds"][kd] = {"viol": a["viol"].tolist(), "viol_frac": (a["viol"] / np.maximum(nhard, 1)).tolist(),
                        "mass": a["mass"].tolist(), "changed_per_world": (a["changed"] / n).tolist(),
                        "worlds_valid": a["worlds_valid"],
                        "stats": {s: [float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(len(x)))] for s, x in a["stats"].items()}}
json.dump(res, open(out, "w"), indent=1)
print("kind".ljust(16), "viol L0..L4".ljust(28), "mass L0..L4".ljust(28), "changed/world L0..L4", "valid", "line_match", "dirt", "rough", "float", "fill_err")
for kd in kinds:
    r = res["kinds"][kd]; s = r["stats"]
    print(kd.ljust(16), str([int(x) for x in r["viol"]]).ljust(28), str([int(x) for x in r["mass"]]).ljust(28),
          str([round(x, 1) for x in r["changed_per_world"]]).ljust(36), r["worlds_valid"],
          f"{s['line_match'][0]:.3f} {s['dirt_frac'][0]:.3f} {s['roughness'][0]:.2f} {s['floating'][0]:.1f} {s['fill_err'][0]:.4f}")
print("time/world", {kd: round(t, 3) for kd, t in res["time_per_world"].items()}, "changed/step", np.round(res["step_changed"], 1).tolist())
