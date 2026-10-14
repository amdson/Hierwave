"""Sweep 2 (fine half): the iterated amortised loop on ground, levels 0-2.
Per iteration: contexts (probe lifts jitter 0-3, and in arm B also the model's
own draws, 50/50 from iteration 2) -> n_cycles exact block-Gibbs cycles
[(2,0) then (1,1)] on levels 0-2 (levels >= 3 stay at the context; lam = 20
prices violations) -> weighted-ML (weights 1) Adam updates of levels 0-2.
usage: run_loop.py <arm A|B|curve> <seed> <budget_s> <n_cycles> <outdir> [nctx] [upd_per_iter]"""
import json, os, pickle, sys, time
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.kernels.amortised import JaxBlockGibbs, make_energy, refine, make_ml_update_drawwise
from castlegen.kernels.logq import make_ml_update, ml_opt_init
from castlegen.variants import _cpass_core as core
from castlegen.variants._cfast_core import train_pass
from castlegen.variants._cln_core import window_of

arm, seed, budget, NC, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
NCTX = int(sys.argv[6]) if len(sys.argv) > 6 else 32
UPI = int(sys.argv[7]) if len(sys.argv) > 7 else 4
B, LR, LAMK = 8, 1e-3, 20.0
os.makedirs(out, exist_ok=True)
t00 = time.time()
p = get("ground"); reg = Region(p.W_train); cells = p.H * reg.W
import castlegen.kernels.logq as _lq; _lq._forced_pass = _lq._forced_pass_ref   # old teacher-forced pass (MLP inside the colour scan)
J20 = JaxBlockGibbs(p, reg, 2, 0, lam=LAMK, fast=False); J11 = JaxBlockGibbs(p, reg, 1, 1, lam=LAMK, fast=False)   # old per-completion kernel
fl = J11.K.flat
E = make_energy(fl, J11.K.insts, fine_max=2)
ck = pickle.load(open("reportables/2026-10-14_cstart2/runs/cstart_f1L3_ln6/params_ground.pkl", "rb"))
lay = core.Layout(p, reg.W, True, 0, win=window_of((-2, 3)))
params = jax.tree_util.tree_map(jnp.asarray, ck["untrained"]["params"])
p0 = jax.device_get(params)
lam_gen = jnp.asarray(ck["trained"]["lam"], jnp.float32)
gen = jax.jit(jax.vmap(lambda pr, s, d: train_pass(lay, pr, s, d, jnp.float32(1.0), lam_gen, record=False)[0],
                       in_axes=(None, 0, 0)))
MLMODE = os.environ.get("MLMODE", "drawwise")
step = make_ml_update_drawwise(lay) if MLMODE == "drawwise" else make_ml_update(lay, tau=1.0, clip=10.0)
opt = ml_opt_init(params)
on = jnp.asarray([1, 1, 1, 0, 0], jnp.float32)
chans = [c.name for c in p.chans]
nfine = sum((p.H >> c.level) * (reg.W >> c.level) for c in p.sampled() if c.level <= 2)
rng = np.random.default_rng(seed)
nseed = [10_000_000 + 1_000_000 * seed]


def unflat_j(v):
    return {c.name: v[:, fl.off[c.name]:fl.off[c.name] + fl.shape[c.name][0] * fl.shape[c.name][1]].reshape((-1,) + fl.shape[c.name])
            for c in p.chans}


def probe_ctx(n):
    vs = []
    for _ in range(n):
        s = nseed[0]; nseed[0] += 1
        k = int(rng.integers(4))
        vs.append(fl.flatten({**p.probes(s, reg, n=k + 1)[k], **p.designed_values(s, reg)}))
    return jnp.asarray(np.stack(vs), jnp.int32)


def own_ctx(params, n):
    ss = np.arange(nseed[0], nseed[0] + n); nseed[0] += n
    ds = [p.designed_values(int(s), reg) for s in ss]
    dj = {k: jnp.asarray(np.stack([d[k] for d in ds]), jnp.int32) for k in ds[0]}
    st = gen(params, jnp.asarray(ss, jnp.int32), dj)
    st = {**{k: np.asarray(v) for k, v in st.items()}, **{k: np.stack([d[k] for d in ds]) for k in ds[0]}}
    return jnp.asarray(np.stack([fl.flatten({k: st[k][b] for k in st}) for b in range(n)]), jnp.int32)


def stats(v):
    s, nv, sf = E(v)
    return float(np.mean(s)) / cells, float(np.mean(nv)) / cells, float(np.mean(sf)) / cells


key = jax.random.PRNGKey(1000 + seed)
if arm == "curve":                                     # long chains: energy per cycle, probe and own-draw contexts
    res = {}
    for name, v in (("probe", probe_ctx(NCTX)), ("own_untrained", own_ctx(params, NCTX))):
        t = time.time()
        _, cur = refine([J20, J11], v, key, NC, energy=E)
        res[name] = [{"cycle": c["cycle"], "soft": float(c["soft"].mean()) / cells, "soft_se": float(c["soft"].std() / np.sqrt(len(c["soft"]))) / cells,
                      "soft_fine": float(c["soft_fine"].mean()) / cells, "nv": float(c["nv"].mean()) / cells,
                      "changed": [float(x.mean()) for x in c.get("changed", [])]} for c in cur]
        print(name, "time", round(time.time() - t, 1), [round(r["soft"], 4) for r in res[name]], flush=True)
    json.dump(res, open(f"{out}/curve.json", "w"), indent=1)
    sys.exit()


import numpy as _np
def sync(x): jax.block_until_ready(x); return time.time()
_ = own_ctx(params, NCTX // 2)
vw = probe_ctx(NCTX)
_ = refine([J20, J11], vw, key, 1, energy=E)
st0 = unflat_j(vw[:B])
_ = step(params, opt, st0, jnp.ones(B), jnp.float32(LR), on)
print("setup", round(time.time()-t00,1), flush=True)
T = {k: 0.0 for k in ["probe_ctx","own_draw","stats","gibbs20","gibbs11","energy","ml","total"]}
t0 = time.time(); it = 0; nupd = 0
while time.time() - t0 < budget:
    it += 1; ti = time.time()
    n_own = NCTX // 2 if it >= 2 else 0
    vp = probe_ctx(NCTX - n_own); t1 = sync(vp); T["probe_ctx"] += t1 - ti
    if n_own:
        vo = own_ctx(params, n_own); t2 = sync(vo); T["own_draw"] += t2 - t1
        _ = stats(vo); t3 = time.time(); T["stats"] += t3 - t2
        v = jnp.concatenate([vp, vo])
    else:
        v = vp; t3 = time.time()
    key, sub = jax.random.split(key)
    for c in range(NC):
        ta = time.time()
        sub, k1, k2 = jax.random.split(sub, 3)
        v, _n = J20.sweep(v, k1); tb = sync(v); T["gibbs20"] += tb - ta
        v, _n = J11.sweep(v, k2); tc = sync(v); T["gibbs11"] += tc - tb
        _e = E(v); np.asarray(_e[0]); td = time.time(); T["energy"] += td - tc
    tr = time.time()
    st = unflat_j(v)
    for u in range(UPI):
        idx = rng.choice(NCTX, B, replace=False)
        params, opt, info = step(params, opt, {k: st[k][idx] for k in chans}, jnp.ones(B), jnp.float32(LR), on)
        float(info["loss"]); nupd += 1
    T["ml"] += time.time() - tr
    T["total"] += time.time() - ti
n = it
print("iters", it, "updates", nupd, "wall", round(time.time()-t0,1))
print(json.dumps({k: round(v/n,3) for k, v in T.items()}))
print("states/s (refined)", it*NCTX/T["total"], "updates/s", nupd/T["total"])
json.dump({"iters": it, "updates": nupd, "per_iter_s": {k: v/n for k, v in T.items()}, "states_per_s": it*NCTX/T["total"], "updates_per_s": nupd/T["total"]}, open(f"{out}/profile.json","w"))
