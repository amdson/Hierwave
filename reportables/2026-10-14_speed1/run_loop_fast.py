"""Sweep 2 (fine half): the iterated amortised loop on ground, levels 0-2.
Per iteration: contexts (probe lifts jitter 0-3, and in arm B also the model's
own draws, 50/50 from iteration 2) -> n_cycles exact block-Gibbs cycles
[(2,0) then (1,1)] on levels 0-2 (levels >= 3 stay at the context; lam = 20
prices violations) -> weighted-ML (weights 1) Adam updates of levels 0-2.
usage: run_loop.py <arm A|B|curve> <seed> <budget_s> <n_cycles> <outdir> [nctx] [upd_per_iter]"""
import json, os, pickle, sys, time
from concurrent.futures import ThreadPoolExecutor
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.kernels.amortised import JaxBlockGibbs, make_energy, refine, make_ml_update_drawwise, make_refine_scan, make_multi_update
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
J20 = JaxBlockGibbs(p, reg, 2, 0, lam=LAMK); J11 = JaxBlockGibbs(p, reg, 1, 1, lam=LAMK)
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
step = make_ml_update_drawwise(lay, score_levels=(0, 1, 2)) if MLMODE == "drawwise" else make_ml_update(lay, tau=1.0, clip=10.0)
multi = make_multi_update(step)
refine_fast = make_refine_scan([J20, J11], NC)
opt = ml_opt_init(params)
on = jnp.asarray([1, 1, 1, 0, 0], jnp.float32)
chans = [c.name for c in p.chans]
nfine = sum((p.H >> c.level) * (reg.W >> c.level) for c in p.sampled() if c.level <= 2)
rng = np.random.default_rng(seed)
nseed = [10_000_000 + 1_000_000 * seed]


def unflat_j(v):
    return {c.name: v[:, fl.off[c.name]:fl.off[c.name] + fl.shape[c.name][0] * fl.shape[c.name][1]].reshape((-1,) + fl.shape[c.name])
            for c in p.chans}


def probe_np(seeds, ks):
    return np.stack([fl.flatten({**p.probes(int(s), reg, n=int(k) + 1)[int(k)], **p.designed_values(int(s), reg)}) for s, k in zip(seeds, ks)]).astype(np.int32)


prng = np.random.default_rng(seed + 777)
pool = ThreadPoolExecutor(1)


def submit_probes(n):
    """Seeds/jitters drawn here (main thread); the host-side probe construction runs in a worker thread."""
    seeds = np.arange(nseed[0], nseed[0] + n); nseed[0] += n
    ks = prng.integers(4, size=n)
    return pool.submit(probe_np, seeds, ks)


def probe_ctx(n):
    return jnp.asarray(submit_probes(n).result(), jnp.int32)


def own_np(ss):
    ds = [p.designed_values(int(x), reg) for x in ss]
    return np.asarray(ss, np.int32), {k: np.stack([d[k] for d in ds]).astype(np.int32) for k in ds[0]}


def submit_own(n):
    ss = np.arange(nseed[0], nseed[0] + n); nseed[0] += n
    return pool.submit(own_np, ss)


@jax.jit
def own_flat(params, ss, dj):
    st = gen(params, ss, dj)
    return jnp.concatenate([(dj[c.name] if c.name in dj else st[c.name]).reshape(ss.shape[0], -1).astype(jnp.int32) for c in p.chans], 1)


def own_ctx(params, n):
    ss, dj = submit_own(n).result()
    return own_flat(params, jnp.asarray(ss), {k: jnp.asarray(v) for k, v in dj.items()})


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

logf = open(f"{out}/loop.jsonl", "w")
PROF = os.environ.get("PROF") == "1"
def sync(x):
    if PROF: jax.block_until_ready(x)
    return time.time()
# warm compiles
_ = own_ctx(params, NCTX // 2)
vw = probe_ctx(NCTX)
_ = refine_fast(vw, key)
_ = E(vw)
idx0 = jnp.asarray(np.stack([rng.choice(NCTX, B, replace=False) for _ in range(UPI)]), jnp.int32)
_ = multi(params, opt, unflat_j(vw), idx0, jnp.ones(B), jnp.float32(LR), on)
jax.block_until_ready(_)
t_setup = time.time() - t00
print("setup", round(t_setup, 1), flush=True)
T = {k: 0.0 for k in ["ctx_probe", "ctx_own", "stats", "gibbs", "ml", "total"]}
t0 = time.time(); it = 0; nupd = 0
fut = submit_probes(NCTX); futo = submit_own(NCTX // 2)
while time.time() - t0 < budget:
    it += 1; ti = time.time()
    n_own = NCTX // 2 if (arm == "B" and it >= 2) else 0
    vp = jnp.asarray(fut.result(), jnp.int32); t1 = sync(vp)
    fut = submit_probes(NCTX - (NCTX // 2 if arm == 'B' else 0))   # next iteration's probes, built while the GPU works
    T["ctx_probe"] += t1 - ti
    rec = {"iter": it}
    t2 = t1
    if n_own:
        ss, dj = futo.result(); futo = submit_own(NCTX // 2); vo = own_flat(params, jnp.asarray(ss), {k: jnp.asarray(v) for k, v in dj.items()}); t2 = sync(vo); T["ctx_own"] += t2 - t1
        s0 = E(vo)
        rec["own_soft"], rec["own_nv"], rec["own_soft_fine"] = [float(np.mean(x)) / cells for x in s0]
        v = jnp.concatenate([vp, vo])
    else:
        v = vp
    t3 = time.time(); T["stats"] += t3 - t2
    key, sub = jax.random.split(key)
    v, chg = refine_fast(v, sub); t4 = sync(v); T["gibbs"] += t4 - t3
    s1 = E(v)
    src = np.array([0] * (NCTX - n_own) + [1] * n_own)
    sf = np.asarray(s1[0]) / cells
    rec["probe_soft_last"] = float(sf[src == 0].mean())
    if n_own:
        rec["own_soft_last"] = float(sf[src == 1].mean())
    rec["changed_last"] = [float(x) for x in np.asarray(chg).mean(-1)]
    t4 = time.time()
    st = unflat_j(v)
    idx = jnp.asarray(np.stack([rng.choice(NCTX, B, replace=False) for _ in range(UPI)]), jnp.int32)
    params, opt, info = multi(params, opt, st, idx, jnp.ones(B), jnp.float32(LR), on)
    nupd += UPI
    losses = np.asarray(info["loss"]) / cells; ninf = int(np.asarray(info["n_infeasible"]).sum()); nfeas = int(np.asarray(info["n_feasible"]).sum())
    T["ml"] += time.time() - t4
    T["total"] += time.time() - ti
    rec.update({"t": time.time() - t0, "updates": nupd, "loss_per_cell_fineonly": float(np.mean(losses)),
                "infeasible_draw_frac": ninf / (UPI * B * nfine), "feasible_state_frac": nfeas / (UPI * B)})
    logf.write(json.dumps(rec) + "\n"); logf.flush()
    if it % 10 == 1:
        print(json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in rec.items()}), flush=True)
pickle.dump({"untrained": {"params": p0, "lam": ck["trained"]["lam"]},
             "trained": {"params": jax.device_get(params), "lam": ck["trained"]["lam"]},
             "updates": nupd, "cfg": {**ck["cfg"], "loop_arm": arm, "n_cycles": NC, "nctx": NCTX, "upi": UPI,
                                      "B": B, "lr": LR, "kernel_lam": LAMK, "ml_mode": MLMODE, "iters": it, "fast": True},
             "train_s": time.time() - t0, "setup_s": t_setup, "devices": str(jax.devices())},
            open(f"{out}/params_ground.pkl", "wb"))
print("done iters", it, "updates", nupd, "train_s", round(time.time() - t0, 1), flush=True)
print(json.dumps({k: round(x / it, 3) for k, x in T.items()}))
json.dump({"iters": it, "updates": nupd, "wall": time.time() - t0, "per_iter_s": {k: x / it for k, x in T.items()},
           "states_per_s": it * NCTX / (time.time() - t0), "updates_per_s": nupd / (time.time() - t0)}, open(f"{out}/profile.json", "w"))
