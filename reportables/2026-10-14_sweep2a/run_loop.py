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

logf = open(f"{out}/loop.jsonl", "w")
# warm compiles
_ = own_ctx(params, NCTX // 2)
vw = probe_ctx(NCTX)
_ = refine([J20, J11], vw, key, 1, energy=E)
st0 = unflat_j(vw[:B])
_ = step(params, opt, st0, jnp.ones(B), jnp.float32(LR), on)
t_setup = time.time() - t00
print("setup", round(t_setup, 1), flush=True)
t0 = time.time(); it = 0; nupd = 0
while time.time() - t0 < budget:
    it += 1; ti = time.time()
    n_own = NCTX // 2 if (arm == "B" and it >= 2) else 0
    vp = probe_ctx(NCTX - n_own)
    rec = {"iter": it}
    if n_own:
        vo = own_ctx(params, n_own)
        rec["own_soft"], rec["own_nv"], rec["own_soft_fine"] = stats(vo)
        v = jnp.concatenate([vp, vo])
    else:
        v = vp
    tg = time.time()
    key, sub = jax.random.split(key)
    v, cur = refine([J20, J11], v, sub, NC, energy=E)
    tr = time.time()
    src = np.array([0] * (NCTX - n_own) + [1] * n_own)
    for c in cur:
        c["soft"] = c["soft"] / cells; c["soft_fine"] = c["soft_fine"] / cells
    rec["probe_soft_curve"] = [float(c["soft"][src == 0].mean()) for c in cur]
    if n_own:
        rec["own_soft_curve"] = [float(c["soft"][src == 1].mean()) for c in cur]
        rec["own_fine_curve"] = [float(c["soft_fine"][src == 1].mean()) for c in cur]
        rec["own_nv_curve"] = [float(c["nv"][src == 1].mean()) / cells for c in cur]
    rec["changed_last"] = [float(x.mean()) for x in cur[-1]["changed"]]
    st = unflat_j(v)
    losses, ninf, nfeas = [], 0, 0
    for u in range(UPI):
        idx = rng.choice(NCTX, B, replace=False)
        params, opt, info = step(params, opt, {k: st[k][idx] for k in chans}, jnp.ones(B), jnp.float32(LR), on)
        losses.append(float(info["loss"]) / cells); ninf += int(info["n_infeasible"]); nfeas += int(info["n_feasible"]); nupd += 1
    rec.update({"t": time.time() - t0, "updates": nupd, "loss_per_cell": float(np.mean(losses)),
                "infeasible_draw_frac": ninf / (UPI * B * nfine), "feasible_state_frac": nfeas / (UPI * B), "time_ctx": tg - ti, "time_refine": tr - tg,
                "time_ml": time.time() - tr})
    logf.write(json.dumps(rec) + "\n"); logf.flush()
    if it % 5 == 1:
        print(json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in rec.items() if "curve" not in k}),
              "probe curve", [round(x, 3) for x in rec["probe_soft_curve"]],
              "own curve", [round(x, 3) for x in rec.get("own_soft_curve", [])], flush=True)
pickle.dump({"untrained": {"params": p0, "lam": ck["trained"]["lam"]},
             "trained": {"params": jax.device_get(params), "lam": ck["trained"]["lam"]},
             "updates": nupd, "cfg": {**ck["cfg"], "loop_arm": arm, "n_cycles": NC, "nctx": NCTX, "upi": UPI,
                                      "B": B, "lr": LR, "kernel_lam": LAMK, "ml_mode": MLMODE, "iters": it},
             "train_s": time.time() - t0, "setup_s": t_setup, "devices": str(jax.devices())},
            open(f"{out}/params_ground.pkl", "wb"))
print("done iters", it, "updates", nupd, "train_s", round(time.time() - t0, 1), flush=True)
