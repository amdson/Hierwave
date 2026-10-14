"""dimers-kernel loop: contexts = own draws (tau = 1); kernel cycle = exact fine-block Gibbs sweeps (k_list)
+ multiple-try footprint redraws from q; draw-wise ML on the refined states.  Logs orientation fractions of
raw own draws and of refined states every iteration (= upi ML updates).
usage: dk_loop.py <init: init|pkl path> <train_s> <out.jsonl> [key=val ...]"""
import json, pickle, sys, time
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.problem import Region
from castlegen.bench.problems import get as get_problem
from castlegen.bench import energy as en
from castlegen.bench.variant import load_variants
from castlegen.kernels import block_mh as bm
from castlegen.kernels.amortised import make_energy, make_ml_update_drawwise
from castlegen.kernels.fine_block import FineBlockKernel
from castlegen.kernels.logq import ml_opt_init
from castlegen.variants._cpass_core import NCOL
from castlegen.variants._cfast_core import train_pass

init, budget, outf = sys.argv[1], float(sys.argv[2]), sys.argv[3]
cfg = dict(k_list="2,3", n_sweeps=3, mt_k=4, mt_moves=8, K_mt=8, nctx=32, upi=8, ml_B=8, ml_lr=1e-3, lam_k=20.0, seed=0,
           mt_on=1, fb_on=1, fb_lam=0.0, wlam=0.0)
for a in sys.argv[4:]:
    k, v = a.split("="); cfg[k] = type(cfg[k])(v) if not isinstance(cfg[k], str) else v
ks = [int(x) for x in cfg["k_list"].split(",") if x]
p = get_problem("dimers"); region = Region(p.W_train); cells = p.H * region.W
c = load_variants()["cfast"](p, seed=cfg["seed"], workdir="/tmp/dk_loop")
if init != "init":
    ck = pickle.load(open(init, "rb"))["trained"]
    c.params = jax.tree_util.tree_map(jnp.asarray, ck["params"]); c.lam = np.asarray(ck["lam"], float)
lay = c.layout(region)
lam_gen = jnp.asarray(c.lam, jnp.float32)
gen = jax.jit(jax.vmap(lambda pr, s, d: train_pass(lay, pr, s, d, jnp.float32(1.0), lam_gen, record=False)[0], in_axes=(None, 0, 0)))
FB = {k: FineBlockKernel(p, region, k, lam=(cfg["fb_lam"] or None), seed=cfg["seed"]) for k in ks}
fl = next(iter(FB.values())).flat
E = make_energy(fl, FB[ks[0]].Ks[0].insts, fine_max=0)
mlstep = make_ml_update_drawwise(lay)

def to_flat(st):
    n = next(iter(st.values())).shape[0]
    return jnp.concatenate([jnp.asarray(st[ch.name]).reshape(n, -1) for ch in p.chans], 1).astype(jnp.int32)
def from_flat(v):
    return {ch.name: v[:, fl.off[ch.name]:fl.off[ch.name] + int(np.prod(fl.shape[ch.name]))].reshape((-1,) + fl.shape[ch.name]) for ch in p.chans}

# --- multiple-try footprint redraw from q (block_mh.block_pass with a k x k footprint mask)
def footprint(h, w, y, x, k):
    yy, xx = jnp.arange(h)[:, None], jnp.arange(w)[None, :]
    return (yy >= y) & (yy < y + k) & (((xx - x) % w) < k)
def mt_move(params, key, state, k, K, lam):
    h, w = lay.hw[0]
    k1, k2, k3, k4 = jax.random.split(key, 4)
    y = jax.random.randint(k1, (), 0, h - k + 1); x = jax.random.randint(k2, (), 0, w)
    inb = {ch.name: footprint(h, w, y, x, k) for ch in p.sampled()}
    keys = jax.random.split(k3, K + 2)
    cst, clq, ces, cnv = jax.vmap(lambda kk: bm.block_pass(lay, params, kk, state, inb, 1.0, lam, 0))(keys[:K])
    _, xlq, xes, xnv = bm.block_pass(lay, params, keys[K], state, inb, 1.0, lam, 0, forced=state)
    lw = bm._logp(ces, cnv, lam) - clq; lwx = bm._logp(xes, xnv, lam) - xlq
    j = jax.random.categorical(keys[K + 1], jnp.where(jnp.isnan(lw), -jnp.inf, lw))
    others = jnp.where(jnp.arange(K) == j, lwx, lw)
    log_r = bm._lse(lw) - bm._lse(others)
    log_r = jnp.where(jnp.isfinite(bm._lse(lw)), log_r, -jnp.inf)
    acc = jnp.log(jax.random.uniform(k4)) < log_r
    new = {n: jnp.where(acc, cst[n][j], v) for n, v in state.items()}
    changed = jnp.any(jnp.stack([(new[n] != state[n]).any() for n in new]))
    return new, (acc, changed)
def make_mt(k, nmoves, K, lam):
    def one(params, key, state):
        def step(carry, i):
            st, kk = carry
            kk, ks_ = jax.random.split(kk)
            st, rec = mt_move(params, ks_, st, k, K, lam)
            return (st, kk), rec
        (st, _), recs = jax.lax.scan(step, (state, key), jnp.arange(nmoves))
        return st, recs
    return jax.jit(jax.vmap(one, in_axes=(None, 0, 0)))
mt = make_mt(cfg["mt_k"], cfg["mt_moves"], cfg["K_mt"], cfg["lam_k"])

def ostats(st):
    o = np.asarray(st["p"]) // 2
    return [float((o == i).mean()) for i in range(3)]
def own_ctx(n, seed0):
    ss = np.arange(seed0, seed0 + n)
    ds = [p.designed_values(int(s), region) for s in ss]
    dj = {k: jnp.asarray(np.stack([d[k] for d in ds]), jnp.int32) for k in ds[0]}
    st = gen(c.params, jnp.asarray(ss, jnp.int32), dj)
    return {ch.name: jnp.asarray(st[ch.name] if ch.name in st else dj[ch.name], jnp.int32) for ch in p.chans}

def refine(st, key, rng):
    d = {}
    v = to_flat(st)
    if cfg["fb_on"]:
        for k in ks:
            ch = []
            for _ in range(cfg["n_sweeps"]):
                key, sub = jax.random.split(key)
                v, n = FB[k].sweep(v, sub); ch.append(float(np.asarray(n).mean()) / FB[k].Ks[0].nb)
            d[f"chg_k{k}"] = float(np.mean(ch))
    st = from_flat(v)
    if cfg["mt_on"]:
        key, sub = jax.random.split(key)
        st, (acc, chg) = mt(c.params, jax.random.split(sub, v.shape[0]), st)
        d["acc_mt"] = float(np.asarray(acc).mean()); d["chg_mt"] = float(np.asarray(chg).mean())
    return st, d

opt = ml_opt_init(c.params); on = jnp.ones(1, jnp.float32)
rng = np.random.default_rng(cfg["seed"]); key = jax.random.PRNGKey(4242 + cfg["seed"])
ctx_seed = 30_000_000 + 1_000_000 * cfg["seed"]
t0 = time.time()
st = own_ctx(cfg["nctx"], ctx_seed); st2, _ = refine(st, key, rng); E(to_flat(st2))
for k in ks:                                     # compile every offset's sweep before the clock
    for J in FB[k].Js:
        jax.block_until_ready(J.sweep(to_flat(st), key))
_ = mlstep(c.params, opt, {k: x[:cfg["ml_B"]] for k, x in st2.items()}, jnp.ones(cfg["ml_B"]), jnp.float32(cfg["ml_lr"]), on)
c.params0 = c.params
ts = time.time(); t_setup = ts - t0
f = open(outf, "w")
def log(d): f.write(json.dumps(d) + "\n"); f.flush()
log({"setup_done": True, "t_setup": t_setup, "init": init, "cfg": cfg, "ks": ks, "ncomps": {k: FB[k].ncomps for k in ks}, "lam_gen": c.lam.tolist()})
it = 0; updates = 0
while time.time() - ts < budget:
    it += 1; ti = time.time()
    st = own_ctx(cfg["nctx"], ctx_seed); ctx_seed += cfg["nctx"]
    s0, nv0, _ = E(to_flat(st)); o_raw = ostats(st)
    key, sub = jax.random.split(key)
    st, kd = refine(st, sub, rng)
    s1, nv1, _ = E(to_flat(st))
    tm = time.time(); losses = []; ninf = 0; nvw = np.asarray(nv1)
    for u in range(cfg["upi"]):
        idx = rng.choice(cfg["nctx"], cfg["ml_B"], replace=False)
        vw = nvw[idx]; w = np.exp(-cfg["wlam"] * (vw - vw.min())); w = w / w.sum() * cfg["ml_B"]
        c.params, opt, info = mlstep(c.params, opt, {k: x[idx] for k, x in st.items()}, jnp.asarray(w, jnp.float32), jnp.float32(cfg["ml_lr"]), on)
        losses.append(float(info["loss"]) / cells); ninf += int(info["n_infeasible"]); updates += 1
    log({"iter": it, "update": updates, "t": time.time() - ts, "orient_raw": o_raw, "orient_refined": ostats(st), **kd, "loss_per_cell": float(np.mean(losses)),
         "infeasible_per_state": ninf / (cfg["upi"] * cfg["ml_B"]), "viol_raw": float(np.asarray(nv0).mean() / cells),
         "viol_refined": float(np.asarray(nv1).mean() / cells), "valid_frac_refined": float((np.asarray(nv1) == 0).mean()), "valid_frac_raw": float((np.asarray(nv0) == 0).mean()), "t_kernel": tm - ti, "t_ml": time.time() - tm})
log({"done": True, "updates": updates, "iters": it})
pickle.dump({"trained": {"params": jax.device_get(c.params), "lam": c.lam.tolist()}}, open(outf.replace(".jsonl", ".pkl"), "wb"))
