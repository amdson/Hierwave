"""ctop: at the top level (level 4) of ground, coupled replays of every top draw with the value
overridden to (probe) a fill-following lifted value and (dirt) all dirt, vs the actual draw.
Downstream cost G = cone return (sum of costs of the draw's dependency cone, later draws only),
split into soft, lam*nv, tau*log q and by level.  Diagnostics only.
usage: ctop_eval.py snaps.pkl out.pkl NW [tau_override]"""
import pickle, sys, time
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench import noise
from castlegen.bench.problem import Region
from castlegen.bench.problems import get as get_problem
from castlegen.bench.problems.ground import cert_index
from castlegen.bench.variant import load_variants
from castlegen.variants import _cpass_core as core
from castlegen.variants._cfast_cone import Cone
from castlegen.variants._cfast_core import train_pass

snapf, outf, NW = sys.argv[1], sys.argv[2], int(sys.argv[3])
taus_over = [float(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [None]
PROBE = 1                                    # probe index (1 row of jitter around the designed line)
p = get_problem("ground"); T = p.T; reg = Region(p.W_train)
V = load_variants()["cfast"]
var = V(p, seed=99, workdir="/tmp/ctop_e")
lay = var.layout(reg)
cone = Cone(lay, lay.actor_win)
CH = lay.chans[T]; topk = [core.key_of(T, c.name) for c in CH]
cells = p.H * reg.W
# top draws table: (ci, k, j, Y, X)
tab = []
for ci, c in enumerate(CH):
    for k in range(core.NCOL):
        Y, X, v = [np.asarray(a) for a in lay.sites(T, k)]
        for j in np.flatnonzero(v): tab.append((ci, k, j, Y[j], X[j]))
tab = np.array(tab, np.int32); ND = len(tab)
idxs = jnp.stack([jnp.asarray(cone.idx[k]) for k in topk])            # (nch,16,n,S,4)
levels = list(range(T + 1))
keys_l = {l: [core.key_of(l, c.name) for c in lay.chans[l]] for l in levels}
allk = [k for l in levels for k in keys_l[l]]
DIRT = {"rho4": 256, "cert4": int(cert_index(16)[16, 16])}
steps = np.array([[lay.step(T, k, ci) for k in range(core.NCOL)] for ci in range(len(CH))], np.int32)


def terms_pass(params, seed, des, tau, lam, ov):
    """train_pass with override; per-key (soft, lam*nv, tau*logq) of the drawn values."""
    state = {c.name: (des[c.name] if c.designed is not None else jnp.full(lay.hw[c.level], -1, jnp.int32)) for c in p.chans}
    out = {}
    for l in lay.levels:
        chans = lay.chans[l]
        if not chans: continue
        def body(state, k, l=l, chans=chans):
            Y, X, valid = lay.sites(l, k); h, w = lay.hw[l]
            Xn = jnp.mod(X, w)
            outs = []
            for ci, c in enumerate(chans):
                soft, nv, look = lay.cond(state, c.name, Y, X)
                v = nv + look; vmin = jnp.min(v, 1, keepdims=True)
                mask = (v == vmin) | ~valid[:, None]
                feat = lay.features(state, l, Y, X, lay.actor_win)
                b = core.mlp(params[core.key_of(l, c.name)], feat)
                logits = jnp.where(mask, -soft / tau - b, -jnp.inf)
                logq = jax.nn.log_softmax(logits, 1)
                g = noise.gumbel(jnp, seed, noise.tag_of(c.name), l, Y, Xn, c.D)
                a = jnp.argmax(logits + g, 1).astype(jnp.int32)
                st = lay.step(l, 0, ci) + k * len(chans)
                if ov is not None:
                    os_, oj, oval = ov
                    a = jnp.where((st == os_) & (jnp.arange(Y.shape[0]) == oj), oval, a)
                ix = jnp.arange(Y.shape[0])
                la = logq[ix, a]
                la = jnp.where(jnp.isfinite(la), la, 0.0)     # value outside A: its own log q is not defined; G never includes the own draw
                outs.append(jnp.stack([jnp.where(valid, soft[ix, a], 0.0), jnp.where(valid, lam[l] * nv[ix, a], 0.0),
                                       jnp.where(valid, tau * la, 0.0)]))
                Ys = jnp.where(valid, Y, h)
                state = {**state, c.name: state[c.name].at[Ys, X].set(a, mode="drop")}
            return state, outs
        state, ys = jax.lax.scan(body, state, jnp.arange(core.NCOL))
        for ci, c in enumerate(chans): out[core.key_of(l, c.name)] = ys[ci]       # (16, 3, n)
    return out


LVM = jnp.asarray([[1.0 if k in keys_l[l] else 0.0 for l in levels] for k in allk], jnp.float32)   # (nkeys, 5)


def split_G(t, ix, lvm):
    """t: {key: (C,16,3,n)}; ix: (C,S,4) cone box corners.  -> (C, 3, 5) per term, level.
    One integral call on a (C*15) batch (term x level masks, traced so nothing constant-folds)."""
    C = ix.shape[0]
    costs = {}
    for ki, k in enumerate(allk):
        x = t[k]                                                       # (C,16,3,n)
        m = lvm[ki]                                                    # (5,)
        y = x[:, :, :, None, :] * m[None, None, None, :, None]         # (C,16,3,5,n)
        costs[k] = jnp.transpose(y, (0, 2, 3, 1, 4)).reshape(C * 15, 16, -1)
    ii = cone.integral(costs)                                          # (C*15, size)
    ixr = jnp.repeat(ix.reshape(C, 1, -1), 15, 1).reshape(C * 15, -1)
    v = jnp.take_along_axis(ii, ixr, -1).reshape(C * 15, *ix.shape[1:])
    g = jnp.sum(v[..., 0] - v[..., 1] - v[..., 2] + v[..., 3], -1)
    return g.reshape(C, 3, 5)


@jax.jit
def run_chunk(params, tau, lam, seeds, des, ov, ix):
    t = jax.vmap(lambda s, d, o: terms_pass(params, s, d, tau, lam, o))(seeds, des, ov)
    return split_G(t, ix, LVM)


@jax.jit
def base_pass(params, tau, lam, seeds, des):
    _, recs = jax.vmap(lambda s, d: train_pass(lay, params, s, d, tau, lam))(seeds, des)
    return recs


seeds = np.arange(4_000_000, 4_000_000 + NW)
dl = [p.designed_values(int(s), reg) for s in seeds]
des = {k: jnp.asarray(np.stack([d[k] for d in dl]), jnp.int32) for k in dl[0]}
probes = [p.probes(int(s), reg)[PROBE] for s in seeds]
snaps = pickle.load(open(snapf, "rb"))
res = {}
for name, sn in snaps.items():
    for to in taus_over:
        tau = sn["tau"] if to is None else to
        params = jax.tree_util.tree_map(jnp.asarray, sn["params"]); lam = jnp.asarray(sn["lam"], jnp.float32)
        t0 = time.time()
        recs = jax.device_get(base_pass(params, jnp.float32(tau), lam, jnp.asarray(seeds, jnp.int32), des))
        # per (world, draw): actual value, in-A flags, own cost terms for each candidate
        R = {f: np.zeros((NW, ND)) for f in ("a",)}
        cand = {"actual": np.zeros((NW, ND), int), "probe": np.zeros((NW, ND), int), "dirt": np.zeros((NW, ND), int)}
        own = {c: np.zeros((NW, ND, 3)) for c in cand}; inA = {c: np.zeros((NW, ND), bool) for c in cand}
        nA = np.zeros((NW, ND))
        for d, (ci, k, j, Y, X) in enumerate(tab):
            key = topk[ci]; r = recs[key]; h, w = lay.hw[T]
            cn = CH[ci].name
            for b in range(NW):
                cand["actual"][b, d] = r["a"][b, k, j]
                cand["probe"][b, d] = probes[b][cn][min(Y, h - 1), X % w]
                cand["dirt"][b, d] = DIRT[cn]
                nA[b, d] = r["mask"][b, k, j].sum()
                for c in cand:
                    v = cand[c][b, d]
                    inA[c][b, d] = r["mask"][b, k, j, v]
                    lq = r["logq"][b, k, j, v]
                    own[c][b, d] = [r["soft"][b, k, j, v], lam[T] * r["nv"][b, k, j, v], tau * (lq if np.isfinite(lq) else np.nan)]
        # coupled replays
        G = {}
        flat_b = np.repeat(np.arange(NW), ND); flat_d = np.tile(np.arange(ND), NW)
        for c in cand:
            ci_ = tab[flat_d, 0]; k_ = tab[flat_d, 1]; j_ = tab[flat_d, 2]
            val = cand[c][flat_b, flat_d]
            ixs = np.asarray(idxs)[ci_, k_, j_]                                           # (N,S,4)
            ov = (jnp.asarray(steps[ci_, k_]), jnp.asarray(j_), jnp.asarray(val, jnp.int32))
            N = len(flat_b); C = 64; out = []
            for s in range(0, N, C):
                sl = slice(s, min(s + C, N)); n = sl.stop - sl.start
                pad = C - n
                pd = lambda x: np.concatenate([x[sl], x[sl][:1].repeat(pad, 0)]) if pad else x[sl]
                bb = pd(flat_b)
                o = run_chunk(params, jnp.float32(tau), lam, jnp.asarray(seeds[bb], jnp.int32),
                              {kk: v[bb] for kk, v in des.items()},
                              tuple(jnp.asarray(pd(np.asarray(x))) for x in ov), jnp.asarray(pd(ixs)))
                out.append(np.asarray(o)[:n])
            G[c] = np.concatenate(out).reshape(NW, ND, 3, 5)
        res[(name, tau)] = {"G": G, "own": own, "inA": inA, "nA": nA, "cand": cand, "tab": tab, "lam": sn["lam"], "tau": tau}
        print(name, tau, "done", time.time() - t0, flush=True)
        pickle.dump(res, open(outf, "wb"))
