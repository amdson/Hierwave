"""Exact block-Gibbs refined contexts on ground (Sweep 1 (a)).
Contexts: probe lifts (jitter 0-3, uniform) and a share of own draws of the
cstart2 LN+SiLU 6x6 model (its own full-depth draws, tau = 1).  Each context
gets n_sweeps cycles of [level-2 single-cell exact Gibbs (rho2, cert2),
level-1 blocks (rho1, cert1, 4 tiles)], hard target (violating completions
excluded); levels >= 3 stay at the context.  States after every cycle are saved.
usage: make_data.py <out.npz> <n_worlds> <own_frac> <n_sweeps> <seed0> <nproc>"""
import sys, time
import numpy as np

from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.bench import energy as en
from castlegen.kernels.exact_block import ExactBlockGibbs, total_energy

out, NW, OWN, NS, S0, NP = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
p = get("ground"); reg = Region(p.W_train)
_K = {}


def kern():
    if not _K:
        _K["a"] = [ExactBlockGibbs(p, reg, level=2, depth=0), ExactBlockGibbs(p, reg, level=1, depth=1)]
    return _K["a"]


def refine(args):
    i, x = args
    Ks = kern()
    rng = np.random.default_rng(10_000 + i)
    v = Ks[0].flat.flatten(x)
    e0 = total_energy(Ks[0].flat, Ks[0].insts, v)
    outs, info = [], {"E0": e0}
    for s in range(NS):
        ch = [K.sweep(v, rng) for K in Ks]
        outs.append(v.copy())
        info[f"changed{s}"] = ch
    info["E"] = total_energy(Ks[0].flat, Ks[0].insts, v)
    return outs, info


if __name__ == "__main__":
    t0 = time.time()
    seeds = np.arange(S0, S0 + NW)
    rng = np.random.default_rng(S0)
    n_own = int(round(OWN * NW))
    ctx = []
    src = []
    for b, s in enumerate(seeds[n_own:]):
        k = int(rng.integers(4))
        x = {**p.probes(int(s), reg, n=k + 1)[k], **p.designed_values(int(s), reg)}
        ctx.append(x); src.append(k)
    if n_own:
        import jax, jax.numpy as jnp, pickle
        from castlegen.variants import _cpass_core as core
        from castlegen.variants._cfast_core import train_pass
        from castlegen.variants._cln_core import window_of
        ck = pickle.load(open("reportables/2026-10-14_cstart2/runs/cstart_f1L3_ln6/params_ground.pkl", "rb"))
        lay = core.Layout(p, reg.W, True, 0, win=window_of((-2, 3)))
        prm = jax.tree_util.tree_map(jnp.asarray, ck["trained"]["params"])
        lam = jnp.asarray(ck["trained"]["lam"], jnp.float32)
        f = jax.jit(lambda s, d: train_pass(lay, prm, s, d, jnp.float32(1.0), lam, record=False)[0])
        for s in seeds[:n_own]:
            d = p.designed_values(int(s), reg)
            st = f(jnp.int32(s), {k: jnp.asarray(v, jnp.int32) for k, v in d.items()})
            x = {**{k: np.asarray(v) for k, v in st.items()}, **d}
            E = en.energy(p, x, reg)
            ctx.append(x); src.append(-1)
        print("own draws", n_own, time.time() - t0, flush=True)
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(NP) as pool:
        res = []
        for r in pool.imap(refine, list(enumerate(ctx)), chunksize=1):   # in order; partial saves
            res.append(r)
            if len(res) % 32 == 0:
                np.savez_compressed(out + ".partial.npz", states=np.array([q[0] for q in res]),
                                    src=np.array(src[:len(res)]))
                print("refined", len(res), time.time() - t0, flush=True)
    K = kern()[0]
    states = np.array([[o for o in r[0]] for r in res])          # (N, NS, n)
    info = [r[1] for r in res]
    np.savez_compressed(out, states=states, src=np.array(src), seeds=np.concatenate([seeds[n_own:], seeds[:n_own]]),
                        E0=np.array([i["E0"] for i in info]), E=np.array([i["E"] for i in info]),
                        changed=np.array([[i[f"changed{s}"] for s in range(NS)] for i in info]))
    ch = np.array([[i[f"changed{s}"] for s in range(NS)] for i in info])
    print("done", len(ctx), "worlds", time.time() - t0, "s; changed blocks per sweep (l2, l1) mean", ch.mean(0).tolist(),
          "E0", np.mean([i["E0"][0] for i in info]), "E", np.mean([i["E"][0] for i in info]),
          "viol0", np.mean([i["E0"][1] for i in info]), "viol", np.mean([i["E"][1] for i in info]), flush=True)
