"""Ablation profile of the vmapped generator: replace one component by a cheap stand-in (outputs meaningless) and time. usage: ablate.py problem W B"""
import sys, time, json
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "reportables/2026-10-14_speed1b")
from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.bench import noise
from castlegen.variants import _cpass_core as core
from castlegen.variants._cpass_core import NCOL, key_of, mlp
import common
pn, W, B = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
p = get(pn); region = Region(W); lay = core.Layout(p, W, True, 0)
params = common.init_params(p, lay, 0, scale=1.0); lam = jnp.full(p.T + 1, 20.0)

def tpass(seed, designed, off=(), only_level=None):
    state = {c.name: (designed[c.name] if c.designed is not None else jnp.full(lay.hw[c.level], -1, jnp.int32)) for c in p.chans}
    for l in lay.levels:
        chans = lay.chans[l]
        if not chans or (only_level is not None and l != only_level): continue
        def body(state, k, l=l, chans=chans):
            Y, X, valid = lay.sites(l, k); h, w = lay.hw[l]
            Xn = jnp.mod(X, w)
            for ci, c in enumerate(chans):
                if "cond" in off:
                    soft = jnp.zeros((Y.shape[0], c.D)); nv = jnp.zeros((Y.shape[0], c.D), jnp.int32); look = nv
                else:
                    soft, nv, look = lay.cond(state, c.name, Y, X)
                v = nv + look
                mask = (v == jnp.min(v, axis=1, keepdims=True)) | ~valid[:, None]
                if "feat" in off:
                    feat = jnp.zeros((Y.shape[0], params[key_of(l, c.name)]["W1"].shape[0]))
                else:
                    feat = lay.features(state, l, Y, X, lay.actor_win)
                b = jnp.zeros_like(soft) if "mlp" in off else mlp(params[key_of(l, c.name)], feat)
                logits = jnp.where(mask, -soft - b, -jnp.inf)
                if "noise" in off:
                    g = jnp.zeros_like(logits)
                else:
                    g = noise.gumbel(jnp, seed, noise.tag_of(c.name), l, Y, Xn, c.D)
                a = jnp.argmax(logits + g, axis=1).astype(jnp.int32)
                Ys = jnp.where(valid, Y, h)
                state = {**state, c.name: state[c.name].at[Ys, X].set(a, mode="drop")}
            return state, None
        state, _ = jax.lax.scan(body, state, jnp.arange(NCOL))
    return state
ss = np.arange(100, 100 + B); ds = [p.designed_values(int(s), region) for s in ss]
dj = {k: jnp.asarray(np.stack([d[k] for d in ds]), jnp.int32) for k in ds[0]}; js = jnp.asarray(ss, jnp.int32)
def timeit(off=(), lev=None):
    f = jax.jit(jax.vmap(lambda s, d: tpass(s, d, off, lev)))
    o = f(js, dj); jax.block_until_ready(o); ts = []
    for _ in range(8):
        t0 = time.time(); o = f(js, dj); jax.block_until_ready(o); ts.append(time.time() - t0)
    return float(np.median(ts))
res = {"full": timeit()}
for off in (("cond",), ("feat",), ("mlp",), ("noise",), ("cond", "feat", "mlp", "noise")):
    res["no_" + "+".join(off)] = timeit(off)
for l in lay.levels:
    res[f"level{l}_only"] = timeit((), l)
print(json.dumps({"problem": pn, "W": W, "B": B, **{k: round(v, 4) for k, v in res.items()}}, indent=1))
