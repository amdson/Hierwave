"""Per-level timing of the ctrunc chain (not a bench): python profile.py <problem> [cfg json]."""
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.variants import _ctrunc_core as tc
from castlegen.variants import ctrunc as m

p = get(sys.argv[1])
cfg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
v = m.CTrunc(p, 0, "/tmp", **cfg)
T = p.T
lay = v.layout(Region(p.W_train))
v.nhard = np.ones(T + 1)
t = time.time()
v._build_trunc(lay)
tau, lam = jnp.float32(1.0), jnp.asarray(v.lam, jnp.float32)
extra = jnp.concatenate([jnp.array([1.0]), lam]).astype(jnp.float32)
vs = jnp.ones(T + 1)
for rep in range(3):
    seeds = np.arange(100 * rep, 100 * rep + v.B[T])
    tim = {}
    region = Region(p.W_train)
    desl = [v.designed(int(s), region) for s in seeds]
    st = {c.name: (jnp.stack([d[c.name] for d in desl]) if c.designed is not None
                   else jnp.full((v.B[T],) + lay.hw[c.level], -1, jnp.int32)) for c in p.chans}
    out = {}
    for l in range(T, -1, -1):
        s0 = time.time()
        st_in = {k: a[:v.B[l]] for k, a in st.items()}
        st, recs, C, Vm = lay.fwd[l](v.params, v.vparams, jnp.asarray(seeds[:v.B[l]], jnp.int32), st_in, tau, lam,
                                     extra, vs)
        jax.block_until_ready(C)
        tim[f"fwd{l}"] = time.time() - s0
        out[l] = {"st_in": st_in, "recs": recs, "C": C, "Vm": Vm}
    U = v._downstream(lay, out, 1)
    for l in range(T + 1):
        s0 = time.time()
        G, y = lay.ret[l](out[l]["C"], U[l])
        jax.block_until_ready(G)
        tim[f"ret{l}"] = time.time() - s0
        s0 = time.time()
        f, keys = lay.loss[l]
        r = f({k: v.params[k] for k in keys}, {k: v.cparams[k] for k in keys},
              v.vparams.get(tc.vkey(l)) if l < T else None, out[l]["recs"], G, y, out[l]["st_in"], tau, lam,
              extra, jnp.float32(1), jnp.float32(1))
        jax.block_until_ready(r[3])
        tim[f"loss{l}"] = time.time() - s0
    print(rep, "total", round(time.time() - t, 2), {k: round(x, 3) for k, x in tim.items()}, flush=True)
    t = time.time()
