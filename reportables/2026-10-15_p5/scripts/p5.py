"""P5: joint block Gibbs vs single-cell for refining clamped coarse levels (cmix sec 2 setup).
usage: p5.py SOURCE ARM SWEEPS [NW]   SOURCE tight|chain ; ARM s1|f2|f4|s2|b11 (see ARMS)
"""
import sys, time, json
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, "reportables/2026-10-14_cmix"); sys.path.insert(0, "reportables/2026-10-14_cgraded")
from refine import coarse_sources, W, LAM
from castlegen.bench.problems import ground as G
from castlegen.bench.problem import Region
from castlegen.kernels import colour_gibbs as cg
from castlegen.kernels.exact_block import Flat
from castlegen.kernels.amortised import JaxBlockGibbs
from castlegen.kernels.fine_block import FineBlockKernel
from castlegen.variants._cpass_core import Layout

src, arm, sweeps = sys.argv[1], sys.argv[2], int(sys.argv[3])
NW = int(sys.argv[4]) if len(sys.argv) > 4 else 32
CK = [5, 10, 20, 40, 100, 200]
p = G.build_ground(); reg = Region(W); lay = Layout(p, W, True)
# d: levels < d free.  s1,f2,f4: d=1 (tiles only); s2,b11: d=2 (levels 0,1 free)
d = 1 if arm in ("s1", "f2", "f4") else 2
# 32 worlds: tight probes seeds 0..31; chain has 16 states -> tile twice? use what exists
import refine as R
names = [c.name for c in p.chans]
if src == "tight":
    xs = [G.probes(s, reg, n=4)[3] for s in range(NW)]
else:
    z = np.load("reportables/2026-10-14_cgraded/final_binary_lam20.npz")
    xs = [{k: z[k][b] for k in names} for b in range(z["t"].shape[0])]
st0 = {k: np.stack([x[k] for x in xs]).astype(np.int32) for k in names}
B = st0["t"].shape[0]
rng = np.random.default_rng(0)
init = dict(st0)
for c in p.sampled():
    if c.level < d: init[c.name] = rng.integers(0, c.D, size=st0[c.name].shape).astype(np.int32)
flat = Flat(p, reg)
def to_v(st):
    v = np.zeros((B, flat.n), np.int32)
    for c in p.chans:
        o = flat.off[c.name]; v[:, o:o + int(np.prod(flat.shape[c.name]))] = np.asarray(st[c.name]).reshape(B, -1)
    return jnp.asarray(v)
def to_st(v):
    v = np.asarray(v); out = {}
    for c in p.chans:
        o = flat.off[c.name]; h, w = flat.shape[c.name]
        out[c.name] = jnp.asarray(v[:, o:o + h * w].reshape(B, h, w))
    return out
if arm in ("s1", "s2"):
    levels = list(range(d - 1, -1, -1))
    step = lambda st, key: cg.sweep_all(lay, st, 1.0, LAM, key, levels=levels)
    cur = {k: jnp.asarray(v) for k, v in init.items()}
    get = lambda s: s; 
else:
    if arm in ("f2", "f4"):
        K = FineBlockKernel(p, reg, k=int(arm[1]), lam=LAM)
        sw = K.sweep
    else:
        K = JaxBlockGibbs(p, reg, 1, 1, lam=LAM)
        sw = lambda v, key: K.sweep(v, key)
    cur = to_v(init)
    def step(v, key):
        v, _ = sw(v, key); return v
    get = to_st
lay0 = lay
def nviol(s):
    _, N = cg.state_energy(lay0, get(s)); return np.asarray(N)
key = jax.random.PRNGKey(0)
# warm compile on a throwaway copy
for _i in range(10): w = step(cur, jax.random.PRNGKey(99+_i))
jax.block_until_ready(w if not isinstance(w, dict) else w["t"]); nviol(w)
first = np.full(B, -1); curve = []; tsw = 0.0
N0 = nviol(cur); first = np.where(N0 == 0, 0, first)
for s in range(1, sweeps + 1):
    key, sub = jax.random.split(key)
    t0 = time.time(); cur = step(cur, sub)
    jax.block_until_ready(cur if not isinstance(cur, dict) else cur["t"]); tsw += time.time() - t0
    N = nviol(cur); first = np.where((first < 0) & (N == 0), s, first)
    if s in CK or s % 10 == 0:
        curve.append([s, float(N.mean()), int((N == 0).sum())])
    if (first >= 0).all(): break
fin = get(cur)
fin = {k: np.asarray(v) for k, v in fin.items()}
nv_at = {k: int(((first >= 0) & (first <= k)).sum()) for k in CK}
r = dict(source=src, arm=arm, d=d, B=B, sweeps_run=s, sec_per_sweep=tsw / s, valid_at=nv_at,
         n_valid_final=int((N == 0).sum()), median_first=float(np.median(first[first >= 0])) if (first >= 0).any() else None,
         tile_changed_frac=float((fin["t"] != st0["t"]).mean()), mean_viol_final=float(N.mean()),
         first=first.tolist(), curve=curve)
print(json.dumps({k: v for k, v in r.items() if k not in ("first",)}), flush=True)
json.dump(r, open(f"reportables/2026-10-15_p5/runs/{src}_{arm}.json", "w"))
