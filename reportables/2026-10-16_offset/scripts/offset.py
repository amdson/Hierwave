"""offset: implied surface offset per level of the pass draw (raw/final), probes, clamped-coarse draws.
usage: offset.py <label> <params.pkl> <n_worlds> <out.json>"""
import json, pickle, sys, time
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.variant import load_variants
from castlegen.bench.problems import get
from castlegen.bench.problems import ground as G
from castlegen.bench.problem import Region
from castlegen.bench.run import EVAL_SEED0
import castlegen.variants._creplay_clamp as _cc
import re as _re
_src = open(_cc.__file__).read().replace("ACTOR_WIN)", "lay.actor_win)")
_ns = dict(_cc.__dict__); exec(compile(_src, "cc_patched", "exec"), _ns); clamped_pass = _ns["clamped_pass"]
label, path, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
p = get("ground"); reg = Region(128); T = p.T; H = p.H; W = 128
V = load_variants()
ck = pickle.load(open(path, "rb")); tr = ck["trained"] if "trained" in ck else ck
v = V["ckpop_hard_pen"](p, 0, "/tmp/off_" + label)
v.params = jax.tree_util.tree_map(jnp.asarray, tr["params"]); v.lam = np.asarray(tr["lam"], float)
lay = v.layout(reg); params = v.params; lam = jnp.asarray(v.lam, jnp.float32)
seeds = [EVAL_SEED0 + i for i in range(n)]
des = [v.designed(s, reg) for s in seeds]
prs = [G.probes(s, reg, n=4) for s in seeds]
names = [c.name for c in p.sampled()]

def stack(ds, keys=None):
    return {k: jnp.asarray(np.stack([d[k] for d in ds]), jnp.int32) for k in ds[0]}

def measure(xs, gs):
    """xs: list of dict (rho1..4,t), gs: list of g (W,). returns per level offset/c1/rough"""
    res = {}
    for l in range(T + 1):
        h = 1 << l; ds = []
        for x, g in zip(xs, gs):
            r = np.asarray(x["t" if l == 0 else f"rho{l}"], np.float64)
            imp = H - r.sum(0) / h                       # implied surface row per level column
            gs_ = g.reshape(-1, h).mean(1)
            ds.append(imp - gs_)
        ds = np.stack(ds); fz = np.fft.rfft(ds, axis=-1) / ds.shape[-1]
        res[f"L{l}"] = {"off": float(ds.mean()), "off_se": float(ds.mean(1).std(ddof=1) / np.sqrt(len(ds))), "c1": float(np.abs(fz[:, 1]).mean())}
    hs, dd = [], []
    for x, g in zip(xs, gs):
        t = np.asarray(x["t"]).astype(bool)
        hh = np.where(t.any(0), t.argmax(0), H).astype(float); hs.append(hh); dd.append(hh - np.floor(g + .5))
    hs = np.stack(hs); dd = np.stack(dd); fz = np.fft.rfft(dd, axis=-1) / W
    res["first_dirt"] = {"off": float(dd.mean()), "rough": float(np.abs(np.diff(hs, axis=-1, append=hs[:, :1])).mean()), "c1": float(np.abs(fz[:, 1]).mean()),
                         "dirt": float(np.mean([np.asarray(x["t"]).mean() for x in xs]))}
    return res

gs = []
for s, d in zip(seeds, des):
    gs.append(G._line_from_fill(np.asarray(d["fill"], np.int64), W))
R = {"label": label, "n": n}
t0 = time.time()
raws, fins = [], []
for s in seeds:
    g = v.generate(s, reg); raws.append(g["raw"]); fins.append(g["final"])
print("gen", time.time() - t0, flush=True)
R["raw"] = measure(raws, gs); R["final"] = measure(fins, gs)
for j in (0, 2):
    R[f"probe_j{j}"] = measure([pr[j] for pr in prs], gs)
R["fill_designed"] = {"off": float(np.mean([(H - np.asarray(d["fill"]).sum(0)).mean() - g.mean() for d, g in zip(des, gs)]))}
print("raw", json.dumps(R["raw"]), "\nfinal", json.dumps(R["final"]), flush=True)

fns = {}
def run_clamp(cl_names, pj):
    key = tuple(cl_names)
    if key not in fns:
        fns[key] = jax.jit(jax.vmap(lambda s, d, c: clamped_pass(lay, params, s, d, 1.0, lam, c)[0], in_axes=(0, 0, 0)))
    outs = []
    for i in range(0, n, 32):
        sl = slice(i, i + 32)
        cl = {k: jnp.asarray(np.stack([prs[b][pj][k] for b in range(n)][sl]), jnp.int32) for k in cl_names}
        st = fns[key](jnp.asarray(seeds[sl], jnp.int32), {k: v_[sl] for k, v_ in stack(des).items()}, cl)
        outs.append(jax.device_get(st))
    st = {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
    return [{k: st[k][b] for k in st} for b in range(n)]
# sanity: no clamp reproduces raw
unc = run_clamp([], 0)
R["sanity_unclamped_same_as_raw"] = float(np.mean([(unc[b]["t"] == raws[b]["t"]).mean() for b in range(n)]))
R["clamp"] = {}
for pj in (0, 2):
    for L in (4, 3, 2, 1):
        cn = [c.name for c in p.sampled() if c.level >= L]
        xs = run_clamp(cn, pj)
        R["clamp"][f"probe_j{pj}_L{L}"] = measure(xs, gs)
        print("clamp", pj, L, json.dumps(R["clamp"][f"probe_j{pj}_L{L}"]), flush=True)
json.dump(R, open(out, "w"), indent=1)
print("DONE", flush=True)
