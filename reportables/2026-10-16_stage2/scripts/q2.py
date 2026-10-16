"""stage Q1 (copy of p10.py + chain observables: cmix offset / circular roughness / Fourier modes)
P10 (cmix proposals): pass with certificates, then delete-and-resample the levels below d.
usage: p10.py <label> <variant|greedy> <params.pkl|-> <n_worlds> <out.json>

Input: the pass draw ('raw', unrepaired) of n held-out eval worlds (seeds EVAL_SEED0+i, W=128, GPU).
Arms (all fixed-count, lam = 20 graded penalty in every sweep, tau = 1, colour_gibbs single-cell):
  v1  ("certificates derived"): levels >= d clamped; R sweeps of levels d-1..0 on ALL channels of the
      full ground problem (rho, cert, t) from the pass draw (row honour of level d stays active because
      cert_{d-1} is carried); then certs < d are deleted (uniform random) and refilled by F
      certificate-only sweeps of levels d-1..1.
  plain (the literal pipeline): certs < d deleted; R sweeps of rho/t below d on the problem with every
      factor touching a deleted channel removed (honour_d dropped; support0 kept); then F
      certificate-only refill as above.  Terrain init = the pass draw in both (del: random terrain).
Measured: violations per level (full ground, host energy), worlds valid, first valid refill sweep,
tile cells changed vs the pass draw, ground stats.
"""
import json, pickle, sys, time
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.variant import load_variants
from castlegen.bench.problems import get
from castlegen.bench.problems import ground as G
from castlegen.bench.problem import Region, Problem
from castlegen.bench.run import EVAL_SEED0, _with_designed
from castlegen.bench.energy import energy
from castlegen.kernels import colour_gibbs as cg
from castlegen.variants._cpass_core import Layout
sys.path.insert(0, "reportables/2026-10-14_cgraded")
from run import graded_problem  # noqa: E402

label, vname, path, n, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
LAM, W = 20.0, 128
p = get("ground"); reg = Region(W)
names = [c.name for c in p.chans]

# ---------------------------------------------------------------- pass draws
t0 = time.time()
raws, finals = [], []
if vname == "greedy":
    from castlegen.variants.greedy import coloured_pass as greedy_pass
    for i in range(n):
        x = greedy_pass(p, EVAL_SEED0 + i, reg)[0]
        raws.append(_with_designed(p, x, EVAL_SEED0 + i, reg)); finals.append(None)
else:
    V = load_variants()
    ck = pickle.load(open(path, "rb")); tr = ck["trained"] if "trained" in ck else ck
    v = V[vname](p, 0, "/tmp/p10_" + label)
    v.params = jax.tree_util.tree_map(jnp.asarray, tr["params"]); v.lam = np.asarray(tr["lam"], float)
    for i in range(n):
        g = v.generate(EVAL_SEED0 + i, reg)
        raws.append(_with_designed(p, g["raw"], EVAL_SEED0 + i, reg))
        finals.append(_with_designed(p, g["final"], EVAL_SEED0 + i, reg))
gen_s = time.time() - t0
print(f"{label}: {n} pass draws in {gen_s:.1f} s", flush=True)
st0 = {k: np.stack([np.asarray(x[k]) for x in raws]).astype(np.int32) for k in names}


def evaluate(xs, ref_t=None):
    pl = np.zeros(p.T + 1, int); nvalid = 0; stats = []; bad = {}
    for b, x in enumerate(xs):
        e = energy(p, x, reg); vl = np.array([e.viol[l] for l in range(p.T + 1)])
        if vl.sum():
            from castlegen import graded as gr
            bad[b] = [vl.tolist(), {k: vv for k, (vv, m, _) in gr.graded(p, x, reg).per_factor.items() if vv}]
        pl += vl; nvalid += int(vl.sum() == 0); stats.append(p.stats(x))
    hs = []; ds = []
    for x in xs:
        t = np.asarray(x["t"]).astype(bool); Hh, Ww = t.shape
        h = np.where(t.any(0), t.argmax(0), Hh).astype(np.float64)
        g = np.floor(G._line_from_fill(np.asarray(x["fill"], np.int64), Ww) + 0.5)
        hs.append(h); ds.append(h - g)
    hs = np.stack(hs); ds = np.stack(ds); fz = np.fft.rfft(ds, axis=-1) / ds.shape[-1]
    chain = {"offset": float(ds.mean()), "rough_circ": float(np.abs(np.diff(hs, axis=-1, append=hs[:, :1])).mean()),
             "c1re": float(fz[:, 1].real.mean()), "c1im": float(fz[:, 1].imag.mean()), "c1abs": float(np.abs(fz[:, 1]).mean()),
             "c2abs": float(np.abs(fz[:, 2]).mean())}
    r = {"viol": pl.tolist(), "chain_obs": chain, "nhard": [energy(p, xs[0], reg).nhard[l] for l in range(p.T + 1)],
         "valid": nvalid, "bad_worlds": bad, "stats": {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}}
    if ref_t is not None:
        r["tile_changed"] = float(np.mean([(np.asarray(x["t"]) != ref_t[b]).mean() for b, x in enumerate(xs)]))
    return r


def unstack(st):
    a = {k: np.asarray(v) for k, v in st.items()}
    return [{k: a[k][b] for k in a} for b in range(n)]


res = {"label": label, "variant": vname, "n": n, "gen_s": gen_s,
       "pass_raw": evaluate(raws), "pass_final": evaluate(finals) if finals[0] is not None else None}
# probe reference (pr[0]: the designed line with jitter, every channel lifted)
res["probe_ref"] = evaluate([_with_designed(p, G.probes(EVAL_SEED0 + i, reg, n=4)[0], EVAL_SEED0 + i, reg) for i in range(n)])
print("pass raw", res["pass_raw"]["viol"], res["pass_raw"]["chain_obs"], "probe", res["probe_ref"]["chain_obs"], res["pass_raw"]["bad_worlds"], res["pass_raw"]["stats"], "\npass final", res["pass_final"], "\nprobe", res["probe_ref"]["stats"], flush=True)

layg = Layout(graded_problem(p, LAM), W, True)
lay0 = Layout(p, W, True)


def plain_problem(d):
    gone = {f"cert{l}" for l in range(1, d)}
    chans = [c for c in p.chans if c.name not in gone]
    fs = [f for f in p.factors if f.a[0] not in gone and f.b[0] not in gone
          and not (f.kind == "count" and f.a[0] == "cert0")]
    # honour_d reads cert_{d-1} (deleted; for d = 1 it reads tiles, kept)
    fs = [f for f in fs if not (f.name.startswith("honour") and f.b[0] == f"cert{d}" and d > 1)]
    return Problem(name=f"ground_plain{d}", T=p.T, H=p.H, chans=chans, factors=fs, W_train=W, W_eval=W, stats=p.stats)


_lays = {}
_B = {}


def _blocks():
    if not _B:
        from castlegen.kernels.amortised import JaxBlockGibbs
        from castlegen.kernels.fine_block import FineBlockKernel
        _B["J1"] = JaxBlockGibbs(p, reg, 1, 1, lam=LAM)
        _B["F2"] = FineBlockKernel(p, reg, k=2, lam=LAM)
    return _B


from castlegen.kernels.exact_block import Flat  # noqa: E402
flat = Flat(p, reg)


def to_v(st):
    parts = [jnp.asarray(st[c.name]).reshape(n, -1) for c in sorted(p.chans, key=lambda c: flat.off[c.name])]
    return jnp.concatenate(parts, 1).astype(jnp.int32)


def to_st(v):
    out = {}
    for c in p.chans:
        o = flat.off[c.name]; h, w = flat.shape[c.name]
        out[c.name] = v[:, o:o + h * w].reshape(n, h, w)
    return out


def arm(kind, d, R, F, seed=0, delete_terrain=False):
    rng = np.random.default_rng(1000 * d + R + F)
    key = jax.random.PRNGKey(seed)
    st = {k: jnp.asarray(v) for k, v in st0.items()}
    lev = list(range(d - 1, -1, -1))
    certs = [c for c in p.sampled() if c.name.startswith("cert") and c.level < d]
    if delete_terrain == 1:      # random start: every sampled channel uniform (cmix section 7 recipe)
        for c in p.sampled() if True else []:
            if c.level < d:
                st[c.name] = jnp.asarray(rng.integers(0, c.D, size=st0[c.name].shape).astype(np.int32))
    elif delete_terrain == 2:    # all-dirt start: t = 1 everywhere, every channel lifted (valid)
        lf = G.lift(np.ones(st0["t"].shape[1:], np.int64))
        for c in p.sampled():
            st[c.name] = jnp.asarray(np.broadcast_to(lf[c.name].astype(np.int32), st0[c.name].shape))
    t1 = time.time()
    # stage 1: terrain resample
    if kind == "v1":
        L = layg
        for _ in range(R):
            key, sub = jax.random.split(key)
            st = cg.sweep_all(L, st, 1.0, 0.0, sub, levels=lev)
    elif kind in ("v1b", "v2", "v1bt"):
        # v2: as v1b at d = T + 1 but the top terrain (rho_T) stays clamped: only certificates are planning-only
        # p5's block kernels (binary lam = 20, exact): per round one single-cell graded sweep of levels < d,
        # one parent+4-children block sweep at level 1 (JaxBlockGibbs, depth 1), one 2x2 tile block sweep
        # (FineBlockKernel k = 2; host-rng offset, diagnostic only)
        blk = _blocks()
        for _ in range(R):
            key, s1_, s2_, s3_ = jax.random.split(key, 4)
            m = None if kind in ("v1b", "v1bt") else {c.name: jnp.full((n,) + st0[c.name].shape[1:], c.name != f"rho{p.T}", bool)
                                             for c in p.sampled() if c.level in lev}
            st = cg.sweep_all(layg, st, 1.0, 0.0, s1_, levels=lev, mask=m)
            v = to_v(st)
            v, _ = blk["J1"].sweep(v, s2_)
            v, _ = blk["F2"].sweep(v, s3_)
            st = to_st(v)
    else:
        if ("plain", d) not in _lays:
            _lays[("plain", d)] = Layout(graded_problem(plain_problem(d), LAM), W, True)
        L = _lays[("plain", d)]
        sp = {k: v for k, v in st.items() if k in L.p.chan}
        for _ in range(R):
            key, sub = jax.random.split(key)
            sp = cg.sweep_all(L, sp, 1.0, 0.0, sub, levels=lev)
        st = {**st, **sp}
    jax.block_until_ready(st["t"])
    s1 = time.time() - t1
    pre = evaluate(unstack(st), st0["t"])
    # stage 2: delete certs < d, refill by F certificate-only sweeps
    if kind == "v1bt":   # diagnostic (p4fill): tight-abstraction certificate start, ground's lift (problem-specific)
        tn = np.asarray(st["t"]); lifted = [G.lift(tn[b]) for b in range(n)]
        for c in certs:
            st[c.name] = jnp.asarray(np.stack([l_[c.name] for l_ in lifted]).astype(np.int32))
    else:
        for c in certs:
            st[c.name] = jnp.asarray(rng.integers(0, c.D, size=st0[c.name].shape).astype(np.int32))
    first = np.full(n, -1)
    if certs:
        cmask = {c.name: jnp.full((n,) + st0[c.name].shape[1:], c.name.startswith("cert"), bool)
                 for c in p.sampled() if 1 <= c.level < d}
        clev = list(range(d - 1, 0, -1))
        for j in range(1, F + 1):
            key, sub = jax.random.split(key)
            st = cg.sweep_all(layg, st, 1.0, 0.0, sub, levels=clev, mask=cmask)
            _, N = cg.state_energy(lay0, st)
            first = np.where((first < 0) & (np.asarray(N) == 0), j, first)
    jax.block_until_ready(st["t"])
    s2 = time.time() - t1 - s1
    r = {"kind": kind, "d": d, "R": R, "F": F, "init": delete_terrain, "stage1_s": s1, "stage2_s": s2,
         "after_stage1": pre, "final": evaluate(unstack(st), st0["t"]), "first_valid": first.tolist()}
    fv = first[first > 0]
    print(f"[{kind} d{d} R{R} F{F}{' init%d' % delete_terrain}] {s1:.1f}+{s2:.1f}s  stage1 viol {pre['viol']} "
          f"| final viol {r['final']['viol']} valid {r['final']['valid']}/{n} tiles chg {r['final']['tile_changed']:.4f} "
          f"lm {r['final']['stats']['line_match']:.3f} rough {r['final']['stats']['roughness']:.3f} "
          f"float {r['final']['stats']['floating']:.2f} dirt {r['final']['stats']['dirt_frac']:.3f} "
          f"chain {r['final']['chain_obs']} fill p50/max {np.median(fv) if fv.size else None}/{fv.max() if fv.size else None} bad {r['final']['bad_worlds']}", flush=True)
    return r


plan = [("v1", 1, 10, 0), ("v1", 2, 10, 40), ("v1", 3, 10, 40), ("v1", 3, 30, 80), ("v1", 2, 30, 80),
        ("plain", 2, 30, 80), ("plain", 3, 30, 80)]
if len(sys.argv) > 6:
    plan = [tuple([a[0]] + [int(x) for x in a[1:]]) for a in (s.split(":") for s in sys.argv[6].split(","))]
res["arms"] = []
for a in plan:
    res["arms"].append(arm(*a[:4], delete_terrain=a[4] if len(a) > 4 else 0))
    json.dump(res, open(out, "w"), indent=1)
print("DONE", flush=True)
