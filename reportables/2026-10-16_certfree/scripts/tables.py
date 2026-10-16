"""stage table for a trained ccertfree.  usage: tables.py <params.pkl> <n_worlds> <out.json> R1,R2 [F]
Per R: generate n worlds (EVAL_SEED0+i, W=128) with stage_R=R, stage_F=F (default 20); report raw (pass + derived certs) and final:
violations per level (bench energy on ground), valid worlds, line_match, roughness, dirt, offset / circular roughness / |c1|
(2026-10-16_stage2/q2.py observables), tiles changed vs the pass draw.  Also times per world."""
import json, pickle, sys, time
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.variant import load_variants
from castlegen.bench.problems import get
from castlegen.bench.problems import ground as G
from castlegen.bench.problem import Region
from castlegen.bench.run import EVAL_SEED0, _with_designed
from castlegen.bench.energy import energy
path, n, out, Rs = sys.argv[1], int(sys.argv[2]), sys.argv[3], [int(r) for r in sys.argv[4].split(",")]
F = int(sys.argv[5]) if len(sys.argv) > 5 else 20
p = get("ground"); reg = Region(128)
V = load_variants(); ck = pickle.load(open(path, "rb"))
v = V["ccertfree"](p, 0, "/tmp/certfree_tables")
v.params = jax.tree_util.tree_map(jnp.asarray, ck["params"]); v.lam = np.asarray(ck["lam"], float)


def evaluate(xs, ref_t=None):
    pl = np.zeros(p.T + 1, int); nvalid = 0; stats = []; hs = []; ds = []
    for x in xs:
        e = energy(p, x, reg); vl = np.array([e.viol[l] for l in range(p.T + 1)]); pl += vl; nvalid += int(vl.sum() == 0)
        stats.append(p.stats(x))
        t = np.asarray(x["t"]).astype(bool); Hh, Ww = t.shape
        h = np.where(t.any(0), t.argmax(0), Hh).astype(np.float64)
        g = np.floor(G._line_from_fill(np.asarray(x["fill"], np.int64), Ww) + 0.5)
        hs.append(h); ds.append(h - g)
    hs = np.stack(hs); ds = np.stack(ds); fz = np.fft.rfft(ds, axis=-1) / ds.shape[-1]
    r = {"viol": pl.tolist(), "valid": nvalid, "offset": float(ds.mean()),
         "rough_circ": float(np.abs(np.diff(hs, axis=-1, append=hs[:, :1])).mean()), "c1abs": float(np.abs(fz[:, 1]).mean()),
         "stats": {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}}
    if ref_t is not None:
        r["tile_changed"] = float(np.mean([(np.asarray(x["t"]) != ref_t[b]).mean() for b, x in enumerate(xs)]))
    return r


res = {"path": path, "n": n, "F": F, "arms": []}
for R in Rs:
    v.cfg["stage_R"], v.cfg["stage_F"] = R, F
    v.__dict__.pop("_gen_cache", None)
    t0 = time.time(); raws, fins = [], []
    for i in range(n):
        g = v.generate(EVAL_SEED0 + i, reg)
        raws.append(_with_designed(p, g["raw"], EVAL_SEED0 + i, reg)); fins.append(_with_designed(p, g["final"], EVAL_SEED0 + i, reg))
    dt = time.time() - t0
    ref_t = [np.asarray(x["t"]) for x in raws]
    a = {"R": R, "F": F, "gen_s_per_world": dt / n, "raw": evaluate(raws), "final": evaluate(fins, ref_t)}
    res["arms"].append(a)
    print(f"R{R} F{F} {dt/n:.2f}s/world raw viol {a['raw']['viol']} lm {a['raw']['stats']['line_match']:.3f} rough {a['raw']['stats']['roughness']:.2f} off {a['raw']['offset']:+.2f} | "
          f"final viol {a['final']['viol']} valid {a['final']['valid']}/{n} lm {a['final']['stats']['line_match']:.3f} rough {a['final']['stats']['roughness']:.2f} "
          f"circ {a['final']['rough_circ']:.2f} off {a['final']['offset']:+.2f} c1 {a['final']['c1abs']:.2f} chg {a['final']['tile_changed']:.3f}", flush=True)
    json.dump(res, open(out, "w"), indent=1)
print("DONE")
