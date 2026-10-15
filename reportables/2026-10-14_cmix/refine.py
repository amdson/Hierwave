"""crefine: given valid coarse levels (>= d), how fast does MCMC find a valid refinement of the
levels below, started from uniform-random values?  (diagnostic, no training)

Coarse sources (16 worlds each, ground W = 128, valid by check):
  tight  probe with 3-row jitter, every coarse channel lifted exactly from the tiles
         (A = full columns, B = occupied columns, rho exact: no slack)
  chain  valid states from the cgraded binary lam = 20 chains (final_binary_lam20.npz): certificates
         and rho sampled by the chain, with whatever slack the target gives
Depths d: levels >= d clamped, levels < d drawn uniformly at random and swept (coarse to fine,
colour_gibbs exact single-cell, tau = 1).  Penalties: bin (lam x count) and grad (lam x distance),
lam = 20.  Recorded every REC sweeps: violations of the free levels' factors (original problem).
"""
from __future__ import annotations

import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench.energy import energy
from castlegen.bench.problem import Region
from castlegen.bench.problems import ground as G
from castlegen.kernels import colour_gibbs as cg
from castlegen.variants._cpass_core import Layout

sys.path.insert(0, "reportables/2026-10-14_cgraded")
from run import graded_problem                                      # noqa: E402

D = "reportables/2026-10-14_cmix"
W, REC, LAM = 128, 5, 20.0


def coarse_sources(p):
    reg = Region(W)
    names = [c.name for c in p.chans]
    tight = []
    for seed in range(16):
        tight.append(G.probes(seed, reg, n=4)[3])
    z = np.load("reportables/2026-10-14_cgraded/final_binary_lam20.npz")
    chain = [{k: z[k][b] for k in names} for b in range(z["t"].shape[0])]
    out = {}
    for name, xs in (("tight", tight), ("chain", chain)):
        for x in xs:
            assert sum(energy(p, x, reg).viol.values()) == 0, name
        out[name] = {k: np.stack([x[k] for x in xs]).astype(np.int32) for k in names}
    return out


def run(sweeps=600, seed=0):
    p = G.build_ground()
    lays = {"bin": Layout(p, W, True), "grad": Layout(graded_problem(p, LAM), W, True)}
    lay0 = lays["bin"]
    src = coarse_sources(p)
    rng = np.random.default_rng(seed)
    res = []
    for sname, st0 in src.items():
        B = st0["t"].shape[0]
        for d in (1, 2, 4):
            free = [c for c in p.sampled() if c.level < d]
            init = dict(st0)
            for c in free:
                init[c.name] = rng.integers(0, c.D, size=st0[c.name].shape).astype(np.int32)
            levels = list(range(d - 1, -1, -1))
            for pen, lay in lays.items():
                st = {k: jnp.asarray(v) for k, v in init.items()}
                key = jax.random.PRNGKey(seed)
                t0 = time.time()
                curve, first = [], np.full(B, -1)
                for s in range(0, sweeps + 1):
                    if s:
                        key, sub = jax.random.split(key)
                        st = cg.sweep_all(lay, st, 1.0, LAM if pen == "bin" else 0.0, sub,
                                          levels=levels)
                    if s % REC == 0:
                        _, N = cg.state_energy(lay0, st)
                        N = np.asarray(N)
                        first = np.where((first < 0) & (N == 0), s, first)
                        curve.append([s, float(N.mean()), int((N == 0).sum())])
                        if (first >= 0).all() and s >= 50:
                            break
                fin = {k: np.asarray(v) for k, v in st.items()}
                # final per-level violations and how far the refinement is from the source tiles
                pl = np.zeros(p.T + 1)
                for b in range(B):
                    e = energy(p, {k: fin[k][b] for k in fin}, Region(W))
                    pl += np.array([e.viol[l] for l in range(p.T + 1)])
                ham = float((fin["t"] != st0["t"]).mean())
                r = {"source": sname, "depth": d, "pen": pen, "B": B, "secs": time.time() - t0,
                     "first_valid": first.tolist(), "n_valid": int((first >= 0).sum()),
                     "median_first": float(np.median(first[first >= 0])) if (first >= 0).any() else None,
                     "final_viol_per_level": pl.tolist(), "tile_diff_vs_source": ham, "curve": curve}
                res.append(r)
                print(sname, "d", d, pen, f"{r['secs']:.0f}s", "valid", r["n_valid"], "/", B,
                      "median first", r["median_first"], "final viol/level", pl.astype(int).tolist(),
                      "tiles differ", round(ham, 3), flush=True)
    json.dump(res, open(f"{D}/refine.json", "w"), indent=1)


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 600)
