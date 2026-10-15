"""Certificate fill length: certificate-free terrains (32 chains, snapshots after 200 and 500 hard
sweeps of mix.stripped), random certificates, then up to MAXF lam = 20 sweeps of certificate cells
only (bin / grad).  First valid sweep per world; violations per level of the worlds still invalid."""
from __future__ import annotations

import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench.energy import energy
from castlegen.bench.problem import Region
from castlegen.kernels import colour_gibbs as cg
from castlegen.variants._cpass_core import Layout

sys.path.insert(0, "reportables/2026-10-14_cgraded")
sys.path.insert(0, "reportables/2026-10-14_cmix")
from run import graded_problem                                      # noqa: E402
from mix import starts, stripped, W                                 # noqa: E402

D = "reportables/2026-10-14_cmix"
MAXF, LAM = 400, 20.0


def main(seed=0):
    p, st0, fam, fseed = starts(8)
    B = fam.size
    ps = stripped(p)
    lay, lays, layg = Layout(p, W, True), Layout(ps, W, True), Layout(graded_problem(p, LAM), W, True)
    certs = [c for c in p.sampled() if c.name.startswith("cert")]
    cmask = {c.name: jnp.full((B,) + st0[c.name].shape[1:], c.name.startswith("cert"), bool)
             for c in p.sampled() if c.level >= 1}
    clev = list(range(p.T, 0, -1))
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    st = {k: v for k, v in st0.items() if k in ps.chan}
    out = []
    s = 0
    for snap in (200, 500):
        while s < snap:
            s += 1
            key, sub = jax.random.split(key)
            st = cg.sweep_all(lays, st, 1.0, None, sub)
        init = dict(st)
        for c in certs:
            init[c.name] = jnp.asarray(rng.integers(0, c.D, size=(B,) + st0[c.name].shape[1:]).astype(np.int32))
        for pen, L, lam in (("bin", lay, LAM), ("grad", layg, 0.0)):
            full = dict(init)
            first = np.full(B, -1)
            for j in range(1, MAXF + 1):
                key, sub = jax.random.split(key)
                full = cg.sweep_all(L, full, 1.0, lam, sub, levels=clev, mask=cmask)
                _, N = cg.state_energy(lay, full)
                first = np.where((first < 0) & (np.asarray(N) == 0), j, first)
                if (first >= 0).all():
                    break
            stuck = np.flatnonzero(first < 0)
            pl = np.zeros(p.T + 1, int)
            fn = {k: np.asarray(v) for k, v in full.items()}
            for b in stuck:
                e = energy(p, {k: fn[k][b] for k in fn}, Region(W))
                pl += np.array([e.viol[l] for l in range(p.T + 1)])
            ok = first >= 0
            q = np.percentile(first[ok], [50, 90, 100]).tolist() if ok.any() else None
            r = {"snap": snap, "pen": pen, "n_valid": int(ok.sum()), "B": B, "first_p50_p90_max": q,
                 "stuck_viol_per_level": pl.tolist(), "stuck_families": fam[stuck].tolist()}
            out.append(r)
            print(json.dumps(r), flush=True)
    json.dump(out, open(f"{D}/fill_len.json", "w"), indent=1)


if __name__ == "__main__":
    main()
