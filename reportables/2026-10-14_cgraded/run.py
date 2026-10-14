"""cgraded: MCMC on ground with a graded vs a binary hard-constraint penalty (ideas.md #21).

Both arms run castlegen.kernels.colour_gibbs (exact single-cell Gibbs, colour classes,
levels coarse to fine) at tau = 1 from the same greedy-pass starts.
  binary:  E = S + lam * (number of violated hard instances)
  graded:  E = S + lam * sum_f d_f, d_f = L1 distance in table-index space from the
           instance's entry to the nearest finite entry of its table (0 iff satisfied).
           Generic: built from the tables, no per-problem code.  On ground: support
           gives B - A, row honour gives the column shortfall / excess.
Violations are always counted on the original problem.  Diagnostic only: no training,
the objective is untouched.
"""
from __future__ import annotations

import copy
import json
import sys
import time

import jax
import numpy as np

from castlegen.bench.energy import energy
from castlegen.bench.problem import Region
from castlegen.bench.problems.ground import build_ground
from castlegen.kernels import colour_gibbs as cg
from castlegen.variants._cpass_core import Layout
from castlegen.variants.greedy import coloured_pass as greedy_pass


def graded_table(t):
    """L1 index distance of every entry to the nearest finite entry (0 on finite entries)."""
    fin = np.argwhere(np.isfinite(t))
    idx = np.indices(t.shape).reshape(t.ndim, -1).T
    d = np.abs(idx[:, None, :] - fin[None, :, :]).sum(-1).min(1)
    return d.reshape(t.shape).astype(np.float64)


def graded_problem(p, lam):
    q = copy.deepcopy(p)
    for f in q.factors:
        if f.hard:
            inf = np.isinf(f.table)
            f.table = np.where(inf, lam * graded_table(f.table), f.table)
    return q


def main(lam=20.0, B=8, sweeps=60, W=128, seed0=0):
    p = build_ground()
    reg = Region(W)
    # same starts for both arms: greedy (untrained) pass draws
    starts = [greedy_pass(p, seed0 + b, reg)[0] for b in range(B)]
    names = [c.name for c in p.chans]
    st0 = {k: np.stack([s[k] for s in starts]).astype(np.int32) for k in names}

    def viol(st):
        out = []
        for b in range(B):
            e = energy(p, {k: np.asarray(st[k][b]) for k in names}, reg)
            out.append([e.viol[l] for l in range(p.T + 1)] + [e.soft])
        return np.array(out)                                  # (B, T + 2)

    arms = {"binary": (Layout(p, W, True), lam), "graded": (Layout(graded_problem(p, lam), W, True), 0.0)}
    res = {"lam": lam, "B": B, "sweeps": sweeps, "start": viol(st0).tolist(), "arms": {}}
    print("start viol/level", res["start"] and np.array(res["start"])[:, :-1].sum(0).tolist(), flush=True)
    for name, (lay, lam_arm) in arms.items():
        st = {k: jax.numpy.asarray(v) for k, v in st0.items()}
        key = jax.random.PRNGKey(seed0)
        trace, t0 = [], time.time()
        for s in range(1, sweeps + 1):
            key, sub = jax.random.split(key)
            st = cg.sweep_all(lay, st, 1.0, lam_arm, sub)
            if s in (1, 2, 5, 10, 20, 50) or s % 25 == 0 or s == sweeps:
                v = viol(st)
                trace.append({"sweep": s, "t": time.time() - t0, "viol": v.tolist()})
                print(name, s, f"{time.time() - t0:.0f}s", "viol/level", v[:, :-1].sum(0).tolist(),
                      "valid worlds", int((v[:, :-1].sum(1) == 0).sum()), flush=True)
        final = {k: np.asarray(v) for k, v in st.items()}
        stats = [p.stats({k: final[k][b] for k in names}) for b in range(B)]
        res["arms"][name] = {"trace": trace, "stats": stats}
        np.savez_compressed(f"reportables/2026-10-14_cgraded/final_{name}_lam{lam:g}.npz", **final)
    with open(f"reportables/2026-10-14_cgraded/result_lam{lam:g}.json", "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main(**{k: type_(v) for k, v, type_ in
            [(a.split("=")[0], a.split("=")[1], float if a.startswith("lam") else int) for a in sys.argv[1:]]})
