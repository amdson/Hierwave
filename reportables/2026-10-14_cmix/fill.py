"""cfill: terrain moved without certificates, then certificates filled given the terrain.
(diagnostic, no training)

Same 32 starts as mix.py (4 valid families x 8 fills), SWEEPS hard sweeps per arm.
  scheme  hard chain on the certificate-free ground (mix.stripped); every SNAP sweeps the snapshot
          gets random certificates and a fill: FILL_G graded lam = 20 sweeps of certificate cells
          only, then FILL_H hard ones (tiles and rho fixed).  Recorded: valid?, sweeps to first
          valid, certificate slack per level, terrain observables.
  full    hard chain on ground itself; the same quantities every SNAP sweeps.
Compared: second-half means of certificate statistics (level: mean A, mean B, B - A,
A_tight - A, B - B_tight) and terrain observables; fill success; wall clock.
"""
from __future__ import annotations

import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.kernels import colour_gibbs as cg
from castlegen.variants._cpass_core import Layout

sys.path.insert(0, "reportables/2026-10-14_cgraded")
sys.path.insert(0, "reportables/2026-10-14_cmix")
from run import graded_problem                                      # noqa: E402
from mix import starts, stripped, W                                 # noqa: E402
from certs import cert_stats                                        # noqa: E402
from analyse import observables                                     # noqa: E402

D = "reportables/2026-10-14_cmix"
SWEEPS, SNAP, FILL_G, FILL_H, LAM = 1000, 50, 30, 10, 20.0
CHECK = ()                      # extra fill sweeps at which valid-only certificate stats are recorded
ARMS = ("scheme", "full")
OUT = "fill"
if len(sys.argv) > 1:              # v2: long fill, scheme only
    SNAP, FILL_G, FILL_H, CHECK, ARMS, OUT = 100, 100, 200, (100, 150, 300), ("scheme",), "fill_v2"


def terrain_obs(st, fseed):
    t = np.asarray(st["t"])
    h = np.where(t.any(1), t.argmax(1), t.shape[1])[None]
    o, _ = observables({"h": h, "fseed": fseed, "S": np.zeros((1, h.shape[1]))})
    return {k: o[k][0] for k in ("offset", "rough", "c1re")}


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
    res = {}
    for arm in ARMS:
        key = jax.random.PRNGKey(seed)
        st = {k: v for k, v in st0.items() if k in ps.chan} if arm == "scheme" else dict(st0)
        L = lays if arm == "scheme" else lay
        snaps, t_chain, t_fill = [], 0.0, 0.0
        for s in range(1, SWEEPS + 1):
            t0 = time.time()
            key, sub = jax.random.split(key)
            st = cg.sweep_all(L, st, 1.0, None, sub)
            jax.block_until_ready(st["t"])
            t_chain += time.time() - t0
            if s % SNAP:
                continue
            if arm == "scheme":
                t0 = time.time()
                full = dict(st)
                for c in certs:
                    full[c.name] = jnp.asarray(rng.integers(0, c.D, size=(B,) + st0[c.name].shape[1:]).astype(np.int32))
                first = np.full(B, -1)
                chk = {}
                for j in range(1, FILL_G + FILL_H + 1):
                    key, sub = jax.random.split(key)
                    if j <= FILL_G:
                        full = cg.sweep_all(layg, full, 1.0, 0.0, sub, levels=clev, mask=cmask)
                    else:
                        full = cg.sweep_all(lay, full, 1.0, None, sub, levels=clev, mask=cmask)
                    _, N = cg.state_energy(lay, full)
                    N = np.asarray(N)
                    first = np.where((first < 0) & (N == 0), j, first)
                    if j in CHECK:
                        ok_ = N == 0
                        fj = {k: np.asarray(v)[ok_] for k, v in full.items()}
                        chk[j] = cert_stats(p, fj) if ok_.any() else None
                t_fill += time.time() - t0
            else:
                full = st
                _, N = cg.state_energy(lay, full)
                N = np.asarray(N)
                first = np.zeros(B, int)
                chk = {}
            fn = {k: np.asarray(v) for k, v in full.items()}
            snaps.append({"s": s, "valid": (N == 0), "first": first, "cs": cert_stats(p, fn),
                          "obs": terrain_obs(fn, fseed), "chk": chk})
            print(arm, s, "valid", int((N == 0).sum()), "/", B,
                  "fill median" if arm == "scheme" else "", float(np.median(first[first >= 0])) if (first >= 0).any() else None,
                  flush=True)
        late = [x for x in snaps if x["s"] > SWEEPS // 2]
        valid = np.array([x["valid"] for x in snaps])
        firsts = np.concatenate([x["first"] for x in snaps])
        res[arm] = {
            "valid_frac": float(valid.mean()),
            "fill_first_median": float(np.median(firsts[firsts >= 0])) if arm == "scheme" else None,
            "fill_first_max": int(firsts.max()) if arm == "scheme" else None,
            "cert_late": {l: np.mean([x["cs"][l] for x in late], 0).round(4).tolist() for l in late[0]["cs"]},
            "obs_late": {k: [float(np.mean([x["obs"][k][fam == f].mean() for x in late])) for f in range(4)]
                         for k in late[0]["obs"]},
            "secs_chain": t_chain, "secs_fill": t_fill,
            "cert_late_valid_at": {j: {l: np.mean([x["chk"][j][l] for x in late if x["chk"].get(j)], 0).round(4).tolist()
                                       for l in range(1, p.T + 1)} for j in CHECK}}
        print(arm, json.dumps(res[arm]), flush=True)
    json.dump(res, open(f"{D}/{OUT}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
