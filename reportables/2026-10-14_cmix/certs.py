"""ccert: are certificates "easy"?  Tiles and rho clamped from valid states, every certificate
channel (cert1..cert4) drawn uniformly at random, then only certificate cells are swept
(colour_gibbs, tau = 1, lam = 20; bin = count, grad = distance).  (diagnostic, no training)

Measured:
  sweeps until valid (all hard factors, original problem);
  certificate statistics per level l (16 worlds): mean A, mean B, mean slack (B - A), and the
  slacks against the tight values (A_tight = full columns, B_tight = occupied columns, lift):
  A_tight - A >= 0 and B - B_tight >= 0;
taken (i) at each world's first valid sweep, (ii) after POST further sweeps (lam = 20),
compared with (ref) an equilibrium reference: from the valid state itself, hard certificate-only
sweeps (violating values excluded), averaged over sweeps REF_BURN..REF_END.
Given tiles and rho, ground's target on certificates is uniform over the valid assignments
(every certificate factor is 0 / inf), so (ref) is the exact conditional's statistics.
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
W, LAM = 128, 20.0
MAXS, POST, REF_BURN, REF_END = 600, 200, 200, 1000


def cert_stats(p, st):
    """{level: [mean A, mean B, mean slack, mean A_tight - A, mean B - B_tight]} over worlds and cells."""
    t = np.asarray(st["t"])
    out = {}
    for l in range(1, p.T + 1):
        c = p.chan[f"cert{l}"]
        v = np.asarray(st[f"cert{l}"])
        A, B = c.views["A"][v], c.views["B"][v]
        h = 1 << l
        b = t.reshape(t.shape[0], t.shape[1] // h, h, t.shape[2] // h, h)
        col = b.sum(2)                                               # (n, hh, ww, h): dirt per column
        At, Bt = (col == h).sum(-1), (col > 0).sum(-1)
        out[l] = [float(A.mean()), float(B.mean()), float((B - A).mean()), float((At - A).mean()), float((B - Bt).mean())]
    return out


def mask_for(p, st, levels):
    n = st["t"].shape[0]
    return {c.name: jnp.full((n,) + st[c.name].shape[1:], c.name.startswith("cert"), bool)
            for c in p.sampled() if c.level in levels}


def main(seed=0):
    p = G.build_ground()
    reg = Region(W)
    names = [c.name for c in p.chans]
    z = np.load("reportables/2026-10-14_cgraded/final_binary_lam20.npz")
    src = {k: z[k].astype(np.int32) for k in names}
    n = src["t"].shape[0]
    for b in range(n):
        assert sum(energy(p, {k: src[k][b] for k in names}, reg).viol.values()) == 0
    levels = list(range(p.T, 0, -1))
    lay0 = Layout(p, W, True)
    lays = {"bin": lay0, "grad": Layout(graded_problem(p, LAM), W, True)}
    st_src = {k: jnp.asarray(v) for k, v in src.items()}
    mask = mask_for(p, st_src, levels)
    res = {"source": cert_stats(p, src)}

    # equilibrium reference: hard certificate-only chain from the valid state
    key = jax.random.PRNGKey(seed)
    st = st_src
    acc = []
    t0 = time.time()
    for s in range(1, REF_END + 1):
        key, sub = jax.random.split(key)
        st = cg.sweep_all(lay0, st, 1.0, None, sub, levels=levels, mask=mask)
        if s >= REF_BURN and s % 10 == 0:
            acc.append(cert_stats(p, st))
    _, N = cg.state_energy(lay0, st)
    assert int(np.asarray(N).sum()) == 0
    res["ref"] = {l: np.mean([a[l] for a in acc], 0).tolist() for l in acc[0]}
    res["ref_sd_over_time"] = {l: np.std([a[l] for a in acc], 0).tolist() for l in acc[0]}
    print("ref", f"{time.time() - t0:.0f}s", flush=True)

    rng = np.random.default_rng(seed)
    init = dict(src)
    for l in range(1, p.T + 1):
        c = p.chan[f"cert{l}"]
        init[c.name] = rng.integers(0, c.D, size=src[c.name].shape).astype(np.int32)
    res["random_init"] = cert_stats(p, init)
    for pen, lay in lays.items():
        st = {k: jnp.asarray(v) for k, v in init.items()}
        key = jax.random.PRNGKey(seed + 1)
        first = np.full(n, -1)
        at_first = {k: np.array(init[k]) for k in names if k.startswith("cert")}
        t0 = time.time()
        s = 0
        while s < MAXS and (first < 0).any():
            s += 1
            key, sub = jax.random.split(key)
            st = cg.sweep_all(lay, st, 1.0, LAM if pen == "bin" else 0.0, sub, levels=levels, mask=mask)
            _, N = cg.state_energy(lay0, st)
            new = (first < 0) & (np.asarray(N) == 0)
            for k in at_first:
                at_first[k][new] = np.asarray(st[k])[new]
            first = np.where(new, s, first)
        ok = first >= 0
        st_first = {**src, **at_first}
        sub_first = {k: v[ok] for k, v in st_first.items()}
        for _ in range(POST):
            key, sub = jax.random.split(key)
            st = cg.sweep_all(lay, st, 1.0, LAM if pen == "bin" else 0.0, sub, levels=levels, mask=mask)
        _, N = cg.state_energy(lay0, st)
        res[pen] = {"first_valid": first.tolist(), "n_valid": int(ok.sum()),
                    "median_first": float(np.median(first[ok])) if ok.any() else None,
                    "at_first": cert_stats(p, sub_first) if ok.any() else None,
                    "post": cert_stats(p, {k: np.asarray(v) for k, v in st.items()}),
                    "post_valid": int((np.asarray(N) == 0).sum()), "secs": time.time() - t0}
        print(pen, f"{time.time() - t0:.0f}s", "valid", int(ok.sum()), "/", n, "median first", res[pen]["median_first"],
              "post valid", res[pen]["post_valid"], flush=True)
    json.dump(res, open(f"{D}/certs.json", "w"), indent=1)
    hdr = "level: [mean A, mean B, slack B-A, A_tight-A, B-B_tight]"
    print(hdr)
    for key_ in ("source", "ref", "random_init"):
        print(key_, {l: np.round(v, 3).tolist() for l, v in res[key_].items()})
    for pen in lays:
        for key_ in ("at_first", "post"):
            if res[pen][key_]:
                print(pen, key_, {l: np.round(v, 3).tolist() for l, v in res[pen][key_].items()})


if __name__ == "__main__":
    main()
