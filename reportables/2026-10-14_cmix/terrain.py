"""cterrain: sampling terrain given certificates.  Certificates at levels >= k clamped (from the 16
valid cgraded states); tiles, every rho and certificates below k free.  k = 1..4, and k = 5
(nothing clamped) as the baseline.  (diagnostic, no training)

(i) mix: from the valid source state, hard single-cell Gibbs on the free cells (stays valid),
    SW_MIX sweeps.  Terrain freedom = per-column temporal sd of the surface height h(x)
    (second half), tiles changed vs the source, autocorrelation time of the surface offset and
    Fourier mode 1 (second half).
(ii) gen: free cells drawn uniformly at random, then lam = 20 sweeps (bin / grad) until valid
    (<= SW_GEN).  Sweeps to valid and how far the valid terrain lands from the source.
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
sys.path.insert(0, "reportables/2026-10-14_cmix")
from run import graded_problem                                      # noqa: E402
from analyse import tau_int                                         # noqa: E402

D = "reportables/2026-10-14_cmix"
W, LAM, SW_MIX, SW_GEN, REC = 128, 20.0, 1000, 600, 5


def heights(t):
    t = np.asarray(t)
    return np.where(t.any(1), t.argmax(1), G.H).astype(np.float64)       # (B, W)


def main(seed=0):
    p = G.build_ground()
    reg = Region(W)
    names = [c.name for c in p.chans]
    z = np.load("reportables/2026-10-14_cgraded/final_binary_lam20.npz")
    src = {k: z[k].astype(np.int32) for k in names}
    n = src["t"].shape[0]
    for b in range(n):
        assert sum(energy(p, {k: src[k][b] for k in names}, reg).viol.values()) == 0
    line = np.stack([G.surface_row(b, np.arange(W), W) for b in range(n)]).astype(np.float64)
    h_src = heights(src["t"])
    lay0 = Layout(p, W, True)
    lays = {"bin": lay0, "grad": Layout(graded_problem(p, LAM), W, True)}
    rng = np.random.default_rng(seed)
    out = []
    for k in (1, 2, 3, 4, 5):
        free = {c.name: (not c.name.startswith("cert")) or c.level < k for c in p.sampled()}
        mask = {c.name: jnp.full((n,) + src[c.name].shape[1:], free[c.name], bool) for c in p.sampled()}
        r = {"k": k, "clamped": [f"cert{l}" for l in range(k, p.T + 1)]}
        # (i) mixing from the valid source
        st = {kk: jnp.asarray(v) for kk, v in src.items()}
        key = jax.random.PRNGKey(seed)
        hs, t0 = [], time.time()
        for s in range(1, SW_MIX + 1):
            key, sub = jax.random.split(key)
            st = cg.sweep_all(lay0, st, 1.0, None, sub, mask=mask)
            if s % REC == 0:
                hs.append(heights(st["t"]))
        hs = np.stack(hs)                                                   # (T, B, W)
        _, N = cg.state_energy(lay0, st)
        half = hs[len(hs) // 2:]
        dev = half - line[None]
        off = dev.mean(-1)
        c1 = (np.fft.rfft(dev, axis=-1)[..., 1] / W).real
        taus_o = [tau_int(off[:, b]) * REC for b in range(n)]
        taus_c = [tau_int(c1[:, b]) * REC for b in range(n)]
        r["mix"] = {"valid_end": int((np.asarray(N) == 0).sum()), "secs": time.time() - t0,
                    "col_sd": float(half.std(0).mean()),
                    "h_change_rms": float(np.sqrt(((hs[-1] - h_src) ** 2).mean())),
                    "tiles_changed": float((np.asarray(st["t"]) != src["t"]).mean()),
                    "tau_offset": float(np.nanmedian(taus_o)), "tau_c1": float(np.nanmedian(taus_c))}
        print("k", k, "mix", {a: (round(b, 4) if isinstance(b, float) else b) for a, b in r["mix"].items()}, flush=True)
        # (ii) generation from random free cells
        init = dict(src)
        for c in p.sampled():
            if free[c.name]:
                init[c.name] = rng.integers(0, c.D, size=src[c.name].shape).astype(np.int32)
        r["gen"] = {}
        for pen, lay in lays.items():
            st = {kk: jnp.asarray(v) for kk, v in init.items()}
            key = jax.random.PRNGKey(seed + 1)
            first = np.full(n, -1)
            h_first = np.zeros((n, W))
            t0 = time.time()
            s = 0
            while s < SW_GEN and (first < 0).any():
                s += 1
                key, sub = jax.random.split(key)
                st = cg.sweep_all(lay, st, 1.0, LAM if pen == "bin" else 0.0, sub, mask=mask)
                if s % REC == 0:
                    _, N = cg.state_energy(lay0, st)
                    new = (first < 0) & (np.asarray(N) == 0)
                    h_first[new] = heights(st["t"])[new]
                    first = np.where(new, s, first)
            _, N = cg.state_energy(lay0, st)
            ok = first >= 0
            g = {"n_valid": int(ok.sum()), "median_first": float(np.median(first[ok])) if ok.any() else None,
                 "mean_N_end": float(np.asarray(N).mean()), "secs": time.time() - t0,
                 "h_vs_source_rms": float(np.sqrt(((h_first[ok] - h_src[ok]) ** 2).mean())) if ok.any() else None}
            r["gen"][pen] = g
            print("k", k, "gen", pen, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in g.items()}, flush=True)
        out.append(r)
    json.dump(out, open(f"{D}/terrain.json", "w"), indent=1)


if __name__ == "__main__":
    main()
