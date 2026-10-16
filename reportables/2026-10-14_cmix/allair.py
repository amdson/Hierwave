"""callair: does certificate-free MCMC alone converge on ground from far starts?  (diagnostic)

Certificate-free ground (mix.stripped), 8 fills x 4 replicate chains = 32 worlds.
Starts:  air    (t = 0, every rho 0: valid)
         dirt   (t = 1, every rho full: valid)
         random (t, rho uniform: invalid; lam = 20 soft sweeps instead of hard)
         probe  (smooth probe surface lifted: near-equilibrium calibration)
Schedules: naive  one sweep of every level per step
           ms     multiscale: 4 sweeps of levels 4, 3, 2, then one full sweep per step
Reference: the long certificate-free chain (trace_strip_hard.npz, second half), per fill seed:
mean surface offset, roughness and soft energy.  Equilibrium time = first full sweep after which
the per-fill means of the 4 chains stay within |d offset| < 0.3 rows, |d rough| < 0.05,
|d S| < 1 % of the reference for the rest of the run.
"""
from __future__ import annotations

import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.kernels import colour_gibbs as cg
from castlegen.bench.problems import ground as G
from castlegen.variants._cpass_core import Layout

sys.path.insert(0, "reportables/2026-10-14_cmix")
sys.path.insert(0, "reportables/2026-10-14_cgraded")
from mix import starts, stripped, W                                 # noqa: E402

D = "reportables/2026-10-14_cmix"
STEPS, REC, LAM = 1500, 5, 20.0


def ref_stats():
    tr = np.load(f"{D}/trace_strip_hard.npz")
    h = tr["h"].astype(float)
    T = h.shape[0]
    fs = tr["fseed"]
    line = np.stack([G.surface_row(int(s), np.arange(W), W) for s in fs])
    off = (h - line[None]).mean(-1)[T // 2:]
    rough = np.abs(np.diff(h, axis=-1, append=h[..., :1])).mean(-1)[T // 2:]
    S = tr["S"][T // 2:]
    return {s: (off[:, fs == s].mean(), rough[:, fs == s].mean(), S[:, fs == s].mean()) for s in np.unique(fs)}


def init_state(kind, ps, base, rng):
    st = {"fill": base["fill"]}
    for c in ps.sampled():
        shp = base[c.name].shape
        full = (1 << c.level) ** 2
        if kind == "air":
            st[c.name] = np.zeros(shp, np.int32)
        elif kind == "dirt":
            st[c.name] = np.full(shp, full if c.name.startswith("rho") else 1, np.int32)
        elif kind == "random":
            st[c.name] = rng.integers(0, c.D, size=shp).astype(np.int32)
        else:
            st[c.name] = np.asarray(base[c.name])
    return {k: jnp.asarray(v) for k, v in st.items()}


def main(arms):
    p, st0, fam, fseed = starts(8)
    ps = stripped(p)
    lay = Layout(ps, W, True)
    base = {k: np.asarray(v) for k, v in st0.items() if k in ps.chan}
    # probe arm = the smooth family (fam 0) replicated 4x per fill, to match the other arms' layout
    sm = np.flatnonzero(fam == 0)
    idx = np.repeat(sm, 4)
    base_probe = {k: v[idx] for k, v in base.items()}
    base_fill = {k: v[idx] for k, v in base.items()}
    fs = fseed[idx]
    line = np.stack([G.surface_row(int(s), np.arange(W), W) for s in fs]).astype(float)
    ref = ref_stats()
    rng = np.random.default_rng(0)
    out = {}
    for arm in arms:
        kind, sched = arm.split("_")
        st = init_state(kind, ps, base_probe if kind == "probe" else base_fill, rng)
        lam = LAM if kind == "random" else None
        key = jax.random.PRNGKey(1)
        rec, t0 = [], time.time()
        for s in range(0, STEPS + 1):
            if s:
                if sched == "ms":
                    for _ in range(4):
                        key, sub = jax.random.split(key)
                        st = cg.sweep_all(lay, st, 1.0, lam, sub, levels=[4, 3, 2])
                key, sub = jax.random.split(key)
                st = cg.sweep_all(lay, st, 1.0, lam, sub)
            if s % REC == 0:
                S, N = cg.state_energy(lay, st)
                t = np.asarray(st["t"])
                h = np.where(t.any(1), t.argmax(1), G.H).astype(float)
                off = (h - line).mean(-1)
                rough = np.abs(np.diff(h, axis=-1, append=h[:, :1])).mean(-1)
                rec.append((s, time.time() - t0, off, rough, np.asarray(S), np.asarray(N)))
        # equilibrium time per fill seed
        steps = np.array([r[0] for r in rec])
        secs = np.array([r[1] for r in rec])
        eq_step, eq_sec = [], []
        for sd in np.unique(fs):
            m = fs == sd
            ro, rr, rS = ref[sd]
            ok = np.array([abs(r[2][m].mean() - ro) < 0.3 and abs(r[3][m].mean() - rr) < 0.05
                           and abs(r[4][m].mean() - rS) < 0.01 * abs(rS) and (r[5][m] == 0).all() for r in rec])
            bad = np.flatnonzero(~ok)
            first = 0 if bad.size == 0 else (bad[-1] + 1 if bad[-1] + 1 < len(ok) else None)
            eq_step.append(None if first is None else int(steps[first]))
            eq_sec.append(None if first is None else float(secs[first]))
        last = rec[-1]
        fin = {k: np.asarray(v) for k, v in st.items()}
        lm = [G.stats({"t": fin["t"][b], "fill": fin["fill"][b], "rho4": fin["rho4"][b]})["line_match"] for b in range(len(fs))]
        v_first = next((r[0] for r in rec if (r[5] == 0).all()), None)
        out[arm] = {"eq_step_per_fill": eq_step, "eq_sec_per_fill": eq_sec,
                    "all_valid_from_step": v_first, "final_valid": int((last[5] == 0).sum()),
                    "final_offset_err": float(np.mean([abs(last[2][fs == sd].mean() - ref[sd][0]) for sd in ref])),
                    "final_rough_err": float(np.mean([abs(last[3][fs == sd].mean() - ref[sd][1]) for sd in ref])),
                    "line_match": float(np.mean(lm)), "secs_total": float(secs[-1]),
                    "curve": [(int(r[0]), float(r[1]), float(r[2].mean()), float(r[3].mean()), float(r[4].mean()), int(r[5].sum()))
                              for r in rec]}
        o = out[arm]
        done = [e for e in eq_step if e is not None]
        print(arm, f"{o['secs_total']:.0f}s", "valid from", o["all_valid_from_step"], "eq steps per fill", eq_step,
              "median", (float(np.median(done)) if len(done) == len(eq_step) else "not all"),
              "eq secs median", (round(float(np.median([e for e in eq_sec if e is not None])), 1) if done else None),
              "line_match", round(o["line_match"], 3), flush=True)
    json.dump(out, open(f"{D}/allair.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1].split(","))
