"""cmix: how well does MCMC move between valid terrains on ground?  (diagnostic, no training)

Worlds.  NF fill seeds x 4 start families = B chains, batched.  Families (all valid, exact lift):
  0 smooth   dirt below the designed line          (probe, jitter 0)
  1 ragged   designed line + 3-row jitter          (probe, jitter 3)
  2 up       designed line shifted 6 rows up
  3 down     designed line shifted 6 rows down
Arms (colour_gibbs exact single-cell sweeps, tau = 1; penalty "bin" = lam x violated count,
"grad" = lam x sum of distances to feasibility, castlegen/graded.py's d on ground):
  hard              violating values excluded (stays valid)
  {bin,grad}_{lam}  fixed soft penalty
  cyc_{bin,grad}    annealing cycles: lam ladder 20 -> 1 -> 20 then 30 sweeps at lam 20, repeated
  pt_{bin,grad}     replica stack on LADDER, one sweep per replica then neighbour swaps
                    (alternating even / odd pairs), log a = (lam_i - lam_j)(M(x_i) - M(x_j)),
                    M = violated count (bin) or sum of distances (grad); the lam = 20 replica is observed
Recorded every REC sweeps per chain: column heights h(x) (first dirt row; H if none), soft energy S,
violations N (original problem), graded mass M.
"""
from __future__ import annotations

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
W = 128
REC = 5
LADDER = [20.0, 12.0, 10.0, 8.0, 7.0, 6.0]
FAM = ["smooth", "ragged", "up", "down"]


def starts(nf):
    p = G.build_ground()
    reg = Region(W)
    xs, fam, fseed = [], [], []
    Xf = np.arange(W)
    s = 1 << G.T
    for seed in range(nf):
        pr = G.probes(seed, reg, n=4)
        fill = pr[0]["fill"]
        sr = G.surface_row(seed, Xf, W)
        for k, x in enumerate([pr[0], pr[3], None, None]):
            if x is None:
                sh = -6 if k == 2 else 6
                t = (np.arange(G.H)[:, None] >= np.clip(sr + sh, 0, G.H)[None, :]).astype(np.int64)
                x = G.lift(t)
                x["fill"] = fill.copy()
            e = energy(p, x, reg)
            assert sum(e.viol.values()) == 0, (seed, k)
            xs.append(x)
            fam.append(k)
            fseed.append(seed)
    names = [c.name for c in p.chans]
    st = {k: jnp.asarray(np.stack([x[k] for x in xs]).astype(np.int32)) for k in names}
    return p, st, np.array(fam), np.array(fseed)


class Arms:
    def __init__(self, p):
        self.p = p
        self.lay = Layout(p, W, True)
        self.g1 = Layout(graded_problem(p, 1.0), W, True)          # energy only: M = S_g1 - S
        self._g = {}

    def glay(self, lam):
        if lam not in self._g:
            self._g[lam] = Layout(graded_problem(self.p, lam), W, True)
        return self._g[lam]

    def sweep(self, st, kind, lam, key):
        if kind == "hard":
            return cg.sweep_all(self.lay, st, 1.0, None, key)
        if kind == "bin":
            return cg.sweep_all(self.lay, st, 1.0, float(lam), key)
        return cg.sweep_all(self.glay(float(lam)), st, 1.0, 0.0, key)

    def obs(self, st):
        S, N = cg.state_energy(self.lay, st)
        Sg, _ = cg.state_energy(self.g1, st)
        t = np.asarray(st["t"])
        has = t.any(1)
        h = np.where(has, t.argmax(1), G.H).astype(np.int16)        # (B, W)
        return h, np.asarray(S), np.asarray(N), np.asarray(Sg - S)


def cyc_lam(s):
    """Per-sweep lam of an annealing cycle (100 sweeps): 10 each at 20, 8, 4, 2, 1, 2, 4, then 30 at 20."""
    sched = [20.0] * 10 + [8.0] * 10 + [4.0] * 10 + [2.0] * 10 + [1.0] * 10 + [2.0] * 10 + [4.0] * 10 + [20.0] * 30
    return sched[(s - 1) % 100]


def run(arm, sweeps, nf=8, seed=0):
    p, st0, fam, fseed = starts(nf)
    A = Arms(p)
    key = jax.random.PRNGKey(seed)
    rec = {"h": [], "S": [], "N": [], "M": [], "sweep": []}
    extra = {}
    t0 = time.time()

    def record(s, st):
        h, S, N, M = A.obs(st)
        for k, v in zip(("h", "S", "N", "M", "sweep"), (h, S, N, M, s)):
            rec[k].append(v)

    record(0, st0)
    if arm.startswith("pt_"):
        kind = arm[3:]
        R = len(LADDER)
        reps = [st0] + [jax.tree_util.tree_map(jnp.copy, st0) for _ in range(R - 1)]
        B = fam.size
        acc, tries = np.zeros(R - 1), np.zeros(R - 1)
        slot = np.tile(np.arange(R)[:, None], (1, B))                # slot[r, b]: replica identity at rung r
        lab = np.zeros((R, B), np.int64) - 1                         # per identity the last end visited: 0 cold, 1 hot
        lab[0] = 0
        trips = np.zeros((R, B))
        for s in range(1, sweeps + 1):
            for r in range(R):
                key, sub = jax.random.split(key)
                reps[r] = A.sweep(reps[r], kind, LADDER[r], sub)
            # swaps
            Ms = []
            for r in range(R):
                S, N = cg.state_energy(A.lay, reps[r])
                if kind == "bin":
                    Ms.append(np.asarray(N, np.float64))
                else:
                    Sg, _ = cg.state_energy(A.g1, reps[r])
                    Ms.append(np.asarray(Sg - S, np.float64))
            key, sub = jax.random.split(key)
            u = np.asarray(jax.random.uniform(sub, (R - 1, B)))
            for r in range(s % 2, R - 1, 2):
                la = (LADDER[r] - LADDER[r + 1]) * (Ms[r] - Ms[r + 1])
                ok = np.log(np.maximum(u[r], 1e-300)) < la
                tries[r] += B
                acc[r] += ok.sum()
                if ok.any():
                    m = jnp.asarray(ok)
                    a, b = reps[r], reps[r + 1]
                    sel = lambda x, y: jnp.where(m.reshape((-1,) + (1,) * (x.ndim - 1)), y, x)
                    reps[r] = {k: sel(a[k], b[k]) for k in a}
                    reps[r + 1] = {k: sel(b[k], a[k]) for k in a}
                    Ms[r], Ms[r + 1] = np.where(ok, Ms[r + 1], Ms[r]), np.where(ok, Ms[r], Ms[r + 1])
                    sr_, sr1 = slot[r].copy(), slot[r + 1].copy()
                    slot[r] = np.where(ok, sr1, sr_)
                    slot[r + 1] = np.where(ok, sr_, sr1)
            # round trips: an identity at the hot end after last being at the cold end, then back
            for ident in range(R):
                at_cold = slot[0] == ident
                at_hot = slot[R - 1] == ident
                trips[ident] += (at_cold & (lab[ident] == 1))
                lab[ident] = np.where(at_cold, 0, np.where(at_hot & (lab[ident] == 0), 1, lab[ident]))
            if s % REC == 0:
                record(s, reps[0])
            if s % 100 == 0:
                print(arm, s, f"{time.time() - t0:.0f}s", "swap acc", np.round(acc / np.maximum(tries, 1), 3).tolist(),
                      "trips", trips.sum(), "cold valid", int((np.asarray(rec['N'][-1]) == 0).sum()), flush=True)
        extra = {"swap_acc": acc / np.maximum(tries, 1), "round_trips": trips.sum(1)}
    else:
        st = st0
        for s in range(1, sweeps + 1):
            key, sub = jax.random.split(key)
            if arm == "hard":
                st = A.sweep(st, "hard", None, sub)
            elif arm.startswith("cyc_"):
                st = A.sweep(st, arm[4:], cyc_lam(s), sub)
            else:
                kind, lam = arm.split("_")
                st = A.sweep(st, kind, float(lam), sub)
            if s % REC == 0:
                record(s, st)
            if s % 250 == 0:
                print(arm, s, f"{time.time() - t0:.0f}s", "valid", int((rec["N"][-1] == 0).sum()), "/", fam.size,
                      "mean N", float(rec["N"][-1].mean()), flush=True)
    out = {k: np.stack(v) if k != "sweep" else np.array(v) for k, v in rec.items()}
    np.savez_compressed(f"{D}/trace_{arm}.npz", fam=fam, fseed=fseed, secs=time.time() - t0, **out, **extra)
    print(arm, "done", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sweeps = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    for arm in sys.argv[1].split(","):
        run(arm, sweeps)
