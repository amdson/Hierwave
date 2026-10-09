"""Legacy (G0, JAX castle sampler era); superseded by castlegen/channels. See the original README.md (git history).

Reference sampler: exact MCMC for the connectivity-constrained model.

Target (the ground truth the coarse levels are later fitted to):

    p(x) ∝ exp(-E(x)) · 1[U(x) = 0] · 1[hard rules]

E is the tile set's energy (pair tables minus log base mass), U the number of
room components not connected to the gate (castlegen.legacy.conn), and the hard rules
are: the gate is fixed, no rooms on the perimeter.

Parallel tempering over rungs k with

    π_k(x) ∝ exp(-(E(x) + λ_k U(x)) / T_k),   T_0 = 1,

so the looser rungs may pass through disconnected states.  Samples from rung 0
with U = 0 are exact samples of p for any λ_0 (the penalty vanishes there).

Moves (each leaves every π_k invariant):
  site    heat-bath proposal from the local conditional of E; accepted with
          exp(-λ ΔU / T) because the proposal cancels the E part.
  window  a w x w window is rebuilt cell by cell in raster order, each cell
          drawn from its conditional given the cells already decided (energy
          terms towards undecided cells are left out, as in wave function
          collapse); Metropolis-Hastings with the forward and reverse
          proposal probabilities, so multi-cell structures can appear whole.
  swap    exchange states between neighbouring rungs.

Sequential numpy; exact and simple rather than fast.
"""
from __future__ import annotations

import math

import numpy as np

from .conn import Connectivity
from .tileset import TileSet


def energy(ts: TileSet, t):
    """E(x): pair energies over neighbour pairs minus log base mass (numpy)."""
    Eh, Ev, logz = ts.np_tables["Eh"], ts.np_tables["Ev"], ts.np_tables["logz"]
    return float(Eh[t[:, :-1], t[:, 1:]].sum() + Ev[t[:-1, :], t[1:, :]].sum() - logz[t].sum())


class Chain:
    def __init__(self, ts: TileSet, tiles, T=1.0, lam=10.0, rng=None):
        self.ts = ts
        self.t = np.array(tiles, np.int32)
        self.H, self.W = self.t.shape
        self.T, self.lam = float(T), float(lam)
        self.rng = rng or np.random.default_rng()
        self.conn = Connectivity(ts, self.t)
        self.Eh, self.Ev = ts.np_tables["Eh"], ts.np_tables["Ev"]
        self.logz = ts.np_tables["logz"]
        S = ts.n_sig
        is_room = np.asarray(ts.np_tables["is_room"], bool)
        not_gate = np.arange(S) != ts.GATE
        self.allow_inner = not_gate                      # interior: anything but the gate
        self.allow_perim = not_gate & ~is_room           # perimeter: no rooms
        self.gate_yx = tuple(np.argwhere(self.t == ts.GATE)[0]) if (self.t == ts.GATE).any() else None
        self.E = energy(ts, self.t)
        self.accepted = {"site": [0, 0], "window": [0, 0]}

    # ------------------------------------------------------------- helpers
    @property
    def U(self):
        return self.conn.unreached_components()

    def _allowed(self, y, x):
        if (y, x) == self.gate_yx:
            return None
        edge = y == 0 or x == 0 or y == self.H - 1 or x == self.W - 1
        return self.allow_perim if edge else self.allow_inner

    def _local_E(self, y, x, decided=None):
        """(S,) energy of each candidate at (y, x) from its neighbours' pair terms
        and its own base mass.  decided: bool grid; undecided neighbours are
        left out (window construction)."""
        t, Eh, Ev = self.t, self.Eh, self.Ev
        e = -self.logz.astype(np.float64)
        if y > 0 and (decided is None or decided[y - 1, x]):
            e = e + Ev[t[y - 1, x], :]
        if y < self.H - 1 and (decided is None or decided[y + 1, x]):
            e = e + Ev[:, t[y + 1, x]]
        if x > 0 and (decided is None or decided[y, x - 1]):
            e = e + Eh[t[y, x - 1], :]
        if x < self.W - 1 and (decided is None or decided[y, x + 1]):
            e = e + Eh[:, t[y, x + 1]]
        return e

    def _log_cond(self, y, x, decided=None):
        """Log conditional probabilities over candidates (−inf where not allowed)."""
        allowed = self._allowed(y, x)
        logit = np.where(allowed, -self._local_E(y, x, decided) / self.T, -np.inf)
        m = logit.max()
        return logit - (m + math.log(np.exp(logit - m).sum()))

    def _set(self, y, x, v):
        self.t[y, x] = v
        self.conn.set(y, x, v)

    # --------------------------------------------------------------- moves
    def site_move(self, y, x):
        if (y, x) == self.gate_yx:
            return
        old = int(self.t[y, x])
        lp = self._log_cond(y, x)
        v = int(self.rng.choice(len(lp), p=np.exp(lp)))
        if v == old:
            return
        e_old = self._local_E(y, x)
        U0 = self.U
        self._set(y, x, v)
        dU = self.U - U0
        self.accepted["site"][1] += 1
        if dU <= 0 or self.rng.random() < math.exp(-self.lam * dU / self.T):
            self.E += e_old[v] - e_old[old]
            self.accepted["site"][0] += 1
        else:
            self._set(y, x, old)

    def window_move(self, y0, x0, w):
        """Rebuild the w x w window at (y0, x0) cell by cell (raster order)."""
        cells = [(y, x) for y in range(y0, y0 + w) for x in range(x0, x0 + w) if (y, x) != self.gate_yx]
        old = {c: int(self.t[c]) for c in cells}
        E0, U0 = self.E, self.U
        decided = np.ones((self.H, self.W), bool)
        for c in cells:
            decided[c] = False
        # reverse proposal probability: rebuild the old contents in the same order
        log_q_old = 0.0
        for c in cells:
            log_q_old += self._log_cond(*c, decided)[old[c]]
            decided[c] = True
        for c in cells:
            decided[c] = False
        # forward proposal: sample the new contents
        log_q_new = 0.0
        for c in cells:
            lp = self._log_cond(*c, decided)
            v = int(self.rng.choice(len(lp), p=np.exp(lp)))
            log_q_new += lp[v]
            self.t[c] = v
            decided[c] = True
        new = {c: int(self.t[c]) for c in cells}
        if new == old:
            return
        for c in cells:                                 # connectivity sees the new window
            self.conn.set(*c, new[c])
        E1, U1 = energy(self.ts, self.t), self.U
        log_a = (-(E1 + self.lam * U1) + (E0 + self.lam * U0)) / self.T + log_q_old - log_q_new
        self.accepted["window"][1] += 1
        if log_a >= 0 or self.rng.random() < math.exp(log_a):
            self.E = E1
            self.accepted["window"][0] += 1
        else:
            for c in cells:
                self._set(*c, old[c])

    def sweep(self, window=0, n_window=0, site=True):
        """One sweep: every cell once in random order (site=True), plus
        n_window window moves of size window."""
        if site:
            for i in self.rng.permutation(self.H * self.W):
                self.site_move(*divmod(int(i), self.W))
        for _ in range(n_window):
            y0 = int(self.rng.integers(0, self.H - window + 1))
            x0 = int(self.rng.integers(0, self.W - window + 1))
            self.window_move(y0, x0, window)

    def log_pi(self, E, U):
        return -(E + self.lam * U) / self.T


def ladder(n, T_max=4.0, lam_max=10.0, lam_min=0.5):
    """(T, λ) per rung: T geometric 1 -> T_max, λ geometric lam_max -> lam_min."""
    if n == 1:
        return [(1.0, lam_max)]
    return [(T_max ** (k / (n - 1)), lam_max * (lam_min / lam_max) ** (k / (n - 1))) for k in range(n)]


def run_pt(ts: TileSet, tiles0, rungs, sweeps, seed=0, window=0, n_window=0, site=True,
           burn=0, thin=1, observe=None, trace=None, log=None):
    """Parallel tempering.  Returns (chains, samples, swap counts, rung-0 U per sweep).

    samples: observe(chain) for rung 0 after each thinned sweep past burn-in,
    only when U = 0 (exact samples of the target); default observe copies the grid.
    trace: optional callback trace(chain) on rung 0 after every sweep."""
    rng = np.random.default_rng(seed)
    chains = [Chain(ts, tiles0, T, lam, np.random.default_rng(rng.integers(1 << 62))) for T, lam in rungs]
    swaps = np.zeros((len(chains) - 1, 2), int)
    samples, rung0_U = [], []
    observe = observe or (lambda c: c.t.copy())
    for s in range(sweeps):
        for c in chains:
            c.sweep(window, n_window, site)
        for k in range(s % 2, len(chains) - 1, 2):      # alternate even / odd pairs
            a, b = chains[k], chains[k + 1]
            Ea, Ua, Eb, Ub = a.E, a.U, b.E, b.U
            log_r = a.log_pi(Eb, Ub) + b.log_pi(Ea, Ua) - a.log_pi(Ea, Ua) - b.log_pi(Eb, Ub)
            swaps[k, 1] += 1
            if log_r >= 0 or rng.random() < math.exp(log_r):
                a.t, b.t = b.t, a.t
                a.conn, b.conn = b.conn, a.conn
                a.E, b.E = b.E, a.E
                swaps[k, 0] += 1
        rung0_U.append(chains[0].U)
        if trace is not None:
            trace(chains[0])
        if s >= burn and (s - burn) % thin == 0 and chains[0].U == 0:
            samples.append(observe(chains[0]))
        if log and (s + 1) % log == 0:
            print(f"sweep {s + 1}: rung-0 E {chains[0].E:.1f} U {chains[0].U}, "
                  f"swap rates {np.round(swaps[:, 0] / np.maximum(swaps[:, 1], 1), 2).tolist()}")
    return chains, samples, swaps, np.array(rung0_U)



# --------------------------------------------------------------------------
# command line: one tempered run on a tile set, trace + samples to an .npz
# --------------------------------------------------------------------------
def observables(c: Chain):
    """Per-sweep summary of a chain: E per cell, U, room fraction, complete
    structures and orphan pieces (summed over structures), kind counts."""
    from .tileset import structure_census
    ts = c.ts
    cen = structure_census(ts, c.t)
    return dict(E=c.E / c.t.size, U=c.U, rooms=float(np.asarray(ts.np_tables["is_room"])[c.t].mean()),
                complete=sum(v[0] for v in cen.values()), orphans=sum(v[1] for v in cen.values()),
                kinds=np.bincount(ts.sig_kind[c.t].ravel(), minlength=len(ts.kinds)))


def initial_tiles(ts: TileSet, size, init="wall"):
    """All wall with the gate on the top row; "planted" adds the first structure
    in the middle (unconnected, so it starts with U = 1)."""
    t = np.full((size, size), ts.WALL, np.int32)
    t[0, size // 2] = ts.GATE
    if init == "planted" and ts.structures:
        _, grid = ts.structures[0]
        sig = {k: int(np.nonzero(ts.sig_kind == k)[0][0]) for k in grid.ravel()}
        R, C = grid.shape
        y, x = size // 2 - R // 2, size // 2 - C // 2
        t[y:y + R, x:x + C] = np.vectorize(sig.get)(grid)
    return t


def main():
    import argparse
    import time
    from . import tileset
    ap = argparse.ArgumentParser(description="Tempered reference sampler (exact, sequential numpy).")
    ap.add_argument("--tileset", default="demo")
    ap.add_argument("--size", type=int, default=20)
    ap.add_argument("--sweeps", type=int, default=500)
    ap.add_argument("--burn", type=int, default=100)
    ap.add_argument("--thin", type=int, default=5)
    ap.add_argument("--rungs", type=int, default=6)
    ap.add_argument("--t-max", type=float, default=3.0)
    ap.add_argument("--lam-max", type=float, default=10.0)
    ap.add_argument("--lam-min", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--n-window", type=int, default=10, help="window moves per chain per sweep")
    ap.add_argument("--init", choices=["wall", "planted"], default="wall")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help=".npz: per-sweep trace, connected samples, swap counts")
    args = ap.parse_args()
    ts = tileset.load(args.tileset)
    rows = []
    t0 = time.time()
    chains, samples, swaps, _ = run_pt(
        ts, initial_tiles(ts, args.size, args.init), ladder(args.rungs, args.t_max, args.lam_max, args.lam_min),
        args.sweeps, seed=args.seed, window=args.window, n_window=args.n_window, burn=args.burn,
        thin=args.thin, trace=lambda c: rows.append(observables(c)), log=max(1, args.sweeps // 10))
    acc = {k: round(a / max(n, 1), 3) for k, (a, n) in chains[0].accepted.items()}
    print(f"done in {time.time() - t0:.0f}s; {len(samples)} connected samples kept; rung-0 acceptance {acc}")
    trace = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    np.savez_compressed(args.out, samples=np.array(samples), swaps=swaps,
                        kind_names=np.array([k.name for k in ts.kinds]), **trace)


if __name__ == "__main__":
    main()
