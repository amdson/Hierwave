"""Legacy (G1b, coarse-to-fine promises era); superseded by castlegen/channels. See notes/history/coarse_to_fine.tex.

Reference sampler for the support-constrained side-view target p_G
(notes/history/coarse_to_fine.tex, quantities/envelope.py), its corpus, promise
tables, and the statistics a generator is compared against.

Target.  On an (n, W) torus with the bottom `ground` rows (the band) clamped
to a ground tile,

    p_G(x) ∝ exp(-E_pair(x) / T + sum_i logz(x_i) + mu * #solid(x)) * 1[G(x)],

which is exactly the model exemplar.gibbs samples (pair tables divided by T,
the log base mass logz = -unary NOT divided by T) restricted to the support
rule G: a solid cell outside the band needs an effective-solid (solid or
band) cell directly below.  mu (default 0) is an optional chemical potential
per solid cell; mu = 0 is the tile set's own model.  The valid maps are the
heightmaps: column c is solid on the rows >= n - H(c), H(c) in [ground, n].

Moves (each leaves p_G invariant; their composition is the chain):

  site    checkerboard heat-bath restricted to the valid set.  The support
          indicator is a product of factors on vertical neighbour pairs, so
          given the other colour a cell's allowed set is local:
            solid allowed  iff the cell below is effective-solid;
            air   allowed  iff the cell above is not solid (row 0 is exempt:
                  the cell above it is band, which needs no support).
          Heat-bath over p's conditional restricted to that set is the exact
          conditional of p_G.  Cells of one colour share no energy term and no
          rule factor (both couple 4-neighbours only), so a colour is updated
          in parallel (needs even n and W on the torus).
  column  Metropolis-Hastings height jump H(c) -> H(c) + d, d uniform in
          ±{1..D}: new solid cells are drawn bottom-up from their conditional
          given left, right and the cell below; removed cells become air.  The
          acceptance uses the exact reverse proposal probability.  Columns of
          one residue mod a stride >= 2 share no energy term, so they are
          updated in parallel.

Irreducibility: from any heightmap, lower every column to H = ground one cell
at a time (the top solid cell may always turn to air), then raise it to any
target heightmap with any materials (the air cell above the top may always
turn to any solid signature).  Heat-bath has positive holding probability, so
the chain is aperiodic; all conditionals are positive (finite energies).

Conditioning (`sample_conditioned`): clamps {(h, by, bx): value} of envelope
promise values.  A promise depends only on the heights of the block's
columns, so materials move freely (class-preserving site heat-bath) and
heights move only by column jumps, rejected when they change a clamped
block's value (stride >= the largest clamped h, so a batch holds at most one
column per clamped block).  Starting heights are built from the clamp
envelopes by a small local search.

Corpus, mining, tables: `corpus`, `mine`, `tables` (cache/ like corpus.py).
Statistics: `stats` (one map), `stats_many`, `compare`.

    python -m castlegen.legacy.refheights --T 0.6,0.9,1.2 --png DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np

from castlegen.legacy import tileset
from castlegen.legacy.promise import Tables
from castlegen.legacy.quantities import envelope as EN

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "cache")
SEEDS = tuple(range(1000, 1024))
LAGS = (1, 2, 4, 8, 16, 32)
SLOPE_CLIP = 8


# ------------------------------------------------------------------- helpers
def _gumbel_argmax(logits, rng):
    u = rng.random(logits.shape)
    return np.argmax(logits - np.log(-np.log(u + 1e-300) + 1e-300), axis=-1)


def _sig(ts, name):
    for s in range(ts.n_sig):
        if ts.sig_name(s).partition("@")[0] == name:
            return s
    raise KeyError(name)


def log_weight(ts, tiles, T=1.0, mu=0.0):
    """log of the unnormalised p (torus): -E_pair / T + sum logz + mu #solid."""
    t = np.asarray(tiles)
    Eh, Ev, logz = ts.np_tables["Eh"], ts.np_tables["Ev"], ts.np_tables["logz"]
    pair = Eh[t, np.roll(t, -1, 1)].sum() + Ev[t, np.roll(t, -1, 0)].sum()
    return float(-pair / T + logz[t].sum() + mu * ts.solid[t].sum())


def energy_parts(ts, tiles):
    """(pair energy, -sum logz) on the torus."""
    t = np.asarray(tiles)
    Eh, Ev, logz = ts.np_tables["Eh"], ts.np_tables["Ev"], ts.np_tables["logz"]
    return float(Eh[t, np.roll(t, -1, 1)].sum() + Ev[t, np.roll(t, -1, 0)].sum()), float(-logz[t].sum())


def violations(ts, tiles, ground):
    """Number of solid cells outside the band without effective-solid below."""
    s = EN.effective_solid(ts, tiles, ground)
    n = s.shape[0]
    band = np.arange(n) >= n - ground
    solid = ts.solid[np.asarray(tiles)] & ~band[:, None]
    return int((solid & ~np.roll(s, -1, 0)).sum())


def heights(ts, tiles, ground):
    """H(c): the effective-solid run length from the bottom row upward (the
    heightmap for a valid map; for an invalid one, the supported part)."""
    s = EN.effective_solid(ts, tiles, ground)
    return np.cumprod(s[::-1], axis=0).sum(0).astype(np.int64)


def from_heights(ts, H, n, ground, fill="stone", ground_tile="stone", air="air"):
    """Valid map from a heightmap: solid `fill` below H, band `ground_tile`."""
    H = np.asarray(H)
    rows = np.arange(n)[:, None]
    t = np.where(rows >= n - H[None, :], _sig(ts, fill), _sig(ts, air)).astype(np.int32)
    t[n - ground:] = _sig(ts, ground_tile)
    return t


def block_values(H, n, h, lg=None):
    """Envelope promise values of level h from heights (valid maps only).
    H: (..., W) heights.  -> (..., n/h, W/h) values."""
    lg = lg or EN.lang()
    H = np.asarray(H, np.int64)
    W = H.shape[-1]
    R = n // h
    base = n - (np.arange(R) + 1) * h                                      # (R,) height of each block row's bottom
    t = np.clip(H[..., None, :] - base[:, None], 0, h)                     # (..., R, W)
    t = t.reshape(t.shape[:-1] + (W // h, lg.I, h // lg.I))
    return lg.encode(lg.bins(t.min(-1), h), lg.bins(t.max(-1), h))


# --------------------------------------------------------------------- chain
class Chain:
    """Exact MCMC for p_G on an (n, W) torus (see the module docstring)."""

    def __init__(self, ts, tiles, ground, T=0.9, mu=0.0, forbid=None, clamps=None, lg=None, seed=0, D=4):
        self.ts = ts
        self.t = np.array(tiles, np.int32)
        self.n, self.W = self.t.shape
        assert self.n % 2 == 0 and self.W % 2 == 0, "checkerboard needs even sides on the torus"
        self.G = int(ground)
        assert 0 < self.G <= self.n
        self.T, self.mu, self.D = float(T), float(mu), int(D)
        self.rng = np.random.default_rng(seed)
        self.lg = lg or EN.lang()
        S = ts.n_sig
        self.Eh, self.Ev = ts.np_tables["Eh"].astype(np.float64), ts.np_tables["Ev"].astype(np.float64)
        self.solid = np.asarray(ts.solid, bool)
        fb = np.zeros(S, bool) if forbid is None else np.asarray(forbid, bool).copy()
        fb[ts.GATE] = True
        self.allowed = ~fb
        self.base = np.where(self.allowed, ts.np_tables["logz"] + self.mu * self.solid, -np.inf)
        air = np.flatnonzero(~self.solid & self.allowed)
        assert len(air) == 1, "exactly one allowed non-solid signature expected"
        self.air = int(air[0])
        ys, xs = np.meshgrid(np.arange(self.n - self.G), np.arange(self.W), indexing="ij")
        par = (ys + xs) % 2
        self.cells = [(ys[par == p], xs[par == p]) for p in (0, 1)]
        self.clamps = None
        if clamps:
            self._set_clamps(clamps)
        assert violations(ts, self.t, self.G) == 0, "start map breaks the support rule"

    # ---- clamps
    def _set_clamps(self, clamps):
        """clamps {(h, by, bx): value} -> per level (mask (R, R), values (R, R))."""
        self.clamps = {}
        for (h, by, bx), v in clamps.items():
            R = self.n // h
            if h not in self.clamps:
                self.clamps[h] = (np.zeros((R, self.W // h), bool), np.zeros((R, self.W // h), np.int64))
            self.clamps[h][0][by, bx] = True
            self.clamps[h][1][by, bx] = v
        self.stride = max(2, max(self.clamps))
        assert self.W % self.stride == 0
        assert self.clamps_ok(), "start map does not honour the clamps"

    def clamps_ok(self, tiles=None):
        if not self.clamps:
            return True
        H = heights(self.ts, self.t if tiles is None else tiles, self.G)
        return all((block_values(H, self.n, h, self.lg)[m] == v[m]).all() for h, (m, v) in self.clamps.items())

    def _clamp_check(self, H, cs, Hn):
        """Would setting H[cs] = Hn keep every clamped block's value?  Needs at
        most one of cs per clamped block (stride >= max h)."""
        ok = np.ones(len(cs), bool)
        for h, (m, v) in self.clamps.items():
            bx = cs // h
            blk = H[(bx * h)[:, None] + np.arange(h)[None, :]].copy()           # (m, h)
            blk[np.arange(len(cs)), cs - bx * h] = Hn
            lg = self.lg
            R = self.n // h
            base = self.n - (np.arange(R) + 1) * h
            t = np.clip(blk[:, None, :] - base[None, :, None], 0, h).reshape(len(cs), R, lg.I, h // lg.I)
            vals = lg.encode(lg.bins(t.min(-1), h), lg.bins(t.max(-1), h))    # (m, R)
            mk = m[:, bx].T                                                    # (m, R)
            ok &= ~(mk & (vals != v[:, bx].T)).any(1)
        return ok

    # ---- moves
    def site_half(self, parity, T=None, keep_class=False):
        T = self.T if T is None else T
        t, n, W, G = self.t, self.n, self.W, self.G
        ys, xs = self.cells[parity]
        L, R = t[ys, (xs - 1) % W], t[ys, (xs + 1) % W]
        U, Dn = t[(ys - 1) % n, xs], t[ys + 1, xs]
        E = self.Eh[L] + self.Eh[:, R].T + self.Ev[U] + self.Ev[:, Dn].T
        logits = -E / T + self.base
        below_es = self.solid[Dn] | (ys + 1 >= n - G)
        air_ok = (ys == 0) | ~self.solid[U]
        ok = np.where(self.solid[None, :], below_es[:, None], air_ok[:, None])
        if keep_class:
            ok &= self.solid[None, :] == self.solid[t[ys, xs]][:, None]
        logits = np.where(ok, logits, -np.inf)
        t[ys, xs] = _gumbel_argmax(logits, self.rng)

    def _fill_logp(self, L, R, below, T):
        """(m, S) log-probabilities of the growth proposal for one cell."""
        lg = -(self.Eh[L] + self.Eh[:, R].T + self.Ev[:, below].T) / T + self.base
        lg = np.where(self.solid[None, :], lg, -np.inf)
        mx = lg.max(1, keepdims=True)
        return lg - mx - np.log(np.exp(lg - mx).sum(1, keepdims=True))

    def _col_logw(self, col, L, R, T):
        """(m,) log weight of the terms touching columns col (n, m)."""
        e = self.Eh[L, col] + self.Eh[col, R]
        v = self.Ev[col, np.roll(col, -1, 0)]
        return (-e.sum(0) - v.sum(0)) / T + self.base[col].sum(0)

    def column_batch(self, cs, T=None):
        T = self.T if T is None else T
        t, n, G, D = self.t, self.n, self.G, self.D
        rng = self.rng
        m = len(cs)
        H = heights(self.ts, t, G)
        Hc = H[cs]
        d = rng.integers(1, D + 1, m) * rng.choice(np.array([-1, 1]), m)
        Hn = Hc + d
        ok = (Hn >= G) & (Hn <= n)
        if self.clamps and ok.any():
            ok &= self._clamp_check(H, cs, np.clip(Hn, G, n))
        if not ok.any():
            return 0
        old = t[:, cs]
        new = old.copy()
        L, R = t[:, (cs - 1) % self.W], t[:, (cs + 1) % self.W]
        lq = np.zeros(m)                                             # log q_rev - log q_fwd
        for k in range(1, D + 1):
            g = np.flatnonzero(ok & (d >= k))                       # grow: row n - Hc - k
            if len(g):
                y = n - Hc[g] - k
                lp = self._fill_logp(L[y, g], R[y, g], new[y + 1, g], T)
                s = _gumbel_argmax(lp, rng)
                new[y, g] = s
                lq[g] -= lp[np.arange(len(g)), s]
            r = np.flatnonzero(ok & (d <= -k))                      # shrink: row n - Hc + k - 1
            if len(r):
                y = n - Hc[r] + k - 1
                lp = self._fill_logp(L[y, r], R[y, r], old[y + 1, r], T)
                lq[r] += lp[np.arange(len(r)), old[y, r]]
                new[y, r] = self.air
        la = self._col_logw(new, L, R, T) - self._col_logw(old, L, R, T) + lq
        if ok.any():
            la = la + self._extra_logw(cs, old, new, T)
        acc = ok & (np.log(rng.random(m) + 1e-300) < la)
        t[:, cs[acc]] = new[:, acc]
        if acc.any():
            self._on_accept(cs[acc], old[:, acc], new[:, acc])
        return int(acc.sum())

    # ---- hooks for extra energy terms on column moves (see castlegen.legacy.exchain)
    def _extra_logw(self, cs, old, new, T):
        return 0.0

    def _on_accept(self, cs, old, new):
        pass

    def column_sweep(self, T=None):
        s = self.stride if self.clamps else 2
        acc = 0
        for off in self.rng.permutation(s):
            acc += self.column_batch(np.arange(off, self.W, s), T)
        return acc / self.W

    def sweep(self, T=None, columns=True):
        keep = bool(self.clamps)
        for p in self.rng.permutation(2):
            self.site_half(p, T, keep_class=keep)
        return self.column_sweep(T) if (columns or keep) else 0.0

    def run(self, sweeps, T_hot=None, anneal=0, columns=True, check=False, callback=None):
        """`anneal` sweeps linearly T_hot -> T, then `sweeps` at T."""
        T_hot = self.T if T_hot is None else T_hot
        for i in range(anneal + sweeps):
            temp = T_hot + (self.T - T_hot) * min(i / max(anneal, 1), 1.0) if i < anneal else self.T
            self.sweep(temp, columns)
            if check:
                assert violations(self.ts, self.t, self.G) == 0
                assert self.clamps_ok()
            if callback is not None:
                callback(i, self)
        return self.t


def band_map(ts, n, ground, W=None, ground_tile="stone", H0=None):
    """Start map: flat ground at height H0 (default: the band only)."""
    W = n if W is None else W
    return from_heights(ts, np.full(W, ground if H0 is None else H0), n, ground, ground_tile, ground_tile)


def sample(ts, n=128, seed=0, T=0.9, ground=8, sweeps=600, anneal=0, T_hot=None, mu=0.0, ground_tile="stone",
           forbid=None, columns=True, init="band"):
    """One sample of p_G on an n x n torus: from the band (init "band") or
    flat ground at a random height ("flat"), anneal T_hot -> T over `anneal`
    sweeps, then `sweeps` at T.

    Default: no anneal.  At mu = 0 the air phase wins in bulk for every T
    below ~1.5 (coexistence mu*(T) ~ 0.09 at T 0.6, 0.11 at 0.9, 0.04 at 1.2),
    so p_G is ground a few rows above the band; a hot start (T_hot = 4 fills
    the map: solid wins by material entropy there) melts down at only ~2
    rows per 100 sweeps, i.e. is far from equilibrium after 400 sweeps.  From
    the band the chain starts next to equilibrium and relaxes locally."""
    rng = np.random.default_rng(seed)
    H0 = ground if init == "band" else int(rng.integers(ground, n))
    ch = Chain(ts, band_map(ts, n, ground, ground_tile=ground_tile, H0=H0), ground, T, mu, forbid, seed=seed)
    return ch.run(sweeps, T_hot, anneal, columns)


# ------------------------------------------------------------- conditioning
def heights_from_clamps(n, W, clamps, ground, lg=None, seed=0, iters=200000):
    """A heightmap (W,) whose envelope values equal every clamp (local search
    from the per-column intersection of the clamp envelopes)."""
    lg = lg or EN.lang()
    rng = np.random.default_rng(seed)
    lo_H, hi_H = np.full(W, ground), np.full(W, n)
    need = []                                         # (cols, H range for the lo bin, H range for the hi bin)

    def hrange(b, h, base):
        if b == 0:
            return 0, base
        if b == lg.F:
            return base + h, n
        tl = (b - 1) * h // lg.Q + 1
        th = min(b * h // lg.Q, h - 1)
        return base + tl, base + th

    for (h, by, bx), v in clamps.items():
        base = n - (by + 1) * h
        w = h // lg.I
        for i in range(lg.I):
            cols = np.arange(bx * h + i * w, bx * h + (i + 1) * w)
            rl, rh = hrange(lg.LO[v, i], h, base), hrange(lg.HI[v, i], h, base)
            lo_H[cols] = np.maximum(lo_H[cols], rl[0])
            hi_H[cols] = np.minimum(hi_H[cols], rh[1])
            need.append((cols, rl, rh))
    if (lo_H > hi_H).any():
        raise ValueError("clamps inconsistent: empty height range")
    H = rng.integers(lo_H, hi_H + 1)

    def bad():
        return [j for j, (cols, rl, rh) in enumerate(need)
                if not (rl[0] <= H[cols].min() <= rl[1] and rh[0] <= H[cols].max() <= rh[1])]

    for _ in range(iters):
        b = bad()
        if not b:
            return H
        cols, rl, rh = need[b[rng.integers(len(b))]]
        c = cols[rng.integers(len(cols))]
        r = rl if rng.random() < 0.5 else rh
        a, z = max(r[0], lo_H[c]), min(r[1], hi_H[c])
        if a <= z:
            H[c] = rng.integers(a, z + 1)
        else:
            H[c] = rng.integers(lo_H[c], hi_H[c] + 1)
    raise ValueError("no heightmap found for the clamps")


def sample_conditioned(ts, clamps, n=128, W=None, seed=0, T=0.9, ground=8, sweeps=400, mu=0.0, ground_tile="stone",
                       forbid=None, init=None, lg=None, return_chain=False):
    """p_G(x | A_b(x) = clamps[b]) with clamps {(h, by, bx): value}; exact
    MCMC from a map that honours the clamps (`init`, or built from them)."""
    W = n if W is None else W
    lg = lg or EN.lang()
    if init is None:
        H = heights_from_clamps(n, W, clamps, ground, lg, seed)
        init = from_heights(ts, H, n, ground, ground_tile, ground_tile)
    ch = Chain(ts, init, ground, T, mu, forbid, clamps=clamps, lg=lg, seed=seed)
    ch.run(sweeps)
    return ch if return_chain else ch.t


# ------------------------------------------------------------------- corpus
def _key(ts, params):
    path = os.path.join(tileset.TILESET_DIR, ts.name + ".json")
    spec = open(path).read() if os.path.exists(path) else ts.name
    return hashlib.sha1((spec + json.dumps(params, sort_keys=True) + "refheights-v1").encode()).hexdigest()[:12]


def corpus(ts, n=128, seeds=SEEDS, T=0.9, ground=8, sweeps=600, anneal=0, T_hot=None, mu=0.0,
           ground_tile="stone", init="band", log=print, rebuild=False):
    """(N, n, n) valid maps, one per seed (`sample`), cached under cache/."""
    params = dict(n=n, seeds=list(seeds), T=T, ground=ground, sweeps=sweeps, anneal=anneal, T_hot=T_hot, mu=mu,
                  ground_tile=ground_tile, init=init)
    path = os.path.join(CACHE_DIR, f"refheights-{_key(ts, params)}.npy")
    if os.path.exists(path) and not rebuild:
        return np.load(path)
    t0 = time.time()
    out = []
    for i, s in enumerate(seeds):
        out.append(sample(ts, n, s, T, ground, sweeps, anneal, T_hot, mu, ground_tile, init=init))
        log(f"refheights: sample {i + 1}/{len(seeds)} ({time.time() - t0:.0f}s)")
    out = np.stack(out)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(path, out)
    log(f"refheights: {len(seeds)} samples {n}x{n} in {time.time() - t0:.1f}s -> {path}")
    return out


def mine(ts, maps, K=16, ground=8, h_top=64, lg=None):
    """{h: [promise grid per map]} for h = K .. h_top (envelope.pyramid)."""
    grids = {}
    for t in maps:
        vals, ok = EN.pyramid(ts, t, K, ground, h_top, lg)
        for h, v in vals.items():
            assert ok[h].all(), "corpus map breaks the support rule"
            grids.setdefault(h, []).append(v)
    return grids


def tables(ts, maps=None, K=16, ground=8, h_top=64, alpha=0.5, lg=None, **corpus_kw):
    """promise.Tables fitted to the mined corpus (maps, or `corpus(**corpus_kw)`)."""
    lg = lg or EN.lang()
    if maps is None:
        maps = corpus(ts, ground=ground, **corpus_kw)
    return Tables.fit(lg.V, mine(ts, maps, K, ground, h_top, lg), alpha)


def value_report(grids, V):
    """{h: (distinct values, entropy in bits, top values with shares)}."""
    out = {}
    for h, gs in sorted(grids.items()):
        c = np.bincount(np.concatenate([g.ravel() for g in gs]), minlength=V)
        p = c / c.sum()
        nz = p[p > 0]
        top = np.argsort(-c)[:5]
        out[h] = (int((c > 0).sum()), float(-(nz * np.log2(nz)).sum()), [(int(v), round(float(p[v]), 3)) for v in top])
    return out


# --------------------------------------------------------------- statistics
def stats(ts, tiles, ground, K=16, h_top=64, lg=None):
    """Statistics of one map (valid or not; H is the supported run)."""
    lg = lg or EN.lang()
    t = np.asarray(tiles)
    n, W = t.shape
    H = heights(ts, t, ground)
    band = np.arange(n) >= n - ground
    solid = ts.solid[t]
    sl = np.roll(H, -1) - H
    Hc = H - H.mean()
    var = float((Hc ** 2).mean())
    S = ts.n_sig
    body = solid & ~band[:, None]
    top = t[np.clip(n - H, 0, n - 1), np.arange(W)]
    pair, un = energy_parts(ts, t)
    vals, ok = EN.pyramid(ts, t, K, ground, h_top, lg)
    return {
        "violations": violations(ts, t, ground) / (n * W),
        "solid": float(solid.mean()),
        "H_mean": float(H.mean()),
        "H_std": float(np.sqrt(var)),
        "H_hist": np.bincount(H, minlength=n + 1),
        "slope_hist": np.bincount(np.clip(sl, -SLOPE_CLIP, SLOPE_CLIP) + SLOPE_CLIP, minlength=2 * SLOPE_CLIP + 1),
        "abs_slope_hist": np.bincount(np.minimum(np.abs(sl), SLOPE_CLIP), minlength=SLOPE_CLIP + 1),
        "abs_slope": float(np.abs(sl).mean()),
        "acov": np.array([(Hc * np.roll(Hc, -k)).mean() for k in LAGS]),
        "var": var,
        "material": np.bincount(t[body], minlength=S),
        "surface": np.bincount(top, minlength=S),
        "E_pair": pair / (n * W),
        "E_unary": un / (n * W),
        "E": (pair + un) / (n * W),
        "promise": {h: np.bincount(v.ravel(), minlength=lg.V) for h, v in vals.items()},
        "promise_ok": {h: float(o.mean()) for h, o in ok.items()},
    }


def stats_many(ts, maps, ground, K=16, h_top=64, lg=None):
    return [stats(ts, t, ground, K, h_top, lg) for t in maps]


SCALARS = ("violations", "solid", "H_mean", "H_std", "abs_slope", "E_pair", "E_unary", "E")
HISTS = ("H_hist", "slope_hist", "abs_slope_hist", "material", "surface")


def _tv(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.sum() == 0 or b.sum() == 0:
        return float("nan")
    return 0.5 * float(np.abs(a / a.sum() - b / b.sum()).sum())


def pooled(sl):
    """Pooled summary of a stats list: scalars as mean (se), histograms as
    distributions, autocorrelation as pooled acov / pooled var."""
    out = {}
    for k in SCALARS:
        x = np.array([s[k] for s in sl], float)
        out[k] = (float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan"))
    for k in HISTS:
        c = np.sum([s[k] for s in sl], 0).astype(float)
        out[k] = c / max(c.sum(), 1)
    v = sum(s["var"] for s in sl)
    out["acf"] = np.sum([s["acov"] for s in sl], 0) / v if v > 0 else np.full(len(LAGS), np.nan)
    out["promise"] = {h: np.sum([s["promise"][h] for s in sl], 0) / sum(s["promise"][h].sum() for s in sl)
                      for h in sl[0]["promise"]}
    return out


def compare(a, b):
    """Distances between two stats lists (a: e.g. generator, b: reference).
    Scalars: (mean_a, mean_b, diff, se, z); histograms: pooled total
    variation; acf: per-lag difference; promise: per-level TV."""
    A, B = pooled(a), pooled(b)
    out = {}
    for k in SCALARS:
        (ma, sa), (mb, sb) = A[k], B[k]
        se = float(np.sqrt(np.nan_to_num(sa) ** 2 + np.nan_to_num(sb) ** 2))
        d = ma - mb
        out[k] = dict(a=ma, b=mb, diff=d, se=se, z=d / se if se > 0 else (0.0 if d == 0 else float("inf")))
    for k in HISTS:
        out[k + "_tv"] = _tv(A[k], B[k])
    out["acf"] = dict(lags=LAGS, a=A["acf"], b=B["acf"], diff=A["acf"] - B["acf"])
    out["promise_tv"] = {h: _tv(A["promise"][h], B["promise"][h]) for h in A["promise"] if h in B["promise"]}
    return out


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tileset", default="cliffs")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--T", default="0.9")
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--ground", type=int, default=8)
    ap.add_argument("--sweeps", type=int, default=600)
    ap.add_argument("--anneal", type=int, default=0)
    ap.add_argument("--init", default="band")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--png", default=None)
    args = ap.parse_args()
    ts = tileset.load(args.tileset)
    for T in [float(x) for x in args.T.split(",")]:
        t0 = time.time()
        maps = [sample(ts, args.n, 1000 + s, T, args.ground, args.sweeps, args.anneal, mu=args.mu, init=args.init)
                for s in range(args.seeds)]
        P = pooled(stats_many(ts, maps, args.ground))
        print(f"T={T} mu={args.mu} ({time.time() - t0:.1f}s): " +
              " ".join(f"{k}={P[k][0]:.3f}" for k in SCALARS) + f" acf={np.round(P['acf'], 2)}")
        if args.png:
            from castlegen.legacy import exemplar as ex
            os.makedirs(args.png, exist_ok=True)
            for i, t in enumerate(maps):
                ex.to_png(ts, t, os.path.join(args.png, f"ref-T{T}-mu{args.mu}-{i}.png"), px=4)


if __name__ == "__main__":
    main()


def column_tables(ts, maps=None, K=16, ground=8, h_top=64, alpha=0.5, lg=None, **corpus_kw):
    """Like `tables`, but the parent term conditions a child only on the
    parent's part over the child's column (Lang.parent_part), which is all
    the language lets a child column depend on: p(child | part(parent, dx),
    dy), expanded to the (4, V, V) [position, child, parent] layout.  Parent
    values unseen in the corpus then still get an informed conditional as
    long as their column part was seen."""
    lg = lg or EN.lang()
    if maps is None:
        maps = corpus(ts, ground=ground, **corpus_kw)
    grids = mine(ts, maps, K, ground, h_top, lg)
    tab = Tables.fit(lg.V, grids, alpha)
    V, Pp = lg.V, lg.P ** (lg.I // 2)
    part = np.stack([lg.parent_part(np.arange(V), dx) for dx in (0, 1)])        # (2, V) parent -> part index
    for h in grids:
        if 2 * h not in grids:
            continue
        cc = np.zeros((4, V, Pp))
        for g, Gp in zip(grids[h], grids[2 * h]):
            for dy in (0, 1):
                for dx in (0, 1):
                    np.add.at(cc[2 * dy + dx], (g[dy::2, dx::2].ravel(), part[dx][Gp.ravel()]), 1)
        pc = (cc + alpha) / (cc.sum(1, keepdims=True) + V * alpha)             # p(child | part, pos)
        logpu = -tab.u[h]
        par = np.empty((4, V, V))
        for pos in range(4):
            dx = pos % 2
            par[pos] = -np.log(pc[pos][:, part[dx]]) - logpu[:, None]
        tab.parent[h] = par
    return tab


def factored_tables(ts, maps=None, K=16, ground=8, h_top=64, alpha=0.5, lg=None, **corpus_kw):
    """Tables whose terms factor over column intervals, so they generalise
    to block values never seen whole in the corpus:
      u(v)            = sum_i -log p_i(pair_i(v))        (per interval position i)
      parent          = column_tables' p(child | parent part over its column)
      pair[0](a, b)   = -PMI of the two intervals meeting at the vertical seam
                        (last interval of a, first of b), fitted on every
                        horizontally adjacent interval pair
      pair[1](u, l)   = sum_i -PMI(pair_i(u), pair_i(l)) of stacked intervals
    Ground-row effects are left to the hard validity rule."""
    lg = lg or EN.lang()
    if maps is None:
        maps = corpus(ts, ground=ground, **corpus_kw)
    grids = mine(ts, maps, K, ground, h_top, lg)
    tab = column_tables(ts, maps, K, ground, h_top, alpha, lg)
    V, P, I = lg.V, lg.P, lg.I
    pidx = np.stack([(np.arange(V) // P ** i) % P for i in range(I)], 1)        # (V, I) pair index per interval

    def pmi(c):
        pp = (c + alpha) / (c.sum() + P * P * alpha)
        return np.log(pp) - np.log(pp.sum(1, keepdims=True)) - np.log(pp.sum(0, keepdims=True))

    for h, gs in grids.items():
        cu = np.zeros((I, P))
        ch = np.zeros((P, P))
        cv = np.zeros((I, P, P))
        for g in gs:
            iv = pidx[g]                                                        # (R, R, I)
            for i in range(I):
                np.add.at(cu[i], iv[..., i].ravel(), 1)
                np.add.at(cv[i], (iv[:-1, :, i].ravel(), iv[1:, :, i].ravel()), 1)
            row = iv.reshape(iv.shape[0], -1)                                   # intervals left to right
            np.add.at(ch, (row.ravel(), np.roll(row, -1, 1).ravel()), 1)
        pu = (cu + alpha) / (cu.sum(1, keepdims=True) + P * alpha)
        tab.u[h] = -np.log(pu[np.arange(I)[None, :], pidx]).sum(1)
        A = -pmi(ch)
        tab.pair[h] = np.stack([A[pidx[:, -1][:, None], pidx[:, 0][None, :]],
                                sum(-pmi(cv[i])[pidx[:, i][:, None], pidx[:, i][None, :]] for i in range(I))])
    return tab
