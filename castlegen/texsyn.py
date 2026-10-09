"""Categorical parallel controllable texture synthesis (Lefebvre & Hoppe 2005)
on a toroidal tile exemplar; see texture_sampling.md, design A steps 1-4.

The output is a map S of exemplar coordinates; tiles are E[S].  Features are
one-hot signatures plus one-hot sockets per side (so a socket mismatch is
visible to the metric), blurred into a Gaussian stack.  Adjacency violations
can only appear at seams (neighbours whose coordinates are not contiguous);
exemplar.repair() fixes them with a few Gibbs sweeps afterwards.

    python -m castlegen.texsyn images/exemplar.npy --size 128 --png images/synth.png
"""
from __future__ import annotations

import argparse

import numpy as np

from castlegen import exemplar as ex
from castlegen import hier
from castlegen import tileset

OFF5 = np.array([(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)])
OFF3 = np.array([(dy, dx) for dy in range(-1, 2) for dx in range(-1, 2)])


def features(ts, E, w_sig=1.0, w_sock=0.5):
    S, ns = ts.n_sig, len(ts.sockets)
    f = [np.eye(S, dtype=np.float32)[E] * np.sqrt(w_sig)]
    for d in range(4):
        f.append(np.eye(ns, dtype=np.float32)[ts.sig_sockets[E, d]] * np.sqrt(w_sock / 4))
    return np.concatenate(f, -1)


def _blur(F, sigma, bounded=False):
    """Circular Gaussian blur of (m, m, D) along the two grid axes; bounded:
    edge-replicating vertically (rows beyond the top / bottom repeat them)."""
    if sigma <= 0:
        return F
    if bounded:                         # vertical pass on a copy extended by 3 sigma of edge rows, then crop
        p = int(np.ceil(3 * sigma))
        Fp = np.pad(F, ((p, p), (0, 0), (0, 0)), mode="edge")
        F = _blur_axis(Fp, sigma, 0)[p:p + F.shape[0]]
    else:
        F = _blur_axis(F, sigma, 0)
    return _blur_axis(F, sigma, 1).astype(np.float32)


def _blur_axis(F, sigma, axis):
    """Circular Gaussian blur along one axis (FFT)."""
    g = np.exp(-2 * (np.pi * sigma * np.fft.fftfreq(F.shape[axis])) ** 2)
    shape = [1] * F.ndim
    shape[axis] = -1
    return np.real(np.fft.ifft(np.fft.fft(F, axis=axis) * g.reshape(shape), axis=axis))


def _pca(X, k, rng, oversample=10, power=2):
    """Top-k principal directions of centred X (randomized range finder)."""
    Q = np.linalg.qr(X @ rng.standard_normal((X.shape[1], k + oversample)).astype(np.float32))[0]
    for _ in range(power):
        Q = np.linalg.qr(X @ (X.T @ Q))[0]
    _, _, Vt = np.linalg.svd(Q.T @ X, full_matrices=False)
    return Vt[:k].T


class Analysis:
    """Per-level stack, PCA'd 5x5 neighbourhoods and k=2 coherence sets.

    bounds=None: the exemplar is a torus.  bounds="edge": it is periodic
    horizontally only, and continues forever above its top row and below its
    bottom row by repeating them (for a side view: sky above, ground below).
    Reads beyond it clamp to its edge rows, which is that continuation;
    coordinates stay inside it (`pad` > 0 would add pad rows of the
    continuation as coordinates).
    levels: the sides h to analyse (default all; the others are None);
    c2=False skips the second coherence candidates (C2 = None).
    self.E is the padded exemplar (my, m); coordinates are (y in [0, my), x mod m)."""

    def __init__(self, ts, E, n_pca=32, min_jump=0.05, seed=0, bounds=None, pad=None, levels=None, c2=True,
                 **feat):
        E = np.asarray(E)
        m = E.shape[0]
        assert E.shape == (m, m) and m & (m - 1) == 0, "exemplar must be square, side a power of 2"
        self.bounded = bounds == "edge"
        self.pad = (pad or 0) if self.bounded else 0
        if self.bounded:
            E = np.concatenate([np.repeat(E[:1], self.pad, 0), E, np.repeat(E[-1:], self.pad, 0)], 0)
        self.E = E
        self.my = E.shape[0]
        self.m, self.L = m, int(np.log2(m))
        self.Eh, self.Ev = ts.np_tables["Eh"], ts.np_tables["Ev"]
        F = features(ts, self.E, **feat)
        rng = np.random.default_rng(seed)
        yy, xx = np.meshgrid(np.arange(self.my), np.arange(m), indexing="ij")
        self.levels = []
        for l in range(self.L + 1):
            h = 2 ** (self.L - l)
            if levels is not None and h not in levels:
                self.levels.append(None)
                continue
            El = _blur(F, 0.5 * h if h > 1 else 0, self.bounded)
            N = self._hood(El, yy, xx, h)                                  # (m*m, 25D)
            mean = N.mean(0)
            P = _pca(N - mean, n_pca, rng)
            NE = (N - mean) @ P                                            # (m*m, k)
            # second coherence candidate: best match at torus distance >= min_jump*m
            C2 = _second_candidate(NE, m, min_jump, rng, my=self.my, bounded=self.bounded) if c2 else None
            self.levels.append(dict(h=h, El=El, mean=mean, P=P, NE=NE, C2=C2))

    def _hood(self, El, ys, xs, h):
        """5x5 neighbourhood at spacing h around exemplar coords (ys, xs)."""
        parts = [El[self.wrap_y(ys + h * dy), (xs + h * dx) % self.m] for dy, dx in OFF5]
        return np.concatenate(parts, -1).reshape(-1, len(OFF5) * El.shape[-1])

    def wrap_y(self, y):
        """Exemplar row of a (possibly out of range) row coordinate."""
        return np.clip(y, 0, self.my - 1) if self.bounded else y % self.m

    def wrap(self, S):
        """Coordinates (..., 2) into the exemplar."""
        return np.stack([self.wrap_y(S[..., 0]), S[..., 1] % self.m], -1)


def _second_candidate(NE, m, min_jump, rng, exact_max=64, d_search=8, k=48, my=None, bounded=False):
    """For every exemplar pixel, the index of its best-matching neighbourhood
    at torus distance >= min_jump * m.  Exact brute force (row chunks, a
    hashed tie-break) up to side exact_max, so small exemplars keep their
    results; above that, a k-d tree on the first d_search PCA dimensions
    proposes k candidates that are re-scored on all dimensions (approximate,
    O(m^2 log m) instead of O(m^4)); a pixel with no far candidate among
    them lies in a homogeneous region (e.g. open sky), where a nearby match
    is as good an alternative as any, so it takes its best candidate."""
    my = m if my is None else my
    yy, xx = np.meshgrid(np.arange(my), np.arange(m), indexing="ij")
    Y, X = yy.ravel(), xx.ravel()
    far = max(1, min_jump * m)
    sq = (NE ** 2).sum(1)
    ty = (lambda d: d) if bounded else (lambda d: np.minimum(d, m - d))           # vertical distance

    def brute(rows, noise):
        d2 = sq[rows, None] + sq[None, :] - 2 * NE[rows] @ NE.T
        if noise:
            d2 += 1e-4 * rng.random(d2.shape)                              # hashed tie-break
        dy = ty(np.abs(Y[rows, None] - Y[None, :]))
        dx = np.abs(X[rows, None] - X[None, :]); dx = np.minimum(dx, m - dx)
        d2[np.maximum(dy, dx) < far] = np.inf
        return np.argmin(d2, 1)

    N = my * m
    if m <= exact_max:
        C2 = np.empty(N, np.int64)
        for a in range(0, N, 2048):
            C2[a:a + 2048] = brute(np.arange(a, min(a + 2048, N)), True)
        return np.stack([C2 // m, C2 % m], -1)
    from scipy.spatial import cKDTree
    _, cand = cKDTree(NE[:, :d_search]).query(NE[:, :d_search], k=k)            # (m*m, k)
    dy = ty(np.abs(Y[:, None] - Y[cand]))
    dx = np.abs(X[:, None] - X[cand]); dx = np.minimum(dx, m - dx)
    d2 = ((NE[:, None, :] - NE[cand]) ** 2).sum(-1)
    d2[cand == np.arange(N)[:, None]] = np.inf                                    # never the pixel itself
    d2_far = np.where(np.maximum(dy, dx) < far, np.inf, d2)
    # a far candidate when one is among the k; otherwise the region is homogeneous
    # (e.g. open sky) and a nearby match is as good an alternative as any
    best = np.where(np.isfinite(d2_far.min(1)), np.argmin(d2_far, 1), np.argmin(d2, 1))
    C2 = cand[np.arange(N), best]
    return np.stack([C2 // m, C2 % m], -1)


def synthesize(an: Analysis, n, r, seed=0, corrections=2, kappa=1.5, first_corrected=3, last_corrected=None,
               final_corrections=None, beta=0.0, T=0.0, jitter=None, ctx=None):
    """-> (n, n, 2) exemplar coords.  r: per-level jitter amplitude, length L+1
    (or a scalar for all levels).  Correction runs on levels
    first_corrected..last_corrected (default: to the finest); finer levels are
    only upsampled, so their patches stay contiguous.  final_corrections
    overrides the pass count at the finest level.  beta weighs the exact pair
    energy of a candidate tile with its four current neighbours into the cost
    at the finest level (the rule repair checks).  T > 0 samples corrections
    (Gumbel-max) instead of taking the argmin.  Output is toroidal.

    Runs CoordVar alone through hier.run; pass a hier.Ctx as ctx to keep
    every level (ctx.levels).  Jitter comes from hashed noise keyed on
    (seed, level, y, x), or from `jitter` (see CoordVar)."""
    assert n % an.m == 0 and n & (n - 1) == 0, "output side must be a power of 2, >= exemplar side"
    v = CoordVar(an, r, kappa=kappa, corrections=corrections, first_corrected=first_corrected,
                 last_corrected=last_corrected, final_corrections=final_corrections, beta=beta, T=T, jitter=jitter)
    ctx = hier.Ctx(seed) if ctx is None else ctx
    return hier.run([v], [], n, ctx).vars[v.name]


class CoordVar(hier.VarType):
    """Exemplar coordinate per cell (Lefebvre & Hoppe correction as a
    hierarchical variable).  Exists at h = m .. 1; texsyn level l has
    h = 2^(L - l).  jitter: None (hashed noise), a list indexed by texsyn
    level l of int arrays (n/h, n/h, 2), or a callable (l, h, shape) -> array."""
    name = "coord"

    def __init__(self, an: Analysis, r, kappa=1.5, corrections=2, first_corrected=3, last_corrected=None,
                 final_corrections=None, beta=0.0, T=0.0, jitter=None):
        self.an = an
        self.h_min, self.h_max = 1, an.m
        self.r = np.broadcast_to(np.asarray(r, float), (an.L + 1,))
        self.kappa, self.corrections, self.beta, self._T = kappa, corrections, beta, T
        self.first, self.last = first_corrected, an.L if last_corrected is None else last_corrected
        self.final = final_corrections
        self.jitter = jitter
        self.weights = np.tile([1.0, kappa], len(OFF3))

    def _l(self, h):
        return self.an.L - hier.level_index(h)

    def _nb(self, S, dy, dx):
        """S[p + (dy, dx)]: periodic horizontally; vertically periodic, or
        with a bounded exemplar the output's boundary: every cell above the
        map is deep sky (exemplar row 0), every cell below it deep ground
        (the last row)."""
        q = np.roll(S, (-dy, -dx), (0, 1))
        if self.an.bounded and dy:
            R = S.shape[0]
            rows = np.arange(R) + dy
            q = q.copy()
            q[rows < 0] = (0, 0)
            q[rows >= R] = (self.an.my - 1, 0)
        return q

    def _jitter(self, level, ctx):
        l, h, shape = self._l(level.h), level.h, level.shape + (2,)
        if self.jitter is not None:
            return np.asarray(self.jitter(l, h, shape) if callable(self.jitter) else self.jitter[l], np.int64)
        if self.r[l] == 0:
            return np.zeros(shape, np.int64)
        ys, xs = np.arange(shape[0])[:, None, None], np.arange(shape[1])[None, :, None]
        u = hier.noise(ctx.seed, hier.level_index(h), hier.INIT_STEP, 0, ys, xs, np.arange(2) + (self.salt << 20))
        return np.floor(h * self.r[l] * (2.0 * u - 1.0) + 0.5).astype(np.int64)

    def top(self, level, ctx):
        J = self._jitter(level, ctx)
        if self.an.bounded:                                               # anywhere in the padded exemplar
            return np.stack([J[..., 0] % self.an.my, J[..., 1] % self.an.m], -1)
        return J % self.an.m

    def init_children(self, parent, level, ctx):
        m, h = self.an.m, level.h
        S = np.empty(level.shape + (2,), np.int64)
        for dy in (0, 1):                                                  # four sub-grids, not cells
            for dx in (0, 1):
                S[dy::2, dx::2] = parent + np.array([(2 * dy - 1) * h // 2, (2 * dx - 1) * h // 2])
        return self.an.wrap(S + self._jitter(level, ctx))

    def sweeps(self, h):
        l = self._l(h)
        if not self.first <= l <= self.last:
            return 0
        return self.final if l == self.an.L and self.final is not None else self.corrections

    def T(self, h):
        return self._T(h) if callable(self._T) else self._T

    def candidates(self, level, colour, ctx):
        """The 3x3 coherent candidates and their C2 partners, minus h*delta."""
        S, lv, m, h = level.vars[self.name], self.an.levels[self._l(level.h)], self.an.m, level.h
        i, j = colour
        cands = []
        for dy, dx in OFF3:
            q = self._nb(S, dy, dx)[i::2, j::2]                            # S[p + delta]
            shift = h * np.array([dy, dx])
            cands.append(self.an.wrap(q - shift))
            cands.append(self.an.wrap(lv["C2"][q[..., 0] * m + q[..., 1]] - shift))
        return np.stack(cands, 2)                                          # (a, b, 18, 2)

    def energy(self, C, level, colour, ctx):
        """PCA'd 5x5 neighbourhood distance x (1 or kappa), plus beta x the
        pair energy with the four current neighbours at h = 1."""
        S, an, m = level.vars[self.name], self.an, self.an.m
        lv = an.levels[self._l(level.h)]
        i, j = colour
        parts = []
        for dy, dx in OFF5:
            q = self._nb(S, dy, dx)[i::2, j::2]
            parts.append(lv["El"][q[..., 0], q[..., 1]])
        NS = (np.concatenate(parts, -1) - lv["mean"]) @ lv["P"]              # (a, b, k)
        NE = lv["NE"][C[..., 0] * m + C[..., 1]]                             # (a, b, 18, k)
        cost = ((NS[:, :, None] - NE) ** 2).sum(-1) * self.weights
        if self.beta and level.h == 1:
            t = an.E[C[..., 0], C[..., 1]]
            Tl = an.E[S[..., 0], S[..., 1]]
            nb = lambda dy, dx: np.roll(Tl, (-dy, -dx), (0, 1))[i::2, j::2][..., None]
            cost = cost + self.beta * (an.Eh[nb(0, -1), t] + an.Eh[t, nb(0, 1)]
                                       + an.Ev[nb(-1, 0), t] + an.Ev[t, nb(1, 0)])
        return cost


def _synthesize_legacy(an: Analysis, n, r, seed=0, corrections=2, kappa=1.5, first_corrected=3, last_corrected=None,
                       final_corrections=None, beta=0.0, jitter=None, levels_out=None):
    """Pre-framework implementation, kept for the equivalence test in
    tests/test_hier.py.  jitter: optional per-level override (see CoordVar);
    levels_out: optional list that receives S after every level.
    -> (n, n, 2) exemplar coords.  r: per-level jitter amplitude, length L+1
    (or a scalar for all levels).  Correction runs on levels
    first_corrected..last_corrected (default: to the finest); finer levels are
    only upsampled, so their patches stay contiguous.  final_corrections
    overrides the pass count at the finest level.  beta weighs the exact pair
    energy of a candidate tile with its four current neighbours into the cost
    at the finest level (the rule repair checks).  Output is toroidal."""
    m, L = an.m, an.L
    assert n % m == 0 and n & (n - 1) == 0, "output side must be a power of 2, >= exemplar side"
    r = np.broadcast_to(np.asarray(r, float), (L + 1,))
    rng = np.random.default_rng(seed)
    S = np.zeros((n // m, n // m, 2), np.int64)
    for l in range(L + 1):
        h = 2 ** (L - l)
        if l > 0:                                                         # upsample
            up = np.zeros((2 * S.shape[0], 2 * S.shape[1], 2), np.int64)
            for dy in (0, 1):
                for dx in (0, 1):
                    off = np.array([(2 * dy - 1) * h // 2, (2 * dx - 1) * h // 2])
                    up[dy::2, dx::2] = (S + off) % m
            S = up
        J = np.floor(h * r[l] * rng.uniform(-1, 1, S.shape) + 0.5).astype(np.int64)  # jitter
        if jitter is not None:
            J = np.asarray(jitter(l, h, S.shape) if callable(jitter) else jitter[l], np.int64)
        S = (S + J) % m
        if first_corrected <= l <= (L if last_corrected is None else last_corrected):
            for _ in range(final_corrections if l == L and final_corrections is not None else corrections):
                S = _correct_legacy(an, an.levels[l], S, kappa, beta if l == L else 0.0)
        if levels_out is not None:
            levels_out.append(S)
    return S


def _correct_legacy(an, lv, S, kappa, beta=0.0):
    m, h = an.m, lv["h"]
    n = S.shape[0]
    for i, j in ((0, 0), (1, 1), (1, 0), (0, 1)):
        # neighbourhood of the synthesized map: E_l at each neighbour's own coord
        parts = []
        for dy, dx in OFF5:
            q = np.roll(S, (-dy, -dx), (0, 1))[i::2, j::2]
            parts.append(lv["El"][q[..., 0], q[..., 1]])
        NS = (np.concatenate(parts, -1) - lv["mean"]) @ lv["P"]              # (n/2, n/2, k)
        cands, weights = [], []
        for dy, dx in OFF3:
            q = np.roll(S, (-dy, -dx), (0, 1))[i::2, j::2]                   # S[p + Δ]
            shift = h * np.array([dy, dx])
            cands.append((q - shift) % m); weights.append(1.0)
            cands.append((lv["C2"][q[..., 0] * m + q[..., 1]] - shift) % m); weights.append(kappa)
        C = np.stack(cands, 2)                                               # (a, b, 18, 2)
        NE = lv["NE"][C[..., 0] * m + C[..., 1]]                             # (a, b, 18, k)
        cost = ((NS[:, :, None] - NE) ** 2).sum(-1) * np.array(weights)
        if beta:
            t = an.E[C[..., 0], C[..., 1]]                                   # candidate tiles (a, b, 18)
            T = an.E[S[..., 0], S[..., 1]]
            nb = lambda dy, dx: np.roll(T, (-dy, -dx), (0, 1))[i::2, j::2][..., None]
            cost += beta * (an.Eh[nb(0, -1), t] + an.Eh[t, nb(0, 1)] + an.Ev[nb(-1, 0), t] + an.Ev[t, nb(1, 0)])
        best = np.argmin(cost, 2)
        S = S.copy()
        S[i::2, j::2] = np.take_along_axis(C, best[..., None, None], 2)[:, :, 0]
    return S


def seam_mask(S, m):
    """Cells with a 4-neighbour whose coordinate is not contiguous."""
    seam = np.zeros(S.shape[:2], bool)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        q = np.roll(S, (-dy, -dx), (0, 1))
        seam |= ((q - S) % m != np.array([dy, dx]) % m).any(-1)
    return seam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exemplar", help="signature grid .npy (from castlegen.exemplar --out)")
    ap.add_argument("--tileset", default="demo")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--r", type=float, nargs="+", default=[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    help="jitter per level, coarse to fine (or one value)")
    ap.add_argument("--kappa", type=float, default=4.0)
    ap.add_argument("--repair", type=int, default=20, help="repair Gibbs rounds")
    ap.add_argument("--repair-T", type=float, default=0.4)
    ap.add_argument("--tau", type=float, default=1.9, help="pair energy counted as a violation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--png", default=None)
    ap.add_argument("--px", type=int, default=8)
    args = ap.parse_args()
    ts = tileset.load(args.tileset)
    E = np.load(args.exemplar)
    an = Analysis(ts, E)
    S = synthesize(an, args.size, args.r if len(args.r) > 1 else args.r[0], args.seed, kappa=args.kappa)
    tiles = E[S[..., 0], S[..., 1]]
    seam = seam_mask(S, an.m)
    bad = ex.bad_cells(ts, tiles, args.tau)
    print(f"seams {seam.mean():.3f}  bad cells {bad.mean():.3f} (outside seams {(bad & ~ex.dilate(seam, 1)).mean():.4f})  "
          f"census {tileset.structure_census(ts, tiles)}")
    if args.png:
        ex.to_png(ts, tiles, args.png.replace(".png", "_raw.png"), args.px)
    fixed = ex.repair(ts, tiles, args.repair, args.seed, T=args.repair_T, tau=args.tau, clear_orphans=True)
    print(f"after repair: bad cells {ex.bad_cells(ts, fixed, args.tau).mean():.4f}  "
          f"changed {(fixed != tiles).mean():.3f}  census {tileset.structure_census(ts, fixed)}")
    if args.png:
        ex.to_png(ts, fixed, args.png, args.px)
        print("wrote", args.png, "and", args.png.replace(".png", "_raw.png"))


if __name__ == "__main__":
    main()
