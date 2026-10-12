"""Infinite tile worlds by annealed checkerboard Gibbs with position-hashed
noise: the value at a world cell is a pure function f(seed, y, x), so any
two requests agree exactly on their overlap.

Model: a pair MRF over T tiles on the 4-neighbour lattice.
    soft   E(x) = sum_c u[x_c] + sum_h Eh[left, right] + sum_v Ev[top, bottom]
    hard   V(x) = number of forbidden pairs (Hh, Hv: 1 forbidden, 0 allowed)
Sweep s draws every cell from exp(-(beta[s] E + gamma[s] V)) restricted to
its four neighbours.  gamma[s] = inf is the lexicographic limit: the cell
keeps only the tiles with the fewest violations and draws among them by
beta[s] E, so a contradiction (no tile fits) leaves the least-bad tile rather
than an undefined state.

Schedule: S sweeps of two half-steps; half-step h = 1 .. 2S updates the
cells of global parity (y + x) & 1 == (h - 1) & 1.  Step 0 draws every cell
from exp(-beta[0] u).  Every draw takes its uniform from
hash(seed, y, x, h), never from a stream.

Light cone: a cell updated at half-step h reads neighbours from h - 1, so
after h half-steps the cells at margin >= h from the edge of the computed
block are exact.  A requested rectangle is therefore computed on itself
padded by PAD = 2S on every side, and each half-step updates only the
shrinking interior (margin >= h), a truncated pyramid.

World: chunks of C x C cells in a dict.  A request for missing chunks is
generated as one batch (their bounding rectangle, padded), which amortises
the padding; a later neighbouring request recomputes its own cone from the
same hashed noise and agrees with what is stored."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

M64 = (1 << 64) - 1


# ------------------------------------------------------------------ noise
@njit(cache=True, inline="always")
def _mix(z):
    """splitmix64 finaliser on uint64."""
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@njit(cache=True, inline="always")
def _u01(seed, y, x, h):
    """Uniform in [0, 1) from (seed, y, x, h); y, x any int64 (negative
    coordinates wrap to distinct uint64 keys)."""
    z = _mix(np.uint64(seed) ^ np.uint64(0x9E3779B97F4A7C15))
    z = _mix(z ^ np.uint64(y))
    z = _mix(z ^ (np.uint64(x) * np.uint64(0xD1B54A32D192ED03)))
    z = _mix(z ^ (np.uint64(h) * np.uint64(0x8CB92BA72F3D8DD7)))
    return np.float64(z >> np.uint64(11)) * (1.0 / 9007199254740992.0)


def u01(seed, y, x, h):
    """Python access to the hashed uniform (tests)."""
    return _u01(np.uint64(seed & M64), np.int64(y), np.int64(x), np.int64(h))


# ------------------------------------------------------------------ kernel
@njit(cache=True)
def _draw(w, n, r):
    """Index drawn from unnormalised weights w[:n] with uniform r."""
    tot = 0.0
    for t in range(n):
        tot += w[t]
    r *= tot
    acc = 0.0
    for t in range(n):
        acc += w[t]
        if r < acc:
            return t
    for t in range(n - 1, -1, -1):          # r == tot by rounding: last positive weight
        if w[t] > 0.0:
            return t
    return 0


@njit(cache=True)
def _anneal(grid, gy0, gx0, seed, u, Eh, Ev, Hh, Hv, beta, gamma, ups):
    """In place on grid (H, W) whose cell (0, 0) is world (gy0, gx0).  After
    return the cells at margin >= 2S are exact.  ups[0] counts updates."""
    H, W = grid.shape
    T = u.shape[0]
    S = beta.shape[0]
    e = np.empty(T)
    v = np.empty(T, np.int64)
    w = np.empty(T)
    # step 0: independent draws from the unary at beta[0]
    b0 = beta[0]
    umin = u.min()
    for t in range(T):
        w[t] = np.exp(-b0 * (u[t] - umin))
    for i in range(H):
        for j in range(W):
            grid[i, j] = _draw(w, T, _u01(seed, gy0 + i, gx0 + j, 0))
    for h in range(1, 2 * S + 1):
        s = (h - 1) >> 1
        b, g = beta[s], gamma[s]
        hard_inf = np.isinf(g)
        par = (h - 1) & 1
        for i in range(h, H - h):
            j0 = h + ((par - (gy0 + i + gx0 + h)) & 1)
            for j in range(j0, W - h, 2):
                up, dn = grid[i - 1, j], grid[i + 1, j]
                lf, rt = grid[i, j - 1], grid[i, j + 1]
                vmin = 1 << 30
                for t in range(T):
                    e[t] = u[t] + Eh[lf, t] + Eh[t, rt] + Ev[up, t] + Ev[t, dn]
                    v[t] = Hh[lf, t] + Hh[t, rt] + Hv[up, t] + Hv[t, dn]
                    if v[t] < vmin:
                        vmin = v[t]
                emin = np.inf
                for t in range(T):
                    if hard_inf:
                        x = b * e[t] if v[t] == vmin else np.inf
                    else:
                        x = b * e[t] + g * v[t]
                    e[t] = x
                    if x < emin:
                        emin = x
                for t in range(T):
                    w[t] = np.exp(-(e[t] - emin))
                grid[i, j] = _draw(w, T, _u01(seed, gy0 + i, gx0 + j, h))
                ups[0] += 1


@njit(cache=True)
def _violations(grid, Hh, Hv):
    """Number of forbidden horizontal + vertical pairs inside grid."""
    H, W = grid.shape
    n = 0
    for i in range(H):
        for j in range(W):
            if j + 1 < W:
                n += Hh[grid[i, j], grid[i, j + 1]]
            if i + 1 < H:
                n += Hv[grid[i, j], grid[i + 1, j]]
    return n


# ------------------------------------------------------------------ model
@dataclass
class TileModel:
    """T tiles.  u (T,) soft unary; Eh[a, b] soft energy of a left of b,
    Ev[a, b] of a above b (finite); Hh, Hv the same pairs forbidden (bool).
    colors (T, 3) uint8 for render."""
    names: list
    u: np.ndarray
    Eh: np.ndarray
    Ev: np.ndarray
    Hh: np.ndarray
    Hv: np.ndarray
    colors: np.ndarray

    def __post_init__(self):
        T = len(self.names)
        self.u = np.ascontiguousarray(self.u, np.float64).reshape(T)
        for k in ("Eh", "Ev"):
            a = np.ascontiguousarray(getattr(self, k), np.float64).reshape(T, T)
            if not np.isfinite(a).all():
                raise ValueError(f"{k}: soft energies must be finite; forbid pairs with Hh / Hv")
            setattr(self, k, a)
        for k in ("Hh", "Hv"):
            setattr(self, k, np.ascontiguousarray(getattr(self, k), np.int64).reshape(T, T))
        self.colors = np.asarray(self.colors, np.uint8).reshape(T, 3)

    @property
    def T(self):
        return len(self.names)

    def run(self, grid, gy0, gx0, seed, sched, ups):
        _anneal(grid, gy0, gx0, seed, self.u, self.Eh, self.Ev, self.Hh, self.Hv, sched.beta, sched.gamma, ups)

    def energy(self, grid):
        """(soft energy, hard violations) of a finite grid, interior pairs only."""
        g = grid
        E = self.u[g].sum() + self.Eh[g[:, :-1], g[:, 1:]].sum() + self.Ev[g[:-1, :], g[1:, :]].sum()
        return float(E), int(_violations(g, self.Hh, self.Hv))

    def render(self, grid, scale=1):
        img = self.colors[grid]
        if scale > 1:
            img = np.repeat(np.repeat(img, scale, 0), scale, 1)
        return img


@njit(cache=True)
def _anneal_sockets(grid, gy0, gx0, seed, base, sN, sE, sS, sW, beta, gamma, ups, cone=True, fix=False):
    """_anneal for socket tiles: tile t fits its left neighbour l iff
    sE[l] == sW[t] (likewise right, up, down); V counts misfits; the soft
    energy is unary only.  base[s, t] = exp(-beta[s] (u[t] - min u)).
    cone=False updates every cell but the border at every half-step (a
    fixed-boundary run for statistics, not exact).
    fix=True adds half-step 2S + 1 on parity 0, lexicographic at beta[S-1]:
    a cell with no tile fitting all four neighbours takes the wildcard T
    (fits anything).  Every pair has one parity-0 cell and parity 1 is not
    updated again, so the result has no misfit; the cone grows by one."""
    H, W = grid.shape
    T = base.shape[1]
    S = beta.shape[0]
    w = np.empty(T)
    v = np.empty(T, np.int64)
    cdf = np.empty(T)
    acc = 0.0
    for t in range(T):
        acc += base[0, t]
        cdf[t] = acc
    for i in range(H):
        for j in range(W):
            r = _u01(seed, gy0 + i, gx0 + j, 0) * acc
            grid[i, j] = min(np.searchsorted(cdf, r, side="right"), T - 1)
    eg = np.empty(5)
    for h in range(1, 2 * S + 1 + fix):
        s = min((h - 1) >> 1, S - 1)
        g = np.inf if h > 2 * S else gamma[s]
        hard_inf = np.isinf(g)
        for k in range(5):
            eg[k] = 0.0 if hard_inf and k > 0 else np.exp(-g * k)
        par = (h - 1) & 1
        bs = base[s]
        m = h if cone else 1
        for i in range(m, H - m):
            j0 = m + ((par - (gy0 + i + gx0 + m)) & 1)
            for j in range(j0, W - m, 2):
                a_up, a_dn = sS[grid[i - 1, j]], sN[grid[i + 1, j]]
                a_lf, a_rt = sE[grid[i, j - 1]], sW[grid[i, j + 1]]
                vmin = 4
                for t in range(T):
                    n = (sN[t] != a_up) + (sS[t] != a_dn) + (sW[t] != a_lf) + (sE[t] != a_rt)
                    v[t] = n
                    if n < vmin:
                        vmin = n
                ups[0] += 1
                if h > 2 * S and vmin > 0:
                    grid[i, j] = T
                    continue
                if hard_inf:
                    for t in range(T):
                        w[t] = bs[t] if v[t] == vmin else 0.0
                else:
                    for t in range(T):
                        w[t] = bs[t] * eg[v[t]]
                grid[i, j] = _draw(w, T, _u01(seed, gy0 + i, gx0 + j, h))


@njit(cache=True)
def _socket_violations(grid, sN, sE, sS, sW):
    """Misfit pairs; socket -1 (the wildcard's) fits anything."""
    H, W = grid.shape
    n = 0
    for i in range(H):
        for j in range(W):
            if j + 1 < W:
                a, b = sE[grid[i, j]], sW[grid[i, j + 1]]
                n += a != b and a >= 0 and b >= 0
            if i + 1 < H:
                a, b = sS[grid[i, j]], sN[grid[i + 1, j]]
                n += a != b and a >= 0 and b >= 0
    return n


@dataclass
class SocketModel:
    """Edge-socket tiles for large T: tile t has sockets sN, sE, sS, sW
    (ints); horizontal neighbours fit iff sE[left] == sW[right], vertical iff
    sS[top] == sN[bottom].  Soft energy unary u (e.g. -log WFC weight).  Cost
    per update O(T) socket compares, no T x T tables.
    wild_u: if set, a final fix half-step puts the wildcard tile T (fits any
    socket, energy wild_u) where no tile fits; output ids then include T."""
    u: np.ndarray
    sN: np.ndarray
    sE: np.ndarray
    sS: np.ndarray
    sW: np.ndarray
    wild_u: float = None

    def __post_init__(self):
        self.u = np.ascontiguousarray(self.u, np.float64)
        for k in ("sN", "sE", "sS", "sW"):
            setattr(self, k, np.ascontiguousarray(getattr(self, k), np.int64))

    @property
    def T(self):
        return self.u.size

    @property
    def extra_pad(self):
        return 0 if self.wild_u is None else 1

    def run(self, grid, gy0, gx0, seed, sched, ups, cone=True):
        base = np.exp(-sched.beta[:, None] * (self.u - self.u.min())[None, :])
        _anneal_sockets(grid, gy0, gx0, seed, base, self.sN, self.sE, self.sS, self.sW,
                        sched.beta, sched.gamma, ups, cone, self.wild_u is not None)

    def energy(self, grid):
        """(soft energy, misfit pairs); the wildcard counts wild_u."""
        ext = lambda a, x: np.append(a, x)
        u = ext(self.u, 0.0 if self.wild_u is None else self.wild_u)
        return float(u[grid].sum()), int(_socket_violations(grid, ext(self.sN, -1), ext(self.sE, -1),
                                                             ext(self.sS, -1), ext(self.sW, -1)))


@dataclass
class Schedule:
    """beta[s], gamma[s] for sweeps s = 0 .. S-1 (gamma may be inf)."""
    beta: np.ndarray
    gamma: np.ndarray

    def __post_init__(self):
        self.beta = np.ascontiguousarray(self.beta, np.float64)
        self.gamma = np.ascontiguousarray(self.gamma, np.float64)
        if self.beta.shape != self.gamma.shape or self.beta.ndim != 1 or self.beta.size == 0:
            raise ValueError("beta and gamma: equal 1-d, non-empty")
        if not np.isfinite(self.beta).all() or (self.beta < 0).any() or (self.gamma < 0).any():
            raise ValueError("beta finite >= 0, gamma >= 0")

    @property
    def S(self):
        return self.beta.size

    @property
    def pad(self):
        return 2 * self.S

    @classmethod
    def anneal(cls, S=64, beta0=0.1, beta1=1.0, gamma0=0.5, gamma1=8.0, quench=4):
        """Geometric ramps over S - quench sweeps, then `quench` sweeps at
        beta1 with hard constraints lexicographic (gamma = inf).  beta1 = 1
        ends at T = 1: a sample of the soft model, not its argmin."""
        n = S - quench
        r = np.linspace(0.0, 1.0, max(n, 1))
        beta = beta0 * (beta1 / beta0) ** r
        gamma = gamma0 * (gamma1 / gamma0) ** r
        beta = np.concatenate([beta[:n], np.full(quench, beta1)])
        gamma = np.concatenate([gamma[:n], np.full(quench, np.inf)])
        return cls(beta, gamma)


def generate(model, sched: Schedule, seed, y0, x0, H, W, pad=None, stats=None):
    """The world cells [y0, y0 + H) x [x0, x0 + W), (H, W) int32.  pad
    defaults to the light cone 2S (smaller pads are for tests: they are not
    exact)."""
    P = sched.pad + getattr(model, "extra_pad", 0) if pad is None else pad
    grid = np.zeros((H + 2 * P, W + 2 * P), np.int32)
    ups = np.zeros(1, np.int64)
    model.run(grid, np.int64(y0 - P), np.int64(x0 - P), np.uint64(seed & M64), sched, ups)
    if stats is not None:
        stats["updates"] = stats.get("updates", 0) + int(ups[0])
        stats["cells"] = stats.get("cells", 0) + H * W
    return grid[P:P + H, P:P + W].copy()


def bulk(model, sched, seed, H, W):
    """Statistics run: an (H, W) block with a fixed random border, every
    interior cell updated at every half-step (SocketModel only).  Not exact
    across requests; measure away from the border."""
    grid = np.zeros((H, W), np.int32)
    ups = np.zeros(1, np.int64)
    model.run(grid, np.int64(0), np.int64(0), np.uint64(seed & M64), sched, ups, cone=False)
    return grid


class World:
    """Chunks of C x C cells generated on demand.  chunk (cy, cx) covers the
    cells [cy C, cy C + C) x [cx C, cx C + C)."""

    def __init__(self, model: TileModel, sched: Schedule, seed=0, C=32):
        self.model, self.sched, self.seed, self.C = model, sched, int(seed), int(C)
        self.chunks: dict = {}
        self.stats: dict = {}

    def ensure(self, keys):
        """Generate the missing chunks among keys as one batch (their bounding
        rectangle), keep every chunk of that rectangle."""
        miss = [k for k in keys if k not in self.chunks]
        if not miss:
            return
        cys = [k[0] for k in miss]
        cxs = [k[1] for k in miss]
        cy0, cy1, cx0, cx1 = min(cys), max(cys) + 1, min(cxs), max(cxs) + 1
        C = self.C
        g = generate(self.model, self.sched, self.seed, cy0 * C, cx0 * C, (cy1 - cy0) * C, (cx1 - cx0) * C,
                     stats=self.stats)
        for cy in range(cy0, cy1):
            for cx in range(cx0, cx1):
                if (cy, cx) not in self.chunks:
                    self.chunks[(cy, cx)] = g[(cy - cy0) * C:(cy - cy0 + 1) * C, (cx - cx0) * C:(cx - cx0 + 1) * C].copy()

    def chunk(self, cy, cx):
        self.ensure([(cy, cx)])
        return self.chunks[(cy, cx)]

    def region(self, y0, x0, H, W):
        """Cells [y0, y0 + H) x [x0, x0 + W) assembled from chunks."""
        C = self.C
        cy0, cy1 = y0 // C, -(-(y0 + H) // C)
        cx0, cx1 = x0 // C, -(-(x0 + W) // C)
        self.ensure([(cy, cx) for cy in range(cy0, cy1) for cx in range(cx0, cx1)])
        out = np.empty(((cy1 - cy0) * C, (cx1 - cx0) * C), np.int32)
        for cy in range(cy0, cy1):
            for cx in range(cx0, cx1):
                out[(cy - cy0) * C:(cy - cy0 + 1) * C, (cx - cx0) * C:(cx - cx0 + 1) * C] = self.chunks[(cy, cx)]
        oy, ox = y0 - cy0 * C, x0 - cx0 * C
        return out[oy:oy + H, ox:ox + W]


# ------------------------------------------------------------------ test tile sets
def socket_model(names, sides, u, colors, eh_soft=None, ev_soft=None):
    """Edge-socket (Wang) tiles: sides[t] = (N, E, S, W) socket ids; a pair is
    forbidden unless the facing sockets are equal.  Optional soft pair
    tables on top."""
    sides = np.asarray(sides)
    T = len(names)
    Hh = (sides[:, 1][:, None] != sides[:, 3][None, :])
    Hv = (sides[:, 2][:, None] != sides[:, 0][None, :])
    Eh = np.zeros((T, T)) if eh_soft is None else eh_soft
    Ev = np.zeros((T, T)) if ev_soft is None else ev_soft
    return TileModel(list(names), np.asarray(u, float), Eh, Ev, Hh, Hv, colors)


def roads(dead_end=None, cross=6.0, tee=5.0, corner=3.5, straight=2.5, empty=0.0):
    """16 road tiles by arm mask (N=1, E=2, S=4, W=8), arms must meet arms.
    dead_end=None leaves out the 1-arm tiles, so the constraint set has
    contradictions (a lone arm cannot be closed): a WFC-style test."""
    names, sides, u = [], [], []
    for m in range(16):
        n = bin(m).count("1")
        if n == 1 and dead_end is None:
            continue
        straight_m = m in (5, 10)
        cost = {0: empty, 1: dead_end, 2: straight if straight_m else corner, 3: tee, 4: cross}[n]
        names.append(f"road{m:x}")
        sides.append([(m >> k) & 1 for k in range(4)])
        u.append(cost)
    colors = [(120, 170, 80) if s == [0, 0, 0, 0] else (200, 180, 130) for s in sides]
    return socket_model(names, sides, u, colors)


def render_roads(model, grid, cell=5):
    """Roads drawn as arms on a cell x cell glyph (odd cell)."""
    T = model.T
    c = cell // 2
    glyph = np.zeros((T, cell, cell), bool)
    for t, nm in enumerate(model.names):
        m = int(nm[4:], 16)
        if m:
            glyph[t, c, c] = True
        if m & 1: glyph[t, :c + 1, c] = True
        if m & 2: glyph[t, c, c:] = True
        if m & 4: glyph[t, c:, c] = True
        if m & 8: glyph[t, c, :c + 1] = True
    H, W = grid.shape
    g = glyph[grid].transpose(0, 2, 1, 3).reshape(H * cell, W * cell)
    img = np.empty(g.shape + (3,), np.uint8)
    img[:] = (120, 170, 80)
    img[g] = (90, 70, 50)
    return img


def biomes(J=0.6, J2=0.25):
    """Eight elevation bands; adjacent tiles may differ by at most one band
    (hard); soft: -J for equal neighbours, +J2 for a one-band step."""
    names = ["deep", "water", "sand", "grass", "forest", "hill", "rock", "snow"]
    colors = [(30, 60, 120), (60, 110, 180), (220, 205, 150), (120, 180, 80), (40, 120, 50),
              (140, 130, 90), (120, 115, 110), (240, 240, 245)]
    u = np.zeros(8)
    T = len(names)
    d = np.abs(np.arange(T)[:, None] - np.arange(T)[None, :])
    E = np.where(d == 0, -J, np.where(d == 1, J2, 0.0))
    H = d > 1
    return TileModel(names, u, E, E, H, H, colors)
