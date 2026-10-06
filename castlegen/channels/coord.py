"""Masked exemplar coordinates for roots (channels.tex sections 5 and 9;
the painted-mask direction of notes/dsl.md).

A level-1 coordinate channel u holds, per cell, an index into a painted
exemplar.  Its view `alpha` is the exemplar's statement at that
coordinate: FREE (masked out: the cell belongs to whoever else speaks,
the ground), EARTH (the ring around the roots: any earth tile, never sky
or root), or a specific root / trunk tile.  One pair factor at offset
(0, 0) couples the tile channel to alpha(u) with cost nu per mismatch, so
the tile kernel needs no change; the certificate, port seam and contact
rules keep acting on the tiles.

The coordinate kernel (sweep) draws each u_p by Gibbs over a candidate
set that does not depend on u_p: the current value, the coherent
continuation of each neighbour's coordinate (u_q + (p - q)), K random
coordinates, and K_t random coordinates whose alpha equals the current
tile at p.  Energy of a candidate c:
    lam  * number of neighbours q with c != u_q + (p - q), unless both
           c and u_q are FREE (free cells owe each other nothing)  coherence
    w    * number of cells s in the (2R+1)^2 window with alpha(c + s)
           not FREE and the tile at p + s not matching it, s != 0  patch
           (default 0: it blocks the copy from spreading into cells
           whose tiles have not followed yet)
    nu   * [tile at p does not match alpha(c)]                      coupling
The energy minimum is a verbatim copy of the exemplar's masked region
around a trunk, with FREE continuations outside its ring; the random
candidates and the soft coupling are where variation comes from.
A Gibbs step over a subset containing the current state leaves the
joint invariant when the subset's choice does not depend on the value
being resampled, which holds here.

The exemplar (EXEMPLAR) is ASCII: T trunk, 1 2 3 root mass, '.' earth,
' ' sky; the mask is the roots plus a ring of RING cells of earth; the
rest is FREE.  Root cells get ports toward every 4-adjacent root cell."""
import numpy as np
from numba import njit

from .core import Channel, Factor, Kinds
from . import roots as R
from . import kernel as KN

FREE, EARTH = 0, 1                                             # alpha codes; a tile kind k is 2 + k
INF = np.inf

EXEMPLAR = [
    "                                            ",
    "                                            ",
    ".....................T......................",
    "................333333333333................",
    ".............3333333333333333333............",
    "...........33333...33333...333333...........",
    ".........2222.......3333......2222..........",
    "........222.........3333........222.........",
    ".......22...........333..........22.........",
    "......11............333...........11........",
    ".....11............2222............1........",
    ".....1.............222.............1........",
    "...................22.......................",
    "..................11........................",
    "..................1.........................",
    "............................................",
    "............................................",
]
CH_MASS = {" ": -1, ".": 0, "T": 3, "1": 1, "2": 2, "3": 3}


# --------------------------------------------------------------- exemplar
def parse_exemplar(kinds: Kinds, rows=EXEMPLAR, ring=1):
    """(alpha (my, mx) int codes, mass grid, mask bool, trunk (y, x)).
    Root cells become root{m}_{ports}; the trunk keeps its S port only."""
    g = np.array([[CH_MASS[c] for c in r] for r in rows], np.int64)
    trunk = np.array([[c == "T" for c in r] for r in rows])
    my, mx = g.shape
    alpha = np.full((my, mx), FREE, np.int64)
    rootish = g > 0
    for y in range(my):
        for x in range(mx):
            if not rootish[y, x]:
                continue
            if trunk[y, x]:
                alpha[y, x] = 2 + kinds.index("trunk")
                continue
            ps = ""
            for s, (dy, dx) in zip("NESW", ((-1, 0), (0, 1), (1, 0), (0, -1))):
                ny, nx = y + dy, x + dx
                if 0 <= ny < my and 0 <= nx < mx and rootish[ny, nx]:
                    ps += s
            if not ps:
                raise ValueError(f"isolated root cell at {(y, x)}")
            alpha[y, x] = 2 + kinds.index(f"root{g[y, x]}_{ps}")
    mask = rootish.copy()
    for _ in range(ring):
        d = mask.copy()
        d[1:] |= mask[:-1]; d[:-1] |= mask[1:]; d[:, 1:] |= mask[:, :-1]; d[:, :-1] |= mask[:, 1:]
        mask = d
    mask &= g >= 0                                             # never claim sky
    alpha[mask & ~rootish] = EARTH
    ty, tx = np.argwhere(trunk)[0]
    return alpha, g, mask, (int(ty), int(tx))


def check_exemplar(kinds: Kinds, alpha, g):
    """Every root cell reachable from the trunk through joined cells of
    non-increasing mass (the certificate is satisfiable on the exemplar)."""
    v = R.views(kinds)
    my, mx = g.shape
    seen = np.zeros_like(g, bool)
    ty, tx = np.argwhere(alpha == 2 + kinds.index("trunk"))[0]
    stack = [(ty, tx)]
    seen[ty, tx] = True
    while stack:
        y, x = stack.pop()
        t = alpha[y, x] - 2
        for i, (dy, dx) in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < my and 0 <= nx < mx) or seen[ny, nx] or alpha[ny, nx] < 2:
                continue
            tq = alpha[ny, nx] - 2
            if v["p" + "NESW"[i]][t] and v["p" + "NESW"[(i + 2) % 4]][tq] and v["mass"][tq] <= v["mass"][t]:
                seen[ny, nx] = True
                stack.append((ny, nx))
    unreached = int(((alpha >= 2) & ~seen).sum())
    return unreached


# --------------------------------------------------------------- channels
def coord_channel(alpha, name="u"):
    """D = my * mx coordinates, view alpha (codes)."""
    my, mx = alpha.shape
    c = Channel(name, 1, my * mx)
    c.add_view("alpha", alpha.ravel(), int(alpha.max()) + 1)
    return c


def coupling(kinds: Kinds, tile: Channel, u: Channel, nu=2.0, name="coord"):
    """Pair factor tile x alpha(u): nu per mismatch; EARTH matches any earth
    tile (root view EARTH); FREE matches anything but a root or trunk."""
    rv = tile.views["root"]
    A = u.nvals("alpha")
    tab = np.zeros((tile.D, A))
    tab[:, FREE] = np.where((rv == R.ROOT) | (rv == R.TRUNK), nu, 0.0)   # FREE: the ground's cell, no root
    tab[:, EARTH] = np.where(rv == R.EARTH, 0.0, nu)
    for k in range(tile.D):
        a = 2 + k
        if a < A:
            tab[:, a] = nu
            tab[k, a] = 0.0
    return Factor.pair((tile.name, "self"), (u.name, "alpha"), (0, 0), tab, name=name), tab


# ----------------------------------------------------------------- kernel
@njit(cache=True)
def seed(s):
    np.random.seed(s)


@njit(cache=True, inline="always")
def _match(t, a, rootv):
    """Tile t matches alpha code a."""
    if a == 0:
        return True
    if a == 1:
        return rootv[t] == 1
    return t == a - 2


@njit(cache=True, inline="always")
def _energy(c, y, x, U, T, alpha, rootv, my, mx, H, W, lam, w, nu, Rr):
    cy, cx = c // mx, c % mx
    e = 0.0
    a_c = alpha[cy, cx]
    for d in range(4):
        dy, dx = ((-1, 0), (0, 1), (1, 0), (0, -1))[d]
        qy, qx = y + dy, x + dx
        if 0 <= qy < H and 0 <= qx < W:
            uq = U[qy, qx]
            if uq // mx - dy != cy or uq % mx - dx != cx:
                if a_c != 0 or alpha[uq // mx, uq % mx] != 0:           # two FREE cells owe each other nothing
                    e += lam
    for sy in range(-Rr, Rr + 1):
        for sx in range(-Rr, Rr + 1):
            ey, ex = cy + sy, cx + sx
            py, px = y + sy, x + sx
            if not (0 <= ey < my and 0 <= ex < mx and 0 <= py < H and 0 <= px < W):
                continue
            a = alpha[ey, ex]
            if a == 0:
                continue
            if not _match(T[py, px], a, rootv):
                e += nu if (sy == 0 and sx == 0) else w
    return e


@njit(cache=True)
def sweep(U, T, alpha, rootv, by_alpha, by_start, lam, w, nu, Rr, K, Kt, fixed):
    """One Gibbs sweep of the coordinates (sequential; sites interact
    through coherence and patches, so there is no exact colouring)."""
    H, W = U.shape
    my, mx = alpha.shape
    N = my * mx
    NC = 1 + 4 + K + Kt
    cand = np.empty(NC, np.int64)
    e = np.empty(NC)
    for y in range(H):
        for x in range(W):
            if fixed[y, x]:
                continue
            n = 0
            cand[n] = U[y, x]; n += 1
            for d in range(4):
                dy, dx = ((-1, 0), (0, 1), (1, 0), (0, -1))[d]
                qy, qx = y + dy, x + dx
                if 0 <= qy < H and 0 <= qx < W:
                    uq = U[qy, qx]
                    cy, cx = uq // mx - dy, uq % mx - dx
                    if 0 <= cy < my and 0 <= cx < mx:
                        cand[n] = cy * mx + cx; n += 1
            for k in range(K):
                cand[n] = np.random.randint(N); n += 1
            t = T[y, x]
            a0, a1 = by_start[t], by_start[t + 1]
            for k in range(Kt):
                if a1 > a0:
                    cand[n] = by_alpha[a0 + np.random.randint(a1 - a0)]; n += 1
            best, arg = -np.inf, 0
            for i in range(n):
                e[i] = _energy(cand[i], y, x, U, T, alpha, rootv, my, mx, H, W, lam, w, nu, Rr)
                v = -e[i] - np.log(-np.log(np.random.random()))
                if v > best:
                    best, arg = v, i
            U[y, x] = cand[arg]


@njit(cache=True, inline="always")
def _energy_u(c, y, x, U, alpha, my, mx, H, W, lam):
    """Coherence only (the coupling is added by the caller)."""
    cy, cx = c // mx, c % mx
    e = 0.0
    a_c = alpha[cy, cx]
    for d in range(4):
        dy, dx = ((-1, 0), (0, 1), (1, 0), (0, -1))[d]
        qy, qx = y + dy, x + dx
        if 0 <= qy < H and 0 <= qx < W:
            uq = U[qy, qx]
            if uq // mx - dy != cy or uq % mx - dx != cx:
                if a_c != 0 or alpha[uq // mx, uq % mx] != 0:
                    e += lam
    return e


@njit(cache=True)
def joint_sweep(U, alpha, by_alpha, by_start, lam, K, Kt, fixed, coup,
                home, grids, hs, views, fac, tabs, cert, joins, delta, T):
    """One sequential sweep drawing (u, t, d) jointly per site: the tile
    energies e(t) from the tile channel's factors (the coupling excluded),
    the certificate weights per t, the coordinate energies per candidate c,
    and the coupling coup[t, alpha(c)] joining them.  A growth step (c = a
    neighbour's continuation, t = the tile it names) is one move."""
    g = grids[home]
    H, W = g.shape
    my, mx = alpha.shape
    N = my * mx
    D = coup.shape[0]
    NC = 1 + 4 + K + Kt
    cand = np.empty(NC, np.int64)
    eu = np.empty(NC)
    e = np.empty(D)
    wt = np.empty(D)
    lo = np.empty(D, np.int64)
    hi = np.empty(D, np.int64)
    need = np.empty(4, np.bool_)
    has_cert = cert[4] == 1
    Dmax = cert[3]
    bad = 0
    for y in range(H):
        for x in range(W):
            if fixed[y, x]:
                continue
            KN._energies(y, x, home, grids, hs, views, fac, tabs, e)
            if has_cert:
                mass, trunk, g_d = views[cert[0]], views[cert[1]], grids[cert[2]]
                KN.site_weights(y, x, e, g, g_d, mass, trunk, joins, Dmax, delta, T, wt, lo, hi, need)
            else:
                for t in range(D):
                    wt[t] = e[t] / T
            n = 0
            cand[n] = U[y, x]; n += 1
            for d in range(4):
                dy, dx = ((-1, 0), (0, 1), (1, 0), (0, -1))[d]
                qy, qx = y + dy, x + dx
                if 0 <= qy < H and 0 <= qx < W:
                    uq = U[qy, qx]
                    cy, cx = uq // mx - dy, uq % mx - dx
                    if 0 <= cy < my and 0 <= cx < mx:
                        cand[n] = cy * mx + cx; n += 1
            for k in range(K):
                cand[n] = np.random.randint(N); n += 1
            t0 = g[y, x]
            a0, a1 = by_start[t0], by_start[t0 + 1]
            for k in range(Kt):
                if a1 > a0:
                    cand[n] = by_alpha[a0 + np.random.randint(a1 - a0)]; n += 1
            for i in range(n):
                eu[i] = _energy_u(cand[i], y, x, U, alpha, my, mx, H, W, lam)
            best, bi, bt = -np.inf, -1, -1
            for i in range(n):
                a = alpha[cand[i] // mx, cand[i] % mx]
                for t in range(D):
                    if wt[t] == np.inf:
                        continue
                    v = -(eu[i] + coup[t, a]) / T - wt[t] - np.log(-np.log(np.random.random()))
                    if v > best:
                        best, bi, bt = v, i, t
            if bi < 0:
                bad += 1
                continue
            U[y, x] = cand[bi]
            g[y, x] = bt
            if has_cert:
                grids[cert[2]][y, x] = lo[bt] + KN._tgeom(hi[bt] - lo[bt], delta / T)
    return bad


def by_alpha_index(alpha, tile_D):
    """For every tile kind k the coordinates whose alpha is exactly k (CSR)."""
    a = alpha.ravel()
    lists = [np.flatnonzero(a == 2 + k) for k in range(tile_D)]
    start = np.zeros(tile_D + 1, np.int64)
    start[1:] = np.cumsum([len(l) for l in lists])
    flat = np.concatenate(lists) if start[-1] else np.zeros(1, np.int64)
    return flat.astype(np.int64), start


class CoordKernel:
    """The coordinate channel's own kernel (channels.tex section 10)."""

    def __init__(self, u: Channel, tile: Channel, alpha, lam=1.0, w=0.0, nu=2.0, radius=2, K=8, Kt=4):
        self.u, self.tile, self.alpha = u, tile, np.ascontiguousarray(alpha, np.int64)
        self.lam, self.w, self.nu, self.radius, self.K, self.Kt = lam, w, nu, radius, K, Kt
        self.by_alpha, self.by_start = by_alpha_index(alpha, tile.D)
        self.rootv = np.ascontiguousarray(tile.views["root"] == R.EARTH).astype(np.int64)

    def sweep(self, n=1, seed_=0):
        seed(seed_)
        for _ in range(n):
            sweep(self.u.grid, self.tile.grid, self.alpha, self.rootv, self.by_alpha, self.by_start,
                  self.lam, self.w, self.nu, self.radius, self.K, self.Kt, self.u.fixed)

    def sweep_joint(self, model, coup, n=1, seed_=0, T=1.0):
        """(u, t, d) jointly per site.  `model` is compiled for the tile
        channel WITHOUT the coupling factor (this kernel adds it from the
        table `coup`, (D_tile, A))."""
        P = model.compile(self.tile.name)
        KN.seed(seed_)
        seed(seed_)
        bad = 0
        for _ in range(n):
            bad = joint_sweep(self.u.grid, self.alpha, self.by_alpha, self.by_start, self.lam, self.K, self.Kt,
                              self.u.fixed, np.ascontiguousarray(coup),
                              P.home, P.grids, P.hs, P.views, P.fac, P.tabs, P.cert, P.joins, P.delta, T)
        return bad

    def init_free(self):
        """Every cell starts on a random FREE coordinate."""
        free = np.flatnonzero(self.alpha.ravel() == FREE)
        self.u.grid[:] = np.random.default_rng(0).choice(free, self.u.grid.shape)


# ---------------------------------------------------------------- metrics
def metrics(u: Channel, tile: Channel, alpha):
    a = alpha.ravel()[u.grid]
    T = tile.grid
    rv = tile.views["root"][T]
    in_mask = a != FREE
    ok = np.where(a == FREE, (rv != R.ROOT) & (rv != R.TRUNK), np.where(a == EARTH, rv == R.EARTH, T == a - 2))
    H, W = T.shape
    my, mx = alpha.shape
    coh = 0
    for dy, dx in ((0, 1), (1, 0)):
        uq = u.grid[dy:, dx:]
        up = u.grid[:H - dy, :W - dx]
        coh += int(((uq // mx - dy == up // mx) & (uq % mx - dx == up % mx)).sum())
    return dict(masked_cells=int(in_mask.sum()), mismatched=int((~ok).sum()),
                coherent_pairs=round(coh / (2 * H * W), 3))


def render_coords(u: Channel, alpha, px=6):
    """Hue by exemplar position; grey for FREE."""
    my, mx = alpha.shape
    a = alpha.ravel()[u.grid]
    cy, cx = u.grid // mx, u.grid % mx
    img = np.zeros(u.grid.shape + (3,), np.uint8)
    img[..., 0] = 40 + 200 * cx / max(mx - 1, 1)
    img[..., 1] = 40 + 200 * cy / max(my - 1, 1)
    img[..., 2] = 120
    img[a == FREE] = (225, 225, 225)
    return np.repeat(np.repeat(img, px, 0), px, 1)
