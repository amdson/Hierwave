"""Masked exemplar coordinates for roots (notes/history/channels.tex sections 5 and 9;
the painted-mask direction of notes/history/dsl.md).

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

from .core import Certificate, Channel, Factor, Kinds
from . import roots as R
from . import kernel as KN
from .sampler import ChannelSampler

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
# A larger, more branching system (40 x 84): a 3-wide taproot thinning to
# mass 1, two main laterals with drops and twigs, mid-depth laterals and
# lower twigs.  Rasterised from strokes (notes/experiments: paint_big),
# every stroke starting on a cell of equal or larger mass.
EXEMPLAR_BIG = [
    "                                                                                    ",
    "                                                                                    ",
    "..........................................T.........................................",
    ".........................................333........................................",
    "........................................333333......................................",
    "....................................33333333333333..................................",
    "................................3333333333333333333333..............................",
    ".............................333333333...333....333333333...........................",
    "..........................33333333.......333......1133333333........................",
    "..........................33333..........333.......1...33333........................",
    ".......................22222..1..........3333......1...22.22222.....................",
    ".....................2222222..1..........3333......1...2..2222222...................",
    "...................222222.....111.......23333.....11...2.....222222.................",
    ".................222222.........11...222223332...11....2......1222222...............",
    "...............22222222............222222233322221.....2......1..222222.............",
    "..............22222..2...........222222...333222222....2......1....22222............",
    "..............2221...2.........222222.....3332.222222..22.....1......222............",
    "............111..1...22......222222.1......222...222222.2.....11.......111..........",
    "...........11...11....2.....22222...1.......22.....222222......1.........11.........",
    "..........11....1.....2.....222.....1.......22......122222.....11.........11........",
    ".........11....11.....22...11.......1.......22......1..222......1..........11.......",
    "........11.....1.......2..11........11.....122......1....21.................111.....",
    ".......11..............2.11..........1...11122......1....111..................11....",
    "......11...............111...........11.11..222.....11...1111..................11...",
    "......1................11.............111...2221.....1....1111......................",
    "......................1111............11.....2211....11....1.11.....................",
    ".....................11..1...........11......22.11....1....11.11....................",
    ".....................1...11.........11.......22..11.........1..11...................",
    "..........................1........11........11...11............1...................",
    "...................................1..........1....11...............................",
    ".............................................11.....11..............................",
    "............................................111......1..............................",
    "...........................................11.11....................................",
    "..........................................11...1....................................",
    "...............................................1....................................",
    "...............................................1....................................",
    "....................................................................................",
    "....................................................................................",
    "....................................................................................",
    "....................................................................................",
]
EXEMPLAR_SHEET = [r + "........" + r[::-1] for r in EXEMPLAR_BIG]    # the big system and its mirror image
EXEMPLARS = {"small": EXEMPLAR, "big": EXEMPLAR_BIG, "sheet": EXEMPLAR_SHEET}
CH_MASS = {" ": -1, ".": 0, "T": 3, "1": 1, "2": 2, "3": 3}


# --------------------------------------------------------------- exemplar
def parse_exemplar(kinds: Kinds, rows=EXEMPLAR, ring=1, tree=True):
    """(alpha (my, mx) int codes, mass grid, mask bool, trunk (y, x)).
    Root cells become root{m}_{ports}.  tree: ports follow a breadth-first
    spanning tree from the trunk over mass-non-increasing adjacencies (a
    thick band becomes a comb), so the exemplar satisfies the tree
    certificate; else every 4-adjacency between root cells is a join."""
    g = np.array([[CH_MASS[c] for c in r] for r in rows], np.int64)
    trunk = np.array([[c == "T" for c in r] for r in rows])
    my, mx = g.shape
    alpha = np.full((my, mx), FREE, np.int64)
    rootish = g > 0
    DIRS4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
    ports = np.zeros((my, mx, 4), bool)
    if tree:
        seen = np.zeros((my, mx), bool)
        queue = [(int(a), int(b)) for a, b in np.argwhere(trunk)]
        for yx in queue:
            seen[yx] = True
        while queue:
            y, x = queue.pop(0)
            for i, (dy, dx) in enumerate(DIRS4):
                ny, nx = y + dy, x + dx
                if 0 <= ny < my and 0 <= nx < mx and rootish[ny, nx] and not seen[ny, nx] and g[ny, nx] <= g[y, x]:
                    seen[ny, nx] = True
                    ports[y, x, i] = ports[ny, nx, (i + 2) % 4] = True
                    queue.append((ny, nx))
        if (rootish & ~seen).any():
            raise ValueError(f"root cells not reachable from the trunk: {np.argwhere(rootish & ~seen).tolist()}")
    else:
        for i, (dy, dx) in enumerate(DIRS4):
            q = np.zeros((my, mx), bool)
            ys, xs = slice(max(-dy, 0), my + min(-dy, 0)), slice(max(-dx, 0), mx + min(-dx, 0))
            q[ys, xs] = rootish[max(dy, 0):my + min(dy, 0), max(dx, 0):mx + min(dx, 0)]
            ports[..., i] = rootish & q
    for y in range(my):
        for x in range(mx):
            if not rootish[y, x]:
                continue
            if trunk[y, x]:
                alpha[y, x] = 2 + kinds.index("trunk")
                continue
            ps = "".join(s for s, on in zip("NESW", ports[y, x]) if on)
            if not ps:
                raise ValueError(f"isolated root cell at {(y, x)}")
            alpha[y, x] = 2 + kinds.index(f"root{g[y, x]}_{ps}")
    mask = rootish.copy()
    for _ in range(ring):
        d = mask.copy()
        d[1:] |= mask[:-1]; d[:-1] |= mask[1:]; d[:, 1:] |= mask[:, :-1]; d[:, :-1] |= mask[:, 1:]
        mask = d
    mask &= g >= 0                                             # never claim sky
    ty0 = int(np.argwhere(trunk)[:, 0].max())
    mask[:ty0 + 1] &= rootish[:ty0 + 1]                        # the ring starts below the trunk row: the surface is the ground's
    alpha[mask & ~rootish] = EARTH
    ty, tx = np.argwhere(trunk)[0]
    return alpha, g, mask, (int(ty), int(tx))


def check_exemplar(kinds: Kinds, alpha, g):
    """Every root cell reachable from the trunk through joined cells of
    non-increasing mass (the certificate is satisfiable on the exemplar)."""
    v = R.views(kinds)
    my, mx = g.shape
    seen = np.zeros_like(g, bool)
    stack = [(int(a), int(b)) for a, b in np.argwhere(alpha == 2 + kinds.index("trunk"))]
    for yx in stack:
        seen[yx] = True
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
                home, grids, hs, views, fac, tabs, cert, joins, delta, T, uref, mu):
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
    NC = 2 + 4 + K + Kt
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
                KN.site_weights(y, x, e, g, g_d, mass, trunk, joins, Dmax, delta, T, wt, lo, hi, need, cert[5] == 1)
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
            r = uref[y, x]
            if r >= 0:
                cand[n] = r; n += 1                                 # the parent's refinement
            for i in range(n):
                eu[i] = _energy_u(cand[i], y, x, U, alpha, my, mx, H, W, lam)
                if r >= 0 and cand[i] != r:
                    eu[i] += mu                                     # parent honour
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
    """The coordinate channel's own kernel (notes/history/channels.tex section 10)."""

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

    def sweep_joint(self, model, coup, n=1, seed_=0, T=1.0, uref=None, mu=0.0):
        """(u, t, d) jointly per site.  `model` is compiled for the tile
        channel WITHOUT the coupling factor (this kernel adds it from the
        table `coup`, (D_tile, A)).  uref: (H, W) coordinate the parent
        level asks for (-1: none), honoured softly at mu per mismatch."""
        P = model.compile(self.tile.name)
        KN.seed(seed_)
        seed(seed_)
        if uref is None:
            uref = np.full(self.u.grid.shape, -1, np.int64)
        bad = 0
        for _ in range(n):
            bad = joint_sweep(self.u.grid, self.alpha, self.by_alpha, self.by_start, self.lam, self.K, self.Kt,
                              self.u.fixed, np.ascontiguousarray(coup),
                              P.home, P.grids, P.hs, P.views, P.fac, P.tabs, P.cert, P.joins, P.delta, T,
                              np.ascontiguousarray(uref, np.int64), float(mu))
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


# ------------------------------------------------- coarse coordinates (level 8)
"""A level-8 coordinate channel: per chunk one exemplar window (its
top-left exemplar cell) or FREE, sampled by the generic tile kernel with
  coherence   a neighbour's window should be this one's shifted by 8
              (two views, ey and ex; FREE windows owe each other nothing)
  surface     hard, against surf: every root and ring cell of the window
              lies below the chunk's first solid row, and a trunk in the
              window sits exactly on it
  unary       a bonus on the trunk window (one per copy), a small cost per
              other masked window
then refined: fine u, tiles, and certificate depth d from the exemplar's
own tree, every cell of a footprint at once (notes/history/channels.tex: a level starts
as a consistent refinement of the level above)."""


def exemplar_depth(kinds, alpha, roots=()):
    """(my, mx) int: d of each root cell along the exemplar's port tree
    (trunk 0), INF (-1) elsewhere.  roots: extra (y, x) sources (the roots
    of floating patches that have no trunk)."""
    v = R.views(kinds)
    my, mx = alpha.shape
    depth = np.full((my, mx), -1, np.int64)
    queue = [(int(a), int(b)) for a, b in np.argwhere(alpha == 2 + kinds.index("trunk"))] + [tuple(r) for r in roots]
    for yx in queue:
        depth[yx] = 0
    while queue:
        y, x = queue.pop(0)
        t = alpha[y, x] - 2
        for i, (dy, dx) in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < my and 0 <= nx < mx) or alpha[ny, nx] < 2 or depth[ny, nx] >= 0:
                continue
            tq = alpha[ny, nx] - 2
            if v["p" + "NESW"[i]][t] and v["p" + "NESW"[(i + 2) % 4]][tq]:
                depth[ny, nx] = depth[y, x] + 1
                queue.append((ny, nx))
    return depth


class Coarse:
    """Windows of the exemplar at chunk size B: domain index 0 = FREE, then
    every B x B window with a masked cell."""

    def __init__(self, kinds, alpha, B=8, leaf_tags=("leaf", "dark"), roots=()):
        self.B = B
        self.leaf_tags = set(leaf_tags)
        self.roots = tuple(roots)
        self._J = None
        my, mx = alpha.shape
        wins = [(ey, ex) for ey in range(my - B + 1) for ex in range(mx - B + 1)
                if (alpha[ey:ey + B, ex:ex + B] != FREE).any()]
        self.wins = np.array([(-1, -1)] + wins, np.int64)              # (D8, 2)
        self.D = len(self.wins)
        trunk_code = 2 + kinds.index("trunk")
        self.has_trunk = np.zeros(self.D, np.int64)
        self.masked = np.zeros(self.D, np.int64)
        self.surface_ok = np.zeros((self.D, B + 1), np.float64)       # [window, surf] 0 / inf
        self.above_ok = np.zeros((self.D, B + 1), np.float64)         # [window, surf of the chunk above]
        self.surface_ok[0] = 0.0
        for w in range(1, self.D):
            ey, ex = self.wins[w]
            win = alpha[ey:ey + B, ex:ex + B]
            self.masked[w] = 1
            ty = np.argwhere(win == trunk_code)
            rows_nonfree = np.flatnonzero((win >= 2).any(1))             # rows holding root cells
            self.has_trunk[w] = int(len(ty) > 0)
            for sv in range(B + 1):
                r0 = B - sv                                             # first solid row in the chunk
                ok = True
                if len(ty):
                    ok = int(ty[0][0]) == r0 and r0 <= B - 1
                    ok = ok and all(r >= r0 for r in rows_nonfree if r != ty[0][0])
                else:
                    ok = all(r >= r0 for r in rows_nonfree)
                self.surface_ok[w, sv] = 0.0 if ok else INF
                if len(ty) and int(ty[0][0]) == 0 and sv > 0:            # a trunk on the top row needs an empty chunk above
                    self.above_ok[w, sv] = INF
        self.ey = np.where(self.wins[:, 0] < 0, my, self.wins[:, 0])   # FREE -> my (never equal to a window row)
        self.ex = np.where(self.wins[:, 1] < 0, mx, self.wins[:, 1])
        self.my, self.mx = my, mx
        # seam codes: per side and position along the edge, what crosses there: 0 the FREE window,
        # 1 no port, 2 + (parent-ward, mass, leaf) otherwise.  Two facing sides are compatible when
        # every position agrees: no port on both, or ports parent-ward on exactly one side with the
        # parent's mass at least the child's (the certificate's rule).  Per position, so the seam
        # check is 8 small pair tables per direction and nothing quadratic in the window count.
        v = R.views(kinds)
        depth = exemplar_depth(kinds, alpha, self.roots)
        is_leaf = np.array([bool(set(t) & self.leaf_tags) for t in kinds.tags])
        self.NCODE = 2 + 2 * 3 * 2
        self.code = np.ones((4, self.D, B), np.int64)
        self.code[:, 0, :] = 0
        DY = ((-1, 0), (0, 1), (1, 0), (0, -1))
        for w in range(1, self.D):
            ey, ex = self.wins[w]
            for d in range(4):
                for k in range(B):
                    y, x = (ey, ex + k) if d == 0 else (ey + k, ex + B - 1) if d == 1 else (ey + B - 1, ex + k) if d == 2 else (ey + k, ex)
                    c = alpha[y, x]
                    if c >= 2 and v["p" + "NESW"[d]][c - 2]:
                        ny, nx = y + DY[d][0], x + DY[d][1]
                        pw = 0 <= ny < my and 0 <= nx < mx and depth[ny, nx] >= 0 and depth[ny, nx] < depth[y, x]
                        self.code[d, w, k] = 2 + int(pw) * 6 + (int(v["mass"][c - 2]) - 1) * 2 + int(is_leaf[c - 2])
        self.has_parent = (self.code >= 8).any(2).astype(np.int64)          # (4, D): a parent-ward crossing on that side
        self.my, self.mx = my, mx

    @staticmethod
    def _decode(c):
        """(has port, parent-ward, mass, leaf) of a seam code."""
        if c < 2:
            return False, 0, 0, 0
        c -= 2
        return True, c // 6, (c % 6) // 2 + 1, c % 2

    def seam_table(self, cut=0.5, cut_leaf=0.1):
        """(NCODE, NCODE): the energy of two facing positions."""
        n = self.NCODE
        T = np.zeros((n, n))
        for a in range(n):
            for b in range(n):
                pa, wa, ma, la = self._decode(a)
                pb, wb, mb, lb = self._decode(b)
                if a == 0 or b == 0:                                    # against the FREE window: a cut per port
                    T[a, b] = (cut_leaf if la else cut) if pa else ((cut_leaf if lb else cut) if pb else 0.0)
                elif pa != pb:                                          # a port facing a placed window's blank
                    T[a, b] = INF
                elif pa:
                    ok = wa != wb and not (wa and mb < ma) and not (wb and ma < mb)
                    T[a, b] = 0.0 if ok else INF
        return T

    def channel(self, name="u8"):
        c = Channel(name, self.B, self.D)
        c.add_view("ey", self.ey, self.my + 1)
        c.add_view("ex", self.ex, self.mx + 1)
        c.add_view("self", np.arange(self.D), self.D)
        c.add_view("masked", self.masked, 2)
        c.add_view("trunk8", self.has_trunk, 2)
        for d, name in enumerate("NESW"):
            for k in range(self.B):
                c.add_view(f"s{name}{k}", self.code[d, :, k], self.NCODE)
        return c

    def factors(self, u8="u8", surf="surf", lam8=INF, f=0.1, bonus=4.0, win_bonus=2.0, cut=0.1, cut_leaf=0.02):
        """lam8: two masked windows that are not each other's shift (soft
        when finite: recombination at a cost); f: a masked window beside a
        FREE one (per view, so 2 f per boundary edge); bonus on the trunk
        window, win_bonus on every other masked window (the density knob).
        Seams are hard between windows (seam_table): the ports crossing a
        shared edge must agree in position and direction with the mass
        rule; a port crossing into the FREE window is a cut at `cut`
        (`cut_leaf` for a leaf's port), per crossing."""
        my, mx, B = self.my, self.mx, self.B
        T = self.seam_table(cut, cut_leaf)

        def shift_table(n, shift):
            """(n + 1, n + 1): 0 where b == a + shift (a, b real), 0 for FREE-FREE,
            f for FREE against a window, lam8 for two windows out of step."""
            t = np.full((n + 1, n + 1), lam8)
            a = np.arange(n)
            ok = a + shift
            m = ok < n
            t[a[m], ok[m]] = 0.0
            t[n, :] = f
            t[:, n] = f
            t[n, n] = 0.0
            return t
        unary = np.where(self.has_trunk == 1, -bonus, np.where(self.masked == 1, -win_bonus, 0.0))
        out = [
            Factor.pair((u8, "ey"), (u8, "ey"), (0, 1), shift_table(my, 0), name="coh_h_ey"),
            Factor.pair((u8, "ex"), (u8, "ex"), (0, 1), shift_table(mx, B), name="coh_h_ex"),
            Factor.pair((u8, "ey"), (u8, "ey"), (1, 0), shift_table(my, B), name="coh_v_ey"),
            Factor.pair((u8, "ex"), (u8, "ex"), (1, 0), shift_table(mx, 0), name="coh_v_ex"),
            Factor.pair((u8, "self"), (surf, "rows"), (0, 0), self.surface_ok, name="surface"),
            Factor.pair((u8, "self"), (surf, "rows"), (-1, 0), self.above_ok, pad_b=0, name="surface_above"),
            Factor.unary((u8, "self"), unary, name="window_unary"),
        ]
        for k in range(B):
            out.append(Factor.pair((u8, f"sE{k}"), (u8, f"sW{k}"), (0, 1), T, pad_b=0, pad_a=0, name=f"seam_h{k}"))
            out.append(Factor.pair((u8, f"sS{k}"), (u8, f"sN{k}"), (1, 0), T, pad_b=0, pad_a=0, name=f"seam_v{k}"))
        return out

    def joins(self):
        """(4, D8, D8) bool: w' on side d of w is joined when every position
        of the shared edge is compatible and w has a parent-ward crossing
        there (witnesses lie parent-ward)."""
        if self._J is not None:
            return self._J
        D, B = self.D, self.B
        okT = np.isfinite(self.seam_table(0.0, 0.0))
        J = np.zeros((4, D, D), np.bool_)
        for d in range(4):
            ok = np.ones((D, D), np.bool_)
            a, b = self.code[d], self.code[(d + 2) % 4]
            for k in range(B):
                ok &= okT[a[:, k][:, None], b[:, k][None, :]]
            J[d] = ok & (self.has_parent[d][:, None] == 1)
        self._J = J
        return J

    def recombined_seams(self, u8: Channel):
        """Adjacent placed windows that are joined but not each other's shift."""
        g, B = u8.grid, self.B
        n = 0
        for (dy, dx) in ((0, 1), (1, 0)):
            a, b = g[:g.shape[0] - dy, :g.shape[1] - dx], g[dy:, dx:]
            both = (a > 0) & (b > 0)
            wa, wb = self.wins[a], self.wins[b]
            shift = (wb[..., 0] == wa[..., 0] + dy * B) & (wb[..., 1] == wa[..., 1] + dx * B)
            d = 1 if dx else 2
            joined = self.joins()[d][a, b]
            n += int((both & joined & ~shift).sum())
        return n

    def cert_channel(self, Dmax=256, name="d8"):
        return Channel(name, self.B, Dmax + 2).add_view("d", np.arange(Dmax + 2))

    def certificate(self, u8="u8", d8="d8", Dmax=256, delta=0.05):
        """Every masked window reaches a trunk window through coherent
        neighbours (the tree rule at level 8, with mass 1 everywhere):
        no headless fragments."""
        return Certificate(u8, "masked", "trunk8", d8, Dmax, delta, joins=self.joins())

    def refine(self, u8: Channel, u: Channel, tile: Channel, cert: Channel, kinds, alpha, depth, earth_kind="soil",
               keep_ref=False):
        """Fine u, tiles and d from the coarse windows: every cell of a masked
        window takes its exemplar coordinate, its tile (root tiles; ring
        cells become earth where the ground had sky or a root; FREE cells
        keep the ground's tile) and its depth.  Returns uref (H, W) for
        the joint kernel's parent-honour term (-1 outside footprints)."""
        B = self.B
        H, W = u.grid.shape
        rv = tile.views["root"]
        uref = np.full((H, W), -1, np.int64)
        free = np.flatnonzero(alpha.ravel() == FREE)
        rng = np.random.default_rng(0)
        for i in range(H // B):
            for j in range(W // B):
                w = u8.grid[i, j]
                ys, xs = slice(i * B, (i + 1) * B), slice(j * B, (j + 1) * B)
                if w == 0:
                    u.grid[ys, xs] = rng.choice(free, (B, B))
                    # a root tile left over from an old footprint is removed
                    blk = tile.grid[ys, xs]
                    blk[(rv[blk] == R.ROOT) | (rv[blk] == R.TRUNK)] = kinds.index(earth_kind)
                    tile.grid[ys, xs] = blk
                    cert.grid[ys, xs] = cert.D - 1
                    continue
                ey, ex = self.wins[w]
                cy, cx = np.mgrid[ey:ey + B, ex:ex + B]
                coords = cy * self.mx + cx
                u.grid[ys, xs] = coords
                uref[ys, xs] = coords
                a = alpha[ey:ey + B, ex:ex + B]
                blk = tile.grid[ys, xs].copy()
                is_root = a >= 2
                blk[is_root] = a[is_root] - 2
                ring = a == EARTH
                fix = ring & (rv[blk] != R.EARTH)
                blk[fix] = kinds.index(earth_kind)
                tile.grid[ys, xs] = blk
                d = depth[ey:ey + B, ex:ex + B]
                cert.grid[ys, xs] = np.where(d >= 0, d, cert.D - 1)
        # depth and pruning over the port tree of the placed tiles themselves (seams may be
        # recombined): breadth-first from every trunk tile; root tiles not reached revert
        v = R.views(kinds)
        rv_g = rv[tile.grid]
        is_root = (rv_g == R.ROOT) | (rv_g == R.TRUNK)
        depth_g = np.full((H, W), -1, np.int64)
        queue = [(int(a), int(b)) for a, b in np.argwhere(rv_g == R.TRUNK)]
        for yx in queue:
            depth_g[yx] = 0
        head = 0
        while head < len(queue):
            y, x = queue[head]; head += 1
            t = tile.grid[y, x]
            for i, (dy, dx) in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
                ny, nx = y + dy, x + dx
                if not (0 <= ny < H and 0 <= nx < W) or depth_g[ny, nx] >= 0 or not is_root[ny, nx]:
                    continue
                tq = tile.grid[ny, nx]
                if v["p" + "NESW"[i]][t] and v["p" + "NESW"[(i + 2) % 4]][tq] and v["mass"][tq] <= v["mass"][t]:
                    depth_g[ny, nx] = depth_g[y, x] + 1
                    queue.append((ny, nx))
        orphan = is_root & (depth_g < 0)
        cert.grid[:] = np.where(depth_g >= 0, depth_g, cert.D - 1)
        if orphan.any():
            tile.grid[orphan] = kinds.index(earth_kind)
            cert.grid[orphan] = cert.D - 1
            u.grid[orphan] = rng.choice(free, int(orphan.sum()))
            if not keep_ref:                                            # keep_ref: the joint kernel's honour term
                uref[orphan] = -1                                       # keeps asking for the pruned cells
        self.pruned = int(orphan.sum())
        return uref


# ------------------------------------------------- the Sampler interface (sampler.ChannelSampler)
@njit(cache=True)
def _candidates(y, x, U, g, alpha, by_alpha, by_start, K, Kt, r):
    """The set sweep / joint_sweep build at (y, x), the same construction
    and RNG draws: current value, each neighbour's coherent continuation,
    K random, Kt with alpha equal to the tile at p, then r (joint, r >= 0)."""
    H, W = U.shape
    my, mx = alpha.shape
    cand = np.empty(2 + 4 + K + Kt, np.int64)
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
        cand[n] = np.random.randint(my * mx); n += 1
    t = g[y, x]
    a0, a1 = by_start[t], by_start[t + 1]
    for k in range(Kt):
        if a1 > a0:
            cand[n] = by_alpha[a0 + np.random.randint(a1 - a0)]; n += 1
    if r >= 0:
        cand[n] = r; n += 1
    return cand[:n].copy()


@njit(cache=True)
def _coord_energy(U, T, alpha, rootv, lam, w, nu, Rr):
    """(own-kernel energy, coherence alone): sum_p of sweep's site energy at
    u_p with the coherence pairs counted once (each pair enters two sites)."""
    H, W = U.shape
    my, mx = alpha.shape
    e, coh = 0.0, 0.0
    for y in range(H):
        for x in range(W):
            c = U[y, x]
            e += _energy(c, y, x, U, T, alpha, rootv, my, mx, H, W, lam, w, nu, Rr)
            coh += _energy_u(c, y, x, U, alpha, my, mx, H, W, lam)
    return e - coh / 2, coh / 2


class CoordSampler(ChannelSampler):
    """sampler.ChannelSampler over CoordKernel; same draws as the existing
    driver for the same seed.  joint=None: the coordinate kernel `sweep`
    (tiles held; patch and coupling from the kernel's own lam, w, nu).
    joint=(model, coup): `sweep_joint`, (u, t, d) per site, the tile
    channel's factors from `model` (compiled without the coupling) plus
    coup[t, alpha(u)]; `uref` (-1: none) honoured at `mu`, set by the
    painter that refines the level above.  Coordinate channels have no
    hard parent rows (the parent's refinement is soft honour), so init()
    is a no-op returning 0.  No relaxation and no tempered kernel."""

    def __init__(self, ck: CoordKernel, joint=None, mu=0.0, seed=0):
        self.ck, self.joint, self.mu = ck, joint, float(mu)
        self.name = ck.u.name
        self.uref = np.full(ck.u.grid.shape, -1, np.int64)
        self.rng = np.random.default_rng(seed)

    def _seed(self, seed):
        return int(self.rng.integers(1 << 30)) if seed is None else int(seed)

    def sweep(self, n=1, T=1.0, seed=None):
        s = self._seed(seed)
        if self.joint is None:
            assert T == 1.0, "the coordinate kernel has no temperature"
            self.ck.sweep(n, s)
            return 0
        model, coup = self.joint
        return self.ck.sweep_joint(model, coup, n, s, T, uref=self.uref, mu=self.mu)

    def candidates(self, y, x):
        """Coordinate candidates at (y, x) (random members drawn from self.rng).
        Joint: each is paired with every tile of finite weight."""
        ck = self.ck
        seed(self._seed(None))
        r = int(self.uref[y, x]) if self.joint is not None else -1
        return _candidates(y, x, ck.u.grid, ck.tile.grid, ck.alpha, ck.by_alpha, ck.by_start, ck.K, ck.Kt, r)

    def energy(self):
        """(finite total, inf count).  Own kernel: coherence + patch + nu
        coupling.  Joint: the tile model's energy + coherence + coup[t,
        alpha(u)] + mu per cell off uref."""
        ck = self.ck
        U, a = ck.u.grid, ck.alpha.ravel()
        if self.joint is None:
            return _coord_energy(U, ck.tile.grid, ck.alpha, ck.rootv, ck.lam, ck.w, ck.nu, ck.radius)[0], 0
        model, coup = self.joint
        e, nv = model.energy(ck.tile.name)
        coh = _coord_energy(U, ck.tile.grid, ck.alpha, ck.rootv, ck.lam, 0.0, 0.0, 0)[1]
        hon = self.mu * int(((self.uref >= 0) & (U != self.uref)).sum())
        return e + coh + float(np.asarray(coup)[ck.tile.grid, a[U]].sum()) + hon, nv
