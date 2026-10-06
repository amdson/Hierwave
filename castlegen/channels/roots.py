"""Root channel set on port tiles (channels.tex, section 14).

A root tile has a mass (1..3) and 1..3 ports among N, E, S, W; a trunk has
mass 3 and a south port.  Ports should meet across every seam (soft: a port
facing a non-port costs `dangle`; see tables), so roots form a graph; the certificate
makes it a forest rooted at trunks with mass non-increasing away from the
trunk: a root cell is valid iff some joined neighbour has mass >= its own
and a smaller d.

Soft terms are the exemplar's statistics on views, the cluster expansion of
channels.tex section 13 counted from a hand-painted picture (EXEMPLAR):
  unary   -log p(root) - log p(mass) - log p(degree) relative to earth, and
          -log 4 p(side | mass) per port (where the ports point)
  pair    -PMI(mass_left, mass_right | joined) and the same vertically
          (thick above thin), in one table with the port seam (dangle per
          unmatched port) and the crowding cost of two adjacent roots that
          are not joined (never seen in the exemplar: -log of the smoothed
          rate)
Hard contact: roots never touch sky; a trunk has sky above and earth
below.  A designed `trees` channel (level 8) asks for exactly one trunk in
each marked chunk.

Views of the tile domain: root (0 sky, 1 earth, 2 trunk, 3 root), mass
(0..3), trunk (0/1), pN pE pS pW (0/1), mN mE mS mW (0 not a root, the
mass if that side has a port, BLANK = 4 for a root without one), self
(identity, for the unary)."""
import itertools

import numpy as np

from .core import Certificate, Channel, Factor, Kinds, view_of_tags

INF = np.inf
CH = 8
SIDES = "NESW"
SKY, EARTH, TRUNK, ROOT = range(4)
BLANK = 4                                                  # seam view: a root cell with no port on that side

EXEMPLAR = [
    "                        ",
    "                        ",
    "...........T............",
    ".........33333..........",
    "........33.3.33.........",
    ".......33..3..33........",
    "......22...2...22.......",
    ".....22....2....2.......",
    "....11.....2....2.......",
    "...11......1....11......",
    "...1.......1.....1......",
    "...........1.....1......",
    "........................",
    "........................",
]
CHARS = {" ": -1, ".": 0, "T": 3, "1": 1, "2": 2, "3": 3}


def make_kinds():
    """trunk, then root{m}_{ports} for m in 1..3 and 1..3 ports."""
    names, tags, cols = ["trunk"], [frozenset({"solid", "wood", "trunk", "mass3", "pS"})], [(60, 30, 10)]
    base = {1: (205, 150, 90), 2: (165, 105, 50), 3: (120, 70, 25)}
    for m in (1, 2, 3):
        for k in (1, 2, 3):
            for ps in itertools.combinations(SIDES, k):
                names.append(f"root{m}_{''.join(ps)}")
                tags.append(frozenset({"solid", "wood", "root", f"mass{m}", f"deg{k}"} | {f"p{s}" for s in ps}))
                cols.append(base[m])
    return Kinds(names, tags, cols)


KINDS = make_kinds()


def views(kinds: Kinds):
    v = {"root": view_of_tags(kinds, {"sky": SKY, "trunk": TRUNK, "root": ROOT}, default=EARTH),
         "mass": view_of_tags(kinds, {"mass1": 1, "mass2": 2, "mass3": 3}),
         "trunk": view_of_tags(kinds, {"trunk": 1})}
    for s in SIDES:
        v["p" + s] = view_of_tags(kinds, {"p" + s: 1})
        v["m" + s] = np.where(v["p" + s] == 1, v["mass"], np.where(v["root"] == ROOT, BLANK, 0))
    return v


VIEW_N = {"root": 4, "mass": 4, "trunk": 2, "pN": 2, "pE": 2, "pS": 2, "pW": 2, "mN": 5, "mE": 5, "mS": 5, "mW": 5}


def add_views(tile: Channel, kinds: Kinds):
    for k, arr in views(kinds).items():
        tile.add_view(k, arr, VIEW_N[k])
    return tile


def cert_channel(Dmax, name="cert"):
    return Channel(name, 1, Dmax + 2).add_view("d", np.arange(Dmax + 2))


def trees_channel(name="trees"):
    return Channel(name, CH, 2).add_view("tree", np.arange(2))


# --------------------------------------------------------------- exemplar
def exemplar_stats(rows=EXEMPLAR, alpha=0.1):
    """Counted on views.  mass grid (-1 sky, 0 earth, 1..3 roots, trunk 3);
    joins = 4-adjacent root cells.  Returns p_root (root cells per earth
    cell), p_mass (4,), p_deg (4,), p_side (4, 4) [mass, side], and the
    joined-pair mass counts Ch (4, 4) [left, right], Cv (4, 4) [top,
    bottom].  Mass and degree are counted as marginals, not jointly: the
    exemplar's tips are all mass 1, and a joint table would price the
    thick tip every growing root passes through at ~8."""
    g = np.array([[CHARS[c] for c in r] for r in rows], np.int64)
    H, W = g.shape
    trunk = np.array([[c == "T" for c in r] for r in rows])
    root = (g > 0) & ~trunk
    nb = np.zeros((H, W, 4), bool)                              # joined on side N, E, S, W
    for i, (dy, dx) in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
        q = np.full((H, W), 0, np.int64)
        ys, xs = slice(max(-dy, 0), H + min(-dy, 0)), slice(max(-dx, 0), W + min(-dx, 0))
        q[ys, xs] = g[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)]
        nb[..., i] = (g > 0) & (q > 0)
    deg = np.clip(nb.sum(-1), 1, 3)
    p_root = (root.sum() + alpha) / ((g == 0).sum() + alpha)
    p_mass = np.bincount(g[root], minlength=4).astype(float) + alpha
    p_mass[0] = 0
    p_mass /= p_mass.sum()
    p_deg = np.bincount(deg[root], minlength=4).astype(float) + alpha
    p_deg[0] = 0
    p_deg /= p_deg.sum()
    p_side = np.full((4, 4), alpha)
    for m in (1, 2, 3):
        p_side[m] += nb[root & (g == m)].sum(0)
    p_side /= p_side.sum(1, keepdims=True)
    Ch, Cv = np.full((4, 4), alpha), np.full((4, 4), alpha)
    np.add.at(Ch, (g[:, :-1][nb[:, :-1, 1]], g[:, 1:][nb[:, :-1, 1]]), 1)
    np.add.at(Cv, (g[:-1][nb[:-1, :, 2]], g[1:][nb[:-1, :, 2]]), 1)
    n_adj = nb.sum() / 2                                        # adjacent root pairs, every one joined here
    p_crowd = alpha / (n_adj + alpha)                           # an adjacent pair that is not joined
    return p_root, p_mass, p_deg, p_side, Ch, Cv, p_crowd


def _pmi(C):
    p = C / C.sum()
    return -np.log(p / (p.sum(1, keepdims=True) * p.sum(0, keepdims=True)))


def tables(kinds: Kinds, alpha=0.1, dangle=4.0):
    """(unary over rclass+ports as a (D,) table on the identity view,
    seam_h over (mE, mW), seam_v over (mS, mN)).  A port facing a non-port
    costs `dangle` (soft: an unfinished tip is how a root grows under
    single-site updates; inf would need two-site moves to nucleate)."""
    v = views(kinds)
    p_root, p_mass, p_deg, p_side, Ch, Cv, p_crowd = exemplar_stats(alpha=alpha)
    deg = sum(v["p" + s] for s in SIDES)
    is_root = v["root"] == ROOT
    with np.errstate(divide="ignore"):
        unary = np.where(is_root, -np.log(p_root) - np.log(p_mass[v["mass"]]) - np.log(p_deg[deg]), 0.0)
    for i, s in enumerate(SIDES):
        unary += np.where((v["root"] == ROOT) & (v["p" + s] == 1), -np.log(4 * p_side[v["mass"], i]), 0.0)
    unary[v["root"] == TRUNK] = 0.0
    seam_h, seam_v = np.zeros((5, 5)), np.zeros((5, 5))
    ph, pv = _pmi(Ch), _pmi(Cv)
    for a in range(5):
        for b in range(5):
            pa, pb = 0 < a < BLANK, 0 < b < BLANK
            if pa != pb:
                seam_h[a, b] = seam_v[a, b] = dangle             # a port facing a non-port
            elif pa:
                seam_h[a, b], seam_v[a, b] = ph[a, b], pv[a, b]
            elif a == BLANK and b == BLANK:
                seam_h[a, b] = seam_v[a, b] = -np.log(p_crowd)   # adjacent roots, not joined
    return unary, seam_h, seam_v


def contact_tables():
    """Hard rules over the root view: (horizontal, vertical [above, below])."""
    Hh, Hv = np.zeros((4, 4)), np.zeros((4, 4))
    Hh[SKY, ROOT] = Hh[ROOT, SKY] = Hv[SKY, ROOT] = Hv[ROOT, SKY] = INF
    Hv[:, TRUNK] = INF                                          # a trunk has sky above ...
    Hv[SKY, TRUNK] = 0.0
    Hv[TRUNK, SKY] = INF                                        # ... and not sky below
    return Hh, Hv


def factors(kinds: Kinds, tile="tile", trees="trees", beta=1.0, lam=6.0, alpha=0.1, dangle=4.0, grow=1.0):
    """grow: energy taken off every root cell (the knob for root density;
    counted marginals alone leave the pairwise model sparser than the
    exemplar)."""
    unary, seam_h, seam_v = tables(kinds, alpha, dangle)
    unary = unary - grow * (views(kinds)["root"] == ROOT)
    Hh, Hv = contact_tables()
    want = np.full((2, CH * CH + 1), INF)
    want[0, 0] = 0.0                                            # no tree: no trunk
    want[1, 0], want[1, 1] = lam, 0.0                           # tree: one trunk (soft on 0, hard above 1)
    seam_h = np.where(np.isinf(seam_h), INF, beta * seam_h)
    seam_v = np.where(np.isinf(seam_v), INF, beta * seam_v)
    return [
        Factor.unary((tile, "self"), beta * unary, name="root_unary"),
        Factor.pair((tile, "mE"), (tile, "mW"), (0, 1), seam_h, pad_b=0, pad_a=0, name="root_seam_h"),
        Factor.pair((tile, "mS"), (tile, "mN"), (1, 0), seam_v, pad_b=0, pad_a=0, name="root_seam_v"),
        Factor.pair((tile, "root"), (tile, "root"), (0, 1), Hh, name="root_contact_h"),
        Factor.pair((tile, "root"), (tile, "root"), (1, 0), Hv, pad_b=EARTH, pad_a=SKY, name="root_contact_v"),
        Factor.count((tile, "trunk"), (trees, "tree"), want, name="trunk_count"),
    ]


def certificate(tile="tile", cert="cert", Dmax=4096, delta=0.05):
    return Certificate(tile, "mass", "trunk", cert, Dmax, delta, ports=("pN", "pE", "pS", "pW"))


def tile_views(tile: Channel, kinds: Kinds):
    """All views the root set needs, plus the identity view `self`."""
    add_views(tile, kinds)
    if "self" not in tile.views:
        tile.add_view("self", np.arange(tile.D))
    return tile


# ---------------------------------------------------------------- designed
def sample_trees(trees: Channel, surface_row, seed, spacing=3, p=0.5):
    """Mark chunks: every `spacing`-th chunk column with probability p, at
    the chunk row holding surface_row(j) (the first solid row in tiles)."""
    rng = np.random.default_rng(seed)
    trees.grid[:] = 0
    rows, cols = trees.grid.shape
    for j in range(cols):
        if j % spacing == spacing // 2 and rng.random() < p:
            i = int(surface_row(j)) // CH
            if 0 <= i < rows:
                trees.grid[i, j] = 1
    return trees


# ---------------------------------------------------------------- metrics
def metrics(tile: Channel, cert: Channel, trees: Channel):
    v = tile.views
    rv, mass, d = v["root"][tile.grid], v["mass"][tile.grid], cert.grid
    INFD = cert.D - 1
    H, W = mass.shape
    roots, trunks = rv == ROOT, rv == TRUNK
    ok = mass == 0
    ok |= trunks & (d == 0)
    dangling = 0
    sky_touch = 0
    for i, (dy, dx) in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
        ys, xs = slice(max(-dy, 0), H + min(-dy, 0)), slice(max(-dx, 0), W + min(-dx, 0))
        yq, xq = slice(max(dy, 0), H + min(dy, 0)), slice(max(dx, 0), W + min(dx, 0))
        pa = v["p" + SIDES[i]][tile.grid]
        pb = np.zeros_like(pa)
        pb[ys, xs] = v["p" + SIDES[(i + 2) % 4]][tile.grid][yq, xq]
        joined = (pa == 1) & (pb == 1)
        dangling += int(((pa == 1) & (pb == 0)).sum())
        mq, dq = np.zeros_like(mass), np.full_like(d, INFD)
        mq[ys, xs], dq[ys, xs] = mass[yq, xq], d[yq, xq]
        ok |= roots & joined & (mq >= mass) & (dq < d) & (d < INFD)
        rb = np.full_like(rv, EARTH)
        rb[ys, xs] = rv[yq, xq]
        sky_touch += int(((rv == ROOT) & (rb == SKY)).sum())
    deg = sum(v["p" + s][tile.grid] for s in SIDES)
    return dict(rule_violations=int((~ok).sum()), dangling_ports=dangling, sky_contacts=sky_touch,
                trunks=int(trunks.sum()), trees_wanted=int(trees.grid.sum()), root_cells=int(roots.sum()),
                mass_hist=np.bincount(mass[roots], minlength=4)[1:].tolist(),
                deg_hist=np.bincount(deg[roots], minlength=4)[1:].tolist(),
                max_d=int(d[roots].max()) if roots.any() else 0,
                mean_d=float(d[roots].mean()) if roots.any() else 0.0)


def render(kinds: Kinds, tile: Channel, px=6, earth=None):
    """Sprites: the kind's colour as background for non-roots; roots draw
    their ports as bars of width by mass over the earth colour."""
    v = views(kinds)
    H, W = tile.grid.shape
    cols = np.array(kinds.colours, np.uint8)
    img = np.repeat(np.repeat(cols[tile.grid], px, 0), px, 1)
    earth_col = np.array(earth if earth is not None else (138, 98, 66), np.uint8)
    half = px // 2
    for y in range(H):
        for x in range(W):
            t = tile.grid[y, x]
            if v["root"][t] != ROOT:
                continue
            m = v["mass"][t]
            wd = max(1, (px * m) // 5)
            a, b = half - wd // 2, half - wd // 2 + wd
            cell = img[y * px:(y + 1) * px, x * px:(x + 1) * px]
            cell[:] = earth_col
            cell[a:b, a:b] = cols[t]
            if v["pN"][t]:
                cell[:half, a:b] = cols[t]
            if v["pS"][t]:
                cell[half:, a:b] = cols[t]
            if v["pW"][t]:
                cell[a:b, :half] = cols[t]
            if v["pE"][t]:
                cell[a:b, half:] = cols[t]
    return img
