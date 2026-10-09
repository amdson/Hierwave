"""The procedural tree exemplar (notes/experiments/legacy/tree_texsyn.py)
copied with guarantees: masked exemplar coordinates (coord.py) under the
tree certificate, on the limb tile set, with the coarse window channel
(coord.Coarse) for seeding.

Everything runs in the root set's frame, upside down: the picture is
flipped vertically, so the ground is the top rows, the trunk sits on the
first row of the medium (air, the root view's EARTH code) and the tree
hangs down from it exactly as a root system does.  The roots machinery
then applies unchanged (the surface rule of Coarse, the level-8
certificate, the refinement, the joint kernel); render() flips back.

Exemplar parse (parse_tree): bark mass = taxicab distance to the nearest
non-bark, non-ground cell (1..3), leaves mass 1; the tree's 4-connected
component holding the trunk cell (the bark row nearest the ground, its
centre) is kept, the rest (leaf specks off the bark) become air; a
breadth-first spanning tree from the trunk gives the ports, and the mass
is clamped non-increasing along it, so the exemplar satisfies
Certificate(tree=True).  alpha: FREE on the ground and outside, EARTH (any
air) on a ring of `ring` cells around the tree, the tree's own kinds on
it.
Kinds (make_kinds): air, grass and soil (tag `sky`: outside the medium),
trunk (mass 3, ports S E W in this frame), limb{m}_{ports}, leaf_{ports},
dark_{ports} (mass 1).
Guarantees, hard in the kernel: the certificate (every tree cell on a join
path to a trunk, mass non-increasing from it, exactly one parent);
contact (tree cells never touch the ground, a trunk has ground on its
ground side and none on the other, a thin limb or leaf never rests on the
ground); free cells hold air or tree cells only (the ground and the
trunks are fixed cells).  Soft: the coupling to the exemplar's statement
(nu per mismatch), coherence (lam), dangling ports, the parent's
refinement (mu)."""
import itertools

import numpy as np
from scipy import ndimage

from . import coord, ground, roots
from .core import Channel, Factor, Kinds, Model

INF = np.inf
SKY, BARK, LEAF, DARK, GRASS, SOIL = range(6)                     # the legacy picture's codes
SIDES = "NESW"
DIRS4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
CH = ground.CH


def procedural_tree(m=64, seed=0, depth=5):
    """The legacy generator verbatim: (m, m) int picture, sky above, soil below."""
    rng = np.random.default_rng(seed)
    A = np.full((m, m), SKY, np.int32)
    g = int(m * 0.86)
    A[g:] = SOIL
    A[g:g + 2] = GRASS
    yy, xx = np.mgrid[:m, :m]

    def seg(y0, x0, y1, x1, w):
        d = np.array([y1 - y0, x1 - x0])
        L2 = max(d @ d, 1e-9)
        t = np.clip(((yy - y0) * d[0] + (xx - x0) * d[1]) / L2, 0, 1)
        A[(yy - y0 - t * d[0]) ** 2 + (xx - x0 - t * d[1]) ** 2 <= (w / 2) ** 2] = BARK

    tips = []

    def grow(y, x, ang, L, w, dep):
        y1, x1 = y - L * np.cos(ang), x + L * np.sin(ang)
        seg(y, x, y1, x1, w)
        if dep == 0:
            tips.append((y1, x1))
            return
        for s in (-1, 1):
            grow(y1, x1, ang + s * rng.uniform(0.35, 0.6), L * rng.uniform(0.66, 0.78), max(w * 0.62, 1.0), dep - 1)

    grow(g + 1, m / 2, 0.0, m * 0.2, m / 14, depth)
    for y, x in tips:
        r = m / 40 * rng.uniform(0.9, 1.2)
        for _ in range(4):
            cy, cx, rr = y + rng.normal(0, r / 2), x + rng.normal(0, r / 2), r * rng.uniform(0.5, 0.8)
            A[(yy - cy) ** 2 + (xx - cx) ** 2 <= rr ** 2] = LEAF
        dk = ((yy - y - r * 0.3) ** 2 + (xx - x) ** 2 <= (0.6 * r) ** 2) & (A == LEAF)
        A[dk] = DARK
    return A


# ------------------------------------------------------------------ kinds
def make_kinds():
    names = ["air", "grass", "soil", "trunk"]
    tags = [frozenset(), frozenset({"sky", "solid", "ground"}), frozenset({"sky", "solid", "ground"}),
            frozenset({"solid", "wood", "trunk", "mass3", "pS", "pE", "pW"})]
    cols = [(169, 212, 240), (106, 168, 79), (138, 98, 66), (70, 40, 15)]
    fam = [("limb1", 1, (120, 80, 40)), ("limb2", 2, (100, 62, 30)), ("limb3", 3, (91, 58, 34)),
           ("leaf", 1, (79, 154, 58)), ("dark", 1, (47, 107, 42))]
    for base, m, col in fam:
        for k in (1, 2, 3, 4):
            for ps in itertools.combinations(SIDES, k):
                names.append(f"{base}_{''.join(ps)}")
                t = {"solid", "root", f"mass{m}", f"deg{k}"} | {f"p{s}" for s in ps}
                t |= {"wood"} if base.startswith("limb") else {base}
                tags.append(frozenset(t))
                cols.append(col)
    return Kinds(names, tags, cols)


KINDS = make_kinds()


def tile_channel(kinds=KINDS, name="tile"):
    tile = Channel(name, 1, len(kinds))
    roots.add_views(tile, kinds)
    tile.add_view("self", np.arange(tile.D))
    tv = tile.views
    base = np.where(tv["root"] == roots.SKY, 0, np.where(tv["root"] == roots.EARTH, 1, np.where(
        tv["root"] == roots.TRUNK, 2, np.where(tv["mass"] == 3, 3, 4))))
    tile.add_view("base", base, 5)                       # 0 ground, 1 air, 2 trunk, 3 thick limb, 4 thin limb / leaf
    return tile


# --------------------------------------------------------------- exemplar
def parse_tree(kinds, A, ring=1):
    """(alpha, mass, mask, trunk (y, x), n_ground) in the flipped frame:
    the ground is rows [0, n_ground), the trunk on row n_ground."""
    A = A[::-1]
    my, mx = A.shape
    treeish = (A == BARK) | (A == LEAF) | (A == DARK)
    bark = A == BARK
    lab, n = ndimage.label(treeish)
    ty = int(np.nonzero(bark.any(1))[0].min())
    xs = np.flatnonzero(bark[ty])
    tx = int(xs[np.argmin(np.abs(xs - mx / 2))])
    keep = lab == lab[ty, tx]
    solid = bark.copy()
    solid[:ty] = True                                            # the ground counts as solid for the thickness
    thick = ndimage.distance_transform_cdt(solid, metric="taxicab")
    mass = np.where(keep & bark, np.clip(thick, 1, 3), np.where(keep, 1, 0)).astype(np.int64)
    seen = np.zeros((my, mx), bool)
    ports = np.zeros((my, mx, 4), bool)
    seen[ty, tx] = True
    mass[ty, tx] = 3
    queue = [(ty, tx)]
    while queue:
        y, x = queue.pop(0)
        for i, (dy, dx) in enumerate(DIRS4):
            ny, nx = y + dy, x + dx
            if 0 <= ny < my and 0 <= nx < mx and keep[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                mass[ny, nx] = min(mass[ny, nx], mass[y, x])
                ports[y, x, i] = ports[ny, nx, (i + 2) % 4] = True
                queue.append((ny, nx))
    assert (keep == seen).all()
    alpha = np.full((my, mx), coord.FREE, np.int64)
    for y in range(my):
        for x in range(mx):
            if not keep[y, x]:
                continue
            if (y, x) == (ty, tx):
                alpha[y, x] = 2 + kinds.index("trunk")
                continue
            ps = "".join(s for s, on in zip(SIDES, ports[y, x]) if on)
            base = f"limb{mass[y, x]}" if A[y, x] == BARK else ("leaf" if A[y, x] == LEAF else "dark")
            alpha[y, x] = 2 + kinds.index(f"{base}_{ps}")
    mask = keep.copy()
    for _ in range(ring):
        d = mask.copy()
        d[1:] |= mask[:-1]
        d[:-1] |= mask[1:]
        d[:, 1:] |= mask[:, :-1]
        d[:, :-1] |= mask[:, 1:]
        mask = d
    mask[:ty] = False
    alpha[mask & ~keep] = coord.EARTH
    return alpha, mass, mask, (ty, tx), ty


# ---------------------------------------------------------------- factors
def contact_tables():
    """Over the view base (0 ground, 1 air, 2 trunk, 3 thick limb, 4 thin
    limb / leaf) in the flipped frame (the ground above): horizontal,
    vertical [above, below]."""
    Hh, Hv = np.zeros((5, 5)), np.zeros((5, 5))
    for t in (3, 4):
        Hh[0, t] = Hh[t, 0] = INF                               # tree cells never beside the ground
        Hv[t, 0] = INF                                           # nor above it
    Hv[0, 4] = INF                                               # a thin limb or leaf never hangs from the ground
    Hv[:, 2] = INF                                               # a trunk has ground on its ground side ...
    Hv[0, 2] = 0.0
    Hv[2, 0] = INF                                               # ... and none on the other
    return Hh, Hv


def factors(kinds=KINDS, tile="tile", dangle=1.0):
    """Structure only: free cells air or tree cells, port seams (dangle per
    unmatched port), contact."""
    unary, seam_h, seam_v = roots.tables(kinds, dangle=dangle, counted=False)
    Hh, Hv = contact_tables()
    free_only = np.array([INF, 0.0, INF, 0.0, 0.0])
    return [
        Factor.unary((tile, "base"), free_only, name="free_only"),
        Factor.pair((tile, "mE"), (tile, "mW"), (0, 1), seam_h, pad_b=0, pad_a=0, name="seam_h"),
        Factor.pair((tile, "mS"), (tile, "mN"), (1, 0), seam_v, pad_b=0, pad_a=0, name="seam_v"),
        Factor.pair((tile, "base"), (tile, "base"), (0, 1), Hh, name="contact_h"),
        Factor.pair((tile, "base"), (tile, "base"), (1, 0), Hv, pad_b=1, pad_a=0, name="contact_v"),
    ]


# ------------------------------------------------------------------ world
class TreeCopy:
    """A world H x W (H = the picture's height) with the ground on the top
    n_ground rows of the flipped frame, trunks fixed at `trunk_xs`.
    Schedules: sweep(n) = sequential growth by the joint (u, t, d) kernel;
    seed_coarse(S8) = the level-8 window channel (trunk windows fixed at
    the trunk chunks, no other trunk window), refinement, then sweep(n,
    mu) honours the refinement softly."""

    def __init__(self, W, trunk_xs, A=None, seed=0, ring=1, nu=6.0, lam=1.0, dangle=1.0, K=1, Kt=1, Dmax=4096,
                 delta=0.05, lam8=INF, f8=0.1, cut=2.0):
        self.kinds = KINDS
        self.A = procedural_tree() if A is None else A
        self.alpha, self.mass, self.mask, self.trunk, self.n_ground = parse_tree(self.kinds, self.A, ring)
        self.depth = coord.exemplar_depth(self.kinds, self.alpha)
        self.H, self.W = self.A.shape[0], W
        assert self.H % CH == 0 and W % CH == 0
        self.trunk_xs = list(trunk_xs)
        self.tile = tile_channel(self.kinds)
        self.u = coord.coord_channel(self.alpha)
        self.cert = roots.cert_channel(Dmax)
        self.surf = ground.surf_channel()
        self.model = Model(self.H, W, [self.tile, self.u, self.cert, self.surf], factors(self.kinds, dangle=dangle),
                           [roots.certificate(Dmax=Dmax, delta=delta, tree=True)])
        self.coupling, self.coup = coord.coupling(self.kinds, self.tile, self.u, nu=nu)
        self.ck = coord.CoordKernel(self.u, self.tile, self.alpha, lam=lam, w=0.0, nu=nu, K=K, Kt=Kt)
        # level 8
        self.coarse = coord.Coarse(self.kinds, self.alpha, CH)
        self.u8, self.d8 = self.coarse.channel(), self.coarse.cert_channel()
        no_trunk = np.where(self.coarse.has_trunk == 1, INF, 0.0)       # trunk windows only where fixed
        self.m8 = Model(self.H, W, [self.u8, self.surf, self.d8],
                        self.coarse.factors(lam8=lam8, f=f8, cut=cut) + [Factor.unary(("u8", "self"), no_trunk, name="no_trunk")],
                        [self.coarse.certificate()])
        ty, tx = self.trunk
        wins = [w for w in range(1, self.coarse.D) if self.coarse.has_trunk[w]
                and self.coarse.wins[w][0] == ty - (ty % CH) and self.coarse.wins[w][1] <= tx < self.coarse.wins[w][1] + CH]
        self.trunk_win = {w: int(tx - self.coarse.wins[w][1]) for w in wins}   # window -> trunk column in the chunk
        self.seed = seed
        self.uref = None
        self.reset()

    # ------------------------------------------------------------ state
    def reset(self):
        k, G = self.kinds, self.n_ground
        self.surf.grid[:] = CH
        self.surf.grid[0] = CH - G
        assert G < CH
        self.tile.grid[:] = k.index("air")
        self.tile.grid[:G] = k.index("soil")
        self.tile.grid[G - 1] = k.index("grass")
        self.tile.fixed[:] = False
        self.tile.fixed[:G] = True
        self.ck.init_free()
        self.u.fixed[:] = False
        self.u.fixed[:G] = True
        self.cert.grid[:] = self.cert.D - 1
        self.u8.grid[:] = 0
        self.u8.fixed[:] = False
        self.d8.grid[:] = self.d8.D - 1
        self.uref = np.full((self.H, self.W), -1, np.int64)
        self._fix_trunks()

    def _fix_trunks(self):
        k, G = self.kinds, self.n_ground
        my, mx = self.alpha.shape
        ty, tx = self.trunk
        for x in self.trunk_xs:
            self.tile.grid[G, x] = k.index("trunk")
            self.tile.fixed[G, x] = True
            self.u.grid[G, x] = ty * mx + tx
            self.u.fixed[G, x] = True
            self.cert.grid[G, x] = 0
            # the trunk window whose trunk column matches x within the chunk
            c = x % CH
            w = [w for w, col in self.trunk_win.items() if col == c]
            assert w, f"no trunk window with the trunk at column {c}"
            self.u8.grid[0, x // CH] = w[0]
            self.u8.fixed[0, x // CH] = True
            self.d8.grid[0, x // CH] = 0

    # -------------------------------------------------------- schedules
    def sweep(self, n, seed=None, mu=0.0):
        return self.ck.sweep_joint(self.model, self.coup, n, self.seed if seed is None else seed,
                                   uref=self.uref, mu=mu)

    def seed_coarse(self, S8=30, seed=None):
        """Level 8 sweeps from the fixed trunk windows, then the refinement
        into u, tiles and d.  Returns (windows placed, pruned cells, no-candidate count)."""
        bad = self.m8.sweep("u8", S8, seed=self.seed if seed is None else seed)
        self.uref = self.coarse.refine(self.u8, self.u, self.tile, self.cert, self.kinds, self.alpha, self.depth,
                                       earth_kind="air")
        G, k = self.n_ground, self.kinds
        self.tile.grid[:G] = k.index("soil")                           # the ground is the ground's
        self.tile.grid[G - 1] = k.index("grass")
        self.cert.grid[:G] = self.cert.D - 1
        self._fix_trunks()
        return int((self.u8.grid > 0).sum()), self.coarse.pruned, bad

    # ------------------------------------------------------- guarantees
    def metrics(self):
        from .limbs import components
        tile, cert, u = self.tile, self.cert, self.u
        trees = roots.trees_channel()
        trees.grid = np.zeros((1, 1), np.int32)
        m = roots.metrics(tile, cert, trees)
        v = tile.views
        g = tile.grid
        rv = v["root"][g]
        treec = rv == roots.ROOT
        total, attached = components(tile)
        leaf = np.array([n.startswith(("leaf", "dark")) for n in self.kinds.names])[g]
        thin = v["mass"] == 1
        H, W = g.shape
        tips_ok, n_leaf = 0, int(leaf.sum())
        for y, x in zip(*np.nonzero(leaf)):
            for i, (dy, dx) in enumerate(DIRS4):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and v["p" + SIDES[i]][g[y, x]] and v["p" + SIDES[(i + 2) % 4]][g[ny, nx]] \
                        and cert.grid[ny, nx] < cert.grid[y, x]:
                    tips_ok += int(thin[g[ny, nx]] == 1)
                    break
        G = self.n_ground
        return dict(rule_violations=m["rule_violations"], multi_parent=m["multi_parent"],
                    dangling_ports=m["dangling_ports"], tree_cells=int(treec.sum()), attached=attached,
                    attached_frac=float(attached / total) if total else 1.0, trunks=m["trunks"],
                    in_ground=int(treec[:G].sum()) + int((rv[G:] == roots.SKY).sum()), leaves=n_leaf,
                    leaves_on_tips=float(tips_ok / n_leaf) if n_leaf else 1.0,
                    mass_hist=m["mass_hist"], deg_hist=m["deg_hist"], max_d=m["max_d"],
                    windows=int((self.u8.grid > 0).sum()), **coord.metrics(u, tile, self.alpha))

    def render(self, px=4):
        """The kinds' colours per cell, the picture's way up."""
        img = np.array(self.kinds.colours, np.uint8)[self.tile.grid][::-1]
        return np.repeat(np.repeat(img, px, 0), px, 1)

    def render_ports(self, px=4):
        """The root set's port sprite (bars toward each port, width by mass):
        shows the join structure, not the picture."""
        return roots.render(self.kinds, self.tile, px, earth=(169, 212, 240))[::-1]

    def render_exemplar(self, px=4):
        cols = np.array([(169, 212, 240), (91, 58, 34), (79, 154, 58), (47, 107, 42), (106, 168, 79), (138, 98, 66)],
                        np.uint8)
        return np.repeat(np.repeat(cols[self.A], px, 0), px, 1)
