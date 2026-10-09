"""Limb channel set: the root set (roots.py) mirrored above the ground.

A limb tile has a mass (1..3) and 1..4 ports among N, E, S, W; a trunk has
mass 3 and a north port.  Ports should meet across every seam (soft:
`dangle`), limbs never touch soil, a trunk has soil below and no soil
above.  The certificate (roots.certificate(tree=True)) makes the join graph
a forest rooted at the trunks with mass non-increasing away from them.
Soft terms are the exemplar's statistics on views, counted by
roots.exemplar_stats through roots.tables from EXEMPLAR: the big root
system of coord.py flipped ('.' sky, ' ' soil, T trunk, 1..3 limbs);
EXEMPLAR_SMALL is the small one mirrored by hand.

Kinds: sky, soil, trunk, limb{m}_{ports}; the limb kinds carry the root
set's tags (root, mass{m}, deg{k}, p{s}) so roots.views applies: the view
`root` is 0 sky, 1 soil (EARTH), 2 trunk, 3 limb."""
import itertools

import numpy as np

from . import coord, roots
from .core import Channel, Factor, Kinds

INF = np.inf
SKY, SOIL, TRUNK, LIMB = roots.SKY, roots.EARTH, roots.TRUNK, roots.ROOT
SIDES = roots.SIDES

# The small root exemplar mirrored by hand (few branch points: 88 % of its
# cells have degree 2, and the limbs it gives are wandering chains).
EXEMPLAR_SMALL = [
    "........................",
    "...1.......1.....1......",
    "...11......1....11......",
    "....11.....2....2.......",
    ".....22....2....2.......",
    "......22...2...22.......",
    ".......33..3..33........",
    "........33.3.33.........",
    ".........33333..........",
    "...........T............",
    "                        ",
    "                        ",
]
# The big root system of coord.py (40 x 84, painted from strokes, 31 % of
# its cells branch points) flipped vertically: '.' stays the medium (sky
# here), the trunk at the bottom.  The default.
EXEMPLAR = coord.EXEMPLAR_BIG[::-1]


def make_kinds():
    """sky, soil, trunk (mass 3, port N), then limb{m}_{ports}."""
    names = ["sky", "soil", "trunk"]
    tags = [frozenset({"sky"}), frozenset({"solid", "earth"}), frozenset({"solid", "wood", "trunk", "mass3", "pN"})]
    cols = [(169, 212, 240), (138, 98, 66), (70, 40, 15)]
    base = {1: (90, 190, 80), 2: (60, 150, 60), 3: (95, 65, 30)}        # thin limbs green, thick ones wood
    for m in (1, 2, 3):
        for k in (1, 2, 3, 4):
            for ps in itertools.combinations(SIDES, k):
                names.append(f"limb{m}_{''.join(ps)}")
                tags.append(frozenset({"solid", "wood", "root", f"mass{m}", f"deg{k}"} | {f"p{s}" for s in ps}))
                cols.append(base[m])
    return Kinds(names, tags, cols)


KINDS = make_kinds()


def tile_channel(kinds: Kinds = KINDS, name="tile"):
    tile = Channel(name, 1, len(kinds))
    roots.add_views(tile, kinds)
    tile.add_view("self", np.arange(tile.D))
    return tile


def contact_tables():
    """Hard rules over the root view (horizontal, vertical [above, below])."""
    Hh, Hv = np.zeros((4, 4)), np.zeros((4, 4))
    Hh[SOIL, LIMB] = Hh[LIMB, SOIL] = Hv[SOIL, LIMB] = Hv[LIMB, SOIL] = INF
    Hv[TRUNK, :] = INF                                          # a trunk has soil below ...
    Hv[TRUNK, SOIL] = 0.0
    Hv[SOIL, TRUNK] = INF                                       # ... and not soil above
    return Hh, Hv


def factors(kinds: Kinds = KINDS, tile="tile", beta=1.0, alpha=0.1, dangle=4.0, grow=1.0, counted=True,
            rows=EXEMPLAR):
    """The limb set's rows on `tile`: unary, port seams, contact.  grow:
    energy taken off every limb cell; rows: the exemplar counted."""
    unary, seam_h, seam_v = roots.tables(kinds, alpha, dangle, counted, rows=rows)
    unary = unary - grow * (roots.views(kinds)["root"] == LIMB)
    Hh, Hv = contact_tables()
    seam_h = np.where(np.isinf(seam_h), INF, beta * seam_h)
    seam_v = np.where(np.isinf(seam_v), INF, beta * seam_v)
    return [
        Factor.unary((tile, "self"), beta * unary, name="limb_unary"),
        Factor.pair((tile, "mE"), (tile, "mW"), (0, 1), seam_h, pad_b=0, pad_a=0, name="limb_seam_h"),
        Factor.pair((tile, "mS"), (tile, "mN"), (1, 0), seam_v, pad_b=0, pad_a=0, name="limb_seam_v"),
        Factor.pair((tile, "root"), (tile, "root"), (0, 1), Hh, name="limb_contact_h"),
        Factor.pair((tile, "root"), (tile, "root"), (1, 0), Hv, pad_b=SOIL, pad_a=SKY, name="limb_contact_v"),
    ]


def certificate(tile="tile", cert="cert", Dmax=4096, delta=0.05):
    return roots.certificate(tile, cert, Dmax, delta, tree=True)


def cert_channel(Dmax=4096, name="cert"):
    return roots.cert_channel(Dmax, name)


def label_components(tile: Channel):
    """(lab, trunks): lab (H, W) int component label of every wood cell
    (ports meeting), -1 elsewhere; trunks {label: (y, x)} of the components
    holding a trunk (the first trunk met)."""
    v = tile.views
    g = tile.grid
    rv = v["root"][g]
    wood = rv >= TRUNK
    H, W = g.shape
    lab = -np.ones((H, W), np.int64)
    trunks = {}
    n = 0
    for y in range(H):
        for x in range(W):
            if not wood[y, x] or lab[y, x] >= 0:
                continue
            stack, lab[y, x] = [(y, x)], n
            while stack:
                a, b = stack.pop()
                if rv[a, b] == TRUNK and n not in trunks:
                    trunks[n] = (a, b)
                for i, (dy, dx) in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
                    c, d = a + dy, b + dx
                    if 0 <= c < H and 0 <= d < W and wood[c, d] and lab[c, d] < 0:
                        if v["p" + SIDES[i]][g[a, b]] and v["p" + SIDES[(i + 2) % 4]][g[c, d]]:
                            lab[c, d] = n
                            stack.append((c, d))
            n += 1
    return lab, trunks


def components(tile: Channel):
    """(n_limb, n_attached): limb cells, and those in a component holding a trunk."""
    lab, trunks = label_components(tile)
    rv = tile.views["root"][tile.grid]
    limb = rv == LIMB
    attached = int(sum((limb & (lab == k)).sum() for k in trunks))
    return int(limb.sum()), attached


def per_tree(tile: Channel):
    """{(y, x) of the trunk: (limb cells, cells left of the trunk column,
    cells right of it)} over the components holding a trunk."""
    lab, trunks = label_components(tile)
    rv = tile.views["root"][tile.grid]
    limb = rv == LIMB
    xs = np.arange(tile.grid.shape[1])[None, :]
    out = {}
    for k, (y, x) in trunks.items():
        m = limb & (lab == k)
        out[(y, x)] = (int(m.sum()), int((m & (xs < x)).sum()), int((m & (xs > x)).sum()))
    return out


def metrics(tile: Channel, cert: Channel = None):
    """Limb cells, the attached fraction, dangling ports, soil contacts,
    mass and degree histograms, and the certificate's rule violations when
    cert is given (roots.metrics with a dummy trees channel)."""
    v = tile.views
    g = tile.grid
    total, attached = components(tile)
    out = dict(limb_cells=total, attached=attached, attached_frac=float(attached / total) if total else 1.0,
               trunks=int((v["trunk"][g] == 1).sum()))
    if cert is not None:
        trees = roots.trees_channel()
        trees.grid = np.zeros((1, 1), np.int32)
        m = roots.metrics(tile, cert, trees)
        out.update(rule_violations=m["rule_violations"], dangling_ports=m["dangling_ports"],
                   mass_hist=m["mass_hist"], deg_hist=m["deg_hist"], max_d=m["max_d"], mean_d=m["mean_d"],
                   multi_parent=m["multi_parent"])
    return out


def render(tile: Channel, px=4, kinds: Kinds = KINDS):
    return roots.render(kinds, tile, px, earth=(169, 212, 240))
