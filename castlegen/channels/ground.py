"""Ground channel set: a designed surface channel, the support rule, a count
honour factor, and soft pair tables over the view `ground`.

Kinds: sky, soil (earth), stone (rock).  Other kinds in a combined domain
(roots) are seen through the views: `solid` by tag, so support and the
count honour apply to them; `ground` maps them to earth, so they take
soil's place in the texture (an earth-like solid).  Making them transparent
instead (a zero row in the texture tables) was tried and is wrong: a trunk
then loses the solid cohesion every surface cell has and is never placed.

surf: level-8 channel, value = rows of the chunk under the surface (0..8),
from 1D value noise over chunk columns (sample_surface).  Chunks are filled
bottom-up in a column, so a chunk above a non-full chunk is empty: the
support rule is honourable by construction."""
import numpy as np

from .core import Channel, Factor, Kinds, view_of_tags

INF = np.inf
CH = 8                                                           # chunk side

KINDS = Kinds(["sky", "soil", "stone"],
              [frozenset({"sky"}), frozenset({"solid", "earth"}), frozenset({"solid", "rock"})],
              [(169, 212, 240), (138, 98, 66), (125, 122, 120)])


def views(kinds: Kinds):
    """{name: (values, n)}."""
    return {"solid": (view_of_tags(kinds, {"solid": 1}), 2),
            "ground": (view_of_tags(kinds, {"sky": 0, "earth": 1, "rock": 2}, default=1), 3)}


def tile_channel(kinds: Kinds, name="tile"):
    c = Channel(name, 1, len(kinds))
    for k, (v, n) in views(kinds).items():
        c.add_view(k, v, n)
    return c


def surf_channel(name="surf"):
    return Channel(name, CH, CH + 1).add_view("rows", np.arange(CH + 1))


# ------------------------------------------------------------ the surface
def value_noise_1d(n, seed, octaves=((64, 36.0), (16, 10.0), (8, 3.0)), base=0.0):
    """n samples of summed cosine-interpolated value noise: (spacing in
    samples, amplitude) per octave."""
    rng = np.random.default_rng(seed)
    x = np.arange(n)
    out = np.full(n, base)
    for sp, amp in octaves:
        knots = rng.uniform(-1, 1, n // sp + 2)
        i, f = x // sp, (x % sp) / sp
        w = (1 - np.cos(np.pi * f)) / 2
        out += amp * ((1 - w) * knots[i] + w * knots[i + 1])
    return out


def sample_surface(surf: Channel, H, W, seed, mean_depth=0.5, **kw):
    """Fill surf.grid: the surface row per chunk column from value noise in
    tiles, then rows under the surface per chunk."""
    rows, cols = surf.grid.shape
    ys = value_noise_1d(cols, seed, base=H * mean_depth, **kw)            # surface row per chunk column
    ys = np.clip(ys, 0, H)
    for j in range(cols):
        for i in range(rows):
            bottom = (i + 1) * CH
            surf.grid[i, j] = int(np.clip(round(bottom - ys[j]), 0, CH))
    return ys


def init_tiles(tile: Channel, surf: Channel, kinds: Kinds, solid="soil", sky="sky"):
    """A consistent refinement of surf: in every chunk the bottom `rows`
    tiles solid, the rest sky.  Supported and count-exact from the start,
    so nothing placed on the surface early (a trunk) is left in a shaft
    when the ground rises later."""
    rows, cols = surf.grid.shape
    tile.grid[:] = kinds.index(sky)
    for i in range(rows):
        for j in range(cols):
            n = int(surf.grid[i, j])
            if n:
                tile.grid[(i + 1) * CH - n:(i + 1) * CH, j * CH:(j + 1) * CH] = kinds.index(solid)
    return tile


def surface_rows(tile: Channel):
    """(W,) first solid row per column (H if none)."""
    solid = tile.views["solid"][tile.grid].astype(bool)
    H = solid.shape[0]
    return np.where(solid.any(0), np.argmax(solid, 0), H)


# ---------------------------------------------------------------- factors
def pair_tables(contact=(("sky", "sky", -1.0), ("solid", "solid", -0.8), ("rock", "rock", -0.9),
                         ("earth", "earth", -0.6), ("sky", "solid", 0.6)),
                beside=(("earth", "earth", -0.3),), above=(("earth", "rock", -0.3),)):
    """(Eh, Ev) over the ground view (0 sky, 1 earth, 2 rock): each rule adds
    its energy for every unordered (contact, beside) or ordered (above) pair
    whose classes match."""
    cls = {"sky": [0], "solid": [1, 2], "earth": [1], "rock": [2]}
    Eh, Ev = np.zeros((3, 3)), np.zeros((3, 3))
    for a, b, e in contact:
        for i in cls[a]:
            for j in cls[b]:
                Eh[i, j] += e; Ev[i, j] += e
                if i != j:
                    Eh[j, i] += e; Ev[j, i] += e
    for a, b, e in beside:
        for i in cls[a]:
            for j in cls[b]:
                Eh[i, j] += e
                if i != j:
                    Eh[j, i] += e
    for a, b, e in above:
        for i in cls[a]:
            for j in cls[b]:
                Ev[i, j] += e
    return Eh, Ev


def factors(tile="tile", surf="surf", kappa=0.5, unary=(0.0, 0.8, 0.6), beta=1.0):
    """The ground set's factors on the tile channel `tile` and the surface
    channel `surf`."""
    Eh, Ev = pair_tables()
    support = np.array([[0.0, 0.0], [INF, 0.0]])                              # [above solid, below solid]
    n = np.arange(CH * CH + 1)
    s = np.arange(CH + 1)
    honour = kappa * (n[None, :] - CH * s[:, None]) ** 2 / CH
    return [
        Factor.pair((tile, "solid"), (tile, "solid"), (1, 0), support, pad_b=1, pad_a=-1, name="support"),
        Factor.pair((tile, "ground"), (tile, "ground"), (0, 1), beta * Eh, name="ground_h"),
        Factor.pair((tile, "ground"), (tile, "ground"), (1, 0), beta * Ev, name="ground_v"),
        Factor.unary((tile, "ground"), beta * np.array(unary), name="ground_unary"),
        Factor.count((tile, "solid"), (surf, "rows"), honour, name="surf_count"),
    ]


# ---------------------------------------------------------------- metrics
def metrics(tile: Channel, surf: Channel):
    solid = tile.views["solid"][tile.grid].astype(bool)
    rows, cols = surf.grid.shape
    cnt = solid.reshape(rows, CH, cols, CH).sum((1, 3))
    target = CH * surf.grid
    unsupported = int((solid[:-1] & ~solid[1:].astype(bool)).sum())
    return dict(unsupported=unsupported, count_mae=float(np.abs(cnt - target).mean()),
                chunks_exact=float((cnt == target).mean()), solid_frac=float(solid.mean()))


def render(kinds: Kinds, tile: Channel, px=4):
    col = np.array(kinds.colours, np.uint8)[tile.grid]
    return np.repeat(np.repeat(col, px, 0), px, 1)
