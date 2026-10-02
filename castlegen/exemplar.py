"""Hand-placed exemplars: macrostructures from a JSON layout, the rest in-filled
by annealed Gibbs on the local energy (pair + unary), optionally cleaned of
violations ("clean") and joined into one component ("connect", see connect()).

Layout JSON:

    {"tileset": "demo", "size": [64, 64], "torus": true,
     "sweeps": 300, "T": 1.0, "T_hot": 4.0, "seed": 0,
     "clean": {"rounds": 30, "T": 0.4, "tau": 1.9}, "connect": true,
     "place": [
       {"room": "courtyard", "at": [4, 4], "size": [12, 14],
        "gates": [["S", 6], ["E", 4]], "interior": "free"},
       {"structure": "greenhouse", "at": [22, 30]},
       {"path": "hallway", "points": [[16, 10], [30, 10], [30, 40]]},
       {"cell": "hall_cross@0", "at": [30, 20]}
     ]}

room       a walled family: <room> interior, <room>_wall, <room>_corner, and an
           entrance kind (<room>_entrance or <room>_gateway, or "entrance").
           gates: [side, offset along the side].  interior "seed" (default)
           starts the inside as the interior kind and samples it only in the
           cold phase, "free" leaves it to the full anneal, "clamp" fixes it.
structure  a structure from the tile set, top-left at "at".
path       axis-aligned polyline of a tag's kinds (default "hallway"); pieces
           are chosen by which sides connect, including doors of anything
           already placed next to the path.
cell       one signature by name, "kind@deg".
grid       rows of characters, top-left at "at"; "legend" maps a character
           to "kind@deg" (characters not in it are left alone).
object     a rigid object, top-left at "at": layout "objects" {name: {"size",
           "place": [...]}} compiles each to a template with its own place
           list (unplaced cells wall).  castlegen/objects.py gives the
           exemplar its object channel from these items.
stroke     axis-aligned polyline of one signature ("kind@deg"), "width" cells
           wide (a square brush): rivers, walls.  A flowing kind (tile set
           "flow") is laid oriented along the polyline: every cell points to
           the neighbour laid latest after it (or ahead, at the end).
disc       one signature filling a disc of "radius" around "at": lakes, plazas.
fill       one signature filling the rectangle "at" [y, x], "size" [h, w].
blob       one signature filling the ellipse around "at" with "radius" [ry, rx].
           fill and blob take "seed": true to start the cells as that
           signature and sample them in the cold phase instead of clamping.
A path over flowing water becomes the bridge that keeps the water's flow.
After placement, placed flowing cells that nothing feeds become springs and
placed cells that flow into dry land become lakes.  Layout "flow": true
also dries the unplaced water that breaks the flow rule after the infill
(prune_flow), so the exemplar keeps the rule.  Layout "support": true (side
view, tile set with "solid" kinds) projects the infill onto the support rule
with the bottom "ground" rows as the band (quantities.support.fix, placed
cells kept where possible).

Placed cells are clamped; everything else is sampled.

    python -m castlegen.exemplar castlegen/exemplars/castle.json --png images/exemplar.png
"""
from __future__ import annotations

import argparse
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from castlegen import render, tileset

SIDE = {"N": 0, "E": 1, "S": 2, "W": 3}
STEP = ((-1, 0), (0, 1), (1, 0), (0, -1))
EXEMPLAR_DIR = os.path.join(os.path.dirname(__file__), "exemplars")


# --------------------------------------------------------------------- lookup
def _kind(ts, name):
    for i, k in enumerate(ts.kinds):
        if k.name == name:
            return i
    raise KeyError(f"no kind {name!r}")


def sig_by_name(ts, name):
    """'kind@deg' (or 'kind') -> signature id."""
    kname, _, deg = name.partition("@")
    k = _kind(ts, kname)
    ri = ts.kinds[k].rotations.index(int(deg or 0))
    return int(ts.np_tables["type_sig_lookup"][k, ri])


def sig_by_sides(ts, kinds, outer):
    """Signature among `kinds` whose non-connecting-to-family sides are exactly
    `outer` (sides whose socket is wall or door), for walled-room rings."""
    wall, door = ts.sockets.index("wall"), ts.sockets.index("door")
    for s in range(ts.n_sig):
        if ts.sig_kind[s] in kinds:
            ext = {d for d in range(4) if ts.sig_sockets[s, d] in (wall, door)}
            if ext == outer:
                return s
    raise KeyError(f"no signature of {[ts.kinds[k].name for k in kinds]} with outer sides {outer}")


def sig_by_doors(ts, tag, doors, flow=None):
    """Signature of a kind carrying `tag` with a door socket on exactly `doors`
    (any other socket elsewhere, e.g. a road's land sides); failing that, any
    signature with doors on `doors` and wall elsewhere (a chamber).  flow:
    also require that flow side (a bridge over water flowing that way)."""
    wall, door = ts.sockets.index("wall"), ts.sockets.index("door")
    is_door = [ts.sig_sockets[:, d] == door for d in range(4)]
    exact = np.all([is_door[d] == (d in doors) for d in range(4)], 0)
    if flow is not None and ts.flow_dir is not None:
        exact &= ts.flow_dir == flow
    tagged = [s for s in np.flatnonzero(exact) if tag in ts.kinds[ts.sig_kind[s]].tags]
    if tagged:
        return int(tagged[0])
    want = tuple(door if d in doors else wall for d in range(4))
    hits = [s for s in range(ts.n_sig) if tuple(ts.sig_sockets[s]) == want]
    if hits:
        return hits[0]
    raise KeyError(f"no {tag!r} signature with doors on {sorted(doors)}")


# --------------------------------------------------------------------- layout
def place(ts, layout):
    """-> (tiles, clamp, seeded): signature grid, placed cells (fixed), and
    seeded room interiors (start as the interior kind, sampled only cold)."""
    H, W = layout["size"]
    tiles = np.full((H, W), ts.WALL, np.int32)
    clamp = np.zeros((H, W), bool)
    seeded = np.zeros((H, W), bool)
    wrap = lambda y, x: (y % H, x % W)

    def put(y, x, s):
        y, x = wrap(y, x)
        tiles[y, x] = s
        clamp[y, x] = True

    paths = []
    for item in layout.get("place", []):
        if "room" in item:
            name = item["room"]
            (y0, x0), (h, w) = item["at"], item["size"]
            ring = [_kind(ts, name + "_wall"), _kind(ts, name + "_corner")]
            ent = item.get("entrance") or next(n for n in (name + "_entrance", name + "_gateway")
                                               if any(k.name == n for k in ts.kinds))
            gates = {(SIDE[s], o) for s, o in item.get("gates", [])}
            for dy in range(h):
                for dx in range(w):
                    outer = {d for d, hit in enumerate((dy == 0, dx == w - 1, dy == h - 1, dx == 0)) if hit}
                    if not outer:
                        mode = item.get("interior", "seed")
                        if mode == "clamp":
                            put(y0 + dy, x0 + dx, sig_by_name(ts, name))
                        elif mode == "seed":
                            yy, xx = wrap(y0 + dy, x0 + dx)
                            tiles[yy, xx] = sig_by_name(ts, name)
                            seeded[yy, xx] = True
                        continue
                    gate = [d for d in outer if (d, dx if d in (0, 2) else dy) in gates]
                    if gate:
                        put(y0 + dy, x0 + dx, sig_by_sides(ts, [_kind(ts, ent)], {gate[0]}))
                    else:
                        put(y0 + dy, x0 + dx, sig_by_sides(ts, ring, outer))
        elif "structure" in item:
            grid = dict(ts.structures)[item["structure"]]
            y0, x0 = item["at"]
            for r in range(grid.shape[0]):
                for c in range(grid.shape[1]):
                    put(y0 + r, x0 + c, sig_by_name(ts, ts.kinds[grid[r, c]].name))
        elif "cell" in item:
            put(*item["at"], sig_by_name(ts, item["cell"]))
        elif "grid" in item:
            y0, x0 = item["at"]
            for r, row in enumerate(item["grid"]):
                for c, ch in enumerate(row):
                    if ch in item["legend"]:
                        put(y0 + r, x0 + c, sig_by_name(ts, item["legend"][ch]))
        elif "object" in item:
            grid = object_grid(ts, layout, item["object"])
            y0, x0 = item["at"]
            for r in range(grid.shape[0]):
                for c in range(grid.shape[1]):
                    put(y0 + r, x0 + c, int(grid[r, c]))
        elif "stroke" in item:
            s, w = sig_by_name(ts, item["stroke"]), item.get("width", 1)
            lo = -(w // 2)
            if ts.flow_dir is not None and ts.flow_dir[s] >= 0:
                for (y, x), f in _flow_stroke(ts, _polyline(item["points"]), w, wrap).items():
                    put(y, x, _with_flow(ts, s, f))
                continue
            for y, x in _polyline(item["points"]):
                for dy in range(lo, lo + w):
                    for dx in range(lo, lo + w):
                        put(y + dy, x + dx, s)
        elif "disc" in item:
            s, (cy, cx), rad = sig_by_name(ts, item["disc"]), item["at"], item["radius"]
            for dy in range(-rad, rad + 1):
                for dx in range(-rad, rad + 1):
                    if dy * dy + dx * dx <= rad * rad + rad:
                        put(cy + dy, cx + dx, s)
        elif "fill" in item or "blob" in item:
            s = sig_by_name(ts, item.get("fill") or item.get("blob"))
            (cy, cx) = item["at"]
            if "fill" in item:
                (h, w) = item["size"]
                cells = [(cy + dy, cx + dx) for dy in range(h) for dx in range(w)]
            else:
                ry, rx = item["radius"]
                cells = [(cy + dy, cx + dx) for dy in range(-ry, ry + 1) for dx in range(-rx, rx + 1)
                         if (dy / (ry + 0.5)) ** 2 + (dx / (rx + 0.5)) ** 2 <= 1.0]
            for y, x in cells:
                if item.get("seed"):
                    yy, xx = wrap(y, x)
                    tiles[yy, xx] = s
                    seeded[yy, xx] = True
                    clamp[yy, xx] = False
                else:
                    put(y, x, s)
        elif "path" in item:
            paths.append(item)
        else:
            raise ValueError(f"unknown placement {item}")

    # paths last, so they can open onto the doors of what is already placed
    door = ts.sockets.index("door") if "door" in ts.sockets else -1
    sides = {}
    tag = {}
    for item in paths:
        cells = [wrap(*c) for c in _polyline(item["points"])]
        for c in cells:
            sides.setdefault(c, set())
            tag[c] = item.get("path") or "hallway"
        for a, b in zip(cells, cells[1:]):
            d = STEP.index(((b[0] - a[0] + 1) % H - 1, (b[1] - a[1] + 1) % W - 1))
            sides[a].add(d)
            sides[b].add((d + 2) % 4)
    for (y, x), ds in sides.items():
        for d, (dy, dx) in enumerate(STEP):
            ny, nx = wrap(y + dy, x + dx)
            if clamp[ny, nx] and (ny, nx) not in sides and ts.sig_sockets[tiles[ny, nx], (d + 2) % 4] == door:
                ds.add(d)
    for (y, x), ds in sides.items():
        f = int(ts.flow_dir[tiles[y, x]]) if ts.flow_dir is not None and clamp[y, x] else -1
        try:
            put(y, x, sig_by_doors(ts, tag[(y, x)], ds, flow=f if f >= 0 else None))
        except KeyError:
            put(y, x, sig_by_doors(ts, tag[(y, x)], ds))
    if ts.water is not None and ts.water.any():
        _fix_placed_flow(ts, tiles, clamp)
    return tiles, clamp, seeded & ~clamp


def object_grid(ts, layout, name):
    """(h, w) signature grid of layout["objects"][name]: its own "place" list
    on a "size" canvas (unplaced cells wall), every cell part of the object."""
    spec = layout["objects"][name]
    return place(ts, {"size": spec["size"], "place": spec["place"], "objects": layout["objects"]})[0]


def object_mask(ts, layout):
    """(H, W) cells covered by the layout's "object" items (wrapped)."""
    H, W = layout["size"]
    mask = np.zeros((H, W), bool)
    for item in layout.get("place", []):
        if "object" in item:
            h, w = object_grid(ts, layout, item["object"]).shape
            y0, x0 = item["at"]
            mask[np.ix_((y0 + np.arange(h)) % H, (x0 + np.arange(w)) % W)] = True
    return mask


def _with_flow(ts, s, f):
    """The signature of s's kind whose flow side is f."""
    k = ts.sig_kind[s]
    for ri in range(len(ts.kinds[k].rotations)):
        t = int(ts.np_tables["type_sig_lookup"][k, ri])
        if ts.flow_dir[t] == f:
            return t
    raise KeyError(f"{ts.kinds[k].name} has no rotation flowing {f}")


def _flow_stroke(ts, cells, w, wrap):
    """{cell: flow side} for a brush of width w along the polyline `cells`:
    each cell points to the neighbour laid latest (a larger polyline index),
    preferring the segment direction; cells with no later neighbour point
    along the last segment."""
    lo = -(w // 2)
    idx, seg = {}, []
    for i, (y, x) in enumerate(cells):
        a, b = (cells[i], cells[i + 1]) if i + 1 < len(cells) else (cells[i - 1], cells[i])
        seg.append(STEP.index((int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1])))))
        for dy in range(lo, lo + w):
            for dx in range(lo, lo + w):
                idx[wrap(y + dy, x + dx)] = i
    out = {}
    for c, i in idx.items():
        best = None
        for d, (dy, dx) in enumerate(STEP):
            n = wrap(c[0] + dy, c[1] + dx)
            if n in idx and idx[n] > i:
                key = (d == seg[i], idx[n])
                if best is None or key > best[0]:
                    best = (key, d)
        out[c] = best[1] if best else seg[i]
    return out


def _fix_placed_flow(ts, tiles, clamp):
    """Placed flowing cells that nothing feeds become springs, placed cells
    that flow into dry land become lakes (repeated until stable)."""
    from castlegen.quantities import flow as F
    lake = next(s for s in range(ts.n_sig) if ts.lake[s] and s != ts.WALL)
    node = np.asarray(ts.np_tables["is_room"], bool)
    for _ in range(4):
        dead, unfed = F._cell_status(ts, tiles)
        dead &= clamp & ~node[tiles]
        unfed &= clamp & ~node[tiles]
        if not (dead | unfed).any():
            return
        tiles[dead] = lake
        for y, x in zip(*np.nonzero(unfed & ~dead)):
            spring = [s for s in range(ts.n_sig) if ts.source[s] and ts.flow_dir[s] == ts.flow_dir[tiles[y, x]]]
            if spring:
                tiles[y, x] = spring[0]


def _polyline(points):
    """Cells of an axis-aligned polyline, in order."""
    pts = [tuple(p) for p in points]
    cells = [pts[0]]
    for (ya, xa), (yb, xb) in zip(pts, pts[1:]):
        if ya != yb and xa != xb:
            raise ValueError(f"segment {(ya, xa)} -> {(yb, xb)} is not axis-aligned")
        n = max(abs(yb - ya), abs(xb - xa))
        sy, sx = np.sign(yb - ya), np.sign(xb - xa)
        cells += [(ya + i * sy, xa + i * sx) for i in range(1, n + 1)]
    return cells


# ---------------------------------------------------------------------- gibbs
def gibbs(ts, tiles, sweeps, seed=0, T=1.0, T_hot=None, update=None, torus=True, forbid=None, keep_class=None):
    """Checkerboard heat-bath on pair + unary energy.  T anneals from T_hot to T
    over the first half of the sweeps.  update: bool mask of sites that may
    change (default all).  The gate is never proposed.  keep_class: (S,) int
    class of every signature; a site only moves within its current class
    (e.g. solid / not solid)."""
    t = ts.jt
    Eh, Ev, logz = t.Eh, t.Ev, t.logz
    EhT, EvT = Eh.T, Ev.T
    bias = logz.at[ts.GATE].set(-jnp.inf)
    if forbid is not None:                                       # (S,) bool: signatures never proposed
        bias = jnp.where(jnp.asarray(forbid, bool), -jnp.inf, bias)
    tiles = jnp.asarray(tiles, jnp.int32)
    H, W = tiles.shape
    upd = jnp.ones((H, W), bool) if update is None else jnp.asarray(update, bool)
    ys, xs = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
    parity = (ys + xs) % 2
    T_hot = T if T_hot is None else T_hot
    WALL = ts.WALL
    kc = None if keep_class is None else jnp.asarray(keep_class, jnp.int32)

    def nbrs(g):
        if torus:
            return jnp.roll(g, 1, 1), jnp.roll(g, -1, 1), jnp.roll(g, 1, 0), jnp.roll(g, -1, 0)
        p = jnp.pad(g, 1, constant_values=WALL)
        return p[1:-1, :-2], p[1:-1, 2:], p[:-2, 1:-1], p[2:, 1:-1]

    def step(carry, s):
        g, key = carry
        key, sub = jax.random.split(key)
        frac = jnp.minimum(s / jnp.maximum(sweeps, 1), 1.0)       # s counts half-sweeps
        temp = T_hot + (T - T_hot) * frac
        left, right, up, down = nbrs(g)
        E = Eh[left] + EhT[right] + Ev[up] + EvT[down]           # (H, W, S)
        logits = -E / temp + bias
        if kc is not None:
            logits = jnp.where(kc[None, None, :] == kc[g][..., None], logits, -jnp.inf)
        new = jax.random.categorical(sub, logits)
        g = jnp.where((parity == s % 2) & upd, new, g)
        return (g, key), None

    (tiles, _), _ = jax.lax.scan(step, (tiles, jax.random.PRNGKey(seed)), jnp.arange(2 * sweeps))
    return np.asarray(tiles)


def pair_energy(ts, tiles, torus=True):
    """(Eh right-pair grid, Ev down-pair grid): energy of each cell with its
    right and lower neighbour."""
    Eh, Ev = ts.np_tables["Eh"], ts.np_tables["Ev"]
    tiles = np.asarray(tiles)
    if torus:
        return Eh[tiles, np.roll(tiles, -1, 1)], Ev[tiles, np.roll(tiles, -1, 0)]
    eh = np.zeros(tiles.shape, np.float32); ev = np.zeros(tiles.shape, np.float32)
    eh[:, :-1] = Eh[tiles[:, :-1], tiles[:, 1:]]
    ev[:-1, :] = Ev[tiles[:-1, :], tiles[1:, :]]
    return eh, ev


def bad_cells(ts, tiles, tau=2.5, torus=True):
    """Cells on a neighbour pair with energy > tau (the hard-rule violations:
    structure seams, dangling court/hall sockets, forbidden contacts)."""
    eh, ev = pair_energy(ts, tiles, torus)
    bh, bv = eh > tau, ev > tau
    bad = bh | bv | np.roll(bh, 1, 1) | np.roll(bv, 1, 0)
    if not torus:
        bad = bh | bv
        bad[:, 1:] |= bh[:, :-1]
        bad[1:, :] |= bv[:-1, :]
    return bad


def dilate(mask, r, torus=True):
    out = mask.copy()
    for _ in range(r):
        m = out.copy()
        if torus:
            out = m | np.roll(m, 1, 0) | np.roll(m, -1, 0) | np.roll(m, 1, 1) | np.roll(m, -1, 1)
        else:
            out[1:] |= m[:-1]; out[:-1] |= m[1:]; out[:, 1:] |= m[:, :-1]; out[:, :-1] |= m[:, 1:]
    return out


def orphan_cells(ts, tiles):
    """Structure pieces not covered by a complete instance."""
    k = ts.sig_kind[np.asarray(tiles)]
    H, W = k.shape
    orphan = np.zeros((H, W), bool)
    for _, grid in ts.structures:
        R, C = grid.shape
        covered = np.zeros((H, W), bool)
        for y, x in zip(*np.nonzero(k == grid[0, 0])):
            if y + R <= H and x + C <= W and np.array_equal(k[y:y + R, x:x + C], grid):
                covered[y:y + R, x:x + C] = True
        orphan |= np.isin(k, grid.ravel()) & ~covered
    return orphan


def repair(ts, tiles, sweeps=5, seed=0, T=0.5, tau=2.5, radius=1, torus=True, clear_orphans=False, update=None,
           forbid=None, keep_class=None):
    """A few Gibbs sweeps on cells within `radius` of a violation, repeated on
    the shrinking violation set.  clear_orphans: first set incomplete
    structure pieces to wall (a partial structure is a local minimum that
    single-site moves cannot leave).  update: cells that may change (default
    all).  Returns the repaired grid."""
    tiles = np.asarray(tiles)
    if clear_orphans:
        tiles = np.where(orphan_cells(ts, tiles), ts.WALL, tiles).astype(np.int32)
    for i in range(sweeps):
        bad = bad_cells(ts, tiles, tau, torus)
        if not bad.any():
            break
        mask = dilate(bad, radius, torus)
        tiles = gibbs(ts, tiles, 1, seed * 1000 + i, T=T, update=mask if update is None else mask & update, torus=torus,
                      forbid=forbid, keep_class=keep_class)
    return tiles


# --------------------------------------------------------------- connectivity
def open_table(ts, tag="hallway"):
    """(S, 4) signature with a door added on side d, keeping the other sockets
    and the kind family (a shared tag other than the kind's own name); failing
    that, a tile with only door and wall sides (a chamber) becomes the `tag`
    piece with its doors plus d.  The signature itself if side d is already a
    door; -1 if there is none (corners, structure pieces)."""
    door, wall = ts.sockets.index("door"), ts.sockets.index("wall")
    fam = [k.tags - {k.name} for k in ts.kinds]
    by_sock = {}
    for s in range(ts.n_sig):
        by_sock.setdefault(tuple(ts.sig_sockets[s]), []).append(s)
    out = -np.ones((ts.n_sig, 4), np.int32)
    for s in range(ts.n_sig):
        if s in (ts.WALL, ts.GATE):
            continue
        for d in range(4):
            if ts.sig_sockets[s, d] == door:
                out[s, d] = s
                continue
            want = list(ts.sig_sockets[s]); want[d] = door
            f = fam[ts.sig_kind[s]]
            hits = [t for t in by_sock.get(tuple(want), []) if f & fam[ts.sig_kind[t]] and "structure" not in f]
            if hits:
                out[s, d] = hits[0]
            elif "structure" not in f and set(ts.sig_sockets[s]) <= {door, wall}:
                out[s, d] = sig_by_doors(ts, tag, {e for e in range(4) if want[e] == door})
    return out


def bridge_jump(ts, tiles, y, x, d, torus=True):
    """Straight bridge from (y, x) moving d over flowing water (not springs):
    -> (run length r >= 1, landing cell, [bridge signature per run cell]) or
    None.  Every run cell must have a bridge with doors on d and its opposite
    that keeps its flow side (so carving it never changes the flow)."""
    if ts.water is None or not ts.water.any():
        return None
    from castlegen.quantities import flow as F
    t = F.tables(ts)
    H, W = tiles.shape
    mask = (1 << d) | (1 << (d + 2) % 4)
    run, cy, cx = [], y, x
    while True:
        cy, cx = cy + STEP[d][0], cx + STEP[d][1]
        if torus:
            cy, cx = cy % H, cx % W
        elif not (0 <= cy < H and 0 <= cx < W):
            return None
        s = tiles[cy, cx]
        if not (t["flowing"][s] and not ts.source[s] and not t["node"][s]):
            break
        b = t["bridge_by"][mask, ts.flow_dir[s]]
        if b < 0 or len(run) > max(H, W):
            return None
        run.append(int(b))
    return (len(run), (cy, cx), run) if run else None


def connect(ts, tiles, clamp=None, torus=True, max_cost=None, seed=0, tag="hallway", w_energy=1.0,
            sources=None, targets=None, delete=True, void=None, fixed=None):
    """Join every component to the main one (the gate's, else the largest) by
    the cheapest set of edits along a shortest-path tree grown from it:

      node -> node, joined         free
      node -> node, not joined     1: open a door on both (open_table)
      node -> wall                 1: open a door on the node, the wall becomes a path piece
      wall -> wall                 1
      wall -> node                 0: open a door on the node (the wall was paid for)

    Only unclamped plain wall cells are carved.  The edits only turn wall
    sockets into matching doors, so they never add a violation.  Components
    farther than max_cost (or unreachable) are set to wall when none of their
    cells is clamped (with the solid tiles next to them, e.g. its trees, but
    never water), set to `void` (a signature, default the wall).  A path may
    also cross flowing water straight on a bridge that keeps the flow
    (bridge_jump, cost 1 per cell).  w_energy adds that weight times the base-mass cost of
    each door opened (a straight becoming a tee, a chamber becoming a path
    piece), so cheap openings win.  sources / targets: component labels (of
    connmetrics.labels) to grow from and to join; default the main component
    and all the others.  delete=False leaves unjoined components as they are.
    fixed: (H, W) cells never edited (no door opened on them).
    Returns (tiles, unreached components left)."""
    import heapq
    from castlegen import connmetrics as cm
    tiles = np.array(tiles, np.int32)
    H, W = tiles.shape
    clamp = np.zeros((H, W), bool) if clamp is None else clamp
    OPEN = open_table(ts, tag)
    fixed = np.zeros((H, W), bool) if fixed is None else fixed
    op = lambda y, x, d: -1 if fixed[y, x] else OPEN[tiles[y, x], d]
    logz = ts.np_tables["logz"]
    ucost = np.where(OPEN >= 0, w_energy * np.maximum(0.0, logz[:, None] - logz[np.maximum(OPEN, 0)]), np.inf)
    node = cm._node(ts)
    lab, n = cm.labels(ts, tiles, torus)
    if sources is None:
        sources = [cm.main_label(ts, tiles, lab, n)]
    if targets is None:
        targets = [c for c in range(n) if c not in sources]
    if not targets or not len(sources):
        return tiles, 0
    Dh, Dv = ts.np_tables["Dh"], ts.np_tables["Dv"]
    rng = np.random.default_rng(seed)
    jitter = 1e-3 * rng.random((H, W))
    carvable = (tiles == ts.WALL) & ~clamp
    void = ts.WALL if void is None else void
    wet = ts.water[tiles] if ts.water is not None else np.zeros((H, W), bool)
    jumps = {}

    def joined(y, x, d, ny, nx):
        a, b = tiles[y, x], tiles[ny, nx]
        return bool((Dh[a, b] if d == 1 else Dh[b, a] if d == 3 else Dv[a, b] if d == 2 else Dv[b, a]))

    dist = np.full((H, W), np.inf)
    parent = {}
    heap = []
    for y, x in zip(*np.nonzero(np.isin(lab, sources))):
        dist[y, x] = 0.0
        heap.append((0.0, y, x))
    heapq.heapify(heap)
    while heap:
        du, y, x = heapq.heappop(heap)
        if du > dist[y, x]:
            continue
        for d, (dy, dx) in enumerate(STEP):
            ny, nx = y + dy, x + dx
            if torus:
                ny, nx = ny % H, nx % W
            elif not (0 <= ny < H and 0 <= nx < W):
                continue
            if wet[ny, nx] and not node[tiles[ny, nx]]:
                jb = bridge_jump(ts, tiles, y, x, d, torus)
                if jb is None:
                    continue
                r, (ly, lx), _ = jb
                a_ok = op(y, x, d) >= 0 if node[tiles[y, x]] else carvable[y, x]
                if not a_ok or not node[tiles[ly, lx]] or op(ly, lx, (d + 2) % 4) < 0:
                    continue
                c = r + (ucost[tiles[y, x], d] if node[tiles[y, x]] else 0.0) + ucost[tiles[ly, lx], (d + 2) % 4]
                nd = du + c + jitter[ly, lx]
                if nd < dist[ly, lx]:
                    dist[ly, lx] = nd
                    parent[ly, lx] = (y, x, d)
                    jumps[ly, lx] = jb
                    heapq.heappush(heap, (nd, ly, lx))
                continue
            if node[tiles[y, x]]:
                if node[tiles[ny, nx]]:
                    if joined(y, x, d, ny, nx):
                        c = 0.0
                    elif op(y, x, d) >= 0 and op(ny, nx, (d + 2) % 4) >= 0:
                        c = 1.0 + ucost[tiles[y, x], d] + ucost[tiles[ny, nx], (d + 2) % 4]
                    else:
                        continue
                elif carvable[ny, nx] and op(y, x, d) >= 0:
                    c = 1.0 + ucost[tiles[y, x], d]
                else:
                    continue
            elif node[tiles[ny, nx]]:
                if op(ny, nx, (d + 2) % 4) < 0:
                    continue
                c = ucost[tiles[ny, nx], (d + 2) % 4]
            elif carvable[ny, nx]:
                c = 1.0
            else:
                continue
            nd = du + c + jitter[ny, nx]
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                parent[ny, nx] = (y, x, d)
                jumps.pop((ny, nx), None)
                heapq.heappush(heap, (nd, ny, nx))

    # one landing cell per component, then walk the tree back to the main component
    opens, paths, done, bridges = {}, {}, set(), {}
    left = 0
    for c in targets:
        cells = np.argwhere(lab == c)
        dc = dist[cells[:, 0], cells[:, 1]]
        best = cells[np.argmin(dc)]
        if not np.isfinite(dc.min()) or (max_cost is not None and dc.min() > max_cost + 0.5):
            if delete and not clamp[lab == c].any():
                gone = lab == c
                tiles[gone] = void
                # trees left standing in it, with no other room next to them
                solid = ~node[tiles] & (tiles != ts.WALL) & (tiles != void) & ~clamp & ~wet
                tiles[dilate(gone, 1, torus) & solid & ~dilate(node[tiles], 1, torus)] = void
            else:
                left += 1
            continue
        y, x = int(best[0]), int(best[1])
        while (y, x) in parent and (y, x) not in done:
            done.add((y, x))
            py, px, d = parent[y, x]
            back = (d + 2) % 4
            a_node, b_node = node[tiles[py, px]], node[tiles[y, x]]
            if (y, x) in jumps:                          # a bridge run between (py, px) and (y, x)
                _, _, run = jumps[y, x]
                for i, b in enumerate(run, 1):
                    by_, bx_ = py + i * STEP[d][0], px + i * STEP[d][1]
                    bridges[(by_ % H, bx_ % W) if torus else (by_, bx_)] = b
                (opens if a_node else paths).setdefault((py, px), set()).add(d)
                opens.setdefault((y, x), set()).add(back)
            elif not (a_node and b_node and joined(py, px, d, y, x)):
                (opens if a_node else paths).setdefault((py, px), set()).add(d)
                (opens if b_node else paths).setdefault((y, x), set()).add(back)
            y, x = py, px
    for (y, x), ds in opens.items():
        s = tiles[y, x]
        for d in ds:
            s = OPEN[s, d] if s >= 0 else s
        if s >= 0:                                       # else a combination the tile set lacks
            tiles[y, x] = s
    for (y, x), ds in paths.items():
        tiles[y, x] = sig_by_doors(ts, tag, ds)
    for (y, x), b in bridges.items():
        tiles[y, x] = b
    return tiles, left


def local_connect(ts, tiles, K=16, r=4, max_cost=4, mode="islands", clamp=None, seed=0, w_energy=1.0,
                  tag="hallway"):
    """Bounded local connectivity repair.  Blocks of side K are visited in four
    phases ((by, bx) mod 2), so windows in one phase (block + margin r, r <
    K/2) never overlap and could run in parallel.  In each window (not wrapped:
    only paths inside it count):

      islands  components meeting the block that touch no window edge are
               joined to anything touching the edge (no island smaller than
               the window survives)
      all      every component meeting the block is joined to the largest of
               them (block connected within its margin)

    Joins cost at most max_cost edits each; a component that cannot be
    joined is set to wall in islands mode (it lies inside the window) and
    left alone in all mode.  tag: the kinds carved paths are made of.  Edits stay inside the window.
    Returns (tiles, stats) with per-window edit counts."""
    from castlegen import connmetrics as cm
    assert 2 * r < K
    tiles = np.array(tiles, np.int32)
    H, W = tiles.shape
    clamp = np.zeros((H, W), bool) if clamp is None else clamp
    edits = []
    for py in (0, 1):
        for px in (0, 1):
            for by in range(py, H // K, 2):
                for bx in range(px, W // K, 2):
                    ys = np.arange(by * K - r, (by + 1) * K + r) % H
                    xs = np.arange(bx * K - r, (bx + 1) * K + r) % W
                    win = tiles[np.ix_(ys, xs)]
                    lab, n = cm.labels(ts, win, torus=False)
                    if n <= 1:
                        edits.append(0)
                        continue
                    core = np.zeros(win.shape, bool)
                    core[r:r + K, r:r + K] = True
                    rim = np.ones(win.shape, bool)
                    rim[1:-1, 1:-1] = False
                    meets = np.unique(lab[core & (lab >= 0)])
                    if mode == "islands":
                        open_ = set(np.unique(lab[rim & (lab >= 0)]).tolist())
                        targets = [int(c) for c in meets if c not in open_]
                        sources = sorted(open_)
                        if targets and not sources:              # nothing leaves the window
                            sizes = np.bincount(lab[lab >= 0], minlength=n)
                            big = int(max(targets, key=lambda c: sizes[c]))
                            sources, targets = [big], [c for c in targets if c != big]
                    else:
                        sizes = np.bincount(lab[lab >= 0], minlength=n)
                        big = int(max(meets, key=lambda c: sizes[c]))
                        sources, targets = [big], [int(c) for c in meets if c != big]
                    if not targets:
                        edits.append(0)
                        continue
                    new, _ = connect(ts, win, clamp[np.ix_(ys, xs)], torus=False, max_cost=max_cost,
                                     seed=seed * 7919 + by * 131 + bx, w_energy=w_energy, tag=tag,
                                     sources=sources, targets=targets, delete=mode == "islands")
                    edits.append(int((new != win).sum()))
                    tiles[np.ix_(ys, xs)] = new
    e = np.array(edits)
    return tiles, dict(windows=len(e), mean_edits=float(e.mean()), max_edits=int(e.max()),
                       windows_edited=float((e > 0).mean()))


# ------------------------------------------------------------------------ run
def load_layout(path):
    if not os.path.exists(path):
        path = os.path.join(EXEMPLAR_DIR, path + ".json")
    with open(path) as f:
        return json.load(f)


def build(layout, ts=None, clean=None, join=None):
    """-> (ts, tiles, clamp).  clean / join override the layout's "clean"
    ({"rounds", "T", "tau"}: masked Gibbs on violations, placed cells held) and
    "connect" (true or {"tag", "w_energy"}: join every component to the
    largest, see connect())."""
    ts = ts or tileset.load(layout.get("tileset", "demo"))
    tiles, clamp, seeded = place(ts, layout)
    torus, sweeps, seed, T = layout.get("torus", True), layout.get("sweeps", 300), layout.get("seed", 0), layout.get("T", 1.0)
    # anneal the open ground with seeded interiors held, then let everything free settle cold
    tiles = gibbs(ts, tiles, sweeps, seed, T, layout.get("T_hot", 4.0), update=~clamp & ~seeded, torus=torus)
    tiles = gibbs(ts, tiles, sweeps // 2, seed + 1, T, update=~clamp, torus=torus)
    clean = layout.get("clean") if clean is None else clean
    if clean:
        c = clean if isinstance(clean, dict) else {}
        tiles = repair(ts, tiles, c.get("rounds", 30), seed + 2, T=c.get("T", 0.4), tau=c.get("tau", 1.9),
                       torus=torus, update=~clamp)
    if layout.get("flow"):                                     # the exemplar keeps the flow rule
        from castlegen.quantities import flow as F
        tiles = F.prune(ts, tiles, clamp, torus)
        c = clean if isinstance(clean, dict) else {}
        tiles = repair(ts, tiles, c.get("rounds", 30), seed + 4, T=c.get("T", 0.4), tau=c.get("tau", 1.9),
                       torus=torus, update=~clamp & ~ts.water[tiles], forbid=ts.water)
    if layout.get("support"):                                  # the exemplar keeps the support rule
        from castlegen.quantities import support as SP
        tiles = SP.fix(ts, tiles, layout.get("ground", 0), keep=clamp)
        c = clean if isinstance(clean, dict) else {}
        tiles = repair(ts, tiles, c.get("rounds", 30), seed + 5, T=c.get("T", 0.4), tau=c.get("tau", 1.9),
                       torus=torus, update=~clamp, keep_class=ts.solid.astype(np.int32))
    join = layout.get("connect", False) if join is None else join
    if join:
        c = join if isinstance(join, dict) else {}
        void = sig_by_name(ts, c["void"]) if "void" in c else None
        tiles, _ = connect(ts, tiles, clamp, torus, seed=seed + 3, tag=c.get("tag", "hallway"),
                           w_energy=c.get("w_energy", 1.0), void=void, fixed=object_mask(ts, layout))
    return ts, tiles, clamp


def to_png(ts, tiles, path, px=8, seed=0):
    render.save_png(render.image(ts, np.asarray(tileset.decorate(ts, seed, jnp.asarray(tiles))), px), path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layout")
    ap.add_argument("--png", default=None)
    ap.add_argument("--px", type=int, default=8)
    ap.add_argument("--out", default=None, help="save the signature grid (.npy)")
    ap.add_argument("--clean", action="store_true", help="masked Gibbs on violations after the infill")
    ap.add_argument("--connect", action="store_true", help="join every component to the largest")
    args = ap.parse_args()
    layout = load_layout(args.layout)
    ts, tiles, clamp = build(layout, clean=args.clean or None, join=args.connect or None)
    print(f"clamped {clamp.mean():.2f}  bad cells {bad_cells(ts, tiles).mean():.3f}  "
          f"census {tileset.structure_census(ts, tiles)}")
    print(render.ascii_grid(ts, np.asarray(tileset.decorate(ts, 0, jnp.asarray(tiles)))))
    if args.out:
        np.save(args.out, tiles)
    if args.png:
        to_png(ts, tiles, args.png, args.px)
        print("wrote", args.png)


if __name__ == "__main__":
    main()
