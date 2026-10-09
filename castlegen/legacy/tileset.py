"""Legacy (G0, JAX castle sampler era); superseded by castlegen/channels. See the original README.md (git history).

Tile sets: a JSON spec of kinds, sockets and tag rules, compiled into the
tables the sampler reads.

The energy sees a tile only through its SIGNATURE: one (kind, distinct
rotation of its sockets).  A kind's variants multiply the number of types
but never change the energy, so a spec can describe millions of types while
the sampler scores a few dozen signatures per site.  The concrete type is
drawn once at the end (`decorate`), per cell, from the signature's members:
rotation uniform, each variant axis from its own weights.

Spec, top level:
  extends   : a base tile set's name: its spec, each list extended and each
              dict updated by this one's (load only)
  sockets   : ["wall", {"name": "door", "draw": "gap"}, ...]; "wall" required
  connects  : [["door", "door"], ...]  unordered pairs that form a door
  dangling  : {"door": 3.0}  energy per socket facing one it does not connect to
  wall      : {"unary": 0.0, "render": {...}}           built-in solid tile
  gate      : {"socket": "door", "render": {...}}       built-in, south side
  kinds     : [kind, ...]
  rules     : [{"a": tag, "b": tag, "when": "door" | "contact" | "above" | "beside", "energy": x}]
  structures: [structure, ...]  multi-cell footprints, expanded into kinds
Structure:
  name, size [rows, cols], tags, unary (per piece), seam (dangling energy of
  every internal socket), doors [{"piece": [r, c], "side": "S", "socket":
  "door"}] (outer sides default to "wall"), glyph, render.  Each piece becomes
  a kind "<name>_r<r>c<c>" tagged with the structure's tags and name; every
  internal edge gets its own socket that connects only to itself, so a piece
  pays `seam` for each neighbour that is not its partner.  No rotations.
Kind:
  name, tags, sides {"N","E","S","W": socket}, rotations [0, 90, ...] | "all",
  solid (not a room: never on a path, no reachability; its sockets still
  count for dangling, so a solid with "court" sides can stand in a courtyard),
  unary (energy; the kind's total mass is
  exp(-unary), shared by its rotations and variants), glyph | glyphs (one per
  rotation), render {"color": "#rrggbb"} | {"image": "file.png"} (default: a
  random colour per variant), variants {axis: [value | {"name", "weight",
  "render"}]}.
  flow: water.  "N" | "E" | "S" | "W": the side the tile flows out of
  (rotated with the tile; a list of sides makes one kind per side, named
  <name>_<side> after the first, sharing the kind's mass); "lake": still
  water (a source and a sink, exempt from the flow rule).  source: true marks
  a spring (needs no inflow).  Flow is part of the signature, so a river's
  four rotations are four signatures although their sockets agree.
  flow_rules (top level): {"into_water", "into_dry", "head_on", "parallel",
  "anti"}: pair energies for a flowing tile pointing into water / into a
  non-water tile, two tiles pointing into each other, and side-by-side
  neighbours flowing the same / opposite way (all soft).
Rule tags match a kind's tags or its name; "*" matches anything and "!x"
anything without x.  A rule applies once per unordered neighbour pair, "door"
only when the facing sockets connect, "contact" always; "above" only when a
is directly above b (vertical pairs, ordered: side-view tile sets), "beside"
only for horizontal neighbours.  All matching rules add.
A kind (or the built-in wall, "wall": {"tags": [...]}) tagged "solid" is
matter for the side-view support rule: `ts.solid` (S,) bool.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .core import noise

SIDES = ("N", "E", "S", "W")
TILESET_DIR = os.path.join(os.path.dirname(__file__), "tilesets")
DECOR_STEP = 1000     # noise step index reserved for the type draw


class Tables(NamedTuple):
    """Device arrays the sampler reads.  S = number of signatures."""
    Eh: jnp.ndarray          # (S, S) energy of (left, right)
    Ev: jnp.ndarray          # (S, S) energy of (top, bottom)
    Dh: jnp.ndarray          # (S, S) float 0/1: a door joins (left, right)
    Dv: jnp.ndarray          # (S, S) float 0/1: a door joins (top, bottom)
    logz: jnp.ndarray        # (S,) log base mass of the signature
    is_room: jnp.ndarray     # (S,) bool: needs a finite d
    plannable: jnp.ndarray   # (S,) float 0/1: counted in the block histogram
    sig_kind: jnp.ndarray    # (S,)
    sig_rot: jnp.ndarray     # (S, 4) member rotation indices, padded
    sig_nrot: jnp.ndarray    # (S,)
    kind_offset: jnp.ndarray  # (K,) first type id of the kind
    kind_V: jnp.ndarray      # (K,) variants per rotation
    axis_start: jnp.ndarray  # (K, A) start of the axis CDF in axis_cdf
    axis_n: jnp.ndarray      # (K, A) values on the axis (1 if absent)
    axis_stride: jnp.ndarray  # (K, A) type-id stride of the axis
    axis_cdf: jnp.ndarray    # (sum of axis sizes,) concatenated CDFs


@dataclass
class Kind:
    name: str
    tags: frozenset
    sides: tuple             # 4 socket indices, rotation 0
    rotations: tuple         # degrees clockwise
    solid: bool
    unary: float
    glyphs: tuple            # one per rotation
    render: dict
    axes: list = field(default_factory=list)   # [(axis, [(value, weight, render)])]
    flow: int = -1           # side the tile flows out of at rotation 0 (-1: none)
    lake: bool = False
    source: bool = False

    @property
    def n_variants(self):
        return math.prod(len(v) for _, v in self.axes) if self.axes else 1


@dataclass
class TileSet:
    name: str
    base_dir: str
    sockets: list            # names
    socket_draw: list        # "line" | "gap" | "none"
    kinds: list              # user kinds, then wall, then gate
    sig_kind: np.ndarray
    sig_rots: list           # per signature, list of rotation indices
    sig_sockets: np.ndarray  # (S, 4)
    kind_offset: np.ndarray
    n_types: int
    WALL: int                # signature ids of the built-ins
    GATE: int
    jt: Tables
    np_tables: dict
    structures: list = field(default_factory=list)   # [(name, (rows, cols) array of kind ids)]
    flow_dir: np.ndarray = None   # (S,) side the signature flows out of, -1 if none
    water: np.ndarray = None      # (S,) bool: flowing water or lake
    lake: np.ndarray = None       # (S,) bool: still water, exempt from the flow rule
    source: np.ndarray = None     # (S,) bool: spring, needs no inflow
    solid: np.ndarray = None      # (S,) bool: tagged "solid" (side-view matter, support rule)

    @property
    def n_sig(self):
        return len(self.sig_kind)

    def sig_name(self, s):
        k = self.kinds[self.sig_kind[s]]
        return f"{k.name}@{k.rotations[self.sig_rots[s][0]]}"

    def decode(self, tid):
        """type ids (any shape) -> kind index, rotation index, (A, ...) variant indices."""
        tid = np.asarray(tid, np.int64)
        k = np.searchsorted(self.kind_offset, tid, side="right") - 1
        V = np.asarray(self.np_tables["kind_V"])[k]
        rem = tid - self.kind_offset[k]
        rot = rem // V
        var = rem % V
        n = self.np_tables["axis_n"][k]              # (..., A)
        stride = self.np_tables["axis_stride"][k]
        vals = np.where(n > 1, (var[..., None] // np.maximum(stride, 1)) % n, 0)
        return k, rot, np.moveaxis(vals, -1, 0)

    def type_name(self, tid):
        k, r, vals = self.decode(tid)
        kind = self.kinds[int(k)]
        s = f"{kind.name}@{kind.rotations[int(r)]}"
        if kind.axes:
            s += "[" + ",".join(vs[int(v)][0] for (_, vs), v in zip(kind.axes, vals)) + "]"
        return s

    def sig_of_type(self, tid):
        k, r, _ = self.decode(tid)
        return self.np_tables["type_sig_lookup"][k, r]


# --------------------------------------------------------------------------
# loading and compiling
# --------------------------------------------------------------------------
def load(name_or_path: str) -> TileSet:
    path = name_or_path
    if not os.path.exists(path):
        path = os.path.join(TILESET_DIR, name_or_path + ".json")
    with open(path) as f:
        spec = json.load(f)
    spec.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    if "extends" in spec:                                                   # a base spec's lists and dicts, extended
        with open(os.path.join(TILESET_DIR, spec.pop("extends") + ".json")) as f:
            base = json.load(f)
        for k, v in spec.items():
            base[k] = (base.get(k, []) + v if isinstance(v, list)
                       else dict(base.get(k, {}), **v) if isinstance(v, dict) else v)
        spec = base
    return compile_spec(spec, base_dir=os.path.dirname(os.path.abspath(path)))


def _rotations(r):
    if r == "all":
        return (0, 90, 180, 270)
    r = tuple(int(x) for x in r)
    if not r or any(x % 90 for x in r) or len(set(x % 360 for x in r)) != len(r):
        raise ValueError(f"rotations must be distinct multiples of 90, got {r}")
    return tuple(x % 360 for x in r)


def _rotate(sides, deg):
    """Sockets (N, E, S, W) after turning the tile deg clockwise."""
    k = deg // 90
    return tuple(sides[(i - k) % 4] for i in range(4))


def _parse_value(v):
    if isinstance(v, str):
        return (v, 1.0, {})
    return (v["name"], float(v.get("weight", 1.0)), v.get("render", {}))


def _expand_structures(spec):
    """Structures -> extra sockets, connects, dangling and kinds (a new spec)."""
    spec = dict(spec)
    sockets = list(spec["sockets"])
    connects = list(spec.get("connects", []))
    dangling = dict(spec.get("dangling", {}))
    kinds = list(spec["kinds"])
    layout = []
    for st in spec.get("structures", []):
        name, (R, C) = st["name"], st["size"]
        sides = {(r, c): {s: "wall" for s in SIDES} for r in range(R) for c in range(C)}
        for r in range(R):
            for c in range(C):
                if c + 1 < C:
                    s = f"{name}.h{r}.{c}"
                    sides[r, c]["E"] = sides[r, c + 1]["W"] = s
                    sockets.append({"name": s, "draw": "none"}); connects.append([s, s])
                    dangling[s] = float(st.get("seam", 10.0))
                if r + 1 < R:
                    s = f"{name}.v{r}.{c}"
                    sides[r, c]["S"] = sides[r + 1, c]["N"] = s
                    sockets.append({"name": s, "draw": "none"}); connects.append([s, s])
                    dangling[s] = float(st.get("seam", 10.0))
        for dspec in st.get("doors", []):
            r, c = dspec["piece"]
            if not (r in (0, R - 1) or c in (0, C - 1)):
                raise ValueError(f"{name}: door on an interior piece {r, c}")
            sides[r, c][dspec["side"]] = dspec.get("socket", "door")
        names = [[f"{name}_r{r}c{c}" for c in range(C)] for r in range(R)]
        for r in range(R):
            for c in range(C):
                kinds.append({"name": names[r][c], "tags": list(st.get("tags", [])) + [name],
                              "unary": st.get("unary", 0.0), "sides": sides[r, c],
                              "glyph": st.get("glyph", name[0]), "render": st.get("render", {})})
        layout.append((name, names))
    spec.update(sockets=sockets, connects=connects, dangling=dangling, kinds=kinds)
    return spec, layout


def _expand_flow(spec):
    """Kinds with a list of flow sides -> one kind per side (mass shared)."""
    kinds = []
    for ks in spec["kinds"]:
        f = ks.get("flow")
        if isinstance(f, list):
            for i, side in enumerate(f):
                kinds.append(dict(ks, flow=side, name=ks["name"] if i == 0 else f"{ks['name']}_{side.lower()}",
                                  unary=float(ks.get("unary", 0.0)) + math.log(len(f))))
        else:
            kinds.append(ks)
    return dict(spec, kinds=kinds)


def compile_spec(spec: dict, base_dir: str = ".") -> TileSet:
    spec, layout = _expand_structures(_expand_flow(spec))
    # sockets
    sockets, draw = [], []
    for s in spec["sockets"]:
        name = s if isinstance(s, str) else s["name"]
        sockets.append(name)
        d = "line" if name == "wall" else "none"
        draw.append(d if isinstance(s, str) else s.get("draw", d))
    if "wall" not in sockets:
        raise ValueError('sockets must include "wall"')
    sid = {n: i for i, n in enumerate(sockets)}
    ns = len(sockets)
    conn = np.zeros((ns, ns), bool)
    for a, b in spec.get("connects", []):
        conn[sid[a], sid[b]] = conn[sid[b], sid[a]] = True
    if conn[sid["wall"]].any():
        raise ValueError('"wall" socket cannot connect')
    dang = np.zeros(ns, np.float32)
    for n, e in spec.get("dangling", {}).items():
        dang[sid[n]] = e

    # kinds, then the two built-ins
    kinds, names = [], set()
    for ks in spec["kinds"]:
        name = ks["name"]
        if name in names or name in ("wall", "gate"):
            raise ValueError(f"duplicate or reserved kind name {name!r}")
        names.add(name)
        rots = _rotations(ks.get("rotations", [0]))
        solid = bool(ks.get("solid", False))
        sides = tuple(sid[ks["sides"][s]] for s in SIDES) if "sides" in ks else (sid["wall"],) * 4
        glyphs = ks.get("glyphs") or [ks.get("glyph", name[0])] * len(rots)
        if len(glyphs) != len(rots):
            raise ValueError(f"{name}: {len(glyphs)} glyphs for {len(rots)} rotations")
        axes = []
        for axis, vals in ks.get("variants", {}).items():
            vals = [_parse_value(v) for v in vals]
            if not vals or any(w <= 0 for _, w, _ in vals):
                raise ValueError(f"{name}.{axis}: variant weights must be positive")
            axes.append((axis, vals))
        flow = ks.get("flow")
        if flow is not None and flow != "lake" and flow not in SIDES:
            raise ValueError(f'{name}: flow must be a side or "lake", got {flow!r}')
        kinds.append(Kind(name, frozenset(ks.get("tags", [])) | {name}, sides, rots, solid,
                          float(ks.get("unary", 0.0)), tuple(glyphs), ks.get("render", {}), axes,
                          SIDES.index(flow) if flow in SIDES else -1, flow == "lake", bool(ks.get("source", False))))
    wall_spec, gate_spec = spec.get("wall", {}), spec.get("gate", {})
    kinds.append(Kind("wall", frozenset({"wall"}) | frozenset(wall_spec.get("tags", [])), (sid["wall"],) * 4, (0,), True,
                      float(wall_spec.get("unary", 0.0)), (wall_spec.get("glyph", "█"),),
                      wall_spec.get("render", {"color": "#3b3632"})))
    gsock = sid[gate_spec.get("socket", "door")]
    kinds.append(Kind("gate", frozenset({"gate"}), (sid["wall"], sid["wall"], gsock, sid["wall"]),
                      (0,), False, 0.0, (gate_spec.get("glyph", "G"),),
                      gate_spec.get("render", {"color": "#c0392b"})))
    K = len(kinds)

    # signatures: one per (kind, distinct rotated sockets and flow side)
    sig_kind, sig_rots, sig_sockets, sig_flow = [], [], [], []
    type_sig = np.zeros((K, 4), np.int32)
    for ki, k in enumerate(kinds):
        seen = {}
        for ri, deg in enumerate(k.rotations):
            f = (k.flow + deg // 90) % 4 if k.flow >= 0 else -1
            key = (_rotate(k.sides, deg), f)
            if key not in seen:
                seen[key] = len(sig_kind)
                sig_kind.append(ki); sig_rots.append([]); sig_sockets.append(key[0]); sig_flow.append(f)
            sig_rots[seen[key]].append(ri)
            type_sig[ki, ri] = seen[key]
    S = len(sig_kind)
    sig_kind = np.asarray(sig_kind, np.int32)
    sig_sockets = np.asarray(sig_sockets, np.int32)
    WALL, GATE = S - 2, S - 1

    # base mass: the kind's exp(-unary), split evenly over its rotations
    logz = np.array([-kinds[sig_kind[s]].unary + math.log(len(sig_rots[s]) / len(kinds[sig_kind[s]].rotations))
                     for s in range(S)], np.float32)
    is_room = np.array([not kinds[k].solid and kinds[k].name != "gate" for k in sig_kind])
    plannable = np.array([kinds[k].name not in ("wall", "gate") for k in sig_kind], np.float32)

    # tag matrix and rules
    all_tags = set().union(*(k.tags for k in kinds))
    def match(expr):
        neg = expr.startswith("!")
        t = expr[1:] if neg else expr
        if t != "*" and t not in all_tags:
            raise ValueError(f"rule refers to unknown tag {t!r}")
        m = np.array([t == "*" or t in kinds[k].tags for k in sig_kind])
        return ~m if neg else m
    R_door = np.zeros((S, S), np.float32)
    R_contact = np.zeros((S, S), np.float32)
    R_above = np.zeros((S, S), np.float32)            # [top, bottom], vertical pairs only
    R_beside = np.zeros((S, S), np.float32)           # horizontal pairs only
    for r in spec.get("rules", []):
        ma, mb = match(r["a"]), match(r["b"])
        M = (ma[:, None] & mb[None, :]) | (mb[:, None] & ma[None, :])
        when = r.get("when", "door")
        if when == "above":
            R_above[ma[:, None] & mb[None, :]] += float(r["energy"])
            continue
        if when not in ("door", "contact", "beside"):
            raise ValueError(f'rule "when" must be "door", "contact", "above" or "beside", got {when!r}')
        {"door": R_door, "contact": R_contact, "beside": R_beside}[when][M] += float(r["energy"])

    flow_dir = np.asarray(sig_flow, np.int32)
    lake = np.array([kinds[k].lake for k in sig_kind])
    source = np.array([kinds[k].source for k in sig_kind])
    water = (flow_dir >= 0) | lake
    solid = np.array(["solid" in kinds[k].tags for k in sig_kind])
    fr = spec.get("flow_rules", {})

    def flow_pair(side_a, side_b):
        """(S, S) flow energy of a (left/top) next to b (right/bottom); side_a
        is a's side facing b."""
        fa, fb = flow_dir[:, None], flow_dir[None, :]
        wa, wb = water[:, None], water[None, :]
        E = np.zeros((S, S), np.float32)
        E += np.where(fa == side_a, np.where(wb, fr.get("into_water", 0.0), fr.get("into_dry", 0.0)), 0.0)
        E += np.where(fb == side_b, np.where(wa, fr.get("into_water", 0.0), fr.get("into_dry", 0.0)), 0.0)
        E += np.where((fa == side_a) & (fb == side_b), fr.get("head_on", 0.0), 0.0)
        perp = (fa >= 0) & (fa % 2 != side_a % 2)
        E += np.where(perp & (fa == fb), fr.get("parallel", 0.0), 0.0)
        E += np.where(perp & (fb == (fa + 2) % 4), fr.get("anti", 0.0), 0.0)
        return E

    def pair_tables(side_a, side_b):
        sa, sb = sig_sockets[:, side_a][:, None], sig_sockets[:, side_b][None, :]
        door = conn[sa, sb]
        E = np.where(door, R_door, dang[sa] + dang[sb]) + R_contact + flow_pair(side_a, side_b)
        return E.astype(np.float32), door
    Eh, Dh = pair_tables(1, 3)   # left's E side against right's W side
    Ev, Dv = pair_tables(2, 0)   # top's S side against bottom's N side
    Eh, Ev = (Eh + R_beside).astype(np.float32), (Ev + R_above).astype(np.float32)

    # type ids: kind-major, then rotation, then variants (first axis most significant)
    kind_V = np.array([k.n_variants for k in kinds], np.int64)
    kind_count = kind_V * np.array([len(k.rotations) for k in kinds], np.int64)
    kind_offset = np.concatenate([[0], np.cumsum(kind_count)[:-1]]).astype(np.int64)
    n_types = int(kind_count.sum())
    if n_types >= 2**31:
        raise ValueError(f"{n_types} types do not fit int32 ids")
    A = max([len(k.axes) for k in kinds] + [1])
    axis_start = np.zeros((K, A), np.int32)
    axis_n = np.ones((K, A), np.int32)
    axis_stride = np.zeros((K, A), np.int32)
    cdfs = [np.ones(1, np.float32)]          # shared one-value CDF for absent axes
    pos = 1
    for ki, k in enumerate(kinds):
        sizes = [len(v) for _, v in k.axes]
        for a, (_, vals) in enumerate(k.axes):
            w = np.array([x[1] for x in vals], np.float64)
            c = np.cumsum(w / w.sum()).astype(np.float32); c[-1] = 1.0
            axis_start[ki, a], axis_n[ki, a] = pos, len(vals)
            axis_stride[ki, a] = math.prod(sizes[a + 1:])
            cdfs.append(c); pos += len(vals)
    sig_rot = np.zeros((S, 4), np.int32)
    for s, rs in enumerate(sig_rots):
        sig_rot[s, :len(rs)] = rs

    np_tables = dict(Eh=Eh, Ev=Ev, Dh=Dh, Dv=Dv, logz=logz, is_room=is_room, kind_V=kind_V,
                     axis_n=axis_n, axis_stride=axis_stride, type_sig_lookup=type_sig,
                     conn=conn)
    jt = Tables(
        Eh=jnp.asarray(Eh), Ev=jnp.asarray(Ev),
        Dh=jnp.asarray(Dh, jnp.float32), Dv=jnp.asarray(Dv, jnp.float32),
        logz=jnp.asarray(logz), is_room=jnp.asarray(is_room), plannable=jnp.asarray(plannable),
        sig_kind=jnp.asarray(sig_kind), sig_rot=jnp.asarray(sig_rot),
        sig_nrot=jnp.asarray([len(r) for r in sig_rots], jnp.int32),
        kind_offset=jnp.asarray(kind_offset, jnp.int32), kind_V=jnp.asarray(kind_V, jnp.int32),
        axis_start=jnp.asarray(axis_start), axis_n=jnp.asarray(axis_n),
        axis_stride=jnp.asarray(axis_stride), axis_cdf=jnp.asarray(np.concatenate(cdfs)))
    kind_id = {k.name: i for i, k in enumerate(kinds)}
    structures = [(name, np.array([[kind_id[n] for n in row] for row in names])) for name, names in layout]
    return TileSet(spec.get("name", "tileset"), base_dir, sockets, draw, kinds, sig_kind, sig_rots,
                   sig_sockets, kind_offset, n_types, WALL, GATE, jt, np_tables, structures,
                   flow_dir, water, lake, source, solid)


def structure_census(ts: TileSet, tiles):
    """{structure: (complete, orphan_cells)} on a signature grid.  A structure is
    complete when every piece sits at its offset from a top-left piece; orphan
    cells are structure pieces not covered by a complete one."""
    k = ts.sig_kind[np.asarray(tiles)]
    H, W = k.shape
    out = {}
    for name, grid in ts.structures:
        R, C = grid.shape
        covered = np.zeros((H, W), bool)
        complete = 0
        for y, x in zip(*np.nonzero(k == grid[0, 0])):
            if y + R <= H and x + C <= W and np.array_equal(k[y:y + R, x:x + C], grid):
                complete += 1
                covered[y:y + R, x:x + C] = True
        pieces = np.isin(k, grid.ravel())
        out[name] = (complete, int((pieces & ~covered).sum()))
    return out


# --------------------------------------------------------------------------
# the type draw
# --------------------------------------------------------------------------
def decorate(ts: TileSet, castle_id, sig):
    """Signature grid -> type-id grid, from the hashed noise (chunk-invariant).

    Rotation is uniform over the signature's member rotations; each variant
    axis is drawn from its weights by a fixed-length binary search in the
    concatenated CDF, so memory does not grow with the number of values.
    """
    t = ts.jt
    H, W = sig.shape
    ys, xs = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
    k = t.sig_kind[sig]
    nrot = t.sig_nrot[sig]
    u = noise(castle_id, 0, DECOR_STEP, 0, ys, xs, 0)
    ri = jnp.minimum((u * nrot).astype(jnp.int32), nrot - 1)
    rot = jnp.take_along_axis(t.sig_rot[sig], ri[..., None], axis=-1)[..., 0]
    tid = t.kind_offset[k] + rot * t.kind_V[k]
    n_steps = max(1, int(np.ceil(np.log2(int(np.asarray(t.axis_n).max()) + 1))))
    for a in range(t.axis_n.shape[1]):
        u = noise(castle_id, 0, DECOR_STEP, 0, ys, xs, 1 + a)
        lo = t.axis_start[k, a]
        hi = lo + t.axis_n[k, a] - 1
        for _ in range(n_steps):              # first index with cdf >= u
            mid = (lo + hi) // 2
            right = t.axis_cdf[mid] < u
            lo = jnp.where(right, mid + 1, lo)
            hi = jnp.where(right, hi, mid)
        tid = tid + (lo - t.axis_start[k, a]) * t.axis_stride[k, a]
    return tid
