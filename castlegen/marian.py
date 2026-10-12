"""Marian Kleineberg's 3D WFC module set (MIT, https://marian42.de/article/wfc/)
as hard pair tables for castlegen.infinite.TileModel3D.

The files under cache/marian are parsed as data (line-based YAML scan), never
executed.

Directions (Orientations.cs), index d and its axis in Unity and in our grid
(L, H, W) = (layer up, row southward, col eastward); Unity is y up, x east,
z north, so Unity +z = row - 1:

    d  Unity           ours
    0  LEFT    -x      col - 1  (west)
    1  DOWN    -y      layer - 1
    2  BACK    -z      row + 1  (south)
    3  RIGHT   +x      col + 1  (east)
    4  UP      +y      layer + 1
    5  FORWARD +z      row - 1  (north)

opposite(d) = (d + 3) % 6.  allowed[d][a, b]: module b may sit next to module
a in direction d (Module.PossibleNeighbors[d] of a contains b).

Bitsets: ModuleSet is a long[] with module i at bit i % 64 of word i // 64;
Unity serialises a primitive array as the hex of its raw little-endian bytes,
so module i is bit i % 8 of byte i // 8 of the hex string (bytes.fromhex +
unpackbits little).

Weights: Slot.CollapseRandom draws among the remaining modules with weight
Prototype.Probability (each rotation of a prototype gets the full weight).

Boundary (Game.unity MapBehaviour, MapHeight 6): InfiniteMap(6) has no
slots outside 0 <= y < 6 (no constraint there) and two BoundaryConstraints,
both EnforceConnector: RelativeY 0, Down, connector 1 (layer 0 modules have a
Down face connector 1: they stand on the ground) and RelativeY -1, Up,
connector 0 (layer 5 modules have an Up face connector 0: open sky).  They are
applied to the default column, a TilingMap of size 1 x 6 x 1 whose slots are
their own horizontal neighbours, and propagated by arc consistency; every
map slot copies its layer's domain.  layer_domains() reproduces that."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DIRS = ("left", "down", "back", "right", "up", "forward")
FACES = ("Left", "Down", "Back", "Right", "Up", "Forward")
HORIZONTAL = (0, 2, 3, 5)
# grid step (dlayer, drow, dcol) for each direction d
STEP = ((0, 0, -1), (-1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0), (0, -1, 0))
MAP_HEIGHT = 6
BOUNDARY = ((0, 1, 1), (-1, 4, 0))      # (RelativeY, direction d, enforced connector)


def opposite(d):
    return (d + 3) % 6


def rotate(d, amount):
    """Orientations.Rotate."""
    if d in (1, 4):
        return d
    return HORIZONTAL[(HORIZONTAL.index(d) + amount) % 4]


@dataclass
class MarianSet:
    names: list                 # T module names ("Empty R0", ...)
    prototype: np.ndarray       # (T,) int64 fileID of the ModulePrototype component
    rotation: np.ndarray        # (T,) int
    weight: np.ndarray          # (T,) Prototype.Probability
    allowed: np.ndarray         # (6, T, T) bool
    faces: dict = field(default_factory=dict)   # prototype id -> list of 6 face dicts
    proto_name: dict = field(default_factory=dict)

    @property
    def T(self):
        return len(self.names)

    def asymmetry(self):
        """Pairs where allowed[d][a, b] != allowed[opposite d][b, a], per d."""
        return [int((self.allowed[d] != self.allowed[opposite(d)].T).sum()) for d in range(6)]

    def face(self, t, d):
        """Face of module t seen in world direction d (Module.GetFace for
        horizontal faces; vertical faces are not rotated)."""
        f = self.faces[int(self.prototype[t])]
        return f[rotate(d, int(self.rotation[t])) if d in HORIZONTAL else d]

    def fits(self, a, d, b):
        """Module.Fits(direction d, module b) from the face connectors."""
        if d in HORIZONTAL:
            f1, f2 = self.face(a, d), self.face(b, opposite(d))
            return f1["Connector"] == f2["Connector"] and (f1["Symmetric"] or f1["Flipped"] != f2["Flipped"])
        f1 = self.faces[int(self.prototype[a])][d]
        f2 = self.faces[int(self.prototype[b])][opposite(d)]
        return f1["Connector"] == f2["Connector"] and (
            f1["Invariant"] or (f1["Rotation"] + self.rotation[a]) % 4 == (f2["Rotation"] + self.rotation[b]) % 4)

    def boundary_ok(self, L=MAP_HEIGHT):
        """(L, T) bool: the raw boundary constraints (Module.Fits(d,
        connector): connector equality on the face) before propagation."""
        ok = np.ones((L, self.T), bool)
        for rel, d, conn in BOUNDARY:
            k = rel % L
            for t in range(self.T):
                if self.face(t, d)["Connector"] != conn:
                    ok[k, t] = False
        return ok

    def layer_domains(self, L=MAP_HEIGHT):
        """(L, T) bool: boundary constraints propagated by arc consistency
        on the 1 x L x 1 default column with horizontal self-neighbours, as
        InfiniteMap does (no constraint beyond layers 0 and L - 1)."""
        dom = self.boundary_ok(L)
        A = self.allowed
        changed = True
        while changed:
            changed = False
            for k in range(L):
                keep = dom[k].copy()
                for d in range(6):
                    dl = STEP[d][0]
                    kk = k + dl
                    if kk < 0 or kk >= L:
                        continue
                    # t needs some neighbour n in dom[kk] with allowed[d][t, n]
                    keep &= (A[d][:, dom[kk]]).any(1)
                if (keep != dom[k]).any():
                    dom[k] = keep
                    changed = True
        return dom

    def air(self):
        """Index of the Empty module (air)."""
        return self.names.index("Empty R0")


def _bits(hexstr, T):
    b = np.frombuffer(bytes.fromhex(hexstr), np.uint8)
    return np.unpackbits(b, bitorder="little")[:T].astype(bool)


def _parse_prototypes(path):
    """{component fileID: (probability, enabled, gameobject id, faces)},
    {gameobject id: name}."""
    protos, gos = {}, {}
    doc_id, kind, lines = None, None, []

    def flush():
        if kind == "114" and any(l.startswith("  Probability:") for l in lines):
            p = float(next(l for l in lines if l.startswith("  Probability:")).split(":")[1])
            en = int(next(l for l in lines if l.startswith("  m_Enabled:")).split(":")[1])
            go = int(re.search(r"fileID: (-?\d+)", next(l for l in lines if l.startswith("  m_GameObject:"))).group(1))
            faces, cur = [None] * 6, None
            for l in lines:
                m = re.match(r"^  (\w+):\s*$", l)
                if m and m.group(1) in FACES:
                    cur = {}
                    faces[FACES.index(m.group(1))] = cur
                    continue
                m = re.match(r"^    (\w+): (-?\d+)\s*$", l)
                if m and cur is not None:
                    cur[m.group(1)] = int(m.group(2))
                elif not l.startswith("    "):
                    cur = None
            protos[doc_id] = (p, en, go, faces)
        elif kind == "1":
            for l in lines:
                if l.startswith("  m_Name:"):
                    gos[doc_id] = l.split(":", 1)[1].strip()

    for line in Path(path).read_text().splitlines():
        m = re.match(r"^--- !u!(\d+) &(-?\d+)", line)
        if m:
            flush()
            kind, doc_id, lines = m.group(1), int(m.group(2)), []
        else:
            lines.append(line)
    flush()
    return protos, gos


def load(root="cache/marian"):
    root = Path(root)
    protos, gos = _parse_prototypes(root / "Assets" / "Prototypes.prefab")
    names, proto, rot, idx, data = [], [], [], [], []
    cur = None
    for line in (root / "Assets" / "ModuleData.asset").read_text().splitlines():
        if line.startswith("  - Name:"):
            cur = {"name": line.split(":", 1)[1].strip(), "data": []}
            names.append(cur)
        elif cur is None:
            continue
        elif line.startswith("    Prototype:"):
            cur["proto"] = int(re.search(r"fileID: (-?\d+)", line).group(1))
        elif line.startswith("    Rotation:"):
            cur["rot"] = int(line.split(":")[1])
        elif line.startswith("    - data:"):
            cur["data"].append(line.split(":")[1].strip())
        elif line.startswith("    Index:"):
            cur["index"] = int(line.split(":")[1])
    T = len(names)
    allowed = np.zeros((6, T, T), bool)
    for a, m in enumerate(names):
        if m["index"] != a or len(m["data"]) != 6:
            raise ValueError(f"module {a}: index {m['index']}, {len(m['data'])} bitsets")
        for d in range(6):
            allowed[d, a] = _bits(m["data"][d], T)
    pid = np.array([m["proto"] for m in names], np.int64)
    missing = set(pid.tolist()) - set(protos)
    if missing:
        raise ValueError(f"prototype ids not in Prototypes.prefab: {sorted(missing)[:5]}")
    w = np.array([protos[p][0] for p in pid], float)
    faces = {p: protos[p][3] for p in set(pid.tolist())}
    pname = {p: gos.get(protos[p][2], "?") for p in faces}
    return MarianSet([m["name"] for m in names], pid, np.array([m["rot"] for m in names]), w, allowed,
                     faces, pname)


# Empty's weight per layer (bottom -> top); None keeps Marian's (10 everywhere).
AIR_WEIGHTS = None


def air_layer_unary(ms: MarianSet, air_weights):
    """(L, T) per-layer unary that gives Empty weight air_weights[l] on layer
    l instead of its prototype weight (other modules unchanged)."""
    L = len(air_weights)
    lu = np.zeros((L, ms.T))
    t = ms.air()
    lu[:, t] = -np.log(np.asarray(air_weights, float)) + np.log(ms.weight[t])
    return lu


def tile_model(ms: MarianSet, L=MAP_HEIGHT, domains="ac", colors=None, air_weights="default"):
    """TileModel3D: u = -log weight, forbidden pairs = not allowed in either
    listing (union of the two directions' bitsets), per-layer domains
    ('ac' propagated, 'raw' boundary only, None unconstrained)."""
    from castlegen.infinite import TileModel3D
    F = np.zeros((6, ms.T, ms.T), np.uint8)
    for d in range(6):
        # F[d][n, t] = 1: t with neighbour n in direction d is forbidden
        ok = ms.allowed[d] & ms.allowed[opposite(d)].T      # ok[t, n]
        F[d] = (~ok).T
    dom = {"ac": ms.layer_domains, "raw": ms.boundary_ok}[domains](L) if domains else np.ones((L, ms.T), bool)
    if colors is None:
        colors = proto_colors(ms)
    if isinstance(air_weights, str):
        air_weights = AIR_WEIGHTS
    lu = None if air_weights is None else air_layer_unary(ms, air_weights)
    return TileModel3D(list(ms.names), -np.log(ms.weight), F, dom, colors, lu=lu)


def proto_colors(ms: MarianSet):
    """(T, 3) uint8: a colour per prototype (hash of its id), air grey-blue,
    solid dark."""
    out = np.empty((ms.T, 3), np.uint8)
    for t, p in enumerate(ms.prototype):
        h = (int(p) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        h ^= h >> 29
        out[t] = (60 + (h & 0xFF) % 180, 60 + ((h >> 8) & 0xFF) % 180, 60 + ((h >> 16) & 0xFF) % 180)
    return out
