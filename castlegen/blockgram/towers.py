"""The tower grammar, compiled by core.py.

  region (owner, 96 columns; split at 32)
    owner variables, Gibbs with hashed noise: n in {absent, 10..15},
    bridge story to the east neighbour (or none); a bridge needs both towers
    to hold that story above their ground floor and the deck to clear the
    terrain (hard), and earns a bonus
    -> tower, bridge (east), trees
  tower (split at 32): base story from the terrain under it (floors sit on a
    global 6-row story lattice, so towers on different ground can share a
    bridge); floor chain drawn exactly (backward messages, forward draws):
    state = (room widths, stair room, stair direction); the hole above a
    stair may not open onto a wall or the next floor's stair; doors to
    bridges, balconies and the entrance keep stairs out of the end rooms;
    the roof's footprint keeps the top hatch clear; soft terms prefer
    floors that differ (interlocking walls), stairs that change room and
    direction
    -> foundation, floors, roof, balconies, entrance
  floor (split at 8) -> shell (walls, doors, windows, ceiling with the hole),
    5 rooms (leaves: stair or furniture by room kind)

Counts are loop bounds: n floors, 5 rooms per floor, one stair per floor
(the top floor's goes to the roof).  Floor pitch 6 and room pitches 7, 9,
10 (6 for a room without a stair) against blocks of 8 and 32."""
from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np

from .core import A, Canvas, Grammar, NodeType, X0, X1, Y0, Y1, SEED, TYPE, hash_grid, mix, record, rng

# tile types (for the checker and the picture)
(SKY, CLOUD, FAR, TERRAIN, TREE, FOUND, EXTW, EXTE, WALL, DOOR, SLAB, HOLE, ROOF, ROOM, STEP, FILL,
 FURN, BRIDGE, ROOFDEC, BALC, WINDOW, EXTDOOR, OUTSTEP, LEDGE) = range(24)
BODY = (EXTW, EXTE, WALL, DOOR, SLAB, HOLE, ROOF, ROOM, STEP, FILL, FURN, WINDOW, EXTDOOR)

H = 224
RW = 96                                     # region (owner) width
PITCH, ROOMH = 6, 5
Y_LAT = 206                                 # story s has its floor slab on row Y_LAT - 6 s
WIDTHS = (5, 6, 8, 9)
NROOM = 5
NS = (0, 10, 11, 12, 13, 14, 15)
SWEEPS = 6
W_SAME_COMP, W_SAME_ROOM, W_SAME_DIR = 1.5, 0.7, 0.8
E_ABSENT, E_BRIDGE = 1.2, 0.0              # E_BRIDGE: -log odds of a bridge; its story is uniform
WORLD = 7


def y_slab(s):
    return Y_LAT - PITCH * s


# ------------------------------------------------------------ the terrain
def _vnoise(x, salt, octaves):
    x = np.asarray(x, np.int64)
    out = np.zeros(x.shape)
    for k, (sp, amp) in enumerate(octaves):
        i, f = np.floor_divide(x, sp), (np.mod(x, sp)) / sp
        a = hash_grid(salt * 31 + k, 0, i)
        b = hash_grid(salt * 31 + k, 0, i + 1)
        t = (1 - np.cos(np.pi * f)) / 2
        out += amp * (2 * (a * (1 - t) + b * t) - 1)
    return out


def surface(x):
    """First terrain row of column x."""
    return np.round(176 + _vnoise(x, 1, ((160, 16.0), (48, 7.0), (12, 1.5)))).astype(np.int64)


def far_ridge(x):
    return np.round(150 + _vnoise(x, 2, ((90, 22.0), (25, 6.0)))).astype(np.int64)


def clouds(ys, xs):
    def vn2(y, x, sy, sx, salt):
        iy, fy = np.floor_divide(y, sy), np.mod(y, sy) / sy
        ix, fx = np.floor_divide(x, sx), np.mod(x, sx) / sx
        v = [[hash_grid(salt, (iy + a) * 1000003, ix + b) for b in (0, 1)] for a in (0, 1)]
        ty, tx = (1 - np.cos(np.pi * fy)) / 2, (1 - np.cos(np.pi * fx)) / 2
        return (v[0][0] * (1 - tx) + v[0][1] * tx) * (1 - ty) + (v[1][0] * (1 - tx) + v[1][1] * tx) * ty
    d = 0.65 * vn2(ys, xs, 10, 28, 3) + 0.35 * vn2(ys, xs, 4, 9, 4)
    return d - np.clip((ys - 20) / 50.0, 0, 1) * 0.5


# --------------------------------------------------------- floor states
@lru_cache(None)
def spans():
    out = {}
    for c in itertools.product(WIDTHS, repeat=NROOM):
        out.setdefault(sum(c) + NROOM - 1, []).append(c)
    return {s: sorted(cs) for s, cs in out.items() if len(cs) >= 40}


def room_x(widths, r):
    return sum(widths[:r]) + r


def wall_cols(widths):
    return [room_x(widths, r + 1) - 1 for r in range(NROOM - 1)]


def stair_cols(widths, r, d):
    """(columns of the stair, from the bottom step to the top step) relative to the span."""
    a, w = room_x(widths, r), widths[r]
    return [a + k for k in range(ROOMH)] if d > 0 else [a + w - 1 - k for k in range(ROOMH)]


@lru_cache(None)
def floor_states(span):
    comps = spans()[span]
    st = [(ci, r, d) for ci, c in enumerate(comps) for r in range(NROOM) for d in (1, -1) if c[r] >= 6]
    S = len(st)
    ci = np.array([s[0] for s in st])
    room = np.array([s[1] for s in st])
    dirn = np.array([s[2] for s in st])
    blocked = np.zeros((S, span), bool)
    hole = np.zeros(S, np.int64)
    for k, (c, r, d) in enumerate(st):
        cols = stair_cols(comps[c], r, d)
        blocked[k, wall_cols(comps[c])] = True
        blocked[k, cols] = True
        hole[k] = cols[-1]
    ok = ~blocked[:, hole].T                                       # ok[a, b]: a's hole lands free in b
    E = W_SAME_COMP * (ci[:, None] == ci[None, :]) + W_SAME_ROOM * (room[:, None] == room[None, :]) \
        + W_SAME_DIR * (dirn[:, None] == dirn[None, :])
    Kmat = np.where(ok, np.exp(-E), 0.0)
    return comps, st, hole, room, Kmat


@lru_cache(None)
def floor_chain(seed, span, n, forbid, roof_block):
    """n floor states drawn exactly from prod_i u_i(x_i) prod_i K(x_i, x_i+1):
    forbid[i] = rooms that may not hold the stair on floor i (door rooms);
    roof_block = span columns the top hatch may not use."""
    comps, st, hole, room, Kmat = floor_states(span)
    S = len(st)
    U = np.ones((n, S))
    for i in range(n):
        for r in forbid[i]:
            U[i, room == r] = 0.0
    if roof_block:
        U[n - 1, np.isin(hole, roof_block)] = 0.0
    beta = np.zeros((n, S))
    beta[n - 1] = U[n - 1]
    for i in range(n - 2, -1, -1):
        b = U[i] * (Kmat @ beta[i + 1])
        beta[i] = b / b.max()
    g = rng(seed, 99)
    out = []
    p = beta[0]
    for i in range(n):
        if i > 0:
            p = Kmat[out[-1]] * beta[i]
        assert p.sum() > 0, "infeasible floor chain"
        out.append(int(g.choice(S, p=p / p.sum())))
    return tuple(out)


# --------------------------------------------------------- owner level
MATERIALS = [  # wall, dark, foundation, slab
    ((152, 152, 158), (118, 118, 126), (100, 96, 94), (96, 92, 90)),
    ((212, 186, 136), (182, 156, 108), (150, 124, 88), (140, 116, 84)),
    ((168, 86, 70), (138, 68, 58), (108, 94, 88), (96, 70, 60)),
    ((92, 92, 104), (68, 68, 80), (62, 58, 58), (58, 56, 60)),
    ((228, 225, 216), (198, 196, 190), (150, 148, 145), (160, 156, 150)),
]
ROOFCOL = [(70, 86, 120), (178, 82, 56), (88, 152, 132), (205, 162, 62), (120, 60, 110)]
BATTLE, PITCHED, TURRET, SPIRE, DOME = range(5)


@lru_cache(None)
def site(r):
    """Pure geometry of region r: tower footprint, base story, style."""
    g = rng(WORLD, 1, r)
    sp = sorted(spans())
    span = int(g.choice(sp))
    total = span + 2
    off = int(g.integers(8, RW - total - 8 + 1))
    x_w = r * RW + off
    x_e = x_w + total - 1
    xs = np.arange(x_w - 1, x_e + 2)
    top = int(surface(xs).min())
    b = -(-(Y_LAT - (top - 2)) // PITCH)                         # floor slab at least 2 rows above the highest ground
    return dict(span=span, x_w=x_w, x_e=x_e, b=b, mat=int(g.integers(len(MATERIALS))),
                roofc=int(g.integers(len(ROOFCOL))), roof=int(g.integers(5)), ledge=int(g.integers(3, 6)),
                turret_side=int(g.integers(2)))


@lru_cache(None)
def _clear(r, s):
    a, c = site(r), site(r + 1)
    xa, xb = a["x_e"] + 1, c["x_w"] - 1
    return xb - xa + 1 >= 4 and int(surface(np.arange(xa, xb + 1)).min()) >= y_slab(s) + 6


def bridge_ok(r, s, n0, n1):
    if s < 0:
        return True
    if n0 == 0 or n1 == 0:
        return False
    a, c = site(r), site(r + 1)
    if not (1 <= s - a["b"] <= n0 - 1 and 1 <= s - c["b"] <= n1 - 1):
        return False
    return _clear(r, s)


def _stories(r):
    a, c = site(r), site(r + 1)
    lo, hi = max(a["b"], c["b"]) + 1, min(a["b"], c["b"]) + 14
    return [-1] + list(range(lo, hi + 1))


def owner_states(r0, r1):
    """{r: (n, k)} for r in [r0, r1]: SWEEPS Gibbs sweeps over regions
    (even then odd), each site drawn by Gumbel-max with noise hashed from
    (region, sweep).  The window carries a margin of 2 SWEEPS + 2 regions,
    beyond the light cone, so the result for r0..r1 does not depend on it."""
    M = 2 * SWEEPS + 2
    rs = list(range(r0 - M, r1 + M + 1))
    st = {}
    for r in rs:
        g = rng(WORLD, 2, r)
        st[r] = (int(g.choice(NS)), -1)
    for t in range(SWEEPS):
        for par in (0, 1):
            for r in rs[1:-1]:
                if r % 2 != par:
                    continue
                nl, kl = st[r - 1]
                nr, _ = st[r + 1]
                best, arg = -np.inf, st[r]
                g = rng(WORLD, 3, r, t, par)
                for n in NS:
                    if not bridge_ok(r - 1, kl, nl, n):
                        continue
                    ks = [k for k in _stories(r) if bridge_ok(r, k, n, nr)]
                    lognk = np.log(max(len(ks) - 1, 1))                 # the story is uniform among the valid ones
                    for k in ks:
                        e = (E_ABSENT if n == 0 else 0.0) + (E_BRIDGE + lognk if k >= 0 else 0.0)
                        v = -e - np.log(-np.log(g.random()))
                        if v > best:
                            best, arg = v, (n, k)
                st[r] = arg
    return {r: st[r] for r in range(r0, r1 + 1)}


# --------------------------------------------------------------- sprites
PAL = {"b": (124, 82, 46), "B": (86, 56, 32), "r": (170, 46, 46), "w": (240, 238, 230), "y": (228, 188, 72),
       "g": (72, 142, 62), "G": (44, 98, 44), "o": (244, 142, 40), "k": (44, 40, 40), "s": (132, 132, 138),
       "c": (92, 92, 96), "u": (62, 92, 172), "p": (124, 72, 150), "l": (255, 244, 170), "t": (196, 160, 110)}
SPR = {
    "bed": ["w   ", "rrrr", "B  B"], "bed2": ["   w", "uuuu", "B  B"],
    "shelf": ["BBB", "ruy", "BBB", "gpr", "BBB"], "shelf2": ["BBB", "yrg", "BBB"],
    "table": ["bbbb", " b b"], "chair": ["b ", "bb", "bb"], "desk": ["  u", "bbb", "b b"],
    "barrel": ["bb", "BB", "bb"], "crate": ["bb", "bb"], "crate2": ["bbb", "bBb", "bbb"],
    "hearth": ["sssss", "s   s", "sooos"], "cauldron": ["k k", "kkk"], "plant": [" g ", "gGg", " b "],
    "armor": [" s ", "sss", " s ", "s s"], "chest": ["yby", "bbb"], "loom": ["b b", "pwp", "b b"],
}
HANG = {"lamp": ["c", "l"], "chandelier": [" c ", "yly"]}
WALLART = {"painting": ["yyy", "yuy", "yyy"], "banner": ["rr", "ry", "rr", "r "], "shield": ["sus", " s "],
           "tapestry": ["ppp", "pyp", "ppp", "p p"]}
KINDS = {
    "bedroom": (["bed", "bed2", "chest", "plant"], ["lamp"], ["painting"], (226, 212, 188)),
    "library": (["shelf", "shelf", "desk", "shelf2"], ["lamp"], [], (206, 186, 150)),
    "dining": (["table", "chair", "chair", "barrel"], ["chandelier"], ["banner"], (232, 214, 180)),
    "storage": (["barrel", "crate", "crate2", "barrel", "crate"], [], [], (186, 170, 150)),
    "kitchen": (["hearth", "cauldron", "barrel", "table"], ["lamp"], [], (214, 196, 170)),
    "hall": (["armor", "plant", "armor"], ["chandelier"], ["banner", "shield", "tapestry"], (222, 218, 206)),
    "workshop": (["loom", "crate", "desk"], ["lamp"], ["shield"], (200, 196, 182)),
}
KIND_NAMES = list(KINDS)


# ------------------------------------------------------------ node types
T_REGION, T_TOWER, T_FLOOR, T_SHELL, T_ROOM, T_FOUND, T_ROOF, T_BALC, T_ENTR, T_BRIDGE, T_TREE = range(11)


class Region(NodeType):
    split = 96

    def children(self, rec):
        r, n, k, kl = (int(v) for v in rec[A:A + 4])
        out = []
        s = site(r)
        if n:
            out.append(Tower.make(r, n, k, kl))
        if k >= 0:
            out.append(Bridge.make(r, k))
        # trees in the gaps (not near towers, not under bridges)
        g = rng(WORLD, 5, r)
        bad = []
        for q in (r - 1, r, r + 1):
            sq = site(q)
            bad.append((sq["x_w"] - 6, sq["x_e"] + 6))
        for q, kq in ((r - 1, kl), (r, k)):                         # bridges that exist (their arcades reach the ground)
            if kq >= 0:
                bad.append((site(q)["x_e"], site(q + 1)["x_w"]))
        for i in range(int(g.integers(3, 9))):
            x = r * RW + int(g.integers(0, RW))
            if any(a - 3 <= x <= b + 3 for a, b in bad):
                continue
            out.append(Tree.make(r, i, x))
        return out


class Tower(NodeType):
    split = 32

    @staticmethod
    def make(r, n, k, kl):
        s = site(r)
        x_w, x_e, b = s["x_w"], s["x_e"], s["b"]
        g = rng(WORLD, 4, r)
        doorW = [0] * n
        doorE = [0] * n
        if kl >= 0:
            doorW[kl - b] = 1
        if k >= 0:
            doorE[k - b] = 1
        sides = [d for d, has in (("W", kl >= 0), ("E", k >= 0)) if not has]
        ent = -1
        if sides:
            ent = 0 if g.choice(sides) == "W" else 1
            (doorW if ent == 0 else doorE)[0] = 3
        for i in range(2, n - 1):                                   # balconies
            if g.random() < 0.18:
                side = int(g.integers(2))
                arr = doorW if side == 0 else doorE
                under_bridge = (kl if side == 0 else k) > b + i            # the arcade fills below a deck
                if arr[i] == 0 and arr[i - 1] != 2 and not under_bridge:
                    arr[i] = 2
        forbid = tuple(tuple(([0] if doorW[i] else []) + ([NROOM - 1] if doorE[i] else [])) for i in range(n))
        roof_block = tuple(roof_blocked(s))
        chain = floor_chain(mix(WORLD, 6, r), s["span"], n, forbid, roof_block)
        y_top = y_slab(b + n)
        attrs = [r, n, k, kl, ent] + [doorW[i] * 4 + doorE[i] if i < n else 0 for i in range(15)] \
            + [chain[i] if i < n else 0 for i in range(15)]
        return record(T_TOWER, y_top - 40, x_w - 48, H, x_e + 49, mix(WORLD, 7, r), attrs)

    def children(self, rec):
        r, n, k, kl, ent = (int(v) for v in rec[A:A + 5])
        doors = [int(v) for v in rec[A + 5:A + 20]]
        chain = [int(v) for v in rec[A + 20:A + 35]]
        s = site(r)
        comps, st, hole, _, _ = floor_states(s["span"])
        out = [Found.make(r, ent)]
        for i in range(n):
            below = int(hole[chain[i - 1]]) if i > 0 else -1
            out.append(Floor.make(r, i, n, chain[i], doors[i], below))
        out.append(Roof.make(r, n, int(hole[chain[n - 1]])))
        for i in range(n):
            dw, de = doors[i] // 4, doors[i] % 4
            if dw == 2:
                out.append(Balcony.make(r, i, 0))
            if de == 2:
                out.append(Balcony.make(r, i, 1))
        if ent >= 0:
            out.append(Entrance.make(r, ent))
        return out


def roof_blocked(s):
    """Span columns where the top floor's hatch may not open."""
    span, roof = s["span"], s["roof"]
    cols = set(c for c in range(span) if (c + 1) % 4 in (0, 1))        # merlons
    if roof == PITCHED:
        cols = set(range(0, 3)) | set(range(span - 3, span))
    elif roof == TURRET:
        cols |= set(range(0, 10)) if s["turret_side"] == 0 else set(range(span - 10, span))
    elif roof == SPIRE:
        m = span // 2
        cols |= set(range(m - 5, m + 6))
    elif roof == DOME:
        cols = set(range(0, 4)) | set(range(span - 4, span))
    return sorted(c for c in cols if 0 <= c < span)


class Floor(NodeType):
    split = 8

    @staticmethod
    def make(r, i, n, state, doors, below):
        s = site(r)
        yf = y_slab(s["b"] + i)
        return record(T_FLOOR, yf - PITCH, s["x_w"] - 1, yf, s["x_e"] + 2, mix(WORLD, 8, r, i),
                      [r, i, n, state, doors, below])

    def children(self, rec):
        r, i, n, state, doors, below = (int(v) for v in rec[A:A + 6])
        s = site(r)
        comps, st, hole, _, _ = floor_states(s["span"])
        ci, sr, d = st[state]
        widths = comps[ci]
        yf = y_slab(s["b"] + i)
        out = [record(T_SHELL, yf - PITCH, s["x_w"] - 1, yf, s["x_e"] + 2, mix(WORLD, 9, r, i),
                      [r, i, n, state, doors, below])]
        g = rng(WORLD, 10, r, i)
        for q in range(NROOM):
            a = s["x_w"] + 1 + room_x(widths, q)
            kind = int(g.integers(len(KIND_NAMES)))
            stair = d if q == sr else 0
            hb = s["x_w"] + 1 + below if below >= 0 and room_x(widths, q) <= below < room_x(widths, q) + widths[q] else -1
            ext = (1 if q == 0 and doors // 4 else 0) + (2 if q == NROOM - 1 and doors % 4 else 0)
            out.append(record(T_ROOM, yf - ROOMH, a, yf, a + widths[q], mix(WORLD, 11, r, i, q),
                              [r, i, q, kind, stair, hb, ext]))
        return out


class Shell(NodeType):
    z = 4

    def raster(self, rec):
        r, i, n, state, doors, below = (int(v) for v in rec[A:A + 6])
        s = site(r)
        comps, st, hole, _, _ = floor_states(s["span"])
        ci, sr, d = st[state]
        widths = comps[ci]
        wall, dark, found, slab = MATERIALS[s["mat"]]
        cv = Canvas(rec)
        yf = y_slab(s["b"] + i)
        yc = yf - PITCH                                              # ceiling row
        x_w, x_e = s["x_w"], s["x_e"]
        top = i == n - 1
        dw, de = doors // 4, doors % 4
        g = rng(WORLD, 12, r, i)
        for x, dside, t in ((x_w, dw, EXTW), (x_e, de, EXTE)):
            for dy in range(ROOMH):
                y = yf - 1 - dy
                if dside and dy <= 2:
                    cv.put(y, x, EXTDOOR, (96, 64, 40) if dy == 2 else (70, 48, 32))
                elif not dside and dy in (2, 3) and (i * 7 + x) % 3 != 0:
                    cv.put(y, x, WINDOW, (150, 200, 232) if dy == 3 else (126, 178, 214), jitter=0.02)
                else:
                    cv.put(y, x, t, dark, salt=3)
            cv.put(yc, x, t, dark, salt=3)
        for c in wall_cols(widths):
            x = x_w + 1 + c
            for dy in range(ROOMH):
                if dy <= 2:
                    cv.put(yf - 1 - dy, x, DOOR, (150, 120, 90) if dy == 2 else (176, 150, 120), jitter=0.03)
                else:
                    cv.put(yf - 1 - dy, x, WALL, wall, salt=2)
        hx = x_w + 1 + int(hole[state])
        for x in range(x_w + 1, x_e):
            if x == hx:
                cv.put(yc, x, HOLE, (70, 52, 36))
            else:
                cv.put(yc, x, ROOF if top else SLAB, ROOFCOL[s["roofc"]] if top else slab, salt=4)
        if not top and (i + 1) % s["ledge"] == 0:
            cv.put(yc, x_w - 1, LEDGE, dark)
            cv.put(yc, x_e + 1, LEDGE, dark)
        del g
        return cv.T, cv.C


class Room(NodeType):
    z = 5

    def raster(self, rec):
        r, i, q, kind, stair, hb, ext = (int(v) for v in rec[A:A + 7])
        cv = Canvas(rec)
        y0, x0, y1, x1 = (int(rec[k]) for k in (Y0, X0, Y1, X1))
        w = x1 - x0
        name = KIND_NAMES[kind]
        floor_items, hang, art, bg = KINDS[name]
        if stair:
            bg = (220, 206, 182)
        cv.rect(y0, x0, y1, x1, ROOM, bg, jitter=0.03, salt=5)
        cv.rect(y1 - 1, x0, y1, x1, ROOM, tuple(int(c * 0.88) for c in bg), jitter=0.03, salt=6)
        g = rng(int(rec[SEED]))
        if stair:
            cols = [x0 + k for k in range(ROOMH)] if stair > 0 else [x1 - 1 - k for k in range(ROOMH)]
            for k, x in enumerate(cols):
                for dy in range(k):
                    cv.put(y1 - 1 - dy, x, FILL, (112, 74, 42), jitter=0.05)
                cv.put(y1 - 1 - k, x, STEP, (164, 112, 60), jitter=0.05)
            return cv.T, cv.C
        free = np.ones(w, bool)
        if hb >= 0:
            free[max(hb - x0 - 1, 0):hb - x0 + 2] = False
        if ext & 1:
            free[:2] = False
        if ext & 2:
            free[-2:] = False
        items = list(floor_items)
        g.shuffle(items)
        for it in items[:int(g.integers(1, 4))]:
            spr = SPR[it]
            sw = len(spr[0])
            starts = [a for a in range(w - sw + 1) if free[a:a + sw].all()]
            if not starts:
                continue
            a = int(g.choice(starts))
            cv.sprite(spr, y1 - 1, x0 + a, FURN, PAL, jitter=0.05)
            free[a:a + sw] = False
        for it in hang:
            spr = HANG[it]
            a = int(g.integers(0, w - len(spr[0]) + 1))
            cv.sprite(spr, y0 + len(spr) - 1, x0 + a, FURN, PAL, jitter=0.03)
        for it in art:
            spr = WALLART[it]
            sw = len(spr[0])
            if w >= sw + 2 and g.random() < 0.8:
                a = int(g.integers(1, w - sw))
                cols = slice(a, a + sw)
                if free[cols].all():                                # behind nothing tall
                    cv.sprite(spr, y0 + len(spr), x0 + a, FURN, PAL, jitter=0.05)
        return cv.T, cv.C


class Found(NodeType):
    z = 3

    @staticmethod
    def make(r, ent):
        s = site(r)
        yf = y_slab(s["b"])
        xs = np.arange(s["x_w"] - 1, s["x_e"] + 2)
        bot = int(surface(xs).max()) + 3
        return record(T_FOUND, yf, s["x_w"] - 1, bot, s["x_e"] + 2, mix(WORLD, 13, r), [r, ent])

    def raster(self, rec):
        r = int(rec[A])
        s = site(r)
        wall, dark, found, slab = MATERIALS[s["mat"]]
        cv = Canvas(rec)
        yf = y_slab(s["b"])
        cv.rect(yf, s["x_w"] + 1, yf + 1, s["x_e"], SLAB, slab, salt=4)
        cv.put(yf, s["x_w"], EXTW, dark)
        cv.put(yf, s["x_e"], EXTE, dark)
        for x in range(s["x_w"] - 1, s["x_e"] + 2):
            y_end = int(surface(x)) + 2
            for y in range(yf + 1, y_end + 1):
                c = found if (y + (x // 3)) % 2 else tuple(int(v * 0.9) for v in found)
                cv.put(y, x, FOUND, c, salt=8)
        return cv.T, cv.C


class Roof(NodeType):
    z = 4

    @staticmethod
    def make(r, n, hole):
        s = site(r)
        yr = y_slab(s["b"] + n)
        return record(T_ROOF, yr - 34, s["x_w"] - 2, yr, s["x_e"] + 3, mix(WORLD, 14, r), [r, n, hole])

    def raster(self, rec):
        r, n, hole = (int(v) for v in rec[A:A + 3])
        s = site(r)
        wall, dark, found, slab = MATERIALS[s["mat"]]
        rc = ROOFCOL[s["roofc"]]
        cv = Canvas(rec)
        yr = y_slab(s["b"] + n)
        x_w, x_e, span = s["x_w"], s["x_e"], s["span"]
        g = rng(int(rec[SEED]))

        def battlements():
            for x in range(x_w, x_e + 1):
                c = x - x_w - 1
                if c < 0 or c >= span or (c + 1) % 4 in (0, 1):
                    cv.put(yr - 1, x, ROOFDEC, wall, salt=9)
                    cv.put(yr - 2, x, ROOFDEC, wall, salt=9)

        def flag(x, y):
            for k in range(1, 7):
                cv.put(y - k, x, ROOFDEC, (60, 50, 40), jitter=0)
            col = PAL["rgupy"[int(g.integers(5))]]
            cv.rect(y - 6, x + 1, y - 4, x + 4, ROOFDEC, col, jitter=0.05)

        def cone(xa, xb, ybase, h, col):
            m = (xa + xb) / 2
            for k in range(h):
                half = (xb - xa) / 2 * (1 - k / h)
                for x in range(int(np.ceil(m - half)), int(np.floor(m + half)) + 1):
                    shade = col if (x + k) % 3 else tuple(int(v * 0.85) for v in col)
                    cv.put(ybase - k, x, ROOFDEC, shade, salt=10)

        if s["roof"] in (BATTLE, TURRET, SPIRE):
            battlements()
        if s["roof"] == BATTLE:
            flag(x_w + 1 + int(g.integers(0, span // 4)) * 4 - 1 if span > 8 else x_w, yr - 2)
        elif s["roof"] == PITCHED:
            cone(x_w - 1, x_e + 1, yr - 1, min((x_e - x_w + 3) // 2, 18), rc)
            cx = x_w + 3 + int(g.integers(0, max(span - 8, 1)))
            cv.rect(yr - 10, cx, yr - 1, cx + 2, ROOFDEC, dark)
        elif s["roof"] == TURRET:
            xa = x_w + 1 if s["turret_side"] == 0 else x_e - 9
            cv.rect(yr - 8, xa, yr, xa + 9, ROOFDEC, wall, salt=11)
            cv.rect(yr - 6, xa + 3, yr - 3, xa + 6, ROOFDEC, (40, 40, 60))
            cone(xa - 1, xa + 9, yr - 9, 9, rc)
            flag(xa + 4, yr - 17)
        elif s["roof"] == SPIRE:
            m = x_w + 1 + span // 2
            cv.rect(yr - 5, m - 4, yr, m + 5, ROOFDEC, wall, salt=11)
            cv.rect(yr - 4, m - 1, yr - 1, m + 2, ROOFDEC, (40, 40, 60))
            cone(m - 4, m + 4, yr - 6, 22, rc)
            cv.put(yr - 28, m, ROOFDEC, PAL["y"], jitter=0)
            cv.put(yr - 29, m, ROOFDEC, PAL["y"], jitter=0)
        elif s["roof"] == DOME:
            m = (x_w + x_e) / 2
            rad = (x_e - x_w) / 2 - 1
            for x in range(x_w - 1, x_e + 2):
                hgt = rad * np.sqrt(max(0.0, 1 - ((x - m) / (rad + 1.5)) ** 2)) * 0.75
                for k in range(int(round(hgt)) + 1):
                    cv.put(yr - 1 - k, x, ROOFDEC, rc if (x + k) % 4 else tuple(int(v * 0.85) for v in rc), salt=12)
            top = yr - 2 - int(round(rad * 0.75))
            cv.rect(top - 4, int(m) - 1, top + 1, int(m) + 2, ROOFDEC, wall)
            cv.put(top - 5, int(m), ROOFDEC, PAL["y"], jitter=0)
        return cv.T, cv.C


class Balcony(NodeType):
    z = 6

    @staticmethod
    def make(r, i, side):
        s = site(r)
        yf = y_slab(s["b"] + i)
        x = s["x_w"] - 5 if side == 0 else s["x_e"] + 1
        return record(T_BALC, yf - 3, x, yf + 3, x + 5, mix(WORLD, 15, r, i, side), [r, i, side])

    def raster(self, rec):
        r, i, side = (int(v) for v in rec[A:A + 3])
        s = site(r)
        cv = Canvas(rec)
        yf = y_slab(s["b"] + i)
        sgn = -1 if side == 0 else 1
        ext = s["x_w"] if side == 0 else s["x_e"]
        for k in range(1, 5):
            x = ext + sgn * k
            cv.put(yf, x, BALC, PAL["b"])
            cv.put(yf - 2, x, BALC, PAL["B"])
        cv.put(yf - 1, ext + sgn * 4, BALC, PAL["B"])
        cv.put(yf - 1, ext + sgn * 2, BALC, PAL["B"])
        cv.put(yf + 1, ext + sgn, BALC, PAL["B"])
        cv.put(yf + 1, ext + sgn * 2, BALC, PAL["B"])
        cv.put(yf + 2, ext + sgn, BALC, PAL["B"])
        if rng(int(rec[SEED])).random() < 0.6:
            cv.put(yf - 3, ext + sgn * 3, BALC, PAL["r"])
            cv.put(yf - 3, ext + sgn * 2, BALC, PAL["g"])
        return cv.T, cv.C


class Entrance(NodeType):
    z = 3

    @staticmethod
    def make(r, side):
        s = site(r)
        yf = y_slab(s["b"])
        x = s["x_w"] - 44 if side == 0 else s["x_e"] + 1
        return record(T_ENTR, yf - 1, x, H, x + 44, mix(WORLD, 16, r), [r, side])

    def raster(self, rec):
        r, side = (int(v) for v in rec[A:A + 2])
        s = site(r)
        wall, dark, found, slab = MATERIALS[s["mat"]]
        cv = Canvas(rec)
        yf = y_slab(s["b"])
        sgn = -1 if side == 0 else 1
        ext = s["x_w"] if side == 0 else s["x_e"]
        x, y = ext + 2 * sgn, yf
        k = 0
        while y < int(surface(x)) and k < 40:
            cv.put(y, x, OUTSTEP, (150, 140, 128))
            for yy in range(y + 1, int(surface(x)) + 1):
                cv.put(yy, x, FOUND, found, salt=8)
            x += sgn
            y += 1
            k += 1
        cv.put(yf, ext + sgn, OUTSTEP, (150, 140, 128))
        for yy in range(yf + 1, int(surface(ext + sgn)) + 1):
            cv.put(yy, ext + sgn, FOUND, found, salt=8)
        return cv.T, cv.C


class Bridge(NodeType):
    z = 2

    @staticmethod
    def make(r, k):
        a, c = site(r), site(r + 1)
        xa, xb = a["x_e"] + 1, c["x_w"]
        yd = y_slab(k)
        bot = int(surface(np.arange(xa, xb)).max()) + 2
        return record(T_BRIDGE, yd - 3, xa, bot, xb, mix(WORLD, 17, r), [r, k])

    def raster(self, rec):
        r, k = (int(v) for v in rec[A:A + 2])
        a = site(r)
        wall, dark, found, slab = MATERIALS[a["mat"]]
        cv = Canvas(rec)
        yd = y_slab(k)
        xa, xb = int(rec[X0]), int(rec[X1])
        L = xb - xa
        for x in range(xa, xb):
            cv.put(yd, x, BRIDGE, PAL["b"] if L <= 16 else dark)
            if (x - xa) % 3 == 1:
                cv.put(yd - 1, x, BRIDGE, PAL["B"])
            cv.put(yd - 2, x, BRIDGE, PAL["B"])
        if L <= 16:                                                 # timber truss
            for x in range(xa, xb):
                d = min(x - xa, xb - 1 - x)
                if d < 3:
                    cv.put(yd + 1 + d, x, BRIDGE, PAL["B"])
                cv.put(yd + 1, x, BRIDGE, PAL["b"])
            return cv.T, cv.C
        nb = max(1, int(round(L / 13)))                             # stone arcade
        piers = np.linspace(xa, xb - 2, nb + 1).round().astype(int)
        for x in range(xa, xb):
            ground = int(surface(x))
            j = np.searchsorted(piers, x, side="right") - 1
            j = min(max(j, 0), nb - 1)
            p0, p1 = piers[j] + 2, piers[j + 1]
            mid, half = (p0 + p1 - 1) / 2, max((p1 - p0) / 2, 1)
            if p0 <= x < p1:
                u = (x - mid) / (half + 0.5)
                arch = yd + 2 + int(round(half * 0.9 * (1 - np.sqrt(max(0.0, 1 - u * u)))))
            else:
                arch = ground + 2
            for y in range(yd + 1, ground + 2):
                if y <= arch:
                    c = wall if (y + x // 2) % 2 else dark
                    cv.put(y, x, BRIDGE, c, salt=13)
        return cv.T, cv.C


class Tree(NodeType):
    z = 1

    @staticmethod
    def make(r, i, x):
        y = int(surface(x))
        return record(T_TREE, y - 16, x - 5, y + 1, x + 6, mix(WORLD, 18, r, i), [r, i, x])

    def raster(self, rec):
        r, i, x = (int(v) for v in rec[A:A + 3])
        cv = Canvas(rec)
        g = rng(int(rec[SEED]))
        y = int(surface(x))
        th = int(g.integers(3, 6))
        for k in range(1, th + 1):
            cv.put(y - k, x, TREE, PAL["B"])
        if g.random() < 0.5:                                        # round
            rad = g.uniform(2.2, 4.2)
            cy = y - th - rad + 1
            ys, xs = np.mgrid[int(cy - rad) - 1:int(cy + rad) + 2, x - 5:x + 6]
            d = ((ys - cy) / rad) ** 2 + ((xs - x) / (rad * 1.15)) ** 2
            m = d + 0.25 * hash_grid(int(rec[SEED]), ys, xs) < 1.0
            cv.put(ys[m], xs[m], TREE, (70, 140, 60), jitter=0.15, salt=14)
        else:                                                       # pine
            h = int(g.integers(6, 11))
            for k in range(h):
                half = int(round((h - k) * 0.45))
                cv.put(np.full(2 * half + 1, y - th + 1 - k), np.arange(x - half, x + half + 1), TREE,
                       (44, 104, 64), jitter=0.12, salt=15)
        return cv.T, cv.C


class Towers(Grammar):
    H = H

    def __init__(self):
        super().__init__()
        self.types = [Region(), Tower(), Floor(), Shell(), Room(), Found(), Roof(), Balcony(), Entrance(),
                      Bridge(), Tree()]

    def roots(self, x0, x1):
        r0, r1 = x0 // RW - 2, (x1 - 1) // RW + 1
        st = owner_states(r0, r1)
        return [record(T_REGION, 0, r * RW - 48, H, (r + 2) * RW, mix(WORLD, 0, r),
                       [r, st[r][0], st[r][1], st[r - 1][1] if r - 1 in st else -1])
                for r in range(r0 + 1, r1 + 1)]

    def paint_field(self, x0, x1, T, C):
        ys, xs = np.mgrid[0:H, x0:x1]
        t = ys / H
        sky = np.stack([92 + 110 * t, 146 + 78 * t, 214 + 26 * t], -1)
        C[:] = sky.astype(np.uint8)
        T[:] = SKY
        cl = clouds(ys, xs) > 0.56
        C[cl] = (246, 248, 252)
        T[cl] = CLOUD
        fr = ys >= far_ridge(xs[0])[None, :]
        C[fr] = (150, 168, 196)
        T[fr] = FAR
        sf = surface(xs[0])[None, :]
        depth = ys - sf
        j = hash_grid(21, ys, xs)
        col = np.where((depth == 0)[..., None], np.array([92, 160, 70]),
                       np.where((depth < 5)[..., None], np.array([134, 98, 62]), np.array([112, 106, 100])))
        col = col * (0.92 + 0.14 * j)[..., None]
        g = depth >= 0
        C[g] = col[g].astype(np.uint8)
        T[g] = TERRAIN
