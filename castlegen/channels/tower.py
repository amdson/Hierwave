"""Towers of interlocking rooms on exemplar windows (a stress test of the
channel layer on counted structure).

Requirements: 10..15 floors, exactly five rooms per floor, exactly one
room per floor a staircase to the next (the top floor has none), floor
pitch 6 and room pitches 6 / 7 / 9 against a block of 8, so floors and
rooms do not align with blocks.

Method: the canonical pipeline of chan_combined3, nothing tower-specific
in the sampler.
  - A regenerable exemplar sheet of towers, drawn by a procedural
    generator, in which every tile carries hidden labels: floor index L,
    room index r, stair state (pre / stair / post / top), row in floor dy,
    and the stair cell's position.  The tile code is (type, labels).
  - Pair rules learned from the sheet: a 4-neighbour pair of codes never
    seen in the sheet is forbidden.  With these labels the local language
    is exact (Giammarresi-Restivo: visible tiles + hidden codes + domino
    checks): L starts at 0 on the ground and steps by one across a slab,
    the roof exists only above L in 9..14, r steps by one across an
    interior wall and the exterior walls stand beside r = 0 and r = 4, the
    stair state can only go pre -> stair -> post, the stair is a rigid
    labelled object and the hole in the slab only sits above its top step.
    So any tiling whose every pair was seen is a set of valid towers.
  - Level 8: one exemplar window per block (u8), sampled by the generic
    kernel.  Edge compatibility = the tile pair rule applied to the 8
    pairs across the block edge (8 pair factors per direction on the
    views E_i / W_i / S_j / N_j, one shared table).  A tower code against
    sky that was never seen is a soft cut (cost `cut`), which is what lets
    a tower nucleate under single-site moves; every other unseen pair is
    hard.  Soft coherence lam8 (shift preference), a bonus per tower
    window, and a hard ground rule against a designed flat ground channel.
  - Refinement: tiles copied from the windows."""
import itertools

import numpy as np

from .core import Channel, Factor

INF = np.inf
B = 8
H_ROOM = 5                                  # interior rows per floor; pitch 6 with the slab
ROOM_W = (5, 6, 8)                          # interior widths; pitch 6, 7, 9 with the wall
NROOM = 5
FLOORS = (10, 15)

# tile types
SKY, GROUND, AIR, WALL, DOOR, STEP, FILL, SLAB, HOLE, ROOF, EXTW, EXTE = range(12)
TOWER_TYPES = (AIR, WALL, DOOR, STEP, FILL, SLAB, HOLE, ROOF, EXTW, EXTE)
PRE, STAIR, POST, TOP = range(4)
COLOURS = {SKY: (169, 212, 240), GROUND: (110, 84, 60), AIR: (236, 226, 204), WALL: (120, 110, 100),
           DOOR: (196, 170, 130), STEP: (150, 90, 50), FILL: (120, 72, 40), SLAB: (90, 82, 76),
           HOLE: (236, 226, 204), ROOF: (150, 50, 40), EXTW: (80, 74, 70), EXTE: (80, 74, 70)}


# --------------------------------------------------------------- exemplar
def _compositions(total):
    return [c for c in itertools.product(ROOM_W, repeat=NROOM) if sum(c) == total]


def gen_tower(rng, n):
    """One tower as a list of floors: (widths, stair room or -1).  The
    interior span is the same on every floor; each floor picks its own
    composition (interlocking walls); the cell above the hole is room air
    or the next floor's first step (resampled until it is)."""
    totals = [t for t in range(NROOM * min(ROOM_W), NROOM * max(ROOM_W) + 1) if len(_compositions(t)) >= 8]
    total = int(rng.choice(totals))
    comps = _compositions(total)
    floors = []
    for k in range(n):
        stair = int(rng.integers(NROOM)) if k < n - 1 else -1
        floors.append([comps[rng.integers(len(comps))], stair])
    for _ in range(1000):                    # hole column must not open under a wall of the floor above
        bad = False
        for k in range(n - 1):
            hx = _hole_x(floors[k])
            if hx in _wall_xs(floors[k + 1][0]):
                floors[k + 1][0] = comps[rng.integers(len(comps))]
                bad = True
        if not bad:
            break
    return [(tuple(w), s) for w, s in floors]


def _wall_xs(widths):
    """x of interior walls, relative to the first interior column."""
    xs, x = [], 0
    for w in widths[:-1]:
        x += w
        xs.append(x)
        x += 1
    return xs


def _room_x0(widths, r):
    return sum(widths[:r]) + r


def _hole_x(floor):
    widths, s = floor
    return _room_x0(widths, s) + H_ROOM - 1


def paint_tower(floors, codes_out, x0, gy):
    """Write (type, L, r, k, dy, sdx) tuples into codes_out (dict (y, x) ->
    tuple), tower's west exterior wall at x0, first ground row gy."""
    n = len(floors)
    span = sum(floors[0][0]) + NROOM - 1
    xe = x0 + span + 1
    for k, (widths, s) in enumerate(floors):
        ybot = gy - 1 - k * (H_ROOM + 1)
        top = k == n - 1
        walls = _wall_xs(widths)
        state = []
        seen = 0
        for r in range(NROOM):
            if top:
                state.append(TOP)
            elif r == s:
                state.append(STAIR); seen = 1
            else:
                state.append(POST if seen else PRE)
        for dy in range(H_ROOM + 1):
            y = ybot - dy
            if dy == H_ROOM:                                     # the slab (roof on the top floor)
                for x in range(x0, xe + 1):
                    if x == x0 or x == xe:
                        codes_out[y, x] = (EXTW if x == x0 else EXTE, k, 0, int(top), dy, 0)
                    elif top:
                        codes_out[y, x] = (ROOF, k, 0, 0, 0, 0)
                    elif x - x0 - 1 == _hole_x((widths, s)):
                        codes_out[y, x] = (HOLE, k, 0, 0, 0, 0)
                    else:
                        codes_out[y, x] = (SLAB, k, 0, 0, 0, 0)
                continue
            codes_out[y, x0] = (EXTW, k, 0, 0, dy, 0)
            codes_out[y, xe] = (EXTE, k, 0, 0, dy, 0)
            r = 0
            for i in range(span):
                x = x0 + 1 + i
                if i in walls:
                    sv = 3 if top else int(state[r] != PRE)
                    codes_out[y, x] = (DOOR if dy < 2 else WALL, k, r, sv, dy, 0)
                    r += 1
                    continue
                sdx = i - _room_x0(widths, r)
                if state[r] == STAIR and sdx < H_ROOM and dy <= sdx:
                    codes_out[y, x] = (STEP if dy == sdx else FILL, k, r, STAIR, dy, sdx)
                else:
                    codes_out[y, x] = (AIR, k, r, state[r], dy, 0)
    return xe


def make_sheet(ntowers=6, seed=0, gap=10, ground=10, sky=8):
    """(codes int (my, mx), code table list of tuples, towers [(x0, floors)]).
    Floor counts cycle through 10..15."""
    rng = np.random.default_rng(seed)
    towers = [gen_tower(rng, FLOORS[0] + i % (FLOORS[1] - FLOORS[0] + 1)) for i in range(ntowers)]
    hmax = max(len(t) for t in towers) * (H_ROOM + 1)
    my = sky + hmax + ground
    gy = my - ground
    cells = {}
    x = gap
    placed = []
    for t in towers:
        xe = paint_tower(t, cells, x, gy)
        placed.append((x, t))
        x = xe + 1 + gap
    mx = x
    table = [(SKY, 0, 0, 0, 0, 0), (GROUND, 0, 0, 0, 0, 0)]
    index = {c: i for i, c in enumerate(table)}
    codes = np.zeros((my, mx), np.int64)
    codes[gy:] = 1
    for (y, xx), c in cells.items():
        if c not in index:
            index[c] = len(table)
            table.append(c)
        codes[y, xx] = index[c]
    return codes, table, placed


def pair_rules(codes, table, cut):
    """(Hrule, Vrule): Hrule[a, b] for a left of b, Vrule[a, b] for a above
    b; 0 if seen in the sheet, `cut` for an unseen tower code against sky,
    inf otherwise."""
    n = len(table)
    types = np.array([c[0] for c in table])
    tower = np.isin(types, TOWER_TYPES)
    rules = []
    for a, b in ((codes[:, :-1], codes[:, 1:]), (codes[:-1, :], codes[1:, :])):
        R = np.full((n, n), INF)
        skypair = (tower[:, None] & (types[None, :] == SKY)) | ((types[:, None] == SKY) & tower[None, :])
        R[skypair] = cut
        R[a.ravel(), b.ravel()] = 0.0
        rules.append(R)
    return rules


# ---------------------------------------------------------------- windows
class Windows:
    """Every distinct B x B window of the sheet (by code content).  Domain
    index 0 = pure sky.  Views: E0..E7, W0..W7 (codes of the east / west
    column), N0..N7, S0..S7, ey / ex (sheet position, for coherence),
    tower (any tower code), gc (ground rows at the bottom)."""

    def __init__(self, codes, table):
        my, mx = codes.shape
        self.codes, self.table = codes, table
        seen, wins, pos = {}, [], []
        sky = np.zeros((B, B), np.int64)
        seen[sky.tobytes()] = 0
        wins.append(sky); pos.append((-1, -1))
        for ey in range(my - B + 1):
            for ex in range(mx - B + 1):
                w = codes[ey:ey + B, ex:ex + B]
                key = w.tobytes()
                if key in seen:
                    continue
                seen[key] = len(wins)
                wins.append(w.copy()); pos.append((ey, ex))
        self.wins = np.array(wins)                                      # (D, B, B)
        self.pos = np.array(pos)
        self.D = len(wins)
        self.my, self.mx = my, mx
        types = np.array([c[0] for c in table])
        self.tower = np.isin(types[self.wins], TOWER_TYPES).any((1, 2)).astype(np.int64)
        g = (types[self.wins] == GROUND)
        full = g.all(2)                                                 # (D, B) rows that are all ground
        gc = np.zeros(self.D, np.int64)
        for d in range(self.D):
            k = 0
            while k < B and full[d, B - 1 - k]:
                k += 1
            gc[d] = k if g[d].sum() == k * B else B + 1                 # ground not a bottom band: never matches
        self.gc = gc

    def channel(self, name="u8"):
        n = len(self.table)
        c = Channel(name, B, self.D)
        for i in range(B):
            c.add_view(f"E{i}", self.wins[:, i, B - 1], n)
            c.add_view(f"W{i}", self.wins[:, i, 0], n)
            c.add_view(f"N{i}", self.wins[:, 0, i], n)
            c.add_view(f"S{i}", self.wins[:, B - 1, i], n)
        c.add_view("ey", np.where(self.pos[:, 0] < 0, self.my, self.pos[:, 0]), self.my + 1)
        c.add_view("ex", np.where(self.pos[:, 1] < 0, self.mx, self.pos[:, 1]), self.mx + 1)
        c.add_view("tower", self.tower, 2)
        c.add_view("gc", self.gc, B + 2)
        return c

    def factors(self, rules, u8="u8", gnd="gnd", lam8=1.0, bonus=1.0):
        Hr, Vr = rules
        f = []
        for i in range(B):
            f.append(Factor.pair((u8, f"E{i}"), (u8, f"W{i}"), (0, 1), Hr, name=f"seam_h{i}"))
            f.append(Factor.pair((u8, f"S{i}"), (u8, f"N{i}"), (1, 0), Vr, pad_b=GROUND, pad_a=SKY, name=f"seam_v{i}"))

        def shift_table(n, shift):
            t = np.full((n + 1, n + 1), lam8)
            a = np.arange(n)
            ok = a + shift
            m = ok < n
            t[a[m], ok[m]] = 0.0
            t[n, n] = 0.0
            return t
        if lam8 > 0:
            f += [Factor.pair((u8, "ey"), (u8, "ey"), (0, 1), shift_table(self.my, 0), name="coh_h_ey"),
                  Factor.pair((u8, "ex"), (u8, "ex"), (0, 1), shift_table(self.mx, B), name="coh_h_ex"),
                  Factor.pair((u8, "ey"), (u8, "ey"), (1, 0), shift_table(self.my, B), name="coh_v_ey"),
                  Factor.pair((u8, "ex"), (u8, "ex"), (1, 0), shift_table(self.mx, 0), name="coh_v_ex")]
        eq = np.full((B + 2, B + 1), INF)
        eq[np.arange(B + 1), np.arange(B + 1)] = 0.0
        f.append(Factor.pair((u8, "gc"), (gnd, "rows"), (0, 0), eq, name="ground"))
        f.append(Factor.unary((u8, "tower"), np.array([0.0, -bonus]), name="bonus"))
        return f

    def certificate(self, rules, u8="u8", d8="d8", Dmax=2048, delta=0.05):
        """Attachment at level 8 (the roots' certificate, plain mode, mass 1
        on tower windows): every tower window reaches a ground-floor window
        (tower codes and ground in one window) through joined neighbours;
        joined = every pair across the shared edge seen in the sheet and at
        least one of them tower-tower.  No floating fragments."""
        types = np.array([c[0] for c in self.table])
        tow = np.isin(types, TOWER_TYPES)
        J = np.zeros((4, self.D, self.D), np.bool_)
        Hr, Vr = rules
        for d, (sa, sb, R) in enumerate(((self.wins[:, 0, :], self.wins[:, B - 1, :], Vr),      # N: my top row under its bottom row
                                         (self.wins[:, :, B - 1], self.wins[:, :, 0], Hr),     # E
                                         (self.wins[:, B - 1, :], self.wins[:, 0, :], Vr),     # S
                                         (self.wins[:, :, 0], self.wins[:, :, B - 1], Hr))):   # W
            J[d] = _joins(sa, sb, R, tow, d in (0, 3))
        self.mass = self.tower.copy()
        self.root = ((self.tower == 1) & (self.gc >= 1) & (self.gc <= B)).astype(np.int64)
        return J, Dmax, delta

    def blank(self, u8, gnd):
        """Sky above, ground below: for each block the window whose ground
        band matches (the purest one: no tower code)."""
        by_gc = {}
        for d in range(self.D):
            if not self.tower[d] and self.gc[d] <= B and self.gc[d] not in by_gc:
                by_gc[self.gc[d]] = d
        u8.grid[:] = np.vectorize(lambda g: by_gc[g])(gnd.grid)

    def copy_tower(self, u8, x0_sheet, x0_world, gy_sheet, gy_world, span):
        """Place the sheet's columns [x0_sheet, x0_sheet + span) at world
        column x0_world, ground rows aligned, by writing the windows that
        tile them (block aligned in the world, so arbitrary in the sheet)."""
        index = {w.tobytes(): d for d, w in enumerate(self.wins)}
        rows, cols = u8.grid.shape
        dy = gy_sheet - gy_world
        for j in range(cols):
            wx0 = j * B
            if wx0 + B <= x0_world or wx0 >= x0_world + span:
                continue
            sx = wx0 - x0_world + x0_sheet
            for i in range(rows):
                sy = i * B + dy
                if 0 <= sy and sy + B <= self.my and 0 <= sx and sx + B <= self.mx:
                    d = index.get(self.codes[sy:sy + B, sx:sx + B].tobytes())
                    if d is not None:
                        u8.grid[i, j] = d

    def tiles(self, u8):
        rows, cols = u8.grid.shape
        return self.wins[u8.grid].transpose(0, 2, 1, 3).reshape(rows * B, cols * B)

    def recombined_seams(self, u8):
        g = u8.grid
        n = 0
        for dy, dx in ((0, 1), (1, 0)):
            a, b = g[:g.shape[0] - dy, :g.shape[1] - dx], g[dy:, dx:]
            both = (self.tower[a] == 1) & (self.tower[b] == 1)
            pa, pb = self.pos[a], self.pos[b]
            shift = (pb[..., 0] == pa[..., 0] + dy * B) & (pb[..., 1] == pa[..., 1] + dx * B)
            n += int((both & ~shift).sum())
        return n


from numba import njit


@njit(cache=True)
def _joins(sa, sb, R, tow, flip):
    """J[a, b]: window b on the given side of a; sa = a's edge codes, sb =
    b's facing edge codes; flip: b comes first in the rule (b above / left of a)."""
    D, n = sa.shape
    J = np.zeros((D, D), np.bool_)
    for a in range(D):
        for b in range(D):
            ok, tt = True, False
            for i in range(n):
                x, y = sa[a, i], sb[b, i]
                e = R[y, x] if flip else R[x, y]
                if e != 0.0:
                    ok = False
                    break
                if tow[x] and tow[y]:
                    tt = True
            J[a, b] = ok and tt
    return J


def gnd_channel(H, W, gy, name="gnd"):
    c = Channel(name, B, B + 1).add_view("rows", np.arange(B + 1))
    rows = np.arange(H // B)
    c.grid = np.repeat(np.clip(rows * B + B - gy, 0, B)[:, None], W // B, 1).astype(np.int32)
    return c


# ---------------------------------------------------------------- metrics
def pair_violations(T, rules):
    """(hard, cut): unseen pairs in the tile grid T (codes)."""
    hard = cut = 0
    for R, a, b in ((rules[0], T[:, :-1], T[:, 1:]), (rules[1], T[:-1, :], T[1:, :])):
        e = R[a, b]
        hard += int(np.isinf(e).sum())
        cut += int(((e > 0) & np.isfinite(e)).sum())
    return hard, cut


def towers_geometry(T, table):
    """Independent check from tile types only (no labels).  Per connected
    component of tower tiles: complete (touches no sky through a room,
    i.e. bounded by exterior walls and a roof), floors, rooms per floor,
    stairs per floor, holes above top steps.  Returns a list of dicts."""
    from scipy import ndimage
    types = np.array([c[0] for c in table])[T]
    tower = np.isin(types, TOWER_TYPES)
    lab, n = ndimage.label(tower)
    out = []
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab == k)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        sub = np.where(lab[y0:y1 + 1, x0:x1 + 1] == k, types[y0:y1 + 1, x0:x1 + 1], SKY)
        rec = dict(bbox=(int(y0), int(x0), int(y1), int(x1)))
        # complete: a rectangle with exterior walls on both sides, roof on top, ground below
        ok = (sub[:, 0] == EXTW).all() and (sub[:, -1] == EXTE).all() and (sub[0, 1:-1] == ROOF).all() \
            and y1 + 1 < T.shape[0] and (types[y1 + 1, x0:x1 + 1] == GROUND).all() and (sub != SKY).all()
        rec["complete"] = bool(ok)
        if ok:
            inner = sub[:, 1:-1]
            slab_rows = [y for y in range(inner.shape[0]) if np.isin(inner[y], (SLAB, HOLE, ROOF)).all()]
            floors, rooms, stairs, holes_ok = [], [], [], True
            bounds = slab_rows + [inner.shape[0]]
            for a, b in zip(bounds[:-1], bounds[1:]):
                rows = inner[a + 1:b]
                if len(rows) == 0:
                    continue
                floors.append(len(rows))
                mid = rows[len(rows) // 2]
                walls = np.isin(rows, (WALL, DOOR)).all(0)
                rooms.append(int(walls.sum()) + 1)
                tops = np.argwhere(rows == STEP)
                stairs.append(int((rows[-1] == STEP).sum()))            # one step per stair on the floor's base row
                # the top step of each stair has a hole right above it
                for c in np.flatnonzero(rows[0] == STEP):
                    holes_ok &= bool(inner[a, c] == HOLE)
                del mid, tops
            floors, rooms, stairs = floors[::-1], rooms[::-1], stairs[::-1]       # floor 0 first
            rec.update(floors=len(floors), heights=floors, rooms=rooms, stairs=stairs, holes_ok=bool(holes_ok))
        out.append(rec)
    return out


def tower_key(T, table, bbox):
    """(n floors, per floor (widths, stair room)) read from labels inside a
    complete tower, for novelty against the sheet."""
    y0, x0, y1, x1 = bbox
    floors = {}
    for y in range(y0, y1 + 1):
        for x in range(x0 + 1, x1):
            c = table[T[y, x]]
            if c[0] in (AIR, STEP, FILL) and c[4] == 2:
                floors.setdefault(c[1], {}).setdefault(c[2], [0, c[3]])[0] += 1
    return tuple((L, tuple((v[0], v[1]) for _, v in sorted(rs.items()))) for L, rs in sorted(floors.items()))


def render(T, table, px=3):
    types = np.array([c[0] for c in table])[T]
    pal = np.zeros((12, 3), np.uint8)
    for k, c in COLOURS.items():
        pal[k] = c
    img = pal[types]
    return np.repeat(np.repeat(img, px, 0), px, 1)
