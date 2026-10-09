"""Trees on a sampled terrain (notes/trees_test.md, stage A).

Side view, rows downward, H = R0 + BT + RB rows, W = (n + 2 m) * BT columns.
Levels (coarsest first):
  tree  h = BT: one slot per block, D = 1 + BT^2.  0 absent; 1 + r * BT + c
        is a trunk at world row R0 + r, column J * BT + c.  Only slot row
        R0 // BT (the band) is sampled; the m margin slot columns per side and
        the other slot rows are fixed absent.
  surf  h = CH = 8: rows of the chunk under the surface, 0..8 (ground.py).
        Chunk rows above the band fixed 0, below it fixed 8, the NB = BT // 8
        band rows sampled.  Column height h = sum of the band values, 0..BT,
        with the fill rule (a chunk above a non-full chunk is empty) when NB
        > 1.  The energy of a column is f(h) through a pair to the painted
        chunk-row channel, and c |h_j - h_k| between neighbouring columns
        through the chunk pair c |a - b| (exact: the fill structure makes
        the per-row differences add up).
  tile  h = 1: sky / soil / trunk, painted from surf and tree (stage A).
Painted channels:
  dem8   h = 8, D = 10: 0 free, 1 + k demands surf = k.  A trunk at row y
         demands surface row y + 1, i.e. column height BT - 1 - r, split over
         the band chunks bottom first.
  rowidx h = 8, D = chunk rows, fixed: the chunk row index.
Designed factors:
  tree_pres  unary over tree.self: -b on present values
  fill       pair surf x surf at (1, 0): INF if a > 0 and the chunk below < 8
  smooth     pair surf x surf at (0, 1): c |a - b|
  prior      pair surf x rowidx at (0, 0): T[a, i]; T[a, bot] = f(a),
             T[a, top] = f(8 + a) - f(8) (NB = 2), so a column costs f(h)
  pin        pair surf x dem8 at (0, 0): INF unless dem8 is 0 or 1 + a
Learned (tree level): tree_u over the view hclass (0 absent, 1 + (c // 8) *
BT + r: which chunk column of the slot and the trunk row), tree_p1 / tree_p2
pairs over hclass at slot offsets (0, 1), (0, 2).

Given the trees, the terrain is a chain over chunk columns with pinned
columns: its log Z and exact conditional samples come from a transfer
matrix (chain_logz, chain_sample), which gives the collapsed target exactly
(the hook of AISTargets) and the oracle's terrain moves."""
import numpy as np

from .core import Channel, Factor, Model

INF = np.inf
CH = 8
SKY, SOIL, TRUNK = 0, 1, 2
TILE_COLOURS = np.array([(169, 212, 240), (138, 98, 66), (60, 30, 10)], np.uint8)
LEARNED = ["tree_u", "tree_p1", "tree_p2"]
LOFFS = {"tree_p1": (0, 1), "tree_p2": (0, 2)}


def _lse(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis) if axis is not None else float(out.reshape(-1)[0])


# ----------------------------------------------------------------- chain
def chain_messages(U, c):
    """Forward messages of a chain with unaries U (n, S) (energies, INF
    allowed) and pair energy c |s - s'|: m[j, s] = log of the sum over
    s_0..s_{j-1} of exp(-E) with s_j = s."""
    n, S = U.shape
    s = np.arange(S)
    W = -c * np.abs(s[:, None] - s[None, :])                      # W[s', s]
    m = np.empty((n, S))
    m[0] = -U[0]
    for j in range(1, n):
        m[j] = -U[j] + _lse(m[j - 1][:, None] + W, axis=0)
    return m, W


def chain_logz(U, c):
    m, _ = chain_messages(U, c)
    return _lse(m[-1])


def chain_sample(U, c, rng):
    """One exact sample of the chain (backward sampling)."""
    m, W = chain_messages(U, c)
    n, S = U.shape
    out = np.empty(n, np.int64)
    p = np.exp(m[-1] - _lse(m[-1]))
    out[-1] = rng.choice(S, p=p / p.sum())
    for j in range(n - 2, -1, -1):
        lw = m[j] + W[:, out[j + 1]]
        p = np.exp(lw - _lse(lw))
        out[j] = rng.choice(S, p=p / p.sum())
    return out


# ------------------------------------------------------------------ model
class Trees:
    def __init__(self, n, margin=2, BT=8, R0=16, RB=16, b=1.0, c=1.0, kappa=0.1, h0=None, reach=1):
        assert BT % CH == 0 and R0 % CH == 0 and RB % CH == 0
        self.n, self.m, self.BT, self.R0, self.RB = n, margin, BT, R0, RB
        self.NB = BT // CH
        self.H, self.W = R0 + BT + RB, (n + 2 * margin) * BT
        self.b, self.c, self.kappa = float(b), float(c), float(kappa)
        self.h0 = BT / 2 if h0 is None else float(h0)
        self.reach = int(reach)                                     # chunk columns of window dilation (hook / AIS)
        self.D = 1 + BT * BT
        self.S = BT + 1                                             # column heights 0..BT
        self.nch = self.H // CH
        self.band = (R0 // CH, R0 // CH + self.NB)                  # chunk rows [b0, b1)
        self.srow = R0 // BT                                        # the sampled slot row
        self.J0, self.J1 = margin, margin + n                       # interior slot columns [J0, J1)
        self.ncol = self.W // CH                                    # chunk columns
        self.f = self.kappa * (np.arange(self.S) - self.h0) ** 2    # terrain height prior f(h)
        # record <-> (r, c)
        v = np.arange(self.D)
        self.rec_r = np.where(v > 0, (v - 1) // BT, -1)
        self.rec_c = np.where(v > 0, (v - 1) % BT, -1)
        self.rec_h = np.where(v > 0, BT - 1 - self.rec_r, -1)      # column height demanded
        self.hclass = np.where(v > 0, 1 + (self.rec_c // CH) * BT + self.rec_r, 0)
        self.NC = 1 + self.NB * BT

    # ------------------------------------------------------- channels
    def channels(self):
        tree = Channel("tree", self.BT, self.D).add_view("self", np.arange(self.D), self.D) \
            .add_view("hclass", self.hclass, self.NC)
        surf = Channel("surf", CH, CH + 1).add_view("val", np.arange(CH + 1), CH + 1)
        dem8 = Channel("dem8", CH, CH + 2).add_view("val", np.arange(CH + 2), CH + 2)
        rowidx = Channel("rowidx", CH, self.nch).add_view("val", np.arange(self.nch), self.nch)
        tile = Channel("tile", 1, 3).add_view("kind", np.arange(3), 3)
        return [tree, surf, dem8, rowidx, tile]

    def init_state(self, chans):
        """Fixed masks and the painted rowidx; a flat terrain at height h0;
        every slot absent."""
        tree, surf, dem8, rowidx, tile = chans
        H, W = self.H, self.W
        for ch in chans:
            if ch.grid is None:
                ch.grid = np.zeros((H // ch.h, W // ch.h), np.int32)
                ch.fixed = np.zeros(ch.grid.shape, bool)
        tree.grid[:] = 0
        tree.fixed[:] = True
        tree.fixed[self.srow, self.J0:self.J1] = False
        b0, b1 = self.band
        surf.fixed[:] = True
        surf.fixed[b0:b1] = False
        surf.grid[:b0] = 0
        surf.grid[b1:] = CH
        self.set_heights(surf, np.full(self.ncol, int(round(self.h0))))
        rowidx.grid[:] = np.arange(self.nch)[:, None]
        rowidx.fixed[:] = True
        dem8.fixed[:] = True
        tile.fixed[:] = True
        self.paint_dem(tree, dem8)
        self.paint_tile(tree, surf, tile)

    # ------------------------------------------------------ heights
    def heights(self, surf):
        """(ncol,) column heights 0..BT from the band chunks."""
        b0, b1 = self.band
        return surf.grid[b0:b1].sum(0)

    def chunks_of(self, h):
        """(NB,) band chunk values (top first) of a column of height h."""
        h = np.asarray(h)
        i = np.arange(self.NB)[::-1]                                 # bottom chunk fills first
        return np.clip(h[..., None] - CH * i, 0, CH)

    def set_heights(self, surf, h, cols=None):
        b0, b1 = self.band
        cols = np.arange(self.ncol) if cols is None else np.asarray(cols)
        surf.grid[b0:b1, cols] = self.chunks_of(np.asarray(h)).T

    # ------------------------------------------------------- painters
    def slot_cols(self, J):
        """Chunk columns of slot column J."""
        return J * self.NB, (J + 1) * self.NB

    def paint_dem(self, tree, dem8, J=None):
        """dem8 over every chunk column (J None) or those of slot column J."""
        b0, b1 = self.band
        g = tree.grid[self.srow]
        Js = range(g.shape[0]) if J is None else [J]
        for j in Js:
            x0, x1 = self.slot_cols(j)
            dem8.grid[b0:b1, x0:x1] = 0
            t = g[j]
            if t > 0:
                cc = j * self.NB + self.rec_c[t] // CH
                dem8.grid[b0:b1, cc] = 1 + self.chunks_of(self.rec_h[t])

    def pins(self, dem8):
        """(ncol,) demanded column height, -1 where free (from dem8)."""
        b0, b1 = self.band
        d = dem8.grid[b0:b1]
        pinned = (d > 0).any(0)
        return np.where(pinned, np.clip(d - 1, 0, CH).sum(0), -1)

    def surface_rows(self, surf):
        return self.R0 + self.BT - self.heights(surf)                # first soil row per chunk column

    def paint_tile(self, tree, surf, tile):
        sr = np.repeat(self.surface_rows(surf), CH)
        tile.grid[:] = np.where(np.arange(self.H)[:, None] >= sr[None, :], SOIL, SKY)
        g = tree.grid[self.srow]
        for j in np.flatnonzero(g > 0):
            t = g[j]
            tile.grid[self.R0 + self.rec_r[t], j * self.BT + self.rec_c[t]] = TRUNK

    # -------------------------------------------------------- factors
    def designed_factors(self):
        S = CH + 1
        pres = np.full(self.D, -self.b)
        pres[0] = 0.0
        fill = np.zeros((S, S))
        fill[1:, :CH] = INF
        a = np.arange(S)
        smooth = self.c * np.abs(a[:, None] - a[None, :]).astype(np.float64)
        prior = np.zeros((S, self.nch))
        b0, b1 = self.band
        prior[:, b1 - 1] = self.f[:S]
        if self.NB == 2:
            prior[:, b0] = self.f[CH + a] - self.f[CH]
        assert self.NB <= 2
        pin = np.zeros((S, CH + 2))
        pin[:, 1:] = INF
        pin[a, 1 + a] = 0.0
        out = [Factor.unary(("tree", "self"), pres, name="tree_pres"),
               Factor.pair(("surf", "val"), ("surf", "val"), (0, 1), smooth, name="smooth"),
               Factor.pair(("surf", "val"), ("rowidx", "val"), (0, 0), prior, name="prior"),
               Factor.pair(("surf", "val"), ("dem8", "val"), (0, 0), pin, name="pin")]
        if self.NB > 1:
            out.append(Factor.pair(("surf", "val"), ("surf", "val"), (1, 0), fill, name="fill"))
        return out

    def learned_factors(self, tables):
        out = [Factor.unary(("tree", "hclass"), tables["tree_u"], name="tree_u")]
        for n, off in LOFFS.items():
            out.append(Factor.pair(("tree", "hclass"), ("tree", "hclass"), off, tables[n], name=n))
        return out

    def theta0(self):
        return {"tree_u": np.zeros(self.NC), "tree_p1": np.zeros((self.NC, self.NC)),
                "tree_p2": np.zeros((self.NC, self.NC))}

    def shapes(self):
        return {n: t.shape for n, t in self.theta0().items()}

    def model(self, chans, tables=None):
        fac = self.designed_factors() + (self.learned_factors(tables) if tables is not None else [])
        return Model(self.H, self.W, chans, fac)

    # -------------------------------------------------- the chain
    def window(self, J):
        """Chunk columns [x0, x1) of slot column J's window: its own columns
        dilated by `reach` (clipped)."""
        x0, x1 = self.slot_cols(J)
        return max(x0 - self.reach, 0), min(x1 + self.reach, self.ncol)

    def unaries(self, x0, x1, pins, h):
        """(x1 - x0, S) chain unaries over the window columns: f(h) plus the
        pins and the boundary terms to the columns outside at heights h."""
        U = np.tile(self.f, (x1 - x0, 1))
        for j in range(x0, x1):
            if pins[j] >= 0:
                U[j - x0] = INF
                U[j - x0, pins[j]] = self.f[pins[j]]
        s = np.arange(self.S)
        if x0 > 0:
            U[0] += self.c * np.abs(h[x0 - 1] - s)
        if x1 < self.ncol:
            U[-1] += self.c * np.abs(h[x1] - s)
        return U

    def window_logz(self, surf, dem8, J):
        """log Z of the window's terrain given the pins and the outside
        columns, in AISTargets' convention (free cells of the window summed,
        every term touching one of them)."""
        x0, x1 = self.window(J)
        return chain_logz(self.unaries(x0, x1, self.pins(dem8), self.heights(surf)), self.c)

    def window_redraw(self, surf, dem8, J, rng):
        x0, x1 = self.window(J)
        hs = chain_sample(self.unaries(x0, x1, self.pins(dem8), self.heights(surf)), self.c, rng)
        self.set_heights(surf, hs, np.arange(x0, x1))

    def chain_redraw(self, surf, dem8, rng):
        """The whole terrain from its exact conditional given the pins."""
        hs = chain_sample(self.unaries(0, self.ncol, self.pins(dem8), self.heights(surf)), self.c, rng)
        self.set_heights(surf, hs)

    # ------------------------------------------------------ checks
    def violations(self, chans):
        """Hard-rule violations of a state: pins, fill, trunk placement."""
        tree, surf, dem8, rowidx, tile = chans
        b0, b1 = self.band
        pins = self.pins(dem8)
        h = self.heights(surf)
        bad_pin = int(((pins >= 0) & (pins != h)).sum())
        bad_fill = 0
        if self.NB > 1:
            band = surf.grid[b0:b1]
            bad_fill = int(((band[:-1] > 0) & (band[1:] < CH)).sum())
        bad_trunk = 0
        g = tree.grid[self.srow]
        sr = self.surface_rows(surf)
        for j in np.flatnonzero(g > 0):
            t = g[j]
            y, x = self.R0 + self.rec_r[t], j * self.BT + self.rec_c[t]
            if sr[x // CH] != y + 1:
                bad_trunk += 1
        return dict(pin=bad_pin, fill=bad_fill, trunk=bad_trunk)

    # ------------------------------------------------------- stats
    def stats(self, tree, surf):
        """Interior moments: presence, the height histogram per slot, the
        adjacent-slot (offset 1, 2) joint classes, terrain roughness."""
        g = tree.grid[self.srow, self.J0:self.J1]
        n = len(g)
        pres = g > 0
        hh = np.where(pres, self.rec_h[g], -1)
        out = dict(present=float(pres.mean()))
        out["h_hist"] = (np.bincount(hh[pres], minlength=self.BT)[:self.BT] / n).tolist()
        for k in (1, 2):
            a, b = hh[:-k], hh[k:]
            both = (a >= 0) & (b >= 0)
            d = np.abs(a - b)
            m = n - k
            out[f"p{k}_both"] = float(both.sum() / m)
            out[f"p{k}_eq"] = float((both & (d == 0)).sum() / m)
            out[f"p{k}_d1"] = float((both & (d == 1)).sum() / m)
            out[f"p{k}_far"] = float((both & (d >= 2)).sum() / m)
        h = self.heights(surf)[self.J0 * self.NB:self.J1 * self.NB]
        out["terr_mean"] = float(h.mean())
        out["terr_var"] = float(h.var())
        out["terr_rough"] = float(np.abs(np.diff(h)).mean())
        return out

    @staticmethod
    def stat_names(s):
        return [k for k in s if k != "h_hist"]

    def render(self, tile, px=4):
        img = TILE_COLOURS[tile.grid]
        return np.repeat(np.repeat(img, px, 0), px, 1)


# ---------------------------------------------------------------- forward
class Forward:
    """The chain on sampler.Sampler: tree sweeps under designed + learned
    tables, dem8 painted, surf init (pinned chunks dormant) + sweeps, tiles
    painted."""

    def __init__(self, T: Trees, tables=None, seed=0, chans=None, K=None):
        from .sampler import Sampler
        self.T = T
        self.tables = T.theta0() if tables is None else tables
        self.chans = T.channels() if chans is None else chans
        self.tree, self.surf, self.dem8, self.rowidx, self.tile = self.chans
        if chans is None:
            T.init_state(self.chans)
        self.rng = np.random.default_rng(seed)
        self.K = K
        self.model = None
        self._Sampler = Sampler

    def build(self):
        self.model = self.T.model(self.chans, self.tables)
        self.S = dict(tree=self._Sampler(self.model, "tree", K=self.K), surf=self._Sampler(self.model, "surf"))

    def run(self, S_tree=30, S_surf=30, fresh=True):
        T = self.T
        self.build()
        if fresh:
            self.tree.grid[:] = 0
            T.set_heights(self.surf, np.full(T.ncol, int(round(T.h0))))
            T.paint_dem(self.tree, self.dem8)
        self.S["tree"].init()
        if S_tree:
            self.S["tree"].sweep(S_tree, seed=int(self.rng.integers(1 << 30)))
        T.paint_dem(self.tree, self.dem8)
        self.dormant_frac = self.S["surf"].init()
        if S_surf:
            self.S["surf"].sweep(S_surf, seed=int(self.rng.integers(1 << 30)))
        T.paint_tile(self.tree, self.surf, self.tile)
        return T.stats(self.tree, self.surf)


# ----------------------------------------------------------------- oracle
class Oracle:
    """Exact p*(tree, surf) by block Gibbs: the terrain from its exact
    conditional given the pins (chain_sample); each interior slot from
    p*(t | terrain): absent, or a trunk at the surface of one of its columns
    (weight e^b each)."""

    def __init__(self, T: Trees, seed=0, chans=None):
        self.T = T
        self.chans = T.channels() if chans is None else chans
        self.tree, self.surf, self.dem8, self.rowidx, self.tile = self.chans
        if chans is None:
            T.init_state(self.chans)
        self.rng = np.random.default_rng(seed)

    def slot_move(self, J):
        T = self.T
        h = T.heights(self.surf)
        x0, x1 = T.slot_cols(J)
        cand, w = [0], [0.0]
        for cc in range(x0, x1):
            if h[cc] < T.BT:                                        # a full band leaves no trunk row
                r = T.BT - 1 - h[cc]
                for c in range(CH):
                    cand.append(1 + r * T.BT + (cc - x0) * CH + c)
                    w.append(T.b)
        w = np.asarray(w)
        p = np.exp(w - w.max())
        self.tree.grid[T.srow, J] = cand[self.rng.choice(len(cand), p=p / p.sum())]
        T.paint_dem(self.tree, self.dem8, J)

    def sweep(self, n=1):
        T = self.T
        for _ in range(n):
            T.chain_redraw(self.surf, self.dem8, self.rng)
            for J in range(T.J0, T.J1):
                self.slot_move(J)

    def moments(self, burn, sweeps, every=1):
        self.sweep(burn)
        acc, k = None, 0
        for s in range(sweeps):
            self.sweep(1)
            if s % every:
                continue
            st = self.T.stats(self.tree, self.surf)
            if acc is None:
                acc = {n: np.asarray(v, float) for n, v in st.items()}
            else:
                for n, v in st.items():
                    acc[n] = acc[n] + np.asarray(v, float)
            k += 1
        return {n: (v / k).tolist() if np.ndim(v) else float(v / k) for n, v in acc.items()}

    def height_tables(self, burn, sweeps):
        """Monte Carlo -log p(h) at a tree column and -PMI(h, h') at slot
        distance 1, 2 among present pairs: the induced potential's shape."""
        T = self.T
        self.sweep(burn)
        u = np.zeros(T.BT)
        pm = {1: np.zeros((T.BT, T.BT)), 2: np.zeros((T.BT, T.BT))}
        for _ in range(sweeps):
            self.sweep(1)
            g = self.tree.grid[T.srow, T.J0:T.J1]
            hh = np.where(g > 0, T.rec_h[g], -1)
            u += np.bincount(hh[hh >= 0], minlength=T.BT)[:T.BT]
            for k in (1, 2):
                a, b = hh[:-k], hh[k:]
                m = (a >= 0) & (b >= 0)
                np.add.at(pm[k], (a[m], b[m]), 1)
        return u, pm


# ------------------------------------------------------------------ bench
class Bench:
    """One state shared by the oracle and the forward chain; the trainer's
    callbacks (train.fit, AISTargets): contexts, designed_e, feats,
    after_change, repaint, region, hook."""

    def __init__(self, T: Trees, seed=0, S=(30, 30)):
        from . import train
        self.T, self.S = T, S
        self.orc = Oracle(T, seed=seed)
        self.fw = Forward(T, chans=self.orc.chans, seed=seed + 1)
        self.tree, self.surf, self.dem8, self.rowidx, self.tile = self.orc.chans
        self.md = T.model(self.orc.chans, None)                       # designed only
        self.feats = train.tabular_feats(LEARNED)
        self.designed_e = train.designed_energies(self.md, "tree")
        self.rng = np.random.default_rng(seed + 2)

    def tables_of(self, theta):
        from . import train
        return self.T.theta0() if theta is None else train.unpack(theta, self.T.shapes())

    def contexts(self, theta, n, rng):
        for _ in range(n):
            self.fw.tables = self.tables_of(theta)
            self.fw.run(*self.S, fresh=True)
            yield self.fw.model

    def region(self, model, home, I, J):
        b0, b1 = self.T.band
        x0, x1 = self.T.window(J)
        return b0, b1, x0, x1

    def repaint(self, model, home, I, J):
        self.T.paint_dem(self.tree, self.dem8, J)

    def move(self, model, home, I, J):
        """K-step after_change: dem8 repainted, the window's terrain redrawn
        from its exact conditional."""
        self.T.paint_dem(self.tree, self.dem8, J)
        self.T.window_redraw(self.surf, self.dem8, J, self.rng)

    def hook(self, model, home, I, J):
        return self.T.window_logz(self.surf, self.dem8, J)
