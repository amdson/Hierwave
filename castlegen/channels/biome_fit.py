"""The bootstrap fit on the biome circles model (notes/circles_biome_test.md,
stages 4-5): the model-specific callbacks train.fit and AISTargets take, and
the materialisation of a stamp-feature potential into the obj tables.

materialise(C, theta)      flat stamp-feature theta (paintpot OFF8 over the dem
                           paint, V = 4) -> obj_h, obj_v, obj_d1, obj_d2 (D, D),
                           obj_u (D,): energies of single objects and pairs
                           painted on a 5 x 5 block canvas (exact for linear psi;
                           no INF: the support is added from the stamps)
mid_tables / all_tables    the theta dict Forward installs
Bench                      an Oracle and a Forward sharing one set of channels,
                           the designed + support model, and the callbacks:
                           contexts, designed_e, feats, after_change (the
                           K-step moves), repaint (the probes), AIS region and
                           painters, and the exact collapsed top hook."""
import numpy as np

from . import paintpot, train
from .circles_biome import DIR8, OFFNAMES, OFFS, Forward, Oracle, _redraw

INF = np.inf
BIO = ["bio_h", "bio_v", "bio_u"]
V = 4


def materialise(C, theta, offsets=paintpot.OFF8):
    """{obj_h, obj_v, obj_d1, obj_d2, obj_u} of psi = features(dem) . theta:
    u[t] = psi(t alone) - psi(empty), g_d[t, t'] = psi(t, t' at +d) - u[t] -
    u[t'] - psi(empty), t at the centre of a 5 x 5 block canvas (every stamp
    of the pair and its partner's spill on the grid, so u[0] and g_d[0, .]
    are exactly 0)."""
    th = np.asarray(theta, float)
    D = C.D
    og = np.zeros((5, 5), np.int32)

    def paint(t, tp=0, d=(0, 0)):
        og[:] = 0
        og[2, 2] = t
        if tp:
            og[2 + d[0], 2 + d[1]] = tp
        return C.dem_of(og)

    e = lambda P: paintpot.features(np.asarray(P), V, offsets) @ th
    e0 = float(e(paint(0)))
    u = e(np.stack([paint(t) for t in range(D)])) - e0
    u[0] = 0.0
    out = dict(obj_u=u)
    for n, d in zip(OFFNAMES, OFFS):
        P = np.stack([paint(t, tp, d) for t in range(D) for tp in range(D)])
        g = e(P).reshape(D, D) - u[:, None] - u[None, :] - e0
        g[0, :] = 0.0
        g[:, 0] = 0.0
        out[n] = g
    return out


def mid_tables(C, theta_mid=None):
    """theta0 with the obj tables materialised from theta_mid (zeros if None)."""
    t = C.theta0()
    if theta_mid is not None:
        t.update(materialise(C, theta_mid))
    return t


def all_tables(C, theta_mid, theta_top=None):
    t = mid_tables(C, theta_mid)
    if theta_top is not None:
        t.update(train.unpack(theta_top, {n: t[n].shape for n in BIO}))
    return t


def pair_total(C, tables):
    """Directed obj pair energy over the 8 neighbours with the support:
    T8[k][t, t'] = t at p, t' at p + DIR8[k]."""
    out = []
    for dy, dx in DIR8:
        if (dy, dx) in OFFS:
            n = OFFNAMES[OFFS.index((dy, dx))]
            out.append(np.where(C.CONF[(dy, dx)], INF, tables[n]))
        else:
            n = OFFNAMES[OFFS.index((-dy, -dx))]
            out.append(np.where(C.CONF[(-dy, -dx)], INF, tables[n]).T)
    return out


def _lse(a):
    m = a.max()
    return -INF if m == -INF else float(m + np.log(np.exp(a - m).sum()))


class Bench:
    """One state shared by the oracle, the forward chain and the
    designed + support model; the stage's callbacks over it."""

    def __init__(self, C, seed=0, S=(30, 30, 20)):
        self.C, self.S = C, S
        self.orc = Oracle(C, seed=seed)
        self.fw = Forward(C, C.theta0(), seed=seed + 1)
        self.fw.chans = self.orc.chans
        self.fw.biome, self.fw.obj, self.fw.tile, self.fw.allow, self.fw.dem = self.orc.chans
        self.biome, self.obj, self.tile, self.allow, self.dem = self.orc.chans
        self.md = C.model(self.orc.chans, None, C.support_factors())       # designed + support
        self.seen = np.zeros((4, C.D, C.D), bool)                          # (t, t') as candidate x neighbour
        self._budget = 0

    # ------------------------------------------------------- contexts
    def contexts(self, tables_of, home, K=0):
        """contexts(theta, n, rng) for train.fit: fresh forward runs at
        tables_of(theta).  Counts the feats calls of a context's data passes
        ((K + 1) x active sites) so that `seen` skips the probes."""
        def gen(theta, n, rng):
            for _ in range(n):
                self.fw.theta = tables_of(theta)
                self.fw.run(*self.S, fresh=True)
                self._budget = (K + 1) * int((~self.fw.model.chan(home).fixed).sum())
                yield self.fw.model
        return gen

    def designed_e(self, home):
        return train.designed_energies(self.md, home)

    # ------------------------------------------------------------ mid
    def mid_feats(self, model, home, i, j, cand):
        X = self.C.stamp_features(self.obj, i, j, cand)
        if self._budget > 0:
            self._budget -= 1
            self._mark(i, j, cand)
        return X

    def _mark(self, i, j, cand):
        og = self.obj.grid
        cand = np.asarray(cand)
        cand = cand[cand > 0]
        for dy, dx in DIR8:
            y, x = i + dy, j + dx
            if 0 <= y < og.shape[0] and 0 <= x < og.shape[1] and og[y, x] > 0:
                w = og[y, x]
                if (dy, dx) in OFFS:
                    self.seen[OFFS.index((dy, dx)), cand, w] = True
                else:
                    self.seen[OFFS.index((-dy, -dx)), w, cand] = True

    def mid_region(self, model, home, i, j):
        BM = self.C.BM
        return i * BM - 1, (i + 1) * BM + 1, j * BM - 1, (j + 1) * BM + 1

    def mid_paint(self, model, home, i, j):
        """Deterministic: dem over the block plus its spill border."""
        self.C.paint_dem(self.obj, self.dem, self.mid_region(model, home, i, j))

    def mid_move(self, model, home, i, j):
        """K-step after_change: dem repainted, the tiles whose demand changed
        redrawn from their exact conditional (Oracle.mid_move's tail)."""
        C = self.C
        y0, y1, x0, x1 = self.mid_region(model, home, i, j)
        ya, xa = max(y0, 0), max(x0, 0)
        old = self.dem.grid[ya:y1, xa:x1].copy()
        C.paint_dem(self.obj, self.dem, (y0, y1, x0, x1))
        _redraw(self.tile.grid, self.dem.grid, old, ya, y1, xa, x1, C.mu, self.orc.rng.random((C.BM + 2) ** 2), True)

    def mid_hook(self, model, home, i, j):
        """Exact log Z of the window tiles: -sum fz[dem] (tiles independent)."""
        y0, y1, x0, x1 = self.mid_region(model, home, i, j)
        return -self.C.fz[self.dem.grid[max(y0, 0):y1, max(x0, 0):x1]].sum()

    # ------------------------------------------------------------ top
    def top_region(self, model, home, I, J):
        BT = self.C.BT
        return I * BT, (I + 1) * BT, J * BT, (J + 1) * BT

    def top_paint(self, model, home, I, J):
        """Probes: allow on the cell only (the slots are left as they are)."""
        self.C.paint_allow(self.biome, self.allow, (I, J))

    def top_dormancy(self, model, home, I, J):
        """AIS after_change: allow painted; the cell's slots fixed at absent
        when it admits no family, freed otherwise."""
        C, BT = self.C, self.C.BT
        C.paint_allow(self.biome, self.allow, (I, J))
        s = (slice(I * BT, (I + 1) * BT), slice(J * BT, (J + 1) * BT))
        dorm = C.dormant_of(self.allow.grid[s])
        self.obj.grid[s][dorm] = 0
        self.obj.fixed[s] = dorm

    def top_move(self, model, home, I, J):
        """K-step after_change: dormancy, then the oracle's collapsed move at
        each of the four slots (dem and tiles follow)."""
        self.top_dormancy(model, home, I, J)
        BT = self.C.BT
        for a in range(I * BT, (I + 1) * BT):
            for b in range(J * BT, (J + 1) * BT):
                self.orc.mid_move(a, b)

    def top_hook(self, tables):
        """Exact log Z of the four slots of a top cell under the installed obj
        rows (pres + obj_u + mask unaries, obj pairs + support among them and
        to the fixed neighbours), by enumeration over the admissible values;
        AISTargets' convention (fixed slots not summed)."""
        C = self.C
        T8 = pair_total(C, tables)
        un = C.pres_e + tables["obj_u"]

        def hook(model, home, I, J):
            BT, og, fx = C.BT, self.obj.grid, self.obj.fixed
            cells = [(a, b) for a in range(I * BT, (I + 1) * BT) for b in range(J * BT, (J + 1) * BT)]
            free = [c for c in cells if not fx[c]]
            if not free:
                return 0.0
            idx = {c: k for k, c in enumerate(free)}
            vals, es = [], []
            for (a, b) in free:
                e = un + C.mask_e[:, self.allow.grid[a, b]]
                for k, (dy, dx) in enumerate(DIR8):
                    y, x = a + dy, b + dx
                    if 0 <= y < og.shape[0] and 0 <= x < og.shape[1] and (y, x) not in idx:
                        e = e + T8[k][:, og[y, x]]
                ok = np.flatnonzero(np.isfinite(e))
                vals.append(ok)
                es.append(e[ok])
            n = len(free)
            tot = np.zeros([len(v) for v in vals])
            for k in range(n):
                sh = [1] * n; sh[k] = -1
                tot = tot + es[k].reshape(sh)
            for k in range(n):
                for l in range(k + 1, n):
                    d = (free[l][0] - free[k][0], free[l][1] - free[k][1])
                    P = T8[DIR8.index(d)][np.ix_(vals[k], vals[l])]
                    sh = [1] * n; sh[k] = len(vals[k]); sh[l] = len(vals[l])
                    tot = tot + P.reshape(sh)
            return _lse(-tot.ravel())
        return hook
