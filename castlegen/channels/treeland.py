"""Trees with limbs on a given terrain (notes/trees_test.md, stage B).

The terrain comes first and the trees read it: level 2 holds surf (chunk
column heights, drawn first) and tree (one slot per chunk, drawn after it,
reading surf at the same level); level 1 holds the limb tiles under the
certificate.  Side view, rows downward, H = R0 + CH + RB, W = (n + 2 m) CH.

  surf   h = 8, D = 9: the band chunk's rows under the surface = the column
         height h (0..8); other chunk rows fixed (0 above, 8 below).  Energy
         f(h) (prior, through the painted rowidx) + c |h_j - h_{j+1}| (smooth).
  tree   h = 8, D = 65: 0 absent, 1 + r * 8 + c a trunk at band row r,
         column c of the slot.  Hard: the trunk sits on the surface, r = 7 -
         h (place, a pair to surf at (0, 0)).  Unary -b per present value.
         Views: self; hclass = 1 + r (0 absent).
  tile   h = 1: limbs.KINDS (sky, soil, trunk, limb{m}_{ports}); hard pairs
         to the painted ground (soil iff under the surface) and tdem (trunk
         iff demanded); limbs.factors (unary, seams, contact);
         limbs.certificate (a forest rooted at the trunks, mass
         non-increasing away from them) in the sampler.
Learned potentials (tabular over views):
  tree level  tree_u (hclass), tree_p1 / tree_p2 (hclass x hclass at slot
              offsets 1, 2), tree_s{p1,m1,p2,m2} (self x surf.val at (0, +1),
              (0, -1), (0, +2), (0, -2)): the limbs' free energy given the
              trunk and the terrain around it (room for limbs)
  surf level  surf_u (val), surf_p1 / surf_p2 (val x val at (0, 1), (0, 2)):
              the trees' and limbs' free energy given the terrain
The forward chain: surf sweeps, tree init (admitted by place) + sweeps,
ground / tdem painted, tiles reset, tile sweeps."""
import numpy as np

from . import limbs
from .core import Channel, Factor, Model

INF = np.inf
CH = 8
TREE_LEARNED = ["tree_u", "tree_p1", "tree_p2", "tree_sp1", "tree_sm1", "tree_sp2", "tree_sm2"]
SURF_LEARNED = ["surf_u", "surf_p1", "surf_p2"]
TREE_OFFS = {"tree_p1": (0, 1), "tree_p2": (0, 2)}
SURF_READ = {"tree_sp1": (0, 1), "tree_sm1": (0, -1), "tree_sp2": (0, 2), "tree_sm2": (0, -2)}
SURF_OFFS = {"surf_p1": (0, 1), "surf_p2": (0, 2)}


class Treeland:
    def __init__(self, n, margin=2, R0=16, RB=8, b=-1.5, c=1.0, kappa=0.1, h0=4.0, win=(12, 8), limb_kw=None,
                 Dmax=64):
        assert R0 % CH == 0 and RB % CH == 0
        self.n, self.m, self.R0, self.RB = n, margin, R0, RB
        self.H, self.W = R0 + CH + RB, (n + 2 * margin) * CH
        self.b, self.c, self.kappa, self.h0 = float(b), float(c), float(kappa), float(h0)
        self.win = win                                              # tile window: rows above, columns each side
        self.limb_kw = dict(limb_kw or {})
        self.Dmax = Dmax
        self.D = 1 + CH * CH
        self.S = CH + 1
        self.nch = self.H // CH
        self.brow = R0 // CH                                        # the band chunk row = the slot row
        self.J0, self.J1 = margin, margin + n
        self.ncol = self.W // CH
        self.f = self.kappa * (np.arange(self.S) - self.h0) ** 2
        v = np.arange(self.D)
        self.rec_r = np.where(v > 0, (v - 1) // CH, -1)
        self.rec_c = np.where(v > 0, (v - 1) % CH, -1)
        self.rec_h = np.where(v > 0, CH - 1 - self.rec_r, -1)
        self.hclass = np.where(v > 0, 1 + self.rec_r, 0)
        self.NC = 1 + CH
        self.kinds = limbs.KINDS
        self.ktrunk, self.ksky, self.ksoil = self.kinds.index("trunk"), self.kinds.index("sky"), self.kinds.index("soil")

    # ------------------------------------------------------- channels
    def channels(self):
        surf = Channel("surf", CH, self.S).add_view("val", np.arange(self.S), self.S)
        tree = Channel("tree", CH, self.D).add_view("self", np.arange(self.D), self.D) \
            .add_view("hclass", self.hclass, self.NC)
        rowidx = Channel("rowidx", CH, self.nch).add_view("val", np.arange(self.nch), self.nch)
        tile = limbs.tile_channel()
        ground = Channel("ground", 1, 2).add_view("val", np.arange(2), 2)
        tdem = Channel("tdem", 1, 2).add_view("val", np.arange(2), 2)
        cert = limbs.cert_channel(self.Dmax)
        return [surf, tree, rowidx, tile, ground, tdem, cert]

    def unpack(self, chans):
        return dict(zip(["surf", "tree", "rowidx", "tile", "ground", "tdem", "cert"], chans))

    def init_state(self, chans):
        c = self.unpack(chans)
        H, W = self.H, self.W
        for ch in chans:
            if ch.grid is None:
                ch.grid = np.zeros((H // ch.h, W // ch.h), np.int32)
                ch.fixed = np.zeros(ch.grid.shape, bool)
        b = self.brow
        c["surf"].fixed[:] = True
        c["surf"].fixed[b] = False
        c["surf"].grid[:b] = 0
        c["surf"].grid[b + 1:] = CH
        c["surf"].grid[b] = int(round(self.h0))
        c["tree"].grid[:] = 0
        c["tree"].fixed[:] = True
        c["tree"].fixed[b, self.J0:self.J1] = False
        c["rowidx"].grid[:] = np.arange(self.nch)[:, None]
        for n in ("rowidx", "ground", "tdem"):
            c[n].fixed[:] = True
        c["cert"].fixed[:] = True                                     # the kernel writes d itself
        self.paint_fine(chans, reset=True)

    # -------------------------------------------------------- painters
    def heights(self, surf):
        return surf.grid[self.brow]

    def surface_rows(self, surf):
        return self.R0 + CH - self.heights(surf)

    def paint_fine(self, chans, reset=False, cols=None):
        """ground from surf, tdem from tree, over every tile column (cols
        None) or the tile columns [x0, x1).  reset: the tiles there become
        soil / sky / trunk with no limbs; otherwise soil and trunk cells are
        forced and former soil now in the sky becomes sky."""
        c = self.unpack(chans)
        x0, x1 = (0, self.W) if cols is None else cols
        sr = np.repeat(self.surface_rows(c["surf"]), CH)[x0:x1]
        g = (np.arange(self.H)[:, None] >= sr[None, :]).astype(np.int32)
        c["ground"].grid[:, x0:x1] = g
        td = np.zeros((self.H, x1 - x0), np.int32)
        tg = c["tree"].grid[self.brow]
        for j in np.flatnonzero(tg > 0):
            t = tg[j]
            y, x = self.R0 + self.rec_r[t], j * CH + self.rec_c[t]
            if x0 <= x < x1:
                td[y, x - x0] = 1
        c["tdem"].grid[:, x0:x1] = td
        tile = c["tile"].grid[:, x0:x1]
        if reset:
            tile[:] = np.where(g == 1, self.ksoil, self.ksky)
        else:
            tile[(g == 1)] = self.ksoil
            tile[(g == 0) & (tile == self.ksoil)] = self.ksky
            tile[(g == 0) & (tile == self.ktrunk)] = self.ksky
        tile[td == 1] = self.ktrunk
        if reset:
            c["cert"].grid[:, x0:x1] = self.Dmax + 1

    # -------------------------------------------------------- factors
    def designed_factors(self):
        S, D = self.S, self.D
        pres = np.full(D, -self.b)
        pres[0] = 0.0
        a = np.arange(S)
        place = np.full((D, S), INF)
        place[0] = 0.0
        for t in range(1, D):
            place[t, self.rec_h[t]] = 0.0
        smooth = self.c * np.abs(a[:, None] - a[None, :]).astype(np.float64)
        prior = np.zeros((S, self.nch))
        prior[:, self.brow] = self.f
        rv = self.kinds                                              # tile view root: 0 sky 1 soil 2 trunk 3 limb
        tg = np.zeros((4, 2))
        tg[limbs.SOIL, 0] = INF
        tg[[limbs.SKY, limbs.TRUNK, limbs.LIMB], 1] = INF
        tt = np.zeros((2, 2))
        tt[0, 1] = INF                                               # demanded: a trunk
        tt[1, 0] = INF                                               # not demanded: no trunk
        out = [Factor.unary(("tree", "self"), pres, name="tree_pres"),
               Factor.pair(("tree", "self"), ("surf", "val"), (0, 0), place, name="place"),
               Factor.pair(("surf", "val"), ("surf", "val"), (0, 1), smooth, name="smooth"),
               Factor.pair(("surf", "val"), ("rowidx", "val"), (0, 0), prior, name="prior"),
               Factor.pair(("tile", "root"), ("ground", "val"), (0, 0), tg, name="tile_ground"),
               Factor.pair(("tile", "trunk"), ("tdem", "val"), (0, 0), tt, name="tile_tdem")]
        return out + limbs.factors(**self.limb_kw)

    def learned_factors(self, tables):
        out = []
        for n, tab in tables.items():
            if n == "tree_u":
                out.append(Factor.unary(("tree", "hclass"), tab, name=n))
            elif n in TREE_OFFS:
                out.append(Factor.pair(("tree", "hclass"), ("tree", "hclass"), TREE_OFFS[n], tab, name=n))
            elif n in SURF_READ:
                out.append(Factor.pair(("tree", "self"), ("surf", "val"), SURF_READ[n], tab, pad_b=-1, name=n))
            elif n == "surf_u":
                out.append(Factor.unary(("surf", "val"), tab, name=n))
            elif n in SURF_OFFS:
                out.append(Factor.pair(("surf", "val"), ("surf", "val"), SURF_OFFS[n], tab, name=n))
            else:
                raise KeyError(n)
        return out

    def theta0(self, level):
        NC, S, D = self.NC, self.S, self.D
        if level == "tree":
            t = {"tree_u": np.zeros(NC), "tree_p1": np.zeros((NC, NC)), "tree_p2": np.zeros((NC, NC))}
            t.update({n: np.zeros((D, S)) for n in SURF_READ})
            return t
        return {"surf_u": np.zeros(S), "surf_p1": np.zeros((S, S)), "surf_p2": np.zeros((S, S))}

    def shapes(self, level):
        return {n: t.shape for n, t in self.theta0(level).items()}

    def model(self, chans, tables=None, cert=True):
        fac = self.designed_factors() + (self.learned_factors(tables) if tables else [])
        return Model(self.H, self.W, chans, fac, [limbs.certificate(Dmax=self.Dmax)] if cert else [])

    # --------------------------------------------------------- windows
    def tile_window(self, J):
        """Tile rows / columns of slot J's limb window: the band and `win[0]`
        rows above it, `win[1]` columns each side."""
        up, side = self.win
        return (max(self.R0 - up, 0), self.R0 + CH, max(J * CH - side, 0), min((J + 1) * CH + side, self.W))

    # --------------------------------------------------------- stats
    def stats(self, chans):
        c = self.unpack(chans)
        tree, surf, tile = c["tree"], c["surf"], c["tile"]
        g = tree.grid[self.brow, self.J0:self.J1]
        n = len(g)
        pres = g > 0
        hh = np.where(pres, self.rec_h[g], -1)
        h = self.heights(surf)
        out = dict(present=float(pres.mean()))
        out["h_hist"] = (np.bincount(hh[pres], minlength=CH)[:CH] / n).tolist()
        for k in (1, 2):
            a, b = hh[:-k], hh[k:]
            both = (a >= 0) & (b >= 0)
            out[f"p{k}_both"] = float(both.sum() / (n - k))
        hi = h[self.J0:self.J1]
        out["terr_mean"], out["terr_var"], out["terr_rough"] = float(hi.mean()), float(hi.var()), float(np.abs(np.diff(hi)).mean())
        # presence against the local relief: max |h_j - h_{j +- 1}| at the slot
        rel = np.maximum(np.abs(h[self.J0:self.J1] - h[self.J0 - 1:self.J1 - 1]),
                         np.abs(h[self.J0:self.J1] - h[self.J0 + 1:self.J1 + 1]))
        for r in (0, 1, 2):
            sel = rel == r if r < 2 else rel >= 2
            out[f"present_rel{r}"] = float(pres[sel].mean()) if sel.any() else float("nan")
            out[f"frac_rel{r}"] = float(sel.mean())
        # limbs: cells per tree, attached fraction, asymmetry against the slope
        rv = tile.views["root"][tile.grid]
        limb = rv == limbs.LIMB
        out["limb_cells"] = float(limb.sum() / max(pres.sum(), 1))
        total, attached = limbs.components(tile)
        out["attached_frac"] = float(attached / total) if total else 1.0
        asym, cnt = 0.0, 0
        for jj in np.flatnonzero(pres):
            J = self.J0 + jj
            x = J * CH + self.rec_c[g[jj]]
            y0, y1, x0, x1 = self.tile_window(J)
            left = limb[y0:y1, max(x - 8, 0):x].sum()
            right = limb[y0:y1, x + 1:x + 9].sum()
            slope = h[min(J + 1, self.ncol - 1)] - h[max(J - 1, 0)]      # > 0: higher ground to the right
            if slope != 0:
                asym += np.sign(slope) * (right - left)
                cnt += 1
        out["limb_uphill_minus_downhill"] = float(asym / cnt) if cnt else float("nan")
        return out

    def render(self, chans, px=4):
        return limbs.render(self.unpack(chans)["tile"], px)


# ---------------------------------------------------------------- forward
class Forward:
    def __init__(self, T: Treeland, tables=None, seed=0, chans=None, cert=True):
        from .sampler import Sampler
        self.T = T
        self.tables = dict(tables or {})
        self.chans = T.channels() if chans is None else chans
        self.c = T.unpack(self.chans)
        if chans is None:
            T.init_state(self.chans)
        self.rng = np.random.default_rng(seed)
        self.cert = cert
        self._Sampler = Sampler
        self.model = None

    def build(self):
        self.model = self.T.model(self.chans, self.tables, cert=self.cert)
        self.S = dict(surf=self._Sampler(self.model, "surf"), tree=self._Sampler(self.model, "tree"),
                      tile=self._Sampler(self.model, "tile"))

    def run(self, S_surf=30, S_tree=20, S_tile=60, fresh=True):
        T, c = self.T, self.c
        self.build()
        if fresh:
            c["surf"].grid[T.brow] = int(round(T.h0))
            c["tree"].grid[:] = 0
        sd = lambda: int(self.rng.integers(1 << 30))
        self.S["surf"].init()
        if S_surf:
            self.S["surf"].sweep(S_surf, seed=sd())
        self.dormant_tree = self.S["tree"].init()
        if S_tree:
            self.S["tree"].sweep(S_tree, seed=sd())
        T.paint_fine(self.chans, reset=True)
        self.dormant_tile = self.S["tile"].init()
        if S_tile:
            self.S["tile"].sweep(S_tile, seed=sd())
        return T.stats(self.chans)


# ------------------------------------------------------------------ bench
class Bench:
    """The trainer's callbacks for the tree level (child = tile, AIS over the
    slot's limb window with the soft limb model, no certificate) and the
    surf level (child = tree)."""

    def __init__(self, T: Treeland, seed=0, S=(30, 20, 60), tables=None):
        from . import train
        self.T, self.S = T, S
        self.fw = Forward(T, tables=tables, seed=seed + 1)
        self.chans, self.c = self.fw.chans, self.fw.c
        self.md = T.model(self.chans, None, cert=False)               # designed only, soft limb model
        self.tree_feats = train.tabular_feats(TREE_LEARNED)
        self.surf_feats = train.tabular_feats(SURF_LEARNED)
        self.tree_designed_e = train.designed_energies(self.md, "tree")
        self.surf_designed_e = train.designed_energies(self.md, "surf")
        self.fixed = dict(tree=None, surf=None)
        self.rng = np.random.default_rng(seed + 2)

    def tables_of(self, level, theta):
        from . import train
        tabs = dict(self.fw.tables)
        tabs.update(self.T.theta0(level) if theta is None else train.unpack(theta, self.T.shapes(level)))
        return tabs

    def soft_model(self):
        """The forward's model without the certificate, over the same
        channels: what the trainer and AISTargets see (the certificate is
        not annealed; generation keeps it)."""
        return self.T.model(self.chans, self.fw.tables, cert=False)

    def contexts(self, level):
        def gen(theta, n, rng):
            for _ in range(n):
                self.fw.tables = self.tables_of(level, theta)
                self.fw.run(*self.S, fresh=True)
                yield self.soft_model()
        return gen

    # tree level: child tile
    def tree_region(self, model, home, I, J):
        return self.T.tile_window(J)

    def tree_repaint(self, model, home, I, J):
        """tdem (and the trunk cell) for the slot's tile columns."""
        self.T.paint_fine(self.chans, reset=False, cols=(J * CH, (J + 1) * CH))

    # surf level: child tree (the slot and its neighbours within the learned reach)
    def surf_region(self, model, home, I, J):
        return I, I + 1, max(J - 2, 0), min(J + 3, self.T.ncol)

    def surf_repaint(self, model, home, I, J):
        pass
