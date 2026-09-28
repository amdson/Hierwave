"""The average-height support promise (quantities/average.py) as a level
variable, its coordinate coupling cost, table fitting, and the tile stage for
the exemplar-defined target (castlegen.exchain).

The level sampler is envelope_var.EnvelopePromise, which only talks to the
language through its interface; this module supplies what is language
specific.
"""
from __future__ import annotations

import numpy as np

from castlegen import exchain as XC
from castlegen import heights as HT
from castlegen import refheights as RH
from castlegen.envelope_var import EnvelopePromise
from castlegen.promise import Tables
from castlegen.quantities import average as AV


class AveragePromise(EnvelopePromise):
    name = "average"

    def __init__(self, K=16, h_top=64, tables=None, T=1.0, sweeps=3, top_sweeps=10, ground=0, clamps=None, lam=None):
        super().__init__(K, h_top, tables, T, sweeps, top_sweeps, ground, clamps=clamps, lam=lam,
                         lang=AV.lang(K, ground))

    def window_costs(self, ts, E, h, viol_w=1.0, viol_cap=4, bounded=False):
        return window_costs(ts, E, h, self.lang, viol_w, viol_cap, bounded)


def window_costs(ts, E, h, lg, viol_w=1.0, viol_cap=4, bounded=False):
    """(m*m, V) cost of the h x h exemplar window centred on each coordinate
    honouring each value: sum over sub-columns of |window fill - value|
    (fill = floor(mean local height * w / h)) plus viol_w * min(columns with
    unsupported matter, viol_cap).  Torus exemplar, or with bounded the
    exemplar continues its top / bottom rows forever (texsyn bounds="edge")."""
    s = ts.solid[np.asarray(E)].astype(np.int64)
    m = s.shape[0]
    if bounded:                                                 # h rows of continuation cover every window
        st = np.concatenate([np.repeat(s[:1], h, 0), s, np.repeat(s[-1:], h, 0)], 0)
    else:
        st = np.concatenate([s, s, s], 0)
    up = np.zeros_like(st)
    for y in range(st.shape[0]):
        up[y] = st[y] * (1 + (up[y - 1] if y else 0))          # consecutive solids ending at y, going up
    if bounded:
        cs = np.concatenate([np.zeros((1, m), np.int64), np.cumsum(st, 0)], 0)
        c0y = np.arange(m) - h // 2 + h                        # window top row in st
    else:
        up = up[m:2 * m]
        cs = np.concatenate([np.zeros((1, m), np.int64), np.cumsum(np.concatenate([s, s], 0), 0)], 0)
        c0y = (np.arange(m) - h // 2) % m
    c0 = (np.arange(m) - h // 2) % m                            # window top-left for a centre index
    ty, tx = np.meshgrid(c0y, c0, indexing="ij")
    ty, tx = ty.ravel(), tx.ravel()
    cols = (tx[:, None] + np.arange(h)[None, :]) % m            # (m*m, h)
    by = ty + h - 1 if bounded else (ty + h - 1) % m
    t = np.minimum(up[by[:, None], cols], h)                    # local height per window column
    tot = cs[ty[:, None] + h, cols] - cs[ty[:, None], cols]     # solids in the window column
    viol = np.minimum((tot > t).sum(1), viol_cap)
    fill = np.minimum((t.reshape(len(ty), AV.M, h // AV.M).mean(-1) * lg.w / h).astype(np.int64), lg.w)
    cost = np.abs(fill[:, None, :] - lg.SUB[None]).sum(-1) + viol_w * viol[:, None]
    return cost.astype(np.float32)


def mine(ts, maps, K=16, ground=8, h_top=64, lg=None):
    lg = lg or AV.lang(K, ground)
    grids = {}
    for t in maps:
        vals, ok = AV.pyramid(ts, t, K, ground, h_top, lg)
        for h, v in vals.items():
            assert ok[h].all(), "corpus map breaks the support rule"
            grids.setdefault(h, []).append(v)
    return grids


def tables(ts, maps, K=16, ground=8, h_top=64, alpha=0.5, lg=None):
    lg = lg or AV.lang(K, ground)
    return Tables.fit(lg.V, mine(ts, maps, K, ground, h_top, lg), alpha)


# ------------------------------------------------------------------ tile level
class AvgExChain(XC.ExChain):
    """ExChain whose column moves must keep every level-K sub-column sum in
    its bin (the average promise's tile contract)."""

    def __init__(self, ts, tiles, ground, E, P, K, lg, **kw):
        super().__init__(ts, tiles, ground, E, **kw)
        self.K, self.lgA = K, lg
        self.lo, self.hi = AV.intervals(P, K, lg)
        self.sums = AV.sums_K(ts, self.t, K, ground)[0]
        assert ((self.sums >= self.lo) & (self.sums <= self.hi)).all(), "start map breaks the promises"
        self.clamps = {"average": True}                     # makes column_batch call _clamp_check
        self.off = self.n - (np.arange(self.n // K) + 1) * K

    def _col_sums(self, Hc):
        return np.clip(np.asarray(Hc)[..., None] - self.off, 0, self.K)      # (..., R) local heights

    def _clamp_check(self, H, cs, Hn):
        ok = np.ones(len(cs), bool)
        for i, (c, hn) in enumerate(zip(cs, Hn)):
            bx, j = c // self.K, (c % self.K) // self.lgA.w
            new = self.sums[:, bx, j] + self._col_sums(hn) - self._col_sums(H[c])
            ok[i] = ((new >= self.lo[:, bx, j]) & (new <= self.hi[:, bx, j])).all()
        return ok

    def _on_accept(self, cs, old, new):
        super()._on_accept(cs, old, new)
        for k, c in enumerate(cs):
            bx, j = c // self.K, (c % self.K) // self.lgA.w
            h_old = RH.heights(self.ts, old[:, k:k + 1], self.G)[0]
            h_new = RH.heights(self.ts, new[:, k:k + 1], self.G)[0]
            self.sums[:, bx, j] += self._col_sums(h_new) - self._col_sums(h_old)

    def clamps_ok(self, tiles=None):
        s = AV.sums_K(self.ts, self.t if tiles is None else tiles, self.K, self.G)[0]
        return bool(((s >= self.lo) & (s <= self.hi)).all())


def project(ts, X_tex, P, K, ground, lg, passes=20):
    """Feasible heightmap near the texture: start from the concentrated
    construction, then greedy +-1 moves toward the texture's best heights
    that keep every sub-column sum in its bin."""
    n = X_tex.shape[0]
    H = AV.construct(P, K, ground, n, lg)
    target = np.argmin(HT.texture_cost(ts, X_tex, ground), 1)            # (n,) best height per column
    lo, hi = AV.intervals(P, K, lg)
    off = n - (np.arange(n // K) + 1) * K
    loc = lambda h: np.clip(h - off, 0, K)
    sums = np.zeros_like(lo)
    for c in range(n):
        sums[:, c // K, (c % K) // lg.w] += loc(H[c])
    for _ in range(passes):
        moved = 0
        for c in np.random.default_rng(_).permutation(n):
            d = int(np.sign(target[c] - H[c]))
            if d == 0:
                continue
            bx, j = c // K, (c % K) // lg.w
            new = sums[:, bx, j] + loc(H[c] + d) - loc(H[c])
            if ((new >= lo[:, bx, j]) & (new <= hi[:, bx, j])).all():
                sums[:, bx, j] = new
                H[c] += d
                moved += 1
        if not moved:
            break
    x = RH.from_heights(ts, H, n, ground, "stone", "stone")
    keep = ts.solid[X_tex] == ts.solid[x]
    keep[n - ground:] = False
    x = np.where(keep & (X_tex != ts.GATE), X_tex, x).astype(np.int32)
    return x


def tile_stage(ts, X_tex, P, K, ground, E, w, lg, T=0.9, sweeps=50, seed=0, feat="solid"):
    x = project(ts, np.asarray(X_tex, np.int32), P, K, ground, lg)
    ch = AvgExChain(ts, x, ground, E, P, K, lg, T=T, w=w, seed=seed, feat=feat)
    for _ in range(sweeps):
        ch.sweep()
    assert ch.clamps_ok()
    return ch.t.copy()


# ------------------------------------------------------------- learned chi
def window_fill(ts, E, h, lg):
    """(m*m, 2) predicted sub-column values of the h x h exemplar window
    centred on each coordinate (the value the distance chi measures from)."""
    cost = window_costs(ts, E, h, lg, viol_w=0.0)
    v = np.argmin(cost, 1)
    return lg.SUB[v]


def fit_chi(ts, E, records, lg, alpha=1.0, hs=None):
    """Learned chi from the loop's records [(h, coords (R, R, 2), realised
    values (R, R)), ...]: per level, the pointwise mutual information between
    a block's predicted sub-column value (from its coordinate's exemplar
    window) and the value it realised, pooled over the two sub-columns,
        chi_h(u, v) = - sum_j [log p(v_j | pred_j(u)) - log p(v_j)],
    as an (m*m, V) cost table like window_costs (additive smoothing alpha)."""
    m = np.asarray(E).shape[0]
    P = lg.P
    counts = {}
    for h, S, vals in records:
        if hs is not None and h not in hs:
            continue
        pred = window_fill(ts, E, h, lg)[S[..., 0] * m + S[..., 1]]          # (R, R, 2)
        real = lg.SUB[vals]                                                 # (R, R, 2)
        c = counts.setdefault(h, np.zeros((P, P)))
        np.add.at(c, (pred.ravel(), real.ravel()), 1)
    out = {}
    for h, c in counts.items():
        pc = (c + alpha) / (c + alpha).sum(1, keepdims=True)                # p(real | pred)
        pr = (c.sum(0) + alpha) / (c.sum() + P * alpha)                     # p(real)
        pmi = np.log(pc) - np.log(pr)[None, :]                              # (pred, real)
        fill = window_fill(ts, E, h, lg)                                    # (m*m, 2)
        S = lg.SUB
        out[h] = -(pmi[fill[:, None, 0], S[None, :, 0]] + pmi[fill[:, None, 1], S[None, :, 1]]).astype(np.float32)
    return out, counts


def linear_tables(ts, maps, K=16, ground=8, h_top=64, l2=1e-2, lg=None, log=print):
    """Tables from the linear predictor (castlegen.envpredict) fitted on
    corpus maps."""
    from castlegen import envpredict
    lg = lg or AV.lang(K, ground)
    return envpredict.to_tables(envpredict.fit(mine(ts, maps, K, ground, h_top, lg), lg, ground, l2, log=log), lg)
