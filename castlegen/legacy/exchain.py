"""Legacy (G1b, coarse-to-fine promises era); superseded by castlegen/channels. See notes/history/coarse_to_fine.tex.

Exemplar-defined target: the tile set's energy plus a multi-scale patch
energy against the exemplar, under the support rule (notes/history/coarse_to_fine.tex,
option (a) of the training discussion: no learned target terms).

    p(x, S) ∝ exp(-E_pair(x)/T + sum_i logz(x_i) - E_ex(x, S)/T) 1[G(x)]

    E_ex(x, S) = sum_h w_h sum_s || N_h(x)(s) - N_E^h(S_h(s)) ||^2

Features are structural only: B_h = the solidity of x blurred with a Gaussian
of sigma h/2 (torus horizontally; vertically air continues above the map and
stone below it, since side-view terrain is not vertically periodic), sampled at the centres of the h x h cells of a coarse
grid; N_h(x)(s) is the 5 x 5 neighbourhood of coarse samples around s
(spacing h).  N_E^h(u) is the same neighbourhood around exemplar position u
(blurred exemplar solidity with the same boundary rule), for every column and
for centre rows from 4h above to 4h below the exemplar.  S_h is a latent exemplar
coordinate per coarse cell, as in texsyn; marginalising it turns the patch
distance into a soft-min, so p(x) is a proper distribution without a
nearest-neighbour search inside the chain.  Materials only enter through
the tile set's energy.

feat="tiles" replaces the solidity by the tile map itself, as a one-hot of
the tile signature scaled by 1/sqrt(2) (so air vs stone costs what it did
with solidity), the categorical form of the image the original texture
synthesis matches: every channel is blurred and matched the same way, the
boundary rows are the air and stone features, and material moves then change
E_ex too, so they include its exact change (`material_sweep`).

Moves (each leaves p(x, S) invariant):
  materials  class-preserving checkerboard heat-bath (refheights.Chain.site_half);
  heights    refheights' column MH moves, one column at a time (patch terms
             couple columns within a footprint), with the exact change of
             E_ex added to the acceptance;
  S          exact heat-bath of every S_h(s) over all exemplar positions
             (given x the S_h(s) are independent), every `s_every` sweeps.

Change of E_ex for a change dB of the coarse samples (exact, E_ex is
quadratic in B):  dE_h = w_h sum_s [ 2 (c(s) B(s) - Q(s)) dB(s) + c(s) dB(s)^2 ],
c(s) = the number of real neighbourhoods sample s lies in (25 inside, fewer
on the virtual rows above and below the map), Q(s) = sum over those of the
matched exemplar value, updated whenever S_h changes.
"""
from __future__ import annotations

import numpy as np

from castlegen.legacy import refheights as RH

OFF = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)]
VPAD = 2                                   # virtual coarse rows above and below the map


def _gauss(d, h):
    sig = h / 2.0
    return np.where(np.abs(d) <= 3 * sig, np.exp(-d ** 2 / (2 * sig ** 2)), 0.0)


def _cols_kernel(n_samples, n, h, offset):
    """(n_samples, n) horizontal weights, torus, rows sum to 1."""
    c = np.arange(n_samples)[:, None] * h + offset
    d = (np.arange(n)[None, :] - c + n / 2) % n - n / 2
    k = _gauss(d, h)
    return k / k.sum(1, keepdims=True)


def _rows_kernel(centres, n, h, above=False):
    """Vertical weights, not periodic: air continues above row 0 and stone
    below row n-1.  -> (weights (len(centres), n) on the real rows, constant
    (len(centres),) = the weight of the virtual stone rows below[, and of the
    virtual air rows above])."""
    r = int(np.ceil(1.5 * h)) + 1
    centres = np.asarray(centres, float)
    ys = np.arange(min(-r - 1, int(np.floor(centres.min())) - r - 1), max(n + r + 1, int(np.ceil(centres.max())) + r + 2))
    k = _gauss(ys[None, :] - np.asarray(centres, float)[:, None], h)
    k = k / k.sum(1, keepdims=True)
    real = (ys >= 0) & (ys < n)
    if above:
        return k[:, real], k[:, ys >= n].sum(1), k[:, ys < 0].sum(1)
    return k[:, real], k[:, ys >= n].sum(1)


def tile_features(ts, feat="solid"):
    """(n_sig, D) feature of every tile signature, and the (D,) features of
    the air above and the stone below the map."""
    if feat == "solid":
        Phi = np.asarray(ts.solid, float)[:, None]
        return Phi, np.zeros(1), np.ones(1)
    assert feat == "tiles", feat
    Phi = np.eye(ts.n_sig) / np.sqrt(2.0)
    return Phi, Phi[RH._sig(ts, "air")], Phi[RH._sig(ts, "stone")]


def exemplar_patches(ts, E, h, feat="solid"):
    """(P, 25 D) neighbourhoods of the blurred exemplar features at spacing h,
    for every exemplar column and every centre row from 4h above the
    exemplar to 4h below it (air above, stone below, torus horizontally), so
    pure-sky and pure-ground patches are available.  Layout (offset, channel)."""
    Phi, f_air, f_stone = tile_features(ts, feat)
    s = Phi[np.asarray(E)]                                            # (m, W, D)
    m, W = s.shape[:2]
    rows = np.arange(-4 * h - 2 * h, m + 4 * h + 2 * h + 1)          # every row a patch sample can sit on
    Ry, c0, c1 = _rows_kernel(rows, m, h, above=True)
    K = np.array([np.roll(_gauss((np.arange(W) + W / 2) % W - W / 2, h), i) for i in range(W)])
    K = K / K.sum(1, keepdims=True)
    B = np.einsum("ry,yxd,cx->rcd", Ry, s, K) + (c0[:, None] * f_stone + c1[:, None] * f_air)[:, None, :]
    off = 2 * h
    cen = np.arange(off, len(rows) - off)                            # centre rows with a full neighbourhood
    cols = [np.roll(B[cen + dy * h], -dx * h, 1).reshape(-1, B.shape[-1]) for dy, dx in OFF]
    return np.stack(cols, 1).reshape(len(cols[0]), -1)


class ExChain(RH.Chain):
    """refheights.Chain plus the exemplar patch energy (module docstring).
    w: {h: weight}; by default lam * h^2 / 25 per scale in `scales`, so every
    scale contributes about lam per unit of blurred solidity mismatch per cell.
    Features are not periodic vertically (air above the map, stone below)."""

    def __init__(self, ts, tiles, ground, E, T=0.9, lam=4.0, scales=(4, 8, 16), w=None, s_every=5, seed=0,
                 feat="solid", **kw):
        super().__init__(ts, tiles, ground, T=T, seed=seed, **kw)
        self.E, self.s_every = np.asarray(E), int(s_every)
        self.feat = feat
        self.Phi, self.f_air, self.f_stone = tile_features(ts, feat)
        self.D = self.Phi.shape[1]
        self.w = dict(w) if w is not None else {h: lam * h * h / 25.0 for h in scales}
        self.sc = {}
        for h, wh in self.w.items():
            assert self.n % h == 0 and self.W % h == 0
            R, C = self.n // h, self.W // h
            Ry, c0, c1 = _rows_kernel(np.arange(-VPAD, R + VPAD) * h + (h - 1) / 2, self.n, h, above=True)
            Cx = _cols_kernel(C, self.W, h, (h - 1) / 2)
            NE = exemplar_patches(ts, self.E, h, feat)
            cv = np.array([sum(0 <= r - VPAD - dy < R for dy in range(-2, 3)) for r in range(R + 2 * VPAD)])
            self.sc[h] = dict(w=wh, R=R, Ry=Ry, Cx=Cx, NE=NE, NE2=(NE ** 2).sum(1),
                              c0=c0[:, None, None] * self.f_stone + c1[:, None, None] * self.f_air,
                              cnt=(5.0 * cv)[:, None, None],
                              qy=(5.0 * cv)[:, None] * Ry ** 2, qx=Cx ** 2)
        self._refresh()
        self._sweeps = 0
        for h in self.sc:
            self.resample_S(h)

    # ---- bookkeeping
    def _refresh(self):
        s = self.Phi[self.t]                                              # (n, W, D)
        for h, c in self.sc.items():
            c["B"] = np.einsum("ry,yxd,cx->rcd", c["Ry"], s, c["Cx"]) + c["c0"]   # (R + 2 VPAD, C, D)

    def _patches(self, h):
        """(R, C, 25 D) neighbourhoods of the real coarse centres (layout of exemplar_patches)."""
        c = self.sc[h]
        B, R = c["B"], c["R"]
        P = np.stack([np.roll(B[VPAD + dy:VPAD + dy + R], -dx, 1) for dy, dx in OFF], 2)   # (R, C, 25, D)
        return P.reshape(P.shape[0], P.shape[1], -1)

    def _Q(self, h):
        c = self.sc[h]
        R = c["R"]
        A = c["NE"][c["S"]].reshape(R, -1, len(OFF), self.D)                          # (R, C, 25, D)
        Q = np.zeros_like(c["B"])
        for k, (dy, dx) in enumerate(OFF):
            Q[VPAD + dy:VPAD + dy + R] += np.roll(A[:, :, k], dx, 1)
        return Q

    def energy_ex(self):
        """E_ex(x, S) (not divided by T)."""
        return sum(c["w"] * float(((self._patches(h) - c["NE"][c["S"]]) ** 2).sum()) for h, c in self.sc.items())

    def resample_S(self, h, T=None):
        """Exact heat-bath of S_h given x."""
        T = self.T if T is None else T
        c = self.sc[h]
        N = self._patches(h).reshape(-1, len(OFF) * self.D)
        e = c["w"] * ((N ** 2).sum(1)[:, None] - 2 * N @ c["NE"].T + c["NE2"][None, :])
        g = self.rng.gumbel(size=e.shape)
        c["S"] = np.argmin(e / T - g, 1).reshape(c["R"], -1)
        c["Q"] = self._Q(h)

    # ---- hooks into refheights.Chain.column_batch
    def _dB(self, c, cs, old, new):
        ds = self.Phi[new] - self.Phi[old]                                            # (n, m, D)
        return [np.einsum("rd,c->rcd", c["Ry"] @ ds[:, j], c["Cx"][:, cs[j]]) for j in range(len(cs))]

    def _extra_logw(self, cs, old, new, T):
        out = np.zeros(len(cs))
        for h, c in self.sc.items():
            G2 = 2 * (c["cnt"] * c["B"] - c["Q"])
            for j, dB in enumerate(self._dB(c, cs, old, new)):
                out[j] -= c["w"] * float((G2 * dB).sum() + (c["cnt"] * dB ** 2).sum()) / T
        return out

    def _on_accept(self, cs, old, new):
        for h, c in self.sc.items():
            for dB in self._dB(c, cs, old, new):
                c["B"] += dB

    # ---- sweep
    def column_sweep(self, T=None):
        acc = 0
        for c in self.rng.permutation(self.W):          # one column at a time: patch terms couple columns
            acc += self.column_batch(np.array([c]), T)
        return acc / self.W

    def material_sweep(self, T=None):
        """Heat-bath of every cell's material (same solidity class), with the
        exact change of E_ex: a cell's change touches coarse samples within
        1.5 h of it, so cells more than 3 h_max apart in both axes are
        updated together exactly."""
        T = self.T if T is None else T
        t, n, W, G = self.t, self.n, self.W, self.G
        st = 3 * max(self.sc) + 1
        for oy, ox in self.rng.permutation([(a, b) for a in range(st) for b in range(st)]):
            ys, xs = np.meshgrid(np.arange(oy, n - G, st), np.arange(ox, W, st), indexing="ij")     # band stays
            ys, xs = ys.ravel(), xs.ravel()
            if not len(ys):
                continue
            cur = t[ys, xs]
            L, R = t[ys, (xs - 1) % W], t[ys, (xs + 1) % W]
            U, Dn = t[(ys - 1) % n, xs], t[ys + 1, xs]
            E = self.Eh[L] + self.Eh[:, R].T + self.Ev[U] + self.Ev[:, Dn].T                    # as site_half
            logits = -E / T + self.base
            below_es = self.solid[Dn] | (ys + 1 >= n - G)
            air_ok = (ys == 0) | ~self.solid[U]
            ok = np.where(self.solid[None, :], below_es[:, None], air_ok[:, None])
            ok &= self.solid[None, :] == self.solid[cur][:, None]
            dphi = self.Phi[None, :, :] - self.Phi[cur][:, None, :]                   # (k, n_sig, D)
            for h, c in self.sc.items():
                G2 = 2 * (c["cnt"] * c["B"] - c["Q"])                                 # (Rr, C, D)
                lin = np.einsum("rk,rcd,ck->kd", c["Ry"][:, ys], G2, c["Cx"][:, xs])  # (k, D)
                quad = c["qy"][:, ys].sum(0) * c["qx"][:, xs].sum(0)                  # (k,)
                logits = logits - c["w"] * (np.einsum("kd,ksd->ks", lin, dphi)
                                            + quad[:, None] * (dphi ** 2).sum(-1)) / T
            logits = np.where(ok, logits, -np.inf)
            new = RH._gumbel_argmax(logits, self.rng)
            ch = new != cur
            if ch.any():
                for h, c in self.sc.items():
                    d = self.Phi[new[ch]] - self.Phi[cur[ch]]                          # (k', D)
                    c["B"] += np.einsum("rk,ck,kd->rcd", c["Ry"][:, ys[ch]], c["Cx"][:, xs[ch]], d)
                t[ys[ch], xs[ch]] = new[ch]

    def sweep(self, T=None, columns=True):
        if self.feat == "solid":
            for p in self.rng.permutation(2):
                self.site_half(p, T, keep_class=True)    # materials only; heights move by column moves
        else:
            self.material_sweep(T)                       # materials change E_ex too
        a = self.column_sweep(T)
        self._sweeps += 1
        if self._sweeps % self.s_every == 0:
            for h in self.sc:
                self.resample_S(h, T)
        return a


def tile_stage(ts, X_tex, P, K, ground, E, w, T=0.9, sweeps=50, seed=0, I=2, Q=2, feat="solid"):
    """Tile level for the exemplar-defined target: project the texture onto
    the maps whose K-blocks keep the promise grid P (heights.project), then
    run ExChain with every K-block clamped to P, so the chain samples the
    same target as the correction step, conditioned on the promises.
    Heights move only by column moves (no witness swaps), so for small
    intervals the chain can be slow to move the witness columns."""
    from castlegen.legacy import heights
    x, _ = heights.project(ts, X_tex, P, K, ground, I=I, Q=Q, seed=seed, T=T)
    P = np.asarray(P)
    clamps = {(K, by, bx): int(P[by, bx]) for by in range(P.shape[0]) for bx in range(P.shape[1])}
    ch = ExChain(ts, x, ground, E, T=T, w=w, seed=seed, clamps=clamps, feat=feat)
    for _ in range(sweeps):
        ch.sweep()
    return ch.t.copy()
