"""JAX port of the generic local conditional (castlegen.variants._local.cond)
plus a one-step lookahead.  Shared helper for variants; not frozen.

    cj = CondJax(problem, W, periodic)
    soft, nv, look = cj(state, chan, Y, X)

state: {channel: int32 (H >> l, W >> l)} for every channel, -1 = unknown.
chan static; Y, X int32 (n,) level coordinates (X in region coordinates; may
be out of range: indices are clipped, the caller masks).  Pure jnp on one
world: jit / vmap over a leading world axis of the state arrays.

soft (n, D) float32  summed finite entries of every instance touching the site
                     whose other members are all known (pads count as known;
                     off-window / off-strip without pad: no instance)
nv   (n, D) int32    inf entries among those instances
look (n, D) int32    over instances touching the site with >= 1 unknown other
                     member: 1 if no completion of the unknown members (each
                     independently over its channel's domain, through its
                     view) makes the entry finite.  A count instance whose
                     parent is unknown while the site is a child adds 0.

All tables are pre-arranged in __init__ so that __call__ only gathers rows of
shape (n, D)."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def _reach_sums(reach, m_max):
    """R[m] = boolean over sums reachable by m members each taking a value in
    `reach` (m-fold Minkowski sum), m = 0..m_max."""
    top = m_max * int(reach.max()) + 1
    R = np.zeros((m_max + 1, top), bool)
    R[0, 0] = True
    for m in range(1, m_max + 1):
        for r in reach:
            R[m, r:] |= R[m - 1, :top - r]
    return R


def _feasible(table, R):
    """G[m, vb, j] = some s in R[m] with table[vb, j + s] finite."""
    fin = np.isfinite(table)
    Vb, S = table.shape
    G = np.zeros((R.shape[0], Vb, S), bool)
    for m in range(R.shape[0]):
        for s in np.flatnonzero(R[m]):
            if s < S:
                G[m, :, :S - s] |= fin[:, s:]
    return G


def _split(M):
    """finite part (inf -> 0) float32, inf indicator int32."""
    inf = np.isinf(M)
    return jnp.asarray(np.where(inf, 0.0, M), jnp.float32), jnp.asarray(inf, jnp.int32)


class CondJax:
    def __init__(self, problem, W, periodic):
        self.p = problem
        self.H, self.W, self.periodic = problem.H, int(W), bool(periodic)
        self.view = {}
        for c in problem.chans:
            for v, arr in c.views.items():
                self.view[(c.name, v)] = jnp.asarray(arr, jnp.int32)
        self.terms = {c.name: self._build(c) for c in problem.chans}

    # ------------------------------------------------------------ precompute
    def _reach(self, cv):
        return np.unique(self.p.chan[cv[0]].views[cv[1]])

    def _build(self, c):
        p, name = self.p, c.name
        terms = []
        for f in p.factors:
            la = p.chan[f.a[0]].level
            tab = f.table
            if f.kind == "unary":
                if f.a[0] == name:
                    terms.append(("unary", _split(tab[c.views[f.a[1]]])))
                continue
            lb = p.chan[f.b[0]].level
            if f.kind == "pair":
                fin = np.isfinite(tab)
                if f.a[0] == name:                       # site is a
                    ta = c.views[f.a[1]]
                    rb = self._reach(f.b)
                    if f.by_pos:
                        M = tab[:, ta, :].transpose(0, 2, 1)             # (4, Vb, D)
                        L = ~fin[:, ta][:, :, rb].any(-1)                # (4, D)
                    else:
                        M = tab[ta, :].T                                 # (Vb, D)
                        L = ~fin[ta][:, rb].any(-1)                      # (D,)
                    terms.append(("pair_a", f, la, lb, _split(M), jnp.asarray(L, jnp.int32)))
                if f.b[0] == name:                       # site is b
                    tb = c.views[f.b[1]]
                    ra = self._reach(f.a)
                    if lb == la:
                        M = tab[:, tb]                                   # (Va, D)
                        L = ~fin[ra][:, tb].any(0)                       # (D,)
                        terms.append(("pair_b", f, la, lb, _split(M), jnp.asarray(L, jnp.int32)))
                    else:
                        t4 = tab if f.by_pos else np.broadcast_to(tab, (4,) + tab.shape)
                        M = t4[:, :, tb]                                 # (4, Va, D)
                        L = ~(np.isfinite(t4)[:, ra][:, :, tb]).any(1)   # (4, D)
                        terms.append(("pair_par", f, la, lb, _split(M), jnp.asarray(L, jnp.int32)))
            elif f.kind == "count":
                ns = len(f.subset)
                R = _reach_sums(self._reach(f.a), ns)
                G = _feasible(tab, R)                                    # (ns + 1, Vb, S)
                S = tab.shape[1]
                if f.a[0] == name:                       # site is a child
                    ta = jnp.asarray(c.views[f.a[1]], jnp.int32)
                    Tf, Ti = _split(tab)
                    terms.append(("count_child", f, la, lb, (Tf, Ti), jnp.asarray(~G, jnp.int32), ta, S))
                if f.b[0] == name:                       # site is the parent
                    tb = c.views[f.b[1]]
                    M = tab[tb, :].T                                     # (S, D)
                    L = ~G[:, tb, :].transpose(0, 2, 1)                  # (ns + 1, S, D)
                    terms.append(("count_par", f, la, lb, _split(M), jnp.asarray(L, jnp.int32), S))
        return terms

    # ------------------------------------------------------------ gathers
    def _read(self, state, cv, Y, X, pad):
        """(view value, exists, known) of channel/view cv at (Y, X)."""
        c = self.p.chan[cv[0]]
        h, w = self.H >> c.level, self.W >> c.level
        arr = state[c.name]
        if self.periodic:
            Xr = jnp.mod(X, w)
            exists = jnp.ones(Y.shape, bool)
        else:
            exists = (X >= 0) & (X < w)
            Xr = jnp.clip(X, 0, w - 1)
        off = (Y < 0) | (Y >= h)
        raw = arr.reshape(-1)[jnp.clip(Y, 0, h - 1) * w + Xr]
        v = self.view[cv][jnp.maximum(raw, 0)]
        known = raw >= 0
        if pad >= 0:
            v = jnp.where(off, pad, v)
            known = known | off
        else:
            exists = exists & ~off
        return v, exists, known

    @staticmethod
    def _within(f, la, Ya, Xa, lb, Yb, Xb):
        if f.within is None:
            return True
        L = f.within
        return (((Ya << la) >> L) == ((Yb << lb) >> L)) & (((Xa << la) >> L) == ((Xb << lb) >> L))

    # ------------------------------------------------------------ call
    def __call__(self, state, chan, Y, X):
        D = self.p.chan[chan].D
        Y = jnp.asarray(Y, jnp.int32)
        X = jnp.asarray(X, jnp.int32)
        n = Y.shape[0]
        soft = jnp.zeros((n, D), jnp.float32)
        nv = jnp.zeros((n, D), jnp.int32)
        look = jnp.zeros((n, D), jnp.int32)

        def add(soft, nv, look, Ms, Mi, Lk, exists, known):
            closed = (exists & known)[:, None]
            soft = soft + jnp.where(closed, Ms, 0.0)
            nv = nv + jnp.where(closed, Mi, 0)
            if Lk is not None:
                look = look + jnp.where((exists & ~known)[:, None], Lk, 0)
            return soft, nv, look

        for term in self.terms[chan]:
            kind = term[0]
            if kind == "unary":
                Uf, Ui = term[1]
                soft = soft + Uf[None, :]
                nv = nv + Ui[None, :]
                continue
            f, la, lb = term[1], term[2], term[3]
            dy, dx = f.off
            if kind == "pair_a":
                (Mf, Mi), L = term[4], term[5]
                if lb == la:
                    Yb, Xb = Y + dy, X + dx
                else:
                    Yb, Xb = (Y >> 1) + dy, (X >> 1) + dx
                vb, ex, kn = self._read(state, f.b, Yb, Xb, f.pad_b)
                ex = ex & self._within(f, la, Y, X, lb, Yb, Xb)
                if f.by_pos:
                    k = 2 * (Y & 1) + (X & 1)
                    soft, nv, look = add(soft, nv, look, Mf[k, vb], Mi[k, vb], L[k], ex, kn)
                else:
                    soft, nv, look = add(soft, nv, look, Mf[vb], Mi[vb], L[None, :], ex, kn)
            elif kind == "pair_b":
                (Mf, Mi), L = term[4], term[5]
                Ya, Xa = Y - dy, X - dx
                va, ex, kn = self._read(state, f.a, Ya, Xa, f.pad_a)
                ex = ex & self._within(f, la, Ya, Xa, lb, Y, X)
                soft, nv, look = add(soft, nv, look, Mf[va], Mi[va], L[None, :], ex, kn)
            elif kind == "pair_par":
                (Mf, Mi), L = term[4], term[5]
                Yp, Xp = Y - dy, X - dx
                hp = self.H >> lb
                inrow = (Yp >= 0) & (Yp < hp)
                for i in (0, 1):
                    for j in (0, 1):
                        Ya, Xa = 2 * Yp + i, 2 * Xp + j
                        va, ex, kn = self._read(state, f.a, Ya, Xa, -1)
                        ex = ex & inrow & self._within(f, la, Ya, Xa, lb, Y, X)
                        k = 2 * i + j
                        soft, nv, look = add(soft, nv, look, Mf[k][va], Mi[k][va], L[k][None, :], ex, kn)
            elif kind == "count_child":
                (Tf, Ti), NG, ta, S = term[4], term[5], term[6], term[7]
                k = 2 * (Y & 1) + (X & 1)
                Yq, Xq = Y >> 1, X >> 1
                vb, ex, knp = self._read(state, f.b, Yq, Xq, -1)
                insub = jnp.zeros(Y.shape, bool)
                for kk in f.subset:
                    insub = insub | (k == kk)
                ex = ex & insub & knp                    # unknown parent: nothing at all
                s = jnp.zeros(Y.shape, jnp.int32)
                m = jnp.zeros(Y.shape, jnp.int32)
                for kk in f.subset:
                    vc, exc, knc = self._read(state, f.a, 2 * Yq + (kk >> 1), 2 * Xq + (kk & 1), -1)
                    mine = k == kk
                    ex = ex & (exc | mine)
                    s = s + jnp.where(mine | ~knc, 0, vc)
                    m = m + (~mine & ~knc).astype(jnp.int32)
                idx = jnp.minimum(s[:, None] + ta[None, :], S - 1)
                vbc = vb[:, None]
                closed = (ex & (m == 0))[:, None]
                soft = soft + jnp.where(closed, Tf[vbc, idx], 0.0)
                nv = nv + jnp.where(closed, Ti[vbc, idx], 0)
                look = look + jnp.where((ex & (m > 0))[:, None], NG[m[:, None], vbc, idx], 0)
            elif kind == "count_par":
                (Mf, Mi), L, S = term[4], term[5], term[6]
                ex = jnp.ones(Y.shape, bool)
                s = jnp.zeros(Y.shape, jnp.int32)
                m = jnp.zeros(Y.shape, jnp.int32)
                for kk in f.subset:
                    vc, exc, knc = self._read(state, f.a, 2 * Y + (kk >> 1), 2 * X + (kk & 1), -1)
                    ex = ex & exc
                    s = s + jnp.where(knc, vc, 0)
                    m = m + (~knc).astype(jnp.int32)
                s = jnp.minimum(s, S - 1)
                closed = (ex & (m == 0))[:, None]
                soft = soft + jnp.where(closed, Mf[s], 0.0)
                nv = nv + jnp.where(closed, Mi[s], 0)
                look = look + jnp.where((ex & (m > 0))[:, None], L[m, s], 0)
        return soft, nv, look
