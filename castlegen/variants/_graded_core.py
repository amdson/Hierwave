"""cgpen training core: cfast's penalty-only update (pen_only, change request #2) with the GRADED
violation mass in place of the binary count (ideas #21; constitution addendum 2026-10-14 (c), training
only).  Objective

    J_gpen = sum_l lam_l E_q[D_l(x)],      D_l = sum of per-instance distances d_f (castlegen.graded),

estimated exactly as cfast's pen_only estimator estimates sum_l lam_l E_q[H_l]: per draw the cost is
lam_l * nvg(a), where nvg(a) is the summed graded distance of the hard instances that this draw CLOSES
(every other member known) when it takes value a, so the draw costs of a world sum to lam . D(x) (cost
identity, tested).  Cone returns, the critic as control variate, the all-action + score-function terms
and the lambda control (driven by the BINARY raw violation rate, unchanged) are cfast's.  The draws
themselves are cfast's: the least-violating set is still argmin of the binary closed violations plus
lookahead, so generation is unchanged.

CondGraded extends _cond_jax.CondJax: beside (soft, nv, look) it returns nvg (n, D), gathered from the
per-factor distance tables of castlegen.graded (L1 to the nearest feasible entry; |.| on monotone-band
axes, Hamming elsewhere; a pad read or a designed channel is a fixed cell: the other side carries the
distance, as in graded.graded).  train_pass_graded is _cfast_core.train_pass (fast path) recording
"nvg" and costing lam * nvg; build_train is cfast's _build_train restricted to K = 0 (no coupled
replays), with ell = lam_l * nvg."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from castlegen import graded as gr
from castlegen.bench import noise

from . import _cpass_core as core
from ._cfast_cone import Cone
from ._cfast_core import _cols, _encode, _geom, fast_features
from ._cln_core import hidden1
from ._cond_jax import CondJax
from ._cpass_core import NCOL, key_of, mlp
from .cpass import adam_step, clip_by_level


def _f32(M):
    return jnp.asarray(np.asarray(M, np.float64), jnp.float32)


class CondGraded(CondJax):
    """CondJax plus the closed graded distance per candidate value.

    __call__(state, chan, Y, X) -> (soft, nv, look, nvg); the first three are CondJax's."""

    def __init__(self, problem, W, periodic):
        super().__init__(problem, W, periodic)
        self.fds = gr.distances(problem)
        self.gterms = {c.name: self._build_graded(c) for c in problem.chans}

    def _build_graded(self, c):
        """One entry per CondJax term of channel c (same order): None for a soft factor, else the
        distance tables gathered like that term's inf-indicator table (plus the fixed-side table)."""
        p, name = self.p, c.name
        out = []
        for i, f in enumerate(p.factors):
            fd = self.fds.get(i)
            la = p.chan[f.a[0]].level
            if f.kind == "unary":
                if f.a[0] == name:
                    out.append(None if fd is None else _f32(fd.d[c.views[f.a[1]]]))
                continue
            lb = p.chan[f.b[0]].level
            a_des = p.chan[f.a[0]].designed is not None
            b_des = p.chan[f.b[0]].designed is not None
            if f.kind == "pair":
                if f.a[0] == name:                       # site is a; reads b (maybe a pad / designed)
                    if fd is None:
                        out.append(None)
                    else:
                        ta = c.views[f.a[1]]
                        both = fd.d_a if b_des else fd.d
                        if f.by_pos:
                            M = both[:, ta, :].transpose(0, 2, 1); Mfix = fd.d_a[:, ta, :].transpose(0, 2, 1)
                        else:
                            M = both[ta, :].T; Mfix = fd.d_a[ta, :].T
                        out.append((_f32(M), _f32(Mfix)))
                if f.b[0] == name:                       # site is b; reads a
                    if fd is None:
                        out.append(None)
                    elif lb == la:
                        tb = c.views[f.b[1]]
                        both = fd.d_b if a_des else fd.d
                        out.append((_f32(both[:, tb]), _f32(fd.d_b[:, tb])))
                    else:
                        tb = c.views[f.b[1]]
                        both = fd.d_b if a_des else fd.d
                        d4 = both if f.by_pos else np.broadcast_to(both, (4,) + both.shape)
                        out.append((_f32(d4[:, :, tb]), None))
            elif f.kind == "count":
                if f.a[0] == name:                       # site is a child: table over (vb, s)
                    out.append(None if fd is None else _f32(fd.d_a if b_des else fd.d))
                if f.b[0] == name:                       # site is the parent: (S, D)
                    if fd is None:
                        out.append(None)
                    else:
                        tb = c.views[f.b[1]]
                        both = fd.d_b if a_des else fd.d
                        out.append(_f32(both[tb, :].T))
        assert len(out) == len(self.terms[name]), (name, len(out), len(self.terms[name]))
        return out

    def _off(self, cv, Y):
        c = self.p.chan[cv[0]]
        h = self.H >> c.level
        return (Y < 0) | (Y >= h)

    def __call__(self, state, chan, Y, X):
        D = self.p.chan[chan].D
        Y = jnp.asarray(Y, jnp.int32)
        X = jnp.asarray(X, jnp.int32)
        n = Y.shape[0]
        soft = jnp.zeros((n, D), jnp.float32)
        nv = jnp.zeros((n, D), jnp.int32)
        look = jnp.zeros((n, D), jnp.int32)
        nvg = jnp.zeros((n, D), jnp.float32)

        def add(soft, nv, look, nvg, Ms, Mi, Lk, Md, exists, known):
            closed = (exists & known)[:, None]
            soft = soft + jnp.where(closed, Ms, 0.0)
            nv = nv + jnp.where(closed, Mi, 0)
            if Md is not None:
                nvg = nvg + jnp.where(closed, Md, 0.0)
            if Lk is not None:
                look = look + jnp.where((exists & ~known)[:, None], Lk, 0)
            return soft, nv, look, nvg

        for term, gt in zip(self.terms[chan], self.gterms[chan]):
            kind = term[0]
            if kind == "unary":
                Uf, Ui = term[1]
                soft = soft + Uf[None, :]
                nv = nv + Ui[None, :]
                if gt is not None:
                    nvg = nvg + gt[None, :]
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
                Md = None
                if gt is not None:
                    Mboth, Mfix = gt
                    padded = (self._off(f.b, Yb) & (f.pad_b >= 0))[:, None]
                    if f.by_pos:
                        k = 2 * (Y & 1) + (X & 1)
                        Md = jnp.where(padded, Mfix[k, vb], Mboth[k, vb])
                    else:
                        Md = jnp.where(padded, Mfix[vb], Mboth[vb])
                if f.by_pos:
                    k = 2 * (Y & 1) + (X & 1)
                    soft, nv, look, nvg = add(soft, nv, look, nvg, Mf[k, vb], Mi[k, vb], L[k], Md, ex, kn)
                else:
                    soft, nv, look, nvg = add(soft, nv, look, nvg, Mf[vb], Mi[vb], L[None, :], Md, ex, kn)
            elif kind == "pair_b":
                (Mf, Mi), L = term[4], term[5]
                Ya, Xa = Y - dy, X - dx
                va, ex, kn = self._read(state, f.a, Ya, Xa, f.pad_a)
                ex = ex & self._within(f, la, Ya, Xa, lb, Y, X)
                Md = None
                if gt is not None:
                    Mboth, Mfix = gt
                    padded = (self._off(f.a, Ya) & (f.pad_a >= 0))[:, None]
                    Md = jnp.where(padded, Mfix[va], Mboth[va])
                soft, nv, look, nvg = add(soft, nv, look, nvg, Mf[va], Mi[va], L[None, :], Md, ex, kn)
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
                        Md = None if gt is None else gt[0][k][va]
                        soft, nv, look, nvg = add(soft, nv, look, nvg, Mf[k][va], Mi[k][va], L[k][None, :], Md, ex, kn)
            elif kind == "count_child":
                (Tf, Ti), NG, ta, S = term[4], term[5], term[6], term[7]
                k = 2 * (Y & 1) + (X & 1)
                Yq, Xq = Y >> 1, X >> 1
                vb, ex, knp = self._read(state, f.b, Yq, Xq, -1)
                insub = jnp.zeros(Y.shape, bool)
                for kk in f.subset:
                    insub = insub | (k == kk)
                ex = ex & insub & knp
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
                if gt is not None:
                    nvg = nvg + jnp.where(closed, gt[vbc, idx], 0.0)
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
                if gt is not None:
                    nvg = nvg + jnp.where(closed, gt[s], 0.0)
                look = look + jnp.where((ex & (m > 0))[:, None], L[m, s], 0)
        return soft, nv, look, nvg


def cond_graded(lay):
    """The layout's CondGraded (built once per layout object)."""
    if "cond_g" not in lay.__dict__:
        lay.cond_g = CondGraded(lay.p, lay.W, lay.periodic)
    return lay.cond_g


def train_pass_graded(lay, params, seed, designed, tau, lam, override=None, record=True):
    """_cfast_core.train_pass (fast path; same draws, features, noise) recording also "nvg" (n, D), the
    closed graded distance per candidate, with cost = lam_l * nvg[a] (the penalty-only graded cost).
    Returns (state, recs) with recs[key] = {feat, soft, nv, nvg, mask, a, valid, logq, cost}, or
    (state, costs) when not record."""
    assert not lay.resid, "cgpen: count-residual layouts are not supported"
    p = lay.p
    cg = cond_graded(lay)
    state = {c.name: (designed[c.name] if c.designed is not None
                      else jnp.full(lay.hw[c.level], -1, jnp.int32)) for c in p.chans}
    recs = {}
    Ep = None
    for l in lay.levels:
        chans = lay.chans[l]
        Es = _encode(lay, l, state)
        if not chans:
            Ep = Es
            continue
        g = {k: jnp.asarray(v) for k, v in _geom(lay, l).items()}
        h, w = lay.hw[l]
        gum = [noise.gumbel(jnp, seed, noise.tag_of(c.name), l, g["Y"], g["Xn"], c.D) for c in chans]
        xs = {**g, "k": jnp.arange(NCOL), "gum": gum}
        Ep_ = Ep

        def body(carry, x, l=l, chans=chans, Ep_=Ep_, h=h, w=w):
            state, Es = carry
            Y, X, valid = x["Y"], x["X"], x["valid"]
            k = x["k"]
            outs = []
            for ci, c in enumerate(chans):
                soft, nv, look, nvg = cg(state, c.name, Y, X)
                v = nv + look
                vmin = jnp.min(v, axis=1, keepdims=True)
                mask = (v == vmin) | ~valid[:, None]
                feat = fast_features(lay, l, Es, Ep_, x)
                b = mlp(params[key_of(l, c.name)], feat)
                logits = jnp.where(mask, -soft / tau - b, -jnp.inf)
                logq = jax.nn.log_softmax(logits, axis=1)
                a = jnp.argmax(logits + x["gum"][ci], axis=1).astype(jnp.int32)
                st = lay.step(l, 0, ci) + k * len(chans)
                if override is not None:
                    os_, oj, oval, on = override
                    hit = on & (st == os_) & (jnp.arange(Y.shape[0]) == oj)
                    a = jnp.where(hit, oval, a)
                idx = jnp.arange(Y.shape[0])
                cost = jnp.where(valid, lam[l] * nvg[idx, a], 0.0)
                if record:
                    outs.append({"feat": feat, "soft": soft, "nv": nv, "nvg": nvg, "mask": mask, "a": a,
                                 "valid": valid, "logq": logq, "cost": cost})
                else:
                    outs.append(cost)
                Ys = jnp.where(valid, Y, h)
                state = {**state, c.name: state[c.name].at[Ys, X].set(a, mode="drop")}
                rows = jnp.concatenate([lay.emb[c.name][a], jnp.ones((a.shape[0], 1), jnp.float32)], -1)
                o, kk = _cols(lay, l)[c.name]
                Es = Es.at[x["sidx"][:, None], jnp.arange(o, o + kk + 1)[None, :]].set(rows, mode="drop")
            return (state, Es), outs

        (state, Es), ys = jax.lax.scan(body, (state, Es), xs)
        Ep = Es
        for ci, c in enumerate(chans):
            recs[key_of(l, c.name)] = ys[ci]
    return state, recs


def build_train(var, lay):
    """cfast's _build_train (K = 0, cone returns, critic h1 / feat, sub) for the graded penalty-only
    objective: ell = lam_l * nvg, cost = lam_l * nvg[a].  Same fn dict keys as cfast's: rec, step, fused,
    repair_b, actor_grad, topkeys, nval, cone.  `viol` returned by rec / fused is the BINARY count per
    level (lambda control unchanged); `tot` is the summed graded cost per world (= lam . D(x))."""
    T, K = var.problem.T, var.cfg["K"]
    assert K == 0, "cgpen: coupled replays (K > 0) are not supported"
    levels = list(range(T + 1))
    keys = [key_of(l, c.name) for l in lay.levels for c in lay.chans[l]]
    topkeys = [key_of(T, c.name) for c in lay.chans[T]]
    level_of = {key_of(l, c.name): l for l in levels for c in lay.chans[l]}
    nval = {}
    for l in levels:
        cnt = sum(int(np.asarray(lay.sites(l, k)[2]).sum()) for k in range(NCOL))
        nval[l] = cnt * len(lay.chans[l])
    cone = Cone(lay, lay.actor_win)
    idx = {k: jnp.asarray(v) for k, v in cone.idx.items()}
    sfrac = {}
    for l in levels:
        for ci, c in enumerate(lay.chans[l]):
            sfrac[key_of(l, c.name)] = jnp.asarray([lay.step(l, k, ci) / lay.nsteps for k in range(NCOL)], jnp.float32)
    use_cone = var.cfg["cone"]

    def returns(costs):
        if use_cone:
            return cone.returns(costs, idx)
        tot = jnp.zeros((next(iter(costs.values())).shape[0], lay.nsteps))
        for key, c in costs.items():
            l = level_of[key]
            ci = [x.name for x in lay.chans[l]].index(key[key.index("_") + 1:])
            st = np.array([lay.step(l, k, ci) for k in range(NCOL)])
            tot = tot.at[:, st].add(c.sum(-1))
        after = jnp.sum(tot, -1, keepdims=True) - jnp.cumsum(tot, -1)
        G = {}
        for key, c in costs.items():
            l = level_of[key]
            ci = [x.name for x in lay.chans[l]].index(key[key.index("_") + 1:])
            st = np.array([lay.step(l, k, ci) for k in range(NCOL)])
            G[key] = jnp.broadcast_to(after[:, st][..., None], c.shape)
        return G

    def _viol(recs, nb):
        viol = jnp.zeros(T + 1)
        for key, r in recs.items():
            nva = jnp.take_along_axis(r["nv"], r["a"][..., None], -1)[..., 0]
            viol = viol.at[level_of[key]].add(jnp.sum(nva * r["valid"]))
        return viol / nb

    def rec_fn(params, seeds, des, tau, lam):
        st, recs = jax.vmap(lambda s, d: train_pass_graded(lay, params, s, d, tau, lam))(seeds, des)
        G = returns({k: r["cost"] for k, r in recs.items()})
        viol = _viol(recs, seeds.shape[0])
        tot = sum(jnp.sum(r["cost"], (1, 2)) for r in recs.values())
        top = {k: {f: recs[k][f] for f in ("valid", "mask", "logq", "a")} for k in topkeys}
        return st, recs, G, viol, tot, top

    critic_h1 = var.cfg["critic"] == "h1"
    sub = var.cfg["sub"]

    def cfeat(r, key, tau, lam, params):
        f = r["feat"]
        if critic_h1:
            f = jax.lax.stop_gradient(hidden1(params[key], f))
        ex = jnp.concatenate([jnp.atleast_1d(tau), lam]).astype(jnp.float32)
        ex = jnp.broadcast_to(ex, f.shape[:-1] + ex.shape)
        sf = jnp.broadcast_to(sfrac[key][None, :, None, None], f.shape[:-1] + (1,))
        return jnp.concatenate([f, ex, sf], -1)

    def actor_loss(params, cparams, recs, G, tau, lam, gs, p0, wgt):
        total = 0.0
        for l in levels:
            lev = 0.0
            for c in lay.chans[l]:
                key = key_of(l, c.name)
                r = recs[key]
                b = mlp(params[key], r["feat"])
                logits = jnp.where(r["mask"], -r["soft"] / tau - b, -jnp.inf)
                logq = jax.nn.log_softmax(logits, -1)
                q = jnp.exp(logq)
                ell = lam[l] * r["nvg"]
                Q = jax.lax.stop_gradient(mlp(cparams[key], cfeat(r, key, tau, lam, p0))) * gs[key]
                t1 = jnp.sum(jnp.where(r["mask"], q * (ell + Q), 0.0), -1)
                la = jnp.take_along_axis(logq, r["a"][..., None], -1)[..., 0]
                Qa = jnp.take_along_axis(Q, r["a"][..., None], -1)[..., 0]
                t2 = la * jax.lax.stop_gradient(G[key] - Qa)
                lev = lev + wgt.get(key, 1.0) * jnp.sum(jnp.where(r["valid"], t1 + t2, 0.0))
            total = total + lev / nval[l]
        return total / recs[keys[0]]["a"].shape[0]

    def critic_loss(cparams, recs, G, tau, lam, gs, p0, wgt):
        total = 0.0
        for l in levels:
            lev = 0.0
            for c in lay.chans[l]:
                key = key_of(l, c.name)
                r = recs[key]
                Q = mlp(cparams[key], cfeat(r, key, tau, lam, p0))
                Qa = jnp.take_along_axis(Q, r["a"][..., None], -1)[..., 0]
                lev = lev + wgt.get(key, 1.0) * jnp.sum(jnp.where(r["valid"], (Qa - G[key] / gs[key]) ** 2, 0.0))
            total = total + lev / nval[l]
        return total / recs[keys[0]]["a"].shape[0]

    def train_step(params, cparams, opt, copt, recs, G, tau, lam, br, gs, lr, lrc, clip, wgt=None):
        wgt = wgt or {}
        p0 = params
        loss, g = jax.value_and_grad(actor_loss)(params, cparams, recs, G, tau, lam, gs, p0, wgt)
        g = clip_by_level(g, levels, clip)
        params, opt = adam_step(params, g, opt, lr)
        closs, gc = jax.value_and_grad(critic_loss)(cparams, recs, G, tau, lam, gs, p0, wgt)
        gc = clip_by_level(gc, levels, clip)
        cparams, copt = adam_step(cparams, gc, copt, lrc)
        return params, cparams, opt, copt, loss, closs

    def fused(params, cparams, opt, copt, gs, seeds, des, tau, lam, lr, lrc, clip, rkey):
        if not sub:
            st, recs, G, viol, tot, _ = rec_fn(params, seeds, des, tau, lam)
            wgt = {}
        else:
            st, recs = jax.vmap(lambda s, d: train_pass_graded(lay, params, s, d, tau, lam))(seeds, des)
            viol = _viol(recs, seeds.shape[0])
            tot = sum(jnp.sum(r["cost"], (1, 2)) for r in recs.values())
            costs = {k: r["cost"] for k, r in recs.items()}
            ii = cone.integral(costs) if use_cone else None
            Gall = None if use_cone else returns(costs)
            Bn = seeds.shape[0]
            G, wgt, rs = {}, {}, {}
            for key, r in recs.items():
                n = r["a"].shape[-1]
                if n <= sub:
                    sel = jnp.broadcast_to(jnp.arange(n), (Bn, NCOL, n))
                else:
                    rkey, k1 = jax.random.split(rkey)
                    sel = jnp.argsort(jax.random.uniform(k1, (Bn, NCOL, n)), -1)[..., :sub]
                    wgt[key] = n / sub
                rs[key] = {f: jnp.take_along_axis(x, sel.reshape(sel.shape + (1,) * (x.ndim - 3)), 2)
                           for f, x in r.items()}
                if use_cone:
                    ix = idx[key][jnp.arange(NCOL)[None, :, None], sel]
                    v = jax.vmap(lambda a, i: a[i])(ii, ix)
                    G[key] = jnp.sum(v[..., 0] - v[..., 1] - v[..., 2] + v[..., 3], -1)
                else:
                    G[key] = jnp.take_along_axis(Gall[key], sel, 2)
            recs = rs
        params, cparams, opt, copt, loss, closs = train_step(params, cparams, opt, copt, recs, G, tau, lam,
                                                             {}, gs, lr, lrc, clip, wgt)
        gstd = {k: jnp.std(v) for k, v in G.items()}
        return st, params, cparams, opt, copt, loss, closs, viol, tot, gstd

    return {"rec": jax.jit(rec_fn), "step": jax.jit(train_step), "fused": jax.jit(fused),
            "repair_b": jax.jit(jax.vmap(lambda st, s: core.repair(lay, st, s))),
            "actor_grad": jax.jit(lambda params, cparams, recs, G, tau, lam, gs: jax.grad(actor_loss)(
                params, cparams, recs, G, tau, lam, gs, params, {})),
            "topkeys": topkeys, "nval": nval, "cone": cone}
