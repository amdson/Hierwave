"""cfine training functions: creplay's coupled replays extended to any set of
levels (the fine ones included), with non-forced selection and a windowed
replay for small cones.

Differences from _creplay_core (same estimator otherwise):

- Branch levels are an explicit tuple `blevs` (e.g. all levels), with R_l
  per level.
- Selection among non-forced draws.  At branch level l, world b, R_l draws are
  chosen uniformly without replacement among the N_nf(b) valid draws with
  |A| > 1, and each selected c_i is weighted N_nf(b) / min(R_l, N_nf(b)).  A
  forced draw (|A| = 1) has q(a) = 1 for its only action, so grad q = 0 and its
  replay term is identically zero: sum_i c_i = sum_{i non-forced} c_i
  pointwise.  The selection depends on the trajectory but not on the replay
  randomness (the extra sampled actions), so E[weighted subset sum | world]
  = sum_{i non-forced} c_i and the estimator stays unbiased (constitution
  hard constraint 3).  Fine levels on ground are 50-99 % forced, so uniform
  selection would waste most replays.
- Windowed replay (levels <= `wlev`).  A replay of draw i at level lb starts
  from the original pass's final state with every cell drawn at a step >=
  base[lb] reset to unknown, walks the steps from base[lb] on, and at each
  step recomputes only the sites inside i's dependency-cone box (padded to a
  static size per (lb, level, channel)); every other site of the step gets
  its original value back (it is outside the cone, so the full replay would
  reproduce it bit-identically).  The cone return G_i^a is the sum of the
  recomputed sites' costs, exactly the cone box sum of a full replay.  Noise
  is coordinate-keyed, so each recomputed draw uses the same Gumbels as the
  full replay.  tests/variants/test_cfine.py checks that the final state equals
  the full replay's bit for bit and that G matches.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench import noise

from . import _cpass_core as core
from ._cfast_cone import Cone
from ._cfast_core import train_pass
from .cpass import adam_step, clip_by_level


def build(var, lay):
    cfg, p = var.cfg, var.problem
    T = p.T
    levels = list(range(T + 1))
    keys = [core.key_of(l, c.name) for l in lay.levels for c in lay.chans[l]]
    level_of = {core.key_of(l, c.name): l for l in levels for c in lay.chans[l]}
    nval = {}
    for l in levels:
        cnt = sum(int(np.asarray(lay.sites(l, k)[2]).sum()) for k in range(core.NCOL))
        nval[l] = cnt * len(lay.chans[l])
    cone = Cone(lay, core.ACTOR_WIN)
    idx = {k: jnp.asarray(v) for k, v in cone.idx.items()}
    sfrac = {}
    for l in levels:
        for ci, c in enumerate(lay.chans[l]):
            sfrac[core.key_of(l, c.name)] = jnp.asarray(
                [lay.step(l, k, ci) / lay.nsteps for k in range(core.NCOL)], jnp.float32)
    Kr, sub, chunk = cfg["Kr"], cfg["sub"], cfg["chunk"]
    nonforced = cfg["nonforced"]
    critic_h1 = cfg["critic"] == "h1"

    # ------------------------------------------------ branch tables per level
    btab = {}
    Rcfg = list(cfg["R"])
    for i, l in enumerate(cfg["blevs"]):
        if l < 0 or l > T or not lay.chans[l]:
            continue
        chans = lay.chans[l]
        cand = []
        for ci in range(len(chans)):
            for k in range(core.NCOL):
                v = np.asarray(lay.sites(l, k)[2])
                cand += [(ci, k, j) for j in np.flatnonzero(v)]
        cand = np.array(cand, np.int32)
        R = Rcfg[i] if i < len(Rcfg) else None
        R = len(cand) if R is None else min(int(R), len(cand))
        btab[l] = {"cand": jnp.asarray(cand), "Nc": len(cand), "R": R,
                   "steps": jnp.asarray([[lay.step(l, k, ci) for k in range(core.NCOL)]
                                         for ci in range(len(chans))], jnp.int32),
                   "idx": jnp.stack([idx[core.key_of(l, c.name)] for c in chans]),     # (nch, 16, n, S, 4)
                   "keys": [core.key_of(l, c.name) for c in chans],
                   "D": [c.D for c in chans], "Dmax": max(c.D for c in chans),
                   "boxes": jnp.asarray(np.stack([cone.boxes[core.key_of(l, c.name)] for c in chans]),
                                        jnp.int32)}                                  # (nch, 16, n, S, 4)
    blevels = sorted(btab, reverse=True)
    wlev = cfg["wlev"]
    wlevels = [l for l in blevels if l <= wlev]
    flevels = [l for l in blevels if l > wlev]
    # static padded box extents (My, Mx) per (branch level lb, level l, channel ci)
    Mbox = {}
    for lb in wlevels:
        bx_all = np.stack([cone.boxes[core.key_of(lb, c.name)] for c in lay.chans[lb]])
        for l in range(lb, -1, -1):
            for ci in range(len(lay.chans[l])):
                ss = [lay.step(l, k, ci) - lay.base[lb] for k in range(core.NCOL)]
                b = bx_all[:, :, :, ss]                                            # (nch, 16, n, 16, 4)
                ny_ = np.maximum(b[..., 1] - b[..., 0] + 1, 0)
                nx_ = np.maximum(b[..., 3] - b[..., 2] + 1, 0)
                Mbox[(lb, l, ci)] = (max(int(ny_.max()), 1), max(int(nx_.max()), 1))
    # step of every cell of every sampled channel (for resetting to the pre-step state)
    stepmap = {}
    for l in levels:
        h, w = lay.hw[l]
        Yg, Xg = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        kg = (Yg % core.P) * core.P + ((Xg + 0) % core.P)
        for ci, c in enumerate(lay.chans[l]):
            stepmap[c.name] = jnp.asarray(lay.base[l] + kg * len(lay.chans[l]) + ci, jnp.int32)
    S_max = max([btab[l]["idx"].shape[3] for l in blevels], default=0)

    def cfeat_base(r, key, tau, lam, params):
        f = r["feat"]
        if critic_h1:                     # the actor's first hidden layer, fixed for this update
            pp = params[key]
            f = jax.lax.stop_gradient(jnp.tanh(f @ pp["W1"] + pp["b1"]))
        ex = jnp.concatenate([jnp.atleast_1d(tau), lam]).astype(jnp.float32)
        return jnp.concatenate([f, jnp.broadcast_to(ex, f.shape[:-1] + ex.shape)], -1)

    def cfeat_std(r, key, tau, lam, params):
        """cfast's critic features of recorded draws laid out (B, 16, m, F)."""
        f = cfeat_base(r, key, tau, lam, params)
        sf = jnp.broadcast_to(sfrac[key][None, :, None, None], f.shape[:-1] + (1,))
        return jnp.concatenate([f, sf], -1)

    def cfeat_sel(r, key, tau, lam, params, kk):
        """The same for selected draws (B, R, F) at colours kk (B, R)."""
        f = cfeat_base(r, key, tau, lam, params)
        return jnp.concatenate([f, sfrac[key][kk][..., None]], -1)

    # ------------------------------------------------------------- the pass
    def run_pass(params, seeds, des, tau, lam):
        st, recs = jax.vmap(lambda s, d: train_pass(lay, params, s, d, tau, lam))(seeds, des)
        viol = jnp.zeros(T + 1)
        for key, r in recs.items():
            nva = jnp.take_along_axis(r["nv"], r["a"][..., None], -1)[..., 0]
            viol = viol.at[level_of[key]].add(jnp.sum(nva * r["valid"]))
        tot = sum(jnp.sum(r["cost"], (1, 2)) for r in recs.values())
        return st, recs, viol / seeds.shape[0], tot

    # ------------------------------------------------------------ branching
    def select(recs, l, rkey):
        """Branch draws and actions of level l: dict of (B, R, ...) arrays."""
        bt = btab[l]
        Bn = recs[bt["keys"][0]]["a"].shape[0]
        k1, k2 = jax.random.split(rkey)
        ib = jnp.arange(Bn)[:, None]
        cc, ck, cj = bt["cand"][:, 0], bt["cand"][:, 1], bt["cand"][:, 2]
        nAs = jnp.stack([jnp.sum(recs[key]["mask"], -1) for key in bt["keys"]])      # (nch, B, 16, n)
        nAc = nAs[cc[None, :], ib, ck[None, :], cj[None, :]]                          # (B, Nc)
        nf = nAc > 1 if nonforced else jnp.ones_like(nAc, bool)
        u = jnp.where(nf, jax.random.uniform(k1, (Bn, bt["Nc"])), 2.0)
        s = jnp.argsort(u, -1)[:, :bt["R"]]
        Nnf = jnp.sum(nf, -1)                                                      # (B,)
        Reff = jnp.minimum(bt["R"], Nnf)
        wsel = jnp.where(jnp.arange(bt["R"])[None, :] < Reff[:, None],
                         (Nnf / jnp.maximum(Reff, 1)).astype(jnp.float32)[:, None], 0.0)
        trip = bt["cand"][s]                                               # (B, R, 3)
        ci, kk, jj = trip[..., 0], trip[..., 1], trip[..., 2]
        Dm = bt["Dmax"]
        masks, lqs, As = [], [], []
        for key, D in zip(bt["keys"], bt["D"]):
            r = recs[key]
            m = r["mask"][ib, kk, jj]
            lq = r["logq"][ib, kk, jj]
            masks.append(jnp.pad(m, ((0, 0), (0, 0), (0, Dm - D))))
            lqs.append(jnp.pad(lq, ((0, 0), (0, 0), (0, Dm - D)), constant_values=-jnp.inf))
            As.append(r["a"][ib, kk, jj])
        ir = jnp.arange(bt["R"])[None, :]
        pick = lambda xs: jnp.stack(xs)[ci, ib, ir]
        mask, lq, a = pick(masks), pick(lqs), pick(As)
        lq = jnp.where(mask, lq, -jnp.inf)
        nA = jnp.sum(mask, -1)
        enum = nA <= Kr
        order = jnp.argsort(~mask, -1, stable=True)                        # A's indices first
        pos = jnp.arange(Kr)
        ea = order[..., jnp.minimum(pos, Dm - 1)]
        ea = jnp.where(pos < nA[..., None], ea, order[..., :1])
        smp = jax.random.categorical(k2, lq[..., None, :], axis=-1, shape=lq.shape[:-1] + (Kr - 1,))
        sa = jnp.concatenate([a[..., None], smp.astype(jnp.int32)], -1)
        acts = jnp.where(enum[..., None], ea, sa).astype(jnp.int32)
        return {"ci": ci, "kk": kk, "jj": jj, "acts": acts, "enum": enum, "nA": nA, "w": wsel}

    def replay_costs(params, seeds, des, tau, lam, step, site, val):
        """Per-site costs {key: (N, 16, n)} of N replays, each with one draw
        (step, site) replaced by val, under the world's own noise."""
        ov = (step, site, val, jnp.ones(step.shape, bool))
        return jax.vmap(lambda s, d, o: train_pass(lay, params, s, d, tau, lam, override=o,
                                                   record=False)[1])(seeds, des, ov)

    def replay_full(params, seeds, des, tau, lam, sels):
        """Cone returns G_i^a for every (world, branched draw, action) of every
        branch level, in one batched, chunked replay: {l: (B, R_l, K)}."""
        Bn = seeds.shape[0]
        parts = []
        ls = sorted(sels, reverse=True)
        for l in ls:
            bt, sel = btab[l], sels[l]
            R = bt["R"]
            e3 = lambda x: jnp.broadcast_to(x[..., None], (Bn, R, Kr)).reshape(-1)
            ix = bt["idx"][sel["ci"], sel["kk"], sel["jj"]]                          # (B, R, S_l, 4)
            ix = jnp.pad(ix, ((0, 0), (0, 0), (0, S_max - ix.shape[2]), (0, 0)))     # index 0: empty box
            parts.append({"b": e3(jnp.broadcast_to(jnp.arange(Bn)[:, None], (Bn, R))),
                          "step": e3(bt["steps"][sel["ci"], sel["kk"]]), "site": e3(sel["jj"]),
                          "val": sel["acts"].reshape(-1),
                          "ix": jnp.broadcast_to(ix[:, :, None], (Bn, R, Kr) + ix.shape[2:]).reshape(
                              (-1,) + ix.shape[2:])})
        flat = {k: jnp.concatenate([q[k] for q in parts], 0) for k in parts[0]}
        N = flat["b"].shape[0]
        C = min(chunk, N)
        nch = -(-N // C)
        pad = nch * C - N
        flat = jax.tree_util.tree_map(lambda x: jnp.concatenate([x, x[:pad]], 0) if pad else x, flat)
        flat = jax.tree_util.tree_map(lambda x: x.reshape((nch, C) + x.shape[1:]), flat)

        def one_chunk(f):
            costs = replay_costs(params, seeds[f["b"]], {k: v[f["b"]] for k, v in des.items()}, tau, lam,
                                 f["step"], f["site"], f["val"])
            ii = cone.integral(costs)                                  # (C, size)
            v = jnp.take_along_axis(ii, f["ix"].reshape(C, -1), -1).reshape(f["ix"].shape)
            return jnp.sum(v[..., 0] - v[..., 1] - v[..., 2] + v[..., 3], -1)

        if nch == 1:                      # one chunk: no lax.map loop around the replay pass
            Gk = one_chunk(jax.tree_util.tree_map(lambda x: x[0], flat)).reshape(-1)[:N]
        else:
            Gk = jax.lax.map(one_chunk, flat).reshape(-1)[:N]
        out, o = {}, 0
        for l in ls:
            n = Bn * btab[l]["R"] * Kr
            out[l] = Gk[o:o + n].reshape(Bn, btab[l]["R"], Kr)
            o += n
        return out

    def wreplay_one(params, seed, des, final, tau, lam, lb, cid, kd, jd, val, ret_state=False):
        """One windowed replay of draw (level lb, channel cid, colour kd, site jd)
        with value val: its cone return (and optionally the final state)."""
        bt = btab[lb]
        s0 = bt["steps"][cid, kd]
        bx = bt["boxes"][cid, kd, jd]                                     # (S, 4)
        nyb, nxb = lay.nyx[lb]
        Y0 = kd // core.P + core.P * (jd // nxb)
        X0 = kd % core.P + core.P * (jd % nxb)
        state = {}
        for c in p.chans:
            if c.designed is not None:
                state[c.name] = des[c.name]
            elif c.level <= lb:
                state[c.name] = jnp.where(stepmap[c.name] >= lay.base[lb], -1, final[c.name])
            else:
                state[c.name] = final[c.name]
        G = jnp.float32(0.0)
        for l in range(lb, -1, -1):
            chans = lay.chans[l]
            if not chans:
                continue
            h, w = lay.hw[l]
            ny, nx = lay.nyx[l]

            def body(carry, k, l=l, chans=chans, h=h, w=w, nx=nx):
                state, G = carry
                cy, cx = k // core.P, k % core.P
                for ci, c in enumerate(chans):
                    st = lay.step(l, 0, ci) + k * len(chans)
                    b = bx[st - lay.base[lb]]
                    My, Mx = Mbox[(lb, l, ci)]
                    iy = b[0] + jnp.arange(My)
                    ix = b[2] + jnp.arange(Mx)
                    ok = (iy[:, None] <= b[1]) & (ix[None, :] <= b[3])
                    Y = jnp.broadcast_to((cy + core.P * iy)[:, None], (My, Mx)).ravel()
                    X = jnp.broadcast_to((cx + core.P * jnp.mod(ix, nx))[None, :], (My, Mx)).ravel()
                    ok = ok.ravel() & (Y < h) & (X < w)
                    Yc = jnp.where(ok, Y, cy)
                    Xc = jnp.where(ok, X, cx)
                    soft, nv, look = lay.cond(state, c.name, Yc, Xc)
                    v = nv + look
                    vmin = jnp.min(v, axis=1, keepdims=True)
                    mask = (v == vmin) | ~ok[:, None]
                    feat = lay.features(state, l, Yc, Xc, lay.actor_win)
                    bb = core.mlp(params[core.key_of(l, c.name)], feat)
                    logits = jnp.where(mask, -soft / tau - bb, -jnp.inf)
                    logq = jax.nn.log_softmax(logits, axis=1)
                    g = noise.gumbel(jnp, seed, noise.tag_of(c.name), l, Yc, jnp.mod(Xc, w), c.D)
                    a = jnp.argmax(logits + g, axis=1).astype(jnp.int32)
                    ii = jnp.arange(Yc.shape[0])
                    cost = jnp.where(ok, soft[ii, a] + lam[l] * nv[ii, a] + tau * logq[ii, a], 0.0)
                    G = G + jnp.sum(cost)
                    # the step's other sites keep their original values; then the override; then the box
                    cur = jnp.where(stepmap[c.name] == st, final[c.name], state[c.name])
                    if l == lb:
                        hit = st == s0
                        Yo = jnp.where(hit, Y0, h)
                        cur = cur.at[Yo, X0].set(val, mode="drop")
                    cur = cur.at[jnp.where(ok, Yc, h), Xc].set(a, mode="drop")
                    state = {**state, c.name: cur}
                return (state, G), None

            (state, G), _ = jax.lax.scan(body, (state, G), jnp.arange(core.NCOL))
        return (G, state) if ret_state else G

    def replay_win(params, seeds, des, final, tau, lam, l, sel):
        """Windowed replays of one level's selected draws: (B, R_l, K)."""
        Bn, R = sel["acts"].shape[:2]
        e3 = lambda x: jnp.broadcast_to(x[..., None], (Bn, R, Kr)).reshape(-1)
        bidx = e3(jnp.broadcast_to(jnp.arange(Bn)[:, None], (Bn, R)))
        f = {"b": bidx, "ci": e3(sel["ci"]), "kk": e3(sel["kk"]), "jj": e3(sel["jj"]),
             "val": sel["acts"].reshape(-1)}
        N = bidx.shape[0]
        C = min(chunk, N)
        nch = -(-N // C)
        pad = nch * C - N
        f = jax.tree_util.tree_map(lambda x: jnp.concatenate([x, x[:pad]], 0) if pad else x, f)
        f = jax.tree_util.tree_map(lambda x: x.reshape((nch, C) + x.shape[1:]), f)

        def one_chunk(q):
            b = q["b"]
            return jax.vmap(lambda s, d, fi, ci, kd, jd, v: wreplay_one(params, s, d, fi, tau, lam, l, ci, kd, jd, v))(
                seeds[b], {k: v[b] for k, v in des.items()}, {k: v[b] for k, v in final.items()},
                q["ci"], q["kk"], q["jj"], q["val"])

        if nch == 1:
            Gk = one_chunk(jax.tree_util.tree_map(lambda x: x[0], f))[:N]
        else:
            Gk = jax.lax.map(one_chunk, f).reshape(-1)[:N]
        return Gk.reshape(Bn, R, Kr)

    def replay_all(params, seeds, des, tau, lam, sels, final=None):
        """{l: (B, R_l, K)}: full-world replays above wlev, windowed at or below."""
        out = {}
        fs = {l: s for l, s in sels.items() if l in flevels or final is None}
        if fs:
            out.update(replay_full(params, seeds, des, tau, lam, fs))
        for l, s in sels.items():
            if l not in out:
                out[l] = replay_win(params, seeds, des, final, tau, lam, l, s)
        return out

    def replay(params, seeds, des, tau, lam, l, sel, final=None):
        """One level's replays: (B, R_l, K)."""
        return replay_all(params, seeds, des, tau, lam, {l: sel}, final)[l]

    # ----------------------------------------------------------------- batch
    def batch(params, seeds, des, tau, lam, rkey):
        """Pass, cone returns, subsample, branches and replays: everything
        the losses need."""
        st, recs, viol, tot = run_pass(params, seeds, des, tau, lam)
        Bn = seeds.shape[0]
        costs = {k: r["cost"] for k, r in recs.items()}
        ii = cone.integral(costs)
        # branches: select at every branch level, then one batched replay
        br, diag = {}, {}
        ib = jnp.arange(Bn)[:, None]
        for l in blevels:
            rkey, k1 = jax.random.split(rkey)
            br[l] = select(recs, l, k1)
        final = {c.name: st[c.name] for c in p.chans if c.designed is None}
        Gks = replay_all(params, seeds, des, tau, lam, br, final) if blevels else {}
        for l in blevels:
            sel = br[l]
            Gk = Gks[l]
            ix = btab[l]["idx"][sel["ci"], sel["kk"], sel["jj"]]             # (B, R, S, 4)
            v = jax.vmap(lambda a, i: a[i])(ii, ix)
            G0 = jnp.sum(v[..., 0] - v[..., 1] - v[..., 2] + v[..., 3], -1)
            sel["Gk"], sel["G0"] = Gk, G0
            real = sel["w"] > 0
            same = ~sel["enum"] & real
            nr = jnp.maximum(jnp.sum(real), 1)
            diag[f"l{l}"] = {"enum_frac": jnp.sum(sel["enum"] & real) / nr, "real_frac": jnp.mean(real),
                             "mean_w": jnp.sum(sel["w"]) / nr,
                             "mean_nA": jnp.sum(jnp.where(real, sel["nA"], 0)) / nr,
                             "replay_err": jnp.max(jnp.where(same, jnp.abs(Gk[..., 0] - G0), 0.0)),
                             "spread": jnp.mean(jnp.std(Gk, -1)), "G0_std": jnp.std(G0)}
        # subsampled standard terms (cfast)
        G, wgt, rs = {}, {}, {}
        for key, r in recs.items():
            n = r["a"].shape[-1]
            if not sub or n <= sub:
                s = jnp.broadcast_to(jnp.arange(n), (Bn, core.NCOL, n))
            else:
                rkey, k1 = jax.random.split(rkey)
                s = jnp.argsort(jax.random.uniform(k1, (Bn, core.NCOL, n)), -1)[..., :sub]
                wgt[key] = n / sub
            rs[key] = {f: jnp.take_along_axis(x, s.reshape(s.shape + (1,) * (x.ndim - 3)), 2) for f, x in r.items()}
            ix = idx[key][jnp.arange(core.NCOL)[None, :, None], s]
            v = jax.vmap(lambda a, i: a[i])(ii, ix)
            G[key] = jnp.sum(v[..., 0] - v[..., 1] - v[..., 2] + v[..., 3], -1)
        # branch-draw records (full, not subsampled)
        brec = {}
        for l in blevels:
            sel = br[l]
            for key in btab[l]["keys"]:
                r = recs[key]
                brec[key] = {f: r[f][ib, sel["kk"], sel["jj"]] for f in ("feat", "soft", "nv", "mask")}
        return st, rs, G, wgt, br, brec, viol, tot, diag

    # ---------------------------------------------------------------- losses
    def branch_terms(params, cparams, br, brec, tau, lam, gs, p0, l):
        """Sum over worlds and selected draws of the coupled correction c_i
        (actor loss) and of the critic's squared errors on the replayed
        actions.  c_i estimates sum_a grad q(a) (G^a - Q(a)) given h_i:
          enum:    sum_a grad q(a) (G^a - Q(a))                      (exact over a_i)
          sampled: (1/K) sum_k grad log q(a_k) [(G^k - Q(a_k)) - mean_{j != k}(G^j - Q(a_j))]
        Each selected draw is weighted by sel["w"] (N_nf / min(R, N_nf), 0 on empty slots)."""
        sel = br[l]
        a_tot, c_tot = 0.0, 0.0
        for c_i, (key, D) in enumerate(zip(btab[l]["keys"], btab[l]["D"])):
            r = brec[key]
            on = sel["ci"] == c_i                                             # (B, R)
            b = core.mlp(params[key], r["feat"])
            mk = r["mask"]
            logits = jnp.where(mk, -r["soft"] / tau - b, -jnp.inf)
            logq = jax.nn.log_softmax(logits, -1)
            q = jnp.exp(logq)
            Qr = core.mlp(cparams[key], cfeat_sel(r, key, tau, lam, p0, sel["kk"]))
            QQ = jax.lax.stop_gradient(Qr) * gs[key]
            acts = jnp.where(on[..., None], jnp.clip(sel["acts"], 0, D - 1), 0)
            Gk = sel["Gk"]
            lqm = jnp.where(mk, logq, 0.0)
            # enumeration: every action of A was replayed (duplicates write the same value)
            Gfull = jnp.zeros_like(QQ)
            Gfull = jax.vmap(jax.vmap(lambda g, a, v: g.at[a].set(v)))(Gfull, acts, Gk)
            ce = jnp.sum(jnp.where(mk, q * jax.lax.stop_gradient(Gfull - QQ), 0.0), -1)
            # sampled: leave-one-out over the K i.i.d. actions
            rk = Gk - jnp.take_along_axis(QQ, acts, -1)
            beta = (jnp.sum(rk, -1, keepdims=True) - rk) / (Kr - 1)
            cs = jnp.mean(jnp.take_along_axis(lqm, acts, -1) * jax.lax.stop_gradient(rk - beta), -1)
            a_tot = a_tot + jnp.sum(jnp.where(on, sel["w"] * jnp.where(sel["enum"], ce, cs), 0.0))
            Qk = jnp.take_along_axis(Qr, acts, -1)
            c_tot = c_tot + jnp.sum(jnp.where(on & (sel["w"] > 0), jnp.sum((Qk - Gk / gs[key]) ** 2, -1), 0.0))
        return a_tot, c_tot

    def actor_loss(params, cparams, recs, G, tau, lam, br, brec, gs, p0, wgt):
        total = 0.0
        for l in levels:
            lev = 0.0
            for c in lay.chans[l]:
                key = core.key_of(l, c.name)
                r = recs[key]
                b = core.mlp(params[key], r["feat"])
                logits = jnp.where(r["mask"], -r["soft"] / tau - b, -jnp.inf)
                logq = jax.nn.log_softmax(logits, -1)
                q = jnp.exp(logq)
                lq_sg = jax.lax.stop_gradient(jnp.where(r["mask"], logq, 0.0))
                ell = r["soft"] + lam[l] * r["nv"] + tau * lq_sg
                Q = jax.lax.stop_gradient(core.mlp(cparams[key], cfeat_std(r, key, tau, lam, p0))) * gs[key]
                t1 = jnp.sum(jnp.where(r["mask"], q * (ell + Q), 0.0), -1)
                la = jnp.take_along_axis(logq, r["a"][..., None], -1)[..., 0]
                Qa = jnp.take_along_axis(Q, r["a"][..., None], -1)[..., 0]
                t2 = la * jax.lax.stop_gradient(G[key] - Qa)
                term = t1 if l in br else t1 + t2       # branch level: every score term replaced
                lev = lev + wgt.get(key, 1.0) * jnp.sum(jnp.where(r["valid"], term, 0.0))
            if l in br:                       # per-world inverse inclusion weights are in sel["w"]
                lev = lev + branch_terms(params, cparams, br, brec, tau, lam, gs, p0, l)[0]
            total = total + lev / nval[l]
        return total / recs[keys[0]]["a"].shape[0]

    def critic_loss(cparams, params, recs, G, tau, lam, br, brec, gs, p0, wgt):
        total = 0.0
        for l in levels:
            lev = 0.0
            for c in lay.chans[l]:
                key = core.key_of(l, c.name)
                r = recs[key]
                Q = core.mlp(cparams[key], cfeat_std(r, key, tau, lam, p0))
                Qa = jnp.take_along_axis(Q, r["a"][..., None], -1)[..., 0]
                lev = lev + wgt.get(key, 1.0) * jnp.sum(jnp.where(r["valid"], (Qa - G[key] / gs[key]) ** 2, 0.0))
            if l in br:
                lev = lev + branch_terms(p0, cparams, br, brec, tau, lam, gs, p0, l)[1]
            total = total + lev / nval[l]
        return total / recs[keys[0]]["a"].shape[0]

    def fused(params, cparams, opt, copt, gs, seeds, des, tau, lam, lr, lrc, clip, rkey):
        st, recs, G, wgt, br, brec, viol, tot, diag = batch(params, seeds, des, tau, lam, rkey)
        p0 = params
        loss, g = jax.value_and_grad(actor_loss)(params, cparams, recs, G, tau, lam, br, brec, gs, p0, wgt)
        g = clip_by_level(g, levels, clip)
        params, opt = adam_step(params, g, opt, lr)
        closs, gc = jax.value_and_grad(critic_loss)(cparams, p0, recs, G, tau, lam, br, brec, gs, p0, wgt)
        gc = clip_by_level(gc, levels, clip)
        cparams, copt = adam_step(cparams, gc, copt, lrc)
        gstd = {k: jnp.std(v) for k, v in G.items()}
        return st, params, cparams, opt, copt, loss, closs, viol, tot, diag, gstd

    def agrad(params, cparams, gs, seeds, des, tau, lam, rkey):
        """The actor gradient of one batch (no update): for tests and SNR."""
        st, recs, G, wgt, br, brec, viol, tot, diag = batch(params, seeds, des, tau, lam, rkey)
        g = jax.grad(actor_loss)(params, cparams, recs, G, tau, lam, br, brec, gs, params, wgt)
        return g, tot, diag

    return {"fused": jax.jit(fused), "agrad": jax.jit(agrad), "batch": jax.jit(batch),
            "repair_b": jax.jit(jax.vmap(lambda st, s: core.repair(lay, st, s))),
            "nval": nval, "cone": cone, "btab": btab, "blevels": blevels,
            "replay": replay, "select": select, "replay_costs": replay_costs, "run_pass": run_pass,
            "wreplay_one": wreplay_one, "replay_full": replay_full, "wlevels": wlevels}
