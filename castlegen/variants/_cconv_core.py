"""Core of cconv (ideas.md #5): a convolutional actor for the coloured pass.

Actor.  Per level l a conv trunk over the whole level grid:

  input planes  every level-l channel embedded as in channel_embedding (zero
                where unknown), its known flag and its exists flag; the same for
                every level-(l + 1) channel, upsampled (nearest, 2 x 2); the
                position in the parent (4 one-hot) and the row fraction Y / (h - 1)
  trunk         3 x 3 conv stem, then r - 1 pre-norm residual blocks
                h <- h + conv3x3(SiLU(LN(h))); features f = SiLU(LN(h))
  heads         per channel: linear logits f W + b (full domain), plus a
                candidate-scoring head  sum_j w_j SiLU(f A + e(t) E + c)_j  where
                e(t) = (embedding of t, s(t), min(nv(t), 3), min(look(t), 3)).
                Both heads start at zero, so the untrained actor is cpass's.

So r = number of 3 x 3 layers is the trunk's radius in level-l cells.  Cells
off the grid are zero at every conv input (zero padding; x wraps on a periodic
strip).

Two evaluations of the same function:
- dense (the pass): the trunk on the current state, every step, read at the
  step's sites.  The state holds exactly the values fixed before the step.
- patch (the loss): the trunk on (2r + 1)^2 windows around chosen sites with
  valid convolutions and every conv input masked off-grid; the state at step s
  is rebuilt from the final state: a sampled cell is known iff its own step is
  earlier than s (static schedule).  test_cconv checks dense == patch.

Finite support.  A draw at (l, Y, X) reads level-l cells within r of the
site and level-(l + 1) cells under them: the Window below, which the cone and
the margin use, together with the factor / lookahead reads."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench import noise

from . import _cpass_core as core
from ._cfast_cone import fine_reach
from ._cpass_core import NCOL, P, Window, key_of

HI = jax.lax.Precision.HIGHEST    # fp32 convs / head matmuls (no TF32): pass and loss agree
TRUNK = "_trunk"           # params key f"l{l}_{TRUNK}" (clip_by_level groups it with level l)


def trunk_key(l):
    return f"l{l}_{TRUNK}"


def conv_window(r):
    """Cells read by the trunk at a site, in cpass's Window convention: same-level
    offsets from the parent-block corner 2 (Y >> 1), parent offsets from Y >> 1."""
    same = tuple((dy, dx) for dy in range(-r, r + 2) for dx in range(-r, r + 2))
    q = (r + 1) // 2
    par = tuple((dy, dx) for dy in range(-q, q + 1) for dx in range(-q, q + 1))
    return Window(same=same, parent=par)


def margin(problem, lay_chans, r):
    """Fine cells of padding bounding the dependency reach of the pass plus one
    repair round: the sum over every step of its reach in x (trunk window,
    parent, factors, lookahead), plus each repair step's (factors only)."""
    win = conv_window(r)
    none = Window(same=(), parent=())
    m = 8
    for l in range(problem.T + 1):
        for c in lay_chans[l]:
            _, _, ax, bx = fine_reach(problem, l, c.name, win)
            _, _, rx, sx = fine_reach(problem, l, c.name, none)
            m += NCOL * (max(ax, bx) + max(rx, sx))
    step = P << problem.T
    return -(-m // step) * step


# ------------------------------------------------------------------ the net
class ConvNet:
    """Static structure for one layout: input channels per level, embeddings."""

    def __init__(self, lay, r=4, width=32, hc=32):
        self.lay, self.r, self.C, self.hc = lay, r, width, hc
        self.cin = {}
        for l in range(lay.T + 1):
            n = sum(lay.emb[c.name].shape[1] + 2 for c in lay.all_chans[l])
            if l < lay.T:
                n += sum(lay.emb[c.name].shape[1] + 2 for c in lay.all_chans[l + 1])
            self.cin[l] = n + 5

    # -------------------------------------------------------------- params
    def init(self, key):
        lay, C, hc = self.lay, self.C, self.hc
        params = {}
        for l in range(lay.T + 1):
            if not lay.chans[l]:
                continue
            key, k0 = jax.random.split(key)
            tr = {"stem": jax.random.normal(k0, (3, 3, self.cin[l], C)) / np.sqrt(9 * self.cin[l]),
                  "stem_b": jnp.zeros(C), "blocks": []}
            for _ in range(self.r - 1):
                key, k1 = jax.random.split(key)
                tr["blocks"].append({"g": jnp.ones(C), "b": jnp.zeros(C),
                                     "W": jax.random.normal(k1, (3, 3, C, C)) / np.sqrt(9 * C) * 0.5,
                                     "c": jnp.zeros(C)})
            tr["g"], tr["b"] = jnp.ones(C), jnp.zeros(C)
            params[trunk_key(l)] = tr
            for c in lay.chans[l]:
                key, k2, k3 = jax.random.split(key, 3)
                ke = lay.emb[c.name].shape[1] + 3
                params[key_of(l, c.name)] = {
                    "W": jnp.zeros((C, c.D)), "bias": jnp.zeros(c.D),
                    "A": jax.random.normal(k2, (C, hc)) / np.sqrt(C), "c": jnp.zeros(hc),
                    "E": jax.random.normal(k3, (ke, hc)) / np.sqrt(ke), "w": jnp.zeros(hc)}
        return params

    # -------------------------------------------------------------- pieces
    @staticmethod
    def _ln(h, g, b):
        mu = jnp.mean(h, -1, keepdims=True)
        var = jnp.mean((h - mu) ** 2, -1, keepdims=True)
        return (h - mu) * jax.lax.rsqrt(var + 1e-5) * g + b

    def _conv(self, x, W, periodic, valid):
        """x (N, h, w, Cin), W (3, 3, Cin, Cout).  valid: no padding (patch);
        else zero padding in y and in x (x wraps when periodic)."""
        if valid:
            pad = ((0, 0), (0, 0))
        elif periodic:
            x = jnp.concatenate([x[:, :, -1:], x, x[:, :, :1]], 2)
            pad = ((1, 1), (0, 0))
        else:
            pad = ((1, 1), (1, 1))
        return jax.lax.conv_general_dilated(x, W, (1, 1), pad, dimension_numbers=("NHWC", "HWIO", "NHWC"),
                                            precision=HI)

    def _trunk(self, tr, x, ok=None, periodic=True):
        """x (N, h, w, Cin).  ok None: dense (zero padding).  ok (N, h, w): patch
        (valid convs, conv inputs masked off-grid).  Returns (N, h', w', C)."""
        valid = ok is not None

        def m(z, i):                                   # mask a conv input whose map is cropped by i
            if not valid:
                return z
            o = ok[:, i:ok.shape[1] - i, i:ok.shape[2] - i]
            return z * o[..., None]

        h = self._conv(m(x, 0), tr["stem"], periodic, valid) + tr["stem_b"]
        for i, bl in enumerate(tr["blocks"], 1):
            z = jax.nn.silu(self._ln(h, bl["g"], bl["b"]))
            d = self._conv(m(z, i), bl["W"], periodic, valid) + bl["c"]
            h = (h[:, 1:-1, 1:-1] if valid else h) + d
        return jax.nn.silu(self._ln(h, tr["g"], tr["b"]))

    def head(self, hp, f, emb, soft, nv, look):
        """f (..., C) site features; soft / nv / look (..., D) -> b (..., D)."""
        mm = lambda a, b: jnp.matmul(a, b, precision=HI)
        b = mm(f, hp["W"]) + hp["bias"]
        e = jnp.concatenate([jnp.broadcast_to(emb, soft.shape + (emb.shape[1],)), soft[..., None],
                             jnp.minimum(nv, 3)[..., None].astype(jnp.float32),
                             jnp.minimum(look, 3)[..., None].astype(jnp.float32)], -1)
        z = jax.nn.silu((mm(f, hp["A"]) + hp["c"])[..., None, :] + mm(e, hp["E"]))
        return b + mm(z, hp["w"])

    # -------------------------------------------------------------- dense
    def _enc(self, v, known, ok, emb):
        e = emb[jnp.maximum(v, 0)] * known[..., None]
        return [e, known[..., None].astype(jnp.float32), ok[..., None].astype(jnp.float32)]

    def dense_planes(self, state, l):
        """(h, w, Cin) input planes of level l from a state (-1 = unknown)."""
        lay = self.lay
        h, w = lay.hw[l]
        one = jnp.ones((h, w), bool)
        out = []
        for c in lay.all_chans[l]:
            v = state[c.name]
            out += self._enc(v, v >= 0, one, lay.emb[c.name])
        if l < lay.T:
            for c in lay.all_chans[l + 1]:
                v = jnp.repeat(jnp.repeat(state[c.name], 2, 0), 2, 1)[:h, :w]
                out += self._enc(v, v >= 0, one, lay.emb[c.name])
        Y, X = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
        out += self._pos(Y, X, h)
        return jnp.concatenate(out, -1).astype(jnp.float32)

    @staticmethod
    def _pos(Y, X, h):
        return [jax.nn.one_hot(2 * (Y & 1) + (X & 1), 4), (Y / max(h - 1, 1))[..., None]]

    def dense_feat(self, params, state, l):
        """(h, w, C) trunk features of level l."""
        x = self.dense_planes(state, l)[None]
        return self._trunk(params[trunk_key(l)], x, None, self.lay.periodic)[0]

    # -------------------------------------------------------------- patch
    def set_step(self, l):
        """{name: (h, w) int} the step at which each sampled level-l cell is written."""
        lay = self.lay
        h, w = lay.hw[l]
        Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        k = (Y % P) * P + (X + (lay.x0 >> l)) % P
        return {c.name: jnp.asarray(lay.base[l] + k * len(lay.chans[l]) + ci, jnp.int32)
                for ci, c in enumerate(lay.chans[l])}

    def _gather(self, a, l, Yc, Xc):
        lay = self.lay
        h, w = lay.hw[l]
        if lay.periodic:
            ok = jnp.ones(Yc.shape, bool)
            Xr = jnp.mod(Xc, w)
        else:
            ok = (Xc >= 0) & (Xc < w)
            Xr = jnp.clip(Xc, 0, w - 1)
        ok = ok & (Yc >= 0) & (Yc < h)
        return a[jnp.clip(Yc, 0, h - 1), Xr], ok

    def patch_planes(self, final, l, Y, X, s, steps):
        """Planes on (2r+1)^2 windows around sites (Y, X) (n,) as the state was
        just before step s (scalar or (n,)).  final: one world's final state.
        Returns x (n, p, p, Cin), ok (n, p, p)."""
        lay, r = self.lay, self.r
        h, _ = lay.hw[l]
        d = jnp.arange(-r, r + 1)
        Yc = Y[:, None, None] + d[None, :, None]
        Xc = X[:, None, None] + d[None, None, :]
        Yc, Xc = jnp.broadcast_arrays(Yc, Xc)
        s = jnp.broadcast_to(jnp.asarray(s), Y.shape)[:, None, None]
        out = []
        okc = None
        for c in lay.all_chans[l]:
            v, ok = self._gather(final[c.name], l, Yc, Xc)
            known = ok & (v >= 0)
            if c.designed is None:
                st, _ = self._gather(steps[c.name], l, Yc, Xc)
                known = known & (st < s)
            okc = ok
            out += self._enc(v, known, ok, lay.emb[c.name])
        if l < lay.T:
            for c in lay.all_chans[l + 1]:
                v, okp = self._gather(final[c.name], l + 1, Yc >> 1, Xc >> 1)
                okp = okp & okc
                out += self._enc(v, okp & (v >= 0), okp, lay.emb[c.name])
        out += self._pos(Yc, Xc, h)
        return jnp.concatenate(out, -1).astype(jnp.float32), okc

    def patch_feat(self, params, final, l, Y, X, s, steps):
        x, ok = self.patch_planes(final, l, Y, X, s, steps)
        return self._trunk(params[trunk_key(l)], x, ok)[:, 0, 0]


# ---------------------------------------------------------------- the pass
def conv_pass(lay, net, params, seed, designed, tau, lam, forced=None, override=None, record=True):
    """cfast's train_pass with the conv actor.  One world.  override: (step,
    site, value, on).  Returns (state, recs) with recs[key] = {soft, nv, look,
    mask, a, valid, logq, cost} each (16, n, ...), or (state, costs) when not
    record; with forced, the draws are the forced values (scoring)."""
    p = lay.p
    state = {c.name: (designed[c.name] if c.designed is not None
                      else jnp.full(lay.hw[c.level], -1, jnp.int32)) for c in p.chans}
    recs = {}
    lqt = jnp.zeros(())
    for l in lay.levels:
        chans = lay.chans[l]
        if not chans:
            continue

        def body(carry, k, l=l, chans=chans):
            state, lqt = carry
            Y, X, valid = lay.sites(l, k)
            h, w = lay.hw[l]
            Xn = jnp.mod(X, w) if lay.periodic else X + (lay.x0 >> l)
            Yi, Xi = jnp.clip(Y, 0, h - 1), jnp.clip(X, 0, w - 1)
            outs = []
            for ci, c in enumerate(chans):
                soft, nv, look = lay.cond(state, c.name, Y, X)
                v = nv + look
                vmin = jnp.min(v, axis=1, keepdims=True)
                mask = (v == vmin) | ~valid[:, None]
                f = net.dense_feat(params, state, l)[Yi, Xi]
                b = net.head(params[key_of(l, c.name)], f, lay.emb[c.name], soft, nv, look)
                logits = jnp.where(mask, -soft / tau - b, -jnp.inf)
                logq = jax.nn.log_softmax(logits, axis=1)
                if forced is not None:
                    a = forced[c.name][Yi, Xi]
                else:
                    g = noise.gumbel(jnp, seed, noise.tag_of(c.name), l, Y, Xn, c.D)
                    a = jnp.argmax(logits + g, axis=1).astype(jnp.int32)
                st = lay.step(l, 0, ci) + k * len(chans)
                if override is not None:
                    os_, oj, oval, on = override
                    hit = on & (st == os_) & (jnp.arange(Y.shape[0]) == oj)
                    a = jnp.where(hit, oval, a)
                idx = jnp.arange(Y.shape[0])
                la = logq[idx, a]
                lqt = lqt + jnp.sum(jnp.where(valid, la, 0.0))
                cost = jnp.where(valid, soft[idx, a] + lam[l] * nv[idx, a] + tau * la, 0.0)
                if record:
                    outs.append({"soft": soft, "nv": nv, "look": look, "mask": mask, "a": a,
                                 "valid": valid, "logq": logq, "cost": cost})
                else:
                    outs.append(cost)
                Ys = jnp.where(valid, Y, h)
                state = {**state, c.name: state[c.name].at[Ys, X].set(a, mode="drop")}
            return (state, lqt), outs

        (state, lqt), ys = jax.lax.scan(body, (state, lqt), jnp.arange(NCOL))
        for ci, c in enumerate(chans):
            recs[key_of(l, c.name)] = ys[ci]
    if forced is not None:
        return state, lqt
    return state, recs
