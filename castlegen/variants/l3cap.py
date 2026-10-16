"""l3cap (2026-10-16, agent l3cap): clongfit continued from a saved clongfit model with a changed actor.

Question: is the high-slack L3 deficit a capacity / context-width limit of the L3 head?  Architecture only (allowed freedom).
Objective, kernels, ML recipe and generation are clongfit's.  The start params (longfit2 big300, seed s) are loaded and
the actor is changed in a function-preserving way, so every arm starts from the same function:
  widen=2, widen_levels=(3,): hidden layers of every level-3 net duplicated (width 64 -> 128).  LayerNorm + SiLU nets are
    exactly invariant to duplicating hidden units if the outgoing weights are split (here asymmetrically at random, so the
    duplicates get different gradients and separate).  Per-level: only l3_* params change shape (core.mlp is shape-generic).
  win=(-3, 4): the same-level window grows from 6 x 6 (-2..3) to 8 x 8.  The window is one Layout-wide setting (cone, margin,
    features), so it cannot be per-level without touching the generic pass structure; it applies to ALL levels.  New W1 rows
    (features of the added cells) start at zero.
nctx is the size of the persistent own-draw population in this recipe, not an actor context width, so "context" here is the window.
"""
from __future__ import annotations

import pickle

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench.variant import register

from ._cln_core import window
from .cfitpen import CFitPen

START = "reportables/2026-10-16_longfit2/runs/big300/params_s{seed}.pkl"
OLD_WIN = (-2, 3)


def widen_net(p, k, rng):
    """Duplicate the hidden units of an LN-SiLU 2-hidden-layer net (k copies), function preserving."""
    W1 = np.asarray(p["W1"]); w = W1.shape[1]
    def split(n):                                    # random positive split of 1 over k copies (rows of the outgoing matrix)
        a = rng.uniform(0.3, 1.0, size=(k, n)); return a / a.sum(0)
    out = {}
    for nm in ("W1",):
        out[nm] = np.tile(W1, (1, k))
    for nm in ("b1", "g1", "c1", "b2", "g2", "c2"):
        out[nm] = np.tile(np.asarray(p[nm]), k)
    s2 = split(w)                                    # (k, w): scale for the copy-j row of W2 (layer-1 copies)
    W2 = np.asarray(p["W2"])                         # (w, w)
    rows = np.concatenate([W2 * s2[j][:, None] for j in range(k)], 0)    # (k w, w)
    out["W2"] = np.tile(rows, (1, k))                # layer-2 columns duplicated
    s3 = split(w)
    W3 = np.asarray(p["W3"])
    out["W3"] = np.concatenate([W3 * s3[j][:, None] for j in range(k)], 0)
    out["b3"] = np.asarray(p["b3"])
    return {n: jnp.asarray(v, jnp.float32) for n, v in out.items()}


def pad_window(p, e, old_cells, new_cells, nold_tail_from):
    """W1 rows: [same cells (cell-major, e each) | rest].  Copy old cells to their new positions, zero for the new ones."""
    W1 = np.asarray(p["W1"]); width = W1.shape[1]
    n_same_old = len(old_cells) * e
    rest = W1[n_same_old:]
    new = np.zeros((len(new_cells) * e, width), np.float32)
    pos = {c: i for i, c in enumerate(new_cells)}
    for i, c in enumerate(old_cells):
        j = pos[c]; new[j * e:(j + 1) * e] = W1[i * e:(i + 1) * e]
    return {**p, "W1": jnp.asarray(np.concatenate([new, rest], 0), jnp.float32)}


class L3Cap(CFitPen):
    name = "l3cap"

    def __init__(self, problem, seed, workdir, **cfg):
        cfg = {"upi": 16, "rkl_n": 0, "nctx": 64, "ml_B": 32, "init": None, **cfg}
        self._wide = cfg.pop("widen", 1); self._wide_levels = cfg.pop("widen_levels", (3,)); start = cfg.pop("start", START)
        super().__init__(problem, seed, workdir, **cfg)
        ck = pickle.load(open(start.format(seed=seed), "rb"))
        pr = {k: jax.tree_util.tree_map(np.asarray, v) for k, v in ck["params"].items()}
        assert set(pr) == set(self.params)
        win = self.cfg.get("win")
        rng = np.random.default_rng(1000 + seed)
        if win is not None and tuple(win) != OLD_WIN:
            from .cfast import Region
            lay = self.layout(Region(self.cfg["W_train"] or problem.W_train))
            old = window(*OLD_WIN)
            for key in pr:
                l = int(key[1:key.index("_")])
                e = (lay.feat_dim(l, lay.actor_win) - lay.feat_dim(l, old)) // (len(lay.actor_win.same) - len(old.same))
                pr[key] = {k: np.asarray(v) for k, v in pad_window(pr[key], e, old.same, lay.actor_win.same, None).items()}
                assert pr[key]["W1"].shape[0] == self.params[key]["W1"].shape[0], (key, pr[key]["W1"].shape, self.params[key]["W1"].shape)
        if self._wide > 1:
            for key in pr:
                if int(key[1:key.index("_")]) in self._wide_levels:
                    pr[key] = widen_net(pr[key], self._wide, rng)
        self.params = jax.tree_util.tree_map(jnp.asarray, pr)
        self.lam = np.asarray(ck["lam"], float)
        self.start = start.format(seed=seed)


register(L3Cap)


def _mk(name, **over):
    cls = type(f"L3Cap_{name}", (L3Cap,), {"name": name})
    orig = cls.__init__

    def init(self, problem, seed, workdir, **cfg):
        orig(self, problem, seed, workdir, **{**over, **cfg})
    cls.__init__ = init
    return register(cls)


_mk("l3cap_ctl")
_mk("l3cap_w2", widen=2, widen_levels=(3,))
_mk("l3cap_c2", win=(-3, 4))
_mk("l3cap_wc2", widen=2, widen_levels=(3,), win=(-3, 4))
