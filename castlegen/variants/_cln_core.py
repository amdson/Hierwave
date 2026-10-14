"""cln: per-variant actor network and actor window for the coloured pass.

- Network.  "tanh" is the cpass MLP (core.mlp_init / core.mlp).  "ln" is the
  same 64 x 2 shape with Linear -> LayerNorm -> SiLU hidden layers (same init
  convention: scaled normal hidden weights, zero output layer, so b = 0 at
  step 0).  Its parameters carry LayerNorm gains "g1", "g2", which is how
  core.mlp dispatches to ln_mlp.
- Window.  window(lo, hi) builds a symmetric same-level window of offsets
  lo..hi from the parent block's corner (default cpass: -1..2, a 4 x 4; the
  6 x 6 option is -2..3), parent offsets -1..1 as before.  The cone
  (_cfast_cone.fine_reach) and core.margin read the window, so both follow it.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def window(lo=-1, hi=2, plo=-1, phi=1):
    from ._cpass_core import Window
    return Window(same=tuple((dy, dx) for dy in range(lo, hi + 1) for dx in range(lo, hi + 1)),
                  parent=tuple((dy, dx) for dy in range(plo, phi + 1) for dx in range(plo, phi + 1)))


def window_of(spec):
    """cfg value -> Window or None (None: core.ACTOR_WIN).  spec: None, (lo, hi)
    or (lo, hi, plo, phi)."""
    if spec is None:
        return None
    return window(*spec)


def win_reach(win):
    """Same-level cells (at the site's level) a draw reads away from its site
    through the window, in either direction and axis."""
    r = 0
    for dy, dx in win.same:                 # corner = 2 (Y >> 1) in {Y - 1, Y}
        for d in (dy, dx):
            r = max(r, 1 - d, d)
    for dy, dx in win.parent:               # parent cell covers rows 2 (P + d) .. 2 (P + d) + 1
        for d in (dy, dx):
            r = max(r, 1 - 2 * d, 2 * d + 1)
    return r


def ln_mlp_init(key, din, dout, width=64):
    k1, k2 = jax.random.split(key)
    return {"W1": jax.random.normal(k1, (din, width)) / np.sqrt(din), "b1": jnp.zeros(width),
            "g1": jnp.ones(width), "c1": jnp.zeros(width),
            "W2": jax.random.normal(k2, (width, width)) / np.sqrt(width), "b2": jnp.zeros(width),
            "g2": jnp.ones(width), "c2": jnp.zeros(width),
            "W3": jnp.zeros((width, dout)), "b3": jnp.zeros(dout)}


def _ln_silu(h, g, c):
    mu = h.mean(-1, keepdims=True)
    var = ((h - mu) ** 2).mean(-1, keepdims=True)
    return jax.nn.silu((h - mu) * jax.lax.rsqrt(var + 1e-5) * g + c)


def ln_hidden1(p, x):
    return _ln_silu(x @ p["W1"] + p["b1"], p["g1"], p["c1"])


def ln_mlp(p, x):
    h = ln_hidden1(p, x)
    h = _ln_silu(h @ p["W2"] + p["b2"], p["g2"], p["c2"])
    return h @ p["W3"] + p["b3"]


def hidden1(p, x):
    """The actor's first hidden layer, whichever network (cfast's h1 critic)."""
    if "g1" in p:
        return ln_hidden1(p, x)
    return jnp.tanh(x @ p["W1"] + p["b1"])


def net_init(net, key, din, dout, width):
    from ._cpass_core import mlp_init
    if net == "ln":
        return ln_mlp_init(key, din, dout, width)
    assert net == "tanh", net
    return mlp_init(key, din, dout, width)
