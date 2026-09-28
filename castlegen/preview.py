"""Preview a tile set's local rules: checkerboard Gibbs on the pairwise and
unary energy only (no reachability, no plan, no gateway), run long enough to
equilibrate on a small grid (annealed from T_hot so that clumped kinds can
nucleate), then decorated and rendered.  This shows what the
rules produce by themselves, which the full castle run cannot while the
coarse levels are missing.

    python -m castlegen.preview [--tileset demo] [--size 64] [--png preview.png]
"""
from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from castlegen import base, render, tileset
from castlegen.core import INF, checkerboard_coords, noise


def sample_local(ts, size=64, sweeps=300, seed=0, T=1.0, T_hot=4.0, init=None):
    """Signature grid (size, size) and its decorated type grid.  init: a
    starting signature grid (default all wall)."""
    P = jnp.zeros((8, ts.n_sig), jnp.float32)
    g = jnp.zeros((1, 8), jnp.float32)
    block_of = jnp.zeros((size, size), jnp.int32)
    d = jnp.full((size, size), INF, jnp.int16)
    no_gate = jnp.array([-10, -10])
    coords = [checkerboard_coords(size, 0), checkerboard_coords(size, 1)]

    start = jnp.full((size, size), ts.WALL, jnp.int32) if init is None else jnp.asarray(init, jnp.int32)

    @jax.jit
    def run(seed):
        tiles = start
        def step(tiles, s):
            colour = s % 2
            c = jnp.where(colour == 0, coords[0], coords[1])
            frac = jnp.minimum(s / sweeps, 1.0)       # anneal over the first half
            p = base.BaseParams(M_reach=jnp.float32(0.0), P=P, lam=jnp.float32(0.0),
                                T=T_hot + (T - T_hot) * frac)
            L = base.base_logits(ts, tiles, d, c, block_of, g, p, no_gate)
            u = noise(seed, 7, s, colour, c[:, 0][:, None], c[:, 1][:, None], jnp.arange(ts.n_sig)[None, :])
            return base.scatter_tiles(tiles, c, base.sample_sites(L, u)), None
        tiles, _ = jax.lax.scan(step, tiles, jnp.arange(2 * sweeps))
        return tiles, tileset.decorate(ts, seed, tiles)
    return run(seed)


def kind_fractions(ts, tiles):
    k = ts.sig_kind[np.asarray(tiles)].ravel()
    counts = np.bincount(k, minlength=len(ts.kinds))
    return {ts.kinds[i].name: round(c / k.size, 3) for i, c in sorted(enumerate(counts), key=lambda x: -x[1]) if c}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tileset", default="demo")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--sweeps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--png", default=None)
    ap.add_argument("--px", type=int, default=8)
    args = ap.parse_args()
    ts = tileset.load(args.tileset)
    tiles, types = sample_local(ts, args.size, args.sweeps, args.seed)
    print(kind_fractions(ts, tiles))
    print(render.ascii_grid(ts, types))
    if args.png:
        render.save_png(render.image(ts, types, args.px), args.png)
        print("wrote", args.png)
