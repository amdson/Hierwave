"""Legacy (G0, JAX castle sampler era); superseded by castlegen/channels. See the original README.md (git history).

Fixed-step base schedule: annealed checkerboard Gibbs with a d Jacobi step
after every colour, then a few extra d relaxations.  No repair: the output
may contain rooms with violated hard terms; `base.violations` reports them.

Coarse levels, window resampling and count splitting are not implemented
yet; this is the minimum needed for the oracle-plan experiment, where the
level-1 plan (h per block, entry-side d) is taken from a reference sample.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import base
from .core import INF, checkerboard_coords, noise
from .tileset import TileSet

SIZE = 20
BLOCK = 4


class Presets(NamedTuple):
    M_reach: float = 20.0
    room_density: float = 0.33
    n_sweeps: int = 1000   # one sweep = both checkerboard colours
    T_start: float = 4.0
    T_end: float = 1.0
    T_frac: float = 0.7   # anneal over this fraction of the run, then hold T_end
    d_iters: int = 64     # Jacobi iterations when d is rebuilt from scratch each step


def block_index(size=SIZE, block=BLOCK):
    ys, xs = jnp.meshgrid(jnp.arange(size), jnp.arange(size), indexing="ij")
    return (ys // block) * (size // block) + (xs // block)


def sig_hist_per_block(ts: TileSet, tiles, block_of, n_blocks):
    """(n_blocks, S) signature histogram per block; wall and gateway not counted."""
    onehot = jax.nn.one_hot(tiles, ts.n_sig) * ts.jt.plannable[tiles][..., None]
    return jax.ops.segment_sum(onehot.reshape(-1, ts.n_sig), block_of.reshape(-1),
                               num_segments=n_blocks) / (BLOCK * BLOCK)


def run_base(ts: TileSet, castle_id, tiles0, d0, gate_yx, P, h_plan, Lam, presets: Presets):
    """Run the base schedule from an initial state.

    tiles0, d0 : initial grids of signature ids and distances (from a plan or a random init)
    P          : (8, S) projection of the signature histogram
    h_plan     : (n_blocks, 8) level-1 plan for the histogram projection
    Lam        : (8, 8) pin precision
    Returns final tiles and d.
    """
    size = tiles0.shape[0]
    block_of = block_index(size)
    n_blocks = (size // BLOCK) ** 2
    coords = [checkerboard_coords(size, 0), checkerboard_coords(size, 1)]

    def step(carry, s):
        tiles, d = carry
        frac = jnp.minimum(s / (presets.T_frac * n_steps), 1.0)
        T = presets.T_start + (presets.T_end - presets.T_start) * frac
        lam = 1.0 - s / n_steps
        colour = s % 2
        c = jnp.where(colour == 0, coords[0], coords[1])
        hist = sig_hist_per_block(ts, tiles, block_of, n_blocks)
        g = (hist @ P.T - h_plan) @ Lam                          # (n_blocks, 8)
        p = base.BaseParams(M_reach=jnp.float32(presets.M_reach), P=P,
                            lam=jnp.float32(lam), T=jnp.float32(T))
        logits = base.base_logits(ts, tiles, d, c, block_of, g, p, gate_yx)
        u = noise(castle_id, 0, s, colour, c[:, 0][:, None], c[:, 1][:, None],
                  jnp.arange(ts.n_sig)[None, :])
        new = base.sample_sites(logits, u)
        tiles = base.scatter_tiles(tiles, c, new)
        d = base.d_relax(ts, tiles, presets.d_iters)
        return (tiles, d), None

    # d is never carried: it is rebuilt from scratch for the current tiles, so a
    # finite value is always a real path (d0 is unused until coarse seeding exists)
    n_steps = 2 * presets.n_sweeps
    d = base.d_relax(ts, tiles0, presets.d_iters)
    (tiles, d), _ = jax.lax.scan(step, (tiles0, d), jnp.arange(n_steps))
    return tiles, d


def init_random(ts: TileSet, castle_id, size, gate_yx, presets: Presets):
    """Random initial state: room signatures uniformly at the target density."""
    ys, xs = jnp.meshgrid(jnp.arange(size), jnp.arange(size), indexing="ij")
    u_room = noise(castle_id, 0, 999, 0, ys, xs, 0)
    u_type = noise(castle_id, 0, 999, 0, ys, xs, 1)
    rooms = jnp.asarray(np.nonzero(np.asarray(ts.jt.is_room))[0], jnp.int32)
    typ = rooms[jnp.minimum((u_type * len(rooms)).astype(jnp.int32), len(rooms) - 1)]
    is_room = u_room < presets.room_density
    perim = (ys == 0) | (ys == size - 1) | (xs == 0) | (xs == size - 1)
    tiles = jnp.where(is_room & ~perim, typ, ts.WALL)
    tiles = tiles.at[gate_yx[0], gate_yx[1]].set(ts.GATE)
    d = jnp.full((size, size), INF, jnp.int16).at[gate_yx[0], gate_yx[1]].set(0)
    return tiles, d

