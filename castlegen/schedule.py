"""Fixed-step base schedule: 20 Gibbs steps, d relaxation, fallback.

Coarse levels, window resampling and count splitting are not implemented
yet; this is the minimum needed for the oracle-plan experiment, where the
level-1 plan (h per block, entry-side d) is taken from a reference sample.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import base
from .core import (GATE, INF, N_TILES, N_TYPES, WALL, checkerboard_coords,
                   noise, split_tile)

SIZE = 192
BLOCK = 4
N_BASE_STEPS = 20
N_D_STEPS_AFTER = 8   # stands in for the validity phase until it exists
N_FALLBACK_ROUNDS = 6


class Presets(NamedTuple):
    M: float = 20.0
    M_door: float = 2.0
    M_reach: float = 20.0
    room_density: float = 0.33
    u_wall: float = 0.7
    w_door: float = 0.0
    T_start: float = 1.5
    T_end: float = 1.0
    T_steps: int = 12


def block_index(size=SIZE, block=BLOCK):
    ys, xs = jnp.meshgrid(jnp.arange(size), jnp.arange(size), indexing="ij")
    return (ys // block) * (size // block) + (xs // block)


def type_hist_per_block(tiles, block_of, n_blocks):
    typ, _, is_room, _, _ = split_tile(tiles)
    onehot = jax.nn.one_hot(typ, N_TYPES) * is_room[..., None]
    return jax.ops.segment_sum(onehot.reshape(-1, N_TYPES), block_of.reshape(-1),
                               num_segments=n_blocks) / (BLOCK * BLOCK)


def run_base(castle_id, tiles0, d0, gate_yx, A, P, h_plan, Lam, presets: Presets):
    """Run the base schedule from an initial state.

    tiles0, d0 : initial grids (from a plan or a random init)
    h_plan     : (n_blocks, 8) level-1 plan for the type histogram projection
    Lam        : (8, 8) pin precision
    Returns final tiles and d (after fallback).
    """
    size = tiles0.shape[0]
    block_of = block_index(size)
    n_blocks = (size // BLOCK) ** 2
    coords = [checkerboard_coords(size, 0), checkerboard_coords(size, 1)]

    def step(carry, s):
        tiles, d = carry
        frac = jnp.minimum(s / presets.T_steps, 1.0)
        T = presets.T_start + (presets.T_end - presets.T_start) * frac
        lam = 1.0 - s / N_BASE_STEPS
        colour = s % 2
        c = jnp.where(colour == 0, coords[0], coords[1])
        hist = type_hist_per_block(tiles, block_of, n_blocks)
        g = (hist @ P.T - h_plan) @ Lam                          # (n_blocks, 8)
        p = base.BaseParams(M=jnp.float32(presets.M), M_door=jnp.float32(presets.M_door),
                            M_reach=jnp.float32(presets.M_reach), A=A,
                            u_type=jnp.zeros(N_TYPES, jnp.float32),
                            u_wall=jnp.float32(presets.u_wall),
                            w_door=jnp.float32(presets.w_door),
                            P=P, lam=jnp.float32(lam), T=jnp.float32(T))
        logits = base.base_logits(tiles, d, c, block_of, g, p, gate_yx)
        u = noise(castle_id, 0, s, colour, c[:, 0][:, None], c[:, 1][:, None],
                  jnp.arange(N_TILES)[None, :])
        new = base.sample_sites(logits, u, T)
        tiles = base.scatter_tiles(tiles, c, new)
        tiles = base.sync_bits(tiles, colour)
        d = base.d_step(d, tiles)
        return (tiles, d), None

    (tiles, d), _ = jax.lax.scan(step, (tiles0, d0), jnp.arange(N_BASE_STEPS))
    for _ in range(N_D_STEPS_AFTER):
        d = base.d_step(d, tiles)

    # fallback: a fixed number of rounds of {wall violating cells, wall unreached
    # rooms, clear dangling door bits, relax d}.  Each round propagates the
    # consequence of a walled cell one step along its chain, so the residual
    # after N_FALLBACK_ROUNDS is the beyond-horizon failure the doc measures.
    tiles_pre = tiles
    def fb_round(carry, _):
        tiles, d = carry
        _, _, is_room, _, is_gate = split_tile(tiles)
        v = base.violations(d, tiles)
        tiles = jnp.where((v > 0) & ~is_gate, WALL, tiles)
        tiles = jnp.where(is_room & (d == INF) & ~is_gate, WALL, tiles)
        tiles = base.clear_half_doors(tiles)
        for _ in range(2):
            d = base.d_step(d, tiles)
        return (tiles, d), None
    (tiles, d), _ = jax.lax.scan(fb_round, (tiles, d), None, length=N_FALLBACK_ROUNDS)
    return tiles, d


def init_random(castle_id, size, gate_yx, presets: Presets):
    """Random initial state from the unary only, drawn from the hashed noise."""
    ys, xs = jnp.meshgrid(jnp.arange(size), jnp.arange(size), indexing="ij")
    u_room = noise(castle_id, 0, 999, 0, ys, xs, 0)
    u_type = noise(castle_id, 0, 999, 0, ys, xs, 1)
    typ = jnp.minimum((u_type * N_TYPES).astype(jnp.int32), N_TYPES - 1)
    mask = jnp.zeros_like(typ)
    for b in range(4):
        mask = mask | ((noise(castle_id, 0, 999, 0, ys, xs, 2 + b) < 0.5).astype(jnp.int32) << b)
    is_room = u_room < presets.room_density
    perim = (ys == 0) | (ys == size - 1) | (xs == 0) | (xs == size - 1)
    tiles = jnp.where(is_room & ~perim, typ * 16 + mask, WALL)
    tiles = tiles.at[gate_yx[0], gate_yx[1]].set(GATE)
    d = jnp.full((size, size), INF, jnp.int16).at[gate_yx[0], gate_yx[1]].set(0)
    return tiles, d


def rooms_lost(tiles_before_fallback, tiles_after):
    _, _, r0, _, _ = split_tile(tiles_before_fallback)
    _, _, r1, _, _ = split_tile(tiles_after)
    return 1.0 - r1.sum() / jnp.maximum(r0.sum(), 1)
