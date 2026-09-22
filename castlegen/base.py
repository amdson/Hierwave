"""Base level: checkerboard Gibbs on tiles and the Jacobi step of the
distance equality d_i = 1 + min over door-neighbours d_j.

Every active site scores all 52 tiles as a dense (n, 52) tensor.  Doors are
a fixed function of the two types (both facing bits set), so the energy is a
plain pairwise table and no variables live on edges.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .core import BIT, DIRS, GATE, INF, N_TILES, N_TYPES, POP, WALL, gumbel, split_tile


class BaseParams(NamedTuple):
    M: jnp.ndarray        # hard penalty, scalar
    M_reach: jnp.ndarray  # penalty for a room tile with no door to a finite-d neighbour
    A: jnp.ndarray        # (50, 50) soft adjacency, applies across a door
    u_type: jnp.ndarray   # (50,)
    u_wall: jnp.ndarray   # scalar
    w_door: jnp.ndarray   # scalar, per door bit of the type (unary)
    P: jnp.ndarray        # (8, 50) projection of the type histogram
    lam: jnp.ndarray      # pin weight (ramped by the schedule)
    T: jnp.ndarray        # temperature


def _pad(a, fill):
    return jnp.pad(a, 1, constant_values=fill)


def base_logits(tiles, d, coords, block_of, g, p: BaseParams, gate_yx):
    """Logits (n, 52) for the active sites.

    tiles    : (S, S) int32 tile indices
    d        : (S, S) int16 current distance field
    coords   : (n, 2) active site coordinates
    block_of : (S, S) int32 level-1 block index per cell
    g        : (n_blocks, 8) pin gradient  Lambda (P hist_B - h_B)
    gate_yx  : (2,) the pinned gateway cell
    """
    S = tiles.shape[0]
    tiles_p = _pad(tiles, WALL)
    d_p = _pad(d, INF)
    ys, xs = coords[:, 0] + 1, coords[:, 1] + 1
    n = coords.shape[0]

    # per-type unary, laid out over all 52 tiles (wall and gate filled below)
    E = jnp.concatenate([p.u_type + p.w_door * POP[:N_TYPES], jnp.zeros(2, jnp.float32)])
    E = jnp.broadcast_to(E[None, :], (n, N_TILES))
    reach = jnp.zeros((n, N_TILES), jnp.float32)
    A52 = jnp.zeros((N_TILES, N_TILES), jnp.float32).at[:N_TYPES, :N_TYPES].set(p.A)
    for dy, dx, own, facing in DIRS:
        nt = tiles_p[ys + dy, xs + dx]                                # (n,) neighbour tile
        door = BIT[own][None, :] * BIT[facing][nt][:, None]           # (n, 52): a door would exist
        E = E + A52[:, nt].T * door                                   # soft adjacency across doors
        finite = (d_p[ys + dy, xs + dx] < INF).astype(jnp.float32)
        reach = jnp.maximum(reach, door * finite[:, None])
    E = E + p.M_reach * (1.0 - reach)

    pin = g[block_of[coords[:, 0], coords[:, 1]]] @ p.P               # (n, 50)
    E = E.at[:, :N_TYPES].add(p.lam * pin)
    E = E.at[:, WALL].set(p.u_wall)

    L = -E
    big = jnp.float32(1e9)
    on_gate = (coords[:, 0] == gate_yx[0]) & (coords[:, 1] == gate_yx[1])
    on_perim = (coords[:, 0] == 0) | (coords[:, 0] == S - 1) | (coords[:, 1] == 0) | (coords[:, 1] == S - 1)
    below = (coords[:, 0] == gate_yx[0] + 1) & (coords[:, 1] == gate_yx[1])
    room_cols = jnp.arange(N_TILES) < N_TYPES
    L = jnp.where((on_gate | on_perim)[:, None] & room_cols[None, :], -big, L)   # enclosure
    L = L.at[:, GATE].set(jnp.where(on_gate, big, -big))
    L = L.at[:, WALL].set(jnp.where(on_gate | below, -big, L[:, WALL]))
    L = jnp.where(below[:, None] & (BIT[0][None, :] == 0) & room_cols[None, :], -big, L)  # north door under the gate
    return L


def sample_sites(logits, u, T):
    """Gumbel-max over the 52 tiles; u are hashed uniforms of logits' shape."""
    return jnp.argmax(logits / T + gumbel(u), axis=1)


def scatter_tiles(tiles, coords, new):
    return tiles.at[coords[:, 0], coords[:, 1]].set(new.astype(tiles.dtype))


def _door_field(tiles):
    """For each direction, (S, S) int: a door exists from this cell to that neighbour."""
    S = tiles.shape[0]
    _, mask, _, _, _ = split_tile(tiles)
    mask_p = _pad(mask, 0)
    out = []
    for dy, dx, own, facing in DIRS:
        nmask = mask_p[1 + dy: S + 1 + dy, 1 + dx: S + 1 + dx]
        out.append(((mask >> own) & 1) & ((nmask >> facing) & 1))
    return out


def d_step(d, tiles):
    """One Jacobi step of d_i = 1 + min_{door-neighbours} d_j.

    Can raise or lower a value; walls stay INF, the gateway stays 0.
    """
    _, _, is_room, _, is_gate = split_tile(tiles)
    S = d.shape[0]
    d_p = _pad(d, INF)
    best = jnp.full_like(d, INF)
    for (dy, dx, own, facing), door in zip(DIRS, _door_field(tiles)):
        nd = d_p[1 + dy: S + 1 + dy, 1 + dx: S + 1 + dx]
        best = jnp.minimum(best, jnp.where(door == 1, nd, INF))
    new = jnp.minimum(best.astype(jnp.int32) + 1, jnp.int32(INF)).astype(jnp.int16)
    return jnp.where(is_gate, jnp.int16(0), jnp.where(is_room, new, INF))


def violations(d, tiles):
    """Per-cell count of violated hard terms (d equality, enclosure)."""
    _, _, is_room, _, is_gate = split_tile(tiles)
    S = d.shape[0]
    v = jnp.zeros_like(d, dtype=jnp.int32)
    supported = d_step(d, tiles) == d
    v = v + jnp.where(is_room & ~is_gate, (~supported).astype(jnp.int32), 0)
    S1 = S - 1
    perim = jnp.zeros_like(v, dtype=bool).at[0, :].set(True).at[S1, :].set(True).at[:, 0].set(True).at[:, S1].set(True)
    v = v + jnp.where(perim & is_room, 1, 0)
    return v
