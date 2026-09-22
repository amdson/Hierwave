"""Base level: checkerboard Gibbs on tiles and the Jacobi step of the
distance equality d_i = 1 + min over door-neighbours d_j.

Everything is dense and static-shape: every active site scores all 802
tiles; infeasible sockets are pushed to -inf by the hard penalty M rather
than enumerated.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .core import (BIT, DIRS, GATE, INF, N_MASKS, N_ROOM_TILES, N_TYPES, POP,
                   WALL, gumbel, split_tile)


class BaseParams(NamedTuple):
    M: jnp.ndarray        # hard penalty, scalar
    M_door: jnp.ndarray   # soft penalty for a half door (bit set, facing bit unset)
    M_reach: jnp.ndarray  # penalty for a room tile no door-neighbour of which has finite d
    A: jnp.ndarray        # (50, 50) soft adjacency, applies across a door
    u_type: jnp.ndarray   # (50,)
    u_wall: jnp.ndarray   # scalar
    w_door: jnp.ndarray   # scalar, per set door bit
    P: jnp.ndarray        # (8, 50) projection of the type histogram
    lam: jnp.ndarray      # pin weight (ramped by the schedule)
    T: jnp.ndarray        # temperature


def _pad(a, fill):
    return jnp.pad(a, 1, constant_values=fill)


def base_logits(tiles, d, coords, block_of, g, p: BaseParams, gate_yx):
    """Logits (n, 802) for the active sites.

    tiles     : (S, S) int32
    d         : (S, S) int16 current distance field
    coords    : (n, 2) active site coordinates
    block_of  : (S, S) int32 level-1 block index per cell
    g         : (n_blocks, 8) pin gradient  Lambda (P hist_B - h_B)
    gate_yx   : (2,) the pinned gateway cell
    """
    S = tiles.shape[0]
    typ, mask, is_room, is_wall, is_gate = split_tile(tiles)
    # pad with walls so the perimeter sees a doorless neighbour outside
    typ_p = _pad(typ, 0)
    room_p = _pad(is_room | is_gate, False)
    d_p = _pad(d, INF)
    ys, xs = coords[:, 0] + 1, coords[:, 1] + 1

    L = p.u_type[None, :, None] + p.w_door * POP[None, None, :]      # (1, 50, 16)
    L = jnp.broadcast_to(L, (coords.shape[0], N_TYPES, N_MASKS))
    reach = jnp.zeros((coords.shape[0], N_MASKS), jnp.float32)   # (n, 16)
    for dy, dx, own, facing in DIRS:
        nt = typ_p[ys + dy, xs + dx]                                     # (n,)
        nroom = room_p[ys + dy, xs + dx].astype(jnp.float32)             # neighbour is a room (or the gate)
        cbit = BIT[own]                                                  # (16,)
        both = nroom[:, None] * cbit[None, :]                            # (n, 16): a door would exist
        # a door bit toward a non-room is infeasible: doors are edge variables
        L = L + p.M * ((1.0 - nroom)[:, None] * cbit[None, :])[:, None, :]
        L = L + p.A[:, nt].T[:, :, None] * both[:, None, :]
        finite = (d_p[ys + dy, xs + dx] < INF).astype(jnp.float32)
        reach = jnp.maximum(reach, both * finite[:, None])
    L = L + p.M_reach * (1.0 - reach)[:, None, :]
    pin = g[block_of[coords[:, 0], coords[:, 1]]] @ p.P                  # (n, 50)
    L = L - p.lam * pin[:, :, None]

    room = -L.reshape(-1, N_ROOM_TILES)          # logits = -energy
    wall = -jnp.broadcast_to(p.u_wall, (coords.shape[0],))
    on_gate = (coords[:, 0] == gate_yx[0]) & (coords[:, 1] == gate_yx[1])
    on_perim = (coords[:, 0] == 0) | (coords[:, 0] == S - 1) | (coords[:, 1] == 0) | (coords[:, 1] == S - 1)
    big = jnp.float32(1e9)
    # gate: only the pinned cell may take it; the pinned cell may take nothing else
    gate = jnp.where(on_gate, big, -big)
    below = (coords[:, 0] == gate_yx[0] + 1) & (coords[:, 1] == gate_yx[1])
    has_north = jnp.broadcast_to(BIT[0][None, :], (N_TYPES, N_MASKS)).reshape(-1) > 0
    room = jnp.where(below[:, None] & ~has_north[None, :], -big, room)
    room = jnp.where((on_gate | on_perim)[:, None], -big, room)   # enclosure: perimeter is wall
    wall = jnp.where(on_gate | below, -big, wall)
    return jnp.concatenate([room, wall[:, None], gate[:, None]], axis=1)


def sample_sites(logits, u, T):
    """Gumbel-max over the 802 tiles; u are hashed uniforms of logits' shape."""
    return jnp.argmax(logits / T + gumbel(u), axis=1)


def scatter_tiles(tiles, coords, new):
    return tiles.at[coords[:, 0], coords[:, 1]].set(new.astype(tiles.dtype))


def sync_bits(tiles, colour):
    """Doors are edge variables owned by the colour just updated: every cell of
    the other colour copies its four facing bits from its neighbours.  After
    this, facing bits agree on every edge."""
    typ, mask, is_room, _, is_gate = split_tile(tiles)
    S = tiles.shape[0]
    mask_p = _pad(mask, 0)
    room_p = _pad(is_room | is_gate, False)
    new_mask = jnp.zeros_like(mask)
    for dy, dx, own, facing in DIRS:
        nmask = mask_p[1 + dy: S + 1 + dy, 1 + dx: S + 1 + dx]
        nroom = room_p[1 + dy: S + 1 + dy, 1 + dx: S + 1 + dx]
        new_mask = new_mask | (jnp.where(nroom, (nmask >> facing) & 1, 0) << own)
    ys, xs = jnp.meshgrid(jnp.arange(S), jnp.arange(S), indexing="ij")
    inactive = ((ys + xs) % 2) != colour
    synced = typ * N_MASKS + new_mask
    return jnp.where(is_room & inactive, synced, tiles)


def d_step(d, tiles):
    """One Jacobi step of d_i = 1 + min_{door-neighbours} d_j.

    Can raise or lower a value; walls stay INF, the gateway stays 0.
    """
    _, mask, is_room, _, is_gate = split_tile(tiles)
    d_p = _pad(d, INF)
    mask_p = _pad(mask, 0)
    S = d.shape[0]
    best = jnp.full_like(d, INF)
    for dy, dx, own, facing in DIRS:
        nmask = mask_p[1 + dy: S + 1 + dy, 1 + dx: S + 1 + dx]
        nd = d_p[1 + dy: S + 1 + dy, 1 + dx: S + 1 + dx]
        door = ((mask >> own) & 1) & ((nmask >> facing) & 1)
        best = jnp.minimum(best, jnp.where(door == 1, nd, INF))
    new = jnp.minimum(best.astype(jnp.int32) + 1, jnp.int32(INF)).astype(jnp.int16)
    return jnp.where(is_gate, jnp.int16(0), jnp.where(is_room, new, INF))


def half_doors(tiles):
    """Per-cell count of set door bits whose facing bit is unset."""
    _, mask, is_room, _, _ = split_tile(tiles)
    S = tiles.shape[0]
    mask_p = _pad(mask, 0)
    v = jnp.zeros(tiles.shape, jnp.int32)
    for dy, dx, own, facing in DIRS:
        nmask = mask_p[1 + dy: S + 1 + dy, 1 + dx: S + 1 + dx]
        v = v + (((mask >> own) & 1) & (1 - ((nmask >> facing) & 1))).astype(jnp.int32)
    return jnp.where(is_room, v, 0)


def clear_half_doors(tiles):
    """Clear every door bit whose facing bit is unset (local repair, no walling)."""
    typ, mask, is_room, _, _ = split_tile(tiles)
    S = tiles.shape[0]
    mask_p = _pad(mask, 0)
    keep = jnp.zeros_like(mask)
    for dy, dx, own, facing in DIRS:
        nmask = mask_p[1 + dy: S + 1 + dy, 1 + dx: S + 1 + dx]
        keep = keep | ((((mask >> own) & 1) & ((nmask >> facing) & 1)) << own)
    return jnp.where(is_room, typ * N_MASKS + keep, tiles)


def violations(d, tiles):
    """Per-cell count of violated hard terms (d equality, enclosure)."""
    _, mask, is_room, _, is_gate = split_tile(tiles)
    S = d.shape[0]
    v = jnp.zeros_like(d, dtype=jnp.int32)
    supported = d_step(d, tiles) == d
    v = v + jnp.where(is_room & ~is_gate, (~supported).astype(jnp.int32), 0)
    S1 = S - 1
    perim = jnp.zeros_like(v, dtype=bool).at[0, :].set(True).at[S1, :].set(True).at[:, 0].set(True).at[:, S1].set(True)
    v = v + jnp.where(perim & is_room, 1, 0)
    return v
