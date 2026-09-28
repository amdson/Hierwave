"""Base level: checkerboard Gibbs on tile signatures and the Jacobi step of
the distance equality d_i = 1 + min over door-neighbours d_j.

Every active site scores all S signatures of the tile set as a dense (n, S)
tensor.  Doors are fixed by the signatures' sockets, so every pairwise term
(coupling across a door, contact, dangling doors) is folded offline into two
tables Eh[left, right] and Ev[top, bottom]; nothing is modelled on edges.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .core import INF, gumbel
from .tileset import TileSet


class BaseParams(NamedTuple):
    M_reach: jnp.ndarray  # penalty for a room with no door to a finite-d neighbour
    P: jnp.ndarray        # (8, S) projection of the signature histogram
    lam: jnp.ndarray      # pin weight (ramped by the schedule)
    T: jnp.ndarray        # temperature (applies to energies, not to the base mass logz)


def _pad(a, fill):
    return jnp.pad(a, 1, constant_values=fill)


def _neighbours(tiles_p, d_p, ys, xs):
    """(up, right, down, left) neighbour signatures and d at padded coords."""
    return ([tiles_p[ys - 1, xs], tiles_p[ys, xs + 1], tiles_p[ys + 1, xs], tiles_p[ys, xs - 1]],
            [d_p[ys - 1, xs], d_p[ys, xs + 1], d_p[ys + 1, xs], d_p[ys, xs - 1]])


def base_logits(ts: TileSet, tiles, d, coords, block_of, g, p: BaseParams, gate_yx, enclose=True):
    """Tempered logits (n, S) for the active sites; sample with argmax(L + Gumbel).

    tiles    : (H, W) int32 signature ids
    d        : (H, W) int16 distance field of the current tiles, relaxed from
               scratch (d_relax), so finite d means a real path to the gate
    coords   : (n, 2) active site coordinates
    block_of : (H, W) int32 level-1 block index per cell
    g        : (n_blocks, 8) pin gradient  Lambda (P hist_B - h_B)
    gate_yx  : (2,) the pinned gateway cell (off the grid for none)
    enclose  : forbid rooms on the grid perimeter (off for coarse levels)

    Connectivity, both terms with weight M_reach:
      own       a room candidate needs a door to a neighbour with finite d that
                is not one of this site's children;
      children  a neighbour k with d_k = d_i + 1 and a door to the current tile
                (i is a shortest-path parent of k) keeps its support only if the
                candidate keeps the door to k and is itself supported.
    Counting every parent (not only sole parents) is conservative, and it is
    what makes simultaneous updates of one checkerboard colour safe: k's other
    parents are active in the same step and protect k too.
    """
    t = ts.jt
    H, W = tiles.shape
    ys, xs = coords[:, 0] + 1, coords[:, 1] + 1
    tiles_p = _pad(tiles, ts.WALL)
    (up, rt, dn, lf), nd = _neighbours(tiles_p, _pad(d, INF), ys, xs)
    cur = tiles_p[ys, xs]
    di = d[coords[:, 0], coords[:, 1]].astype(jnp.int32)

    # candidate v at this site: above is (up, v), right is (v, rt), ...
    E = t.Ev[up, :] + t.Eh[:, rt].T + t.Ev[:, dn].T + t.Eh[lf, :]
    doors = (t.Dv[up, :], t.Dh[:, rt].T, t.Dv[:, dn].T, t.Dh[lf, :])        # (n, S) each
    cur_doors = (t.Dv[up, cur], t.Dh[cur, rt], t.Dv[cur, dn], t.Dh[lf, cur])  # (n,) each
    finite = [(dj < INF).astype(jnp.float32) for dj in nd]
    child = [cd * f * (di < INF) * (dj.astype(jnp.int32) == di + 1)
             for cd, f, dj in zip(cur_doors, finite, nd)]
    reach = jnp.zeros_like(E)
    for door, f, c in zip(doors, finite, child):
        reach = jnp.maximum(reach, door * (f * (1.0 - c))[:, None])
    E = E + p.M_reach * (1.0 - reach) * t.is_room[None, :]
    for door, c in zip(doors, child):
        E = E + p.M_reach * c[:, None] * (1.0 - door * reach)

    pin = g[block_of[coords[:, 0], coords[:, 1]]] @ p.P               # (n, S)
    E = E + p.lam * pin * t.plannable[None, :]
    L = -E / p.T + t.logz[None, :]

    big = jnp.float32(1e9)
    on_gate = (coords[:, 0] == gate_yx[0]) & (coords[:, 1] == gate_yx[1])
    on_perim = (coords[:, 0] == 0) | (coords[:, 0] == H - 1) | (coords[:, 1] == 0) | (coords[:, 1] == W - 1)
    on_perim = on_perim & enclose
    below = (coords[:, 0] == gate_yx[0] + 1) & (coords[:, 1] == gate_yx[1])
    L = jnp.where((on_gate | on_perim)[:, None] & t.is_room[None, :], -big, L)     # enclosure
    L = L.at[:, ts.GATE].set(jnp.where(on_gate, big, -big))
    L = jnp.where(below[:, None] & (t.Dv[ts.GATE, :] == 0)[None, :], -big, L)     # a door under the gate
    return L


def sample_sites(logits, u):
    """Gumbel-max over the signatures; u are hashed uniforms of logits' shape."""
    return jnp.argmax(logits + gumbel(u), axis=1)


def scatter_tiles(tiles, coords, new):
    return tiles.at[coords[:, 0], coords[:, 1]].set(new.astype(tiles.dtype))


def door_field(ts: TileSet, tiles):
    """(up, right, down, left) float 0/1 grids: a door joins this cell to that neighbour."""
    t = ts.jt
    tp = _pad(tiles, ts.WALL)
    up, dn, lf, rt = tp[:-2, 1:-1], tp[2:, 1:-1], tp[1:-1, :-2], tp[1:-1, 2:]
    return t.Dv[up, tiles], t.Dh[tiles, rt], t.Dv[tiles, dn], t.Dh[lf, tiles]


def d_step(ts: TileSet, d, tiles):
    """One Jacobi step of d_i = 1 + min_{door-neighbours} d_j.

    Can raise or lower a value; non-rooms stay INF, the gateway stays 0.
    """
    dp = _pad(d, INF)
    nds = (dp[:-2, 1:-1], dp[1:-1, 2:], dp[2:, 1:-1], dp[1:-1, :-2])
    best = jnp.full_like(d, INF)
    for door, nd in zip(door_field(ts, tiles), nds):
        best = jnp.minimum(best, jnp.where(door > 0, nd, INF))
    new = jnp.minimum(best.astype(jnp.int32) + 1, jnp.int32(INF)).astype(jnp.int16)
    is_room = ts.jt.is_room[tiles]
    return jnp.where(tiles == ts.GATE, jnp.int16(0), jnp.where(is_room, new, INF))


def d_relax(ts: TileSet, tiles, n_iters: int):
    """d from scratch: all INF but the gate, then n_iters Jacobi steps.  Values
    only decrease, so a finite d_i is the length of a real door path (exact
    for paths up to n_iters long; longer ones read INF)."""
    d0 = jnp.where(tiles == ts.GATE, jnp.int16(0), INF)
    return jax.lax.fori_loop(0, n_iters, lambda _, d: d_step(ts, d, tiles), d0)


def total_energy(ts: TileSet, tiles):
    """E(x) = sum of pair energies over neighbour pairs - sum of log base mass:
    -log of the unnormalised probability at T = 1 (hard constraints excluded)."""
    t = ts.jt
    return (t.Eh[tiles[:, :-1], tiles[:, 1:]].sum() + t.Ev[tiles[:-1, :], tiles[1:, :]].sum()
            - t.logz[tiles].sum())


def violations(ts: TileSet, d, tiles):
    """Per-cell count of violated hard terms (d equality, enclosure)."""
    is_room = ts.jt.is_room[tiles]
    H, W = d.shape
    supported = d_step(ts, d, tiles) == d
    v = jnp.where(is_room, (~supported).astype(jnp.int32), 0)
    perim = jnp.zeros((H, W), bool).at[0, :].set(True).at[-1, :].set(True).at[:, 0].set(True).at[:, -1].set(True)
    return v + jnp.where(perim & is_room, 1, 0)
