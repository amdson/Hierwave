"""Tile encoding, grid constants and coordinate-hashed noise.

Tile alphabet: 0..799 are room tiles, index = type * 16 + door_mask;
800 is wall, 801 is gateway.  Door mask bits: 0 = north, 1 = east,
2 = south, 3 = west.  d uses INF = 32767.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

N_TYPES = 50
N_MASKS = 16
N_ROOM_TILES = N_TYPES * N_MASKS  # 800
WALL = N_ROOM_TILES               # 800
GATE = N_ROOM_TILES + 1           # 801
GATE_MASK = 4                     # gateway sits on the top row, door faces south
N_TILES = N_ROOM_TILES + 2        # 802
INF = jnp.int16(32767)

# (dy, dx, own door bit, facing bit of the neighbour)
DIRS = ((-1, 0, 0, 2), (1, 0, 2, 0), (0, 1, 1, 3), (0, -1, 3, 1))

POP = jnp.array([bin(m).count("1") for m in range(N_MASKS)], jnp.float32)
BIT = jnp.array([[(m >> b) & 1 for m in range(N_MASKS)] for b in range(4)],
                jnp.float32)  # (4, 16)


def split_tile(tile):
    """tile int32 array -> (type, mask, is_room, is_wall, is_gate)."""
    is_room = tile < N_ROOM_TILES
    is_wall = tile == WALL
    is_gate = tile == GATE
    typ = jnp.where(is_room, tile // N_MASKS, 0)
    mask = jnp.where(is_room, tile % N_MASKS, jnp.where(is_gate, GATE_MASK, 0))
    return typ, mask, is_room, is_wall, is_gate


# --------------------------------------------------------------------------
# Noise: murmur3 finaliser on (castle, level, step, colour, y, x, slot).
# Identical numbers for a cell whether it is generated inside a chunk or a
# whole castle, and trivially reproducible in C++.
# --------------------------------------------------------------------------
def fmix32(x):
    x = x.astype(jnp.uint32)
    x = x ^ (x >> 16)
    x = x * jnp.uint32(0x85EBCA6B)
    x = x ^ (x >> 13)
    x = x * jnp.uint32(0xC2B2AE35)
    x = x ^ (x >> 16)
    return x


def noise(castle, level, step, colour, y, x, slot):
    """Uniform in (0, 1), float32, shape = broadcast of the integer inputs."""
    u = jnp.uint32
    h = fmix32(u(castle) * u(0x9E3779B1) + u(level))
    h = fmix32(h ^ (u(step) * u(0x85EBCA6B) + u(colour)))
    h = fmix32(h ^ (jnp.asarray(y, jnp.uint32) * u(0xC2B2AE35) + jnp.asarray(x, jnp.uint32)))
    h = fmix32(h ^ jnp.asarray(slot, jnp.uint32))
    return (h.astype(jnp.float32) + 0.5) / 4294967296.0


def gumbel(u):
    return -jnp.log(-jnp.log(u))


def checkerboard_coords(size: int, colour: int):
    """Coordinates (n, 2) of cells with (y + x) % 2 == colour, fixed order."""
    ys, xs = jnp.meshgrid(jnp.arange(size), jnp.arange(size), indexing="ij")
    ys, xs = ys.reshape(-1), xs.reshape(-1)
    keep = (ys + xs) % 2 == colour
    # static shape: exactly size*size/2 cells per colour for even size
    idx = jnp.nonzero(keep, size=size * size // 2)[0]
    return jnp.stack([ys[idx], xs[idx]], axis=1)
