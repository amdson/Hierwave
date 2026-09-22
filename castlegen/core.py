"""Tile encoding, grid constants and coordinate-hashed noise.

Tile alphabet: 0..49 are room types, 50 is wall, 51 is gateway.  Each room
type has a FIXED entrance/exit mask (bits: 0 north, 1 east, 2 south, 3 west);
a door exists on an edge iff both facing bits are set.  Doors are therefore a
function of the two types and nothing is modelled on edges.  d uses
INF = 32767.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

N_TYPES = 50
WALL = N_TYPES          # 50
GATE = N_TYPES + 1      # 51
N_TILES = N_TYPES + 2   # 52
INF = jnp.int16(32767)
GATE_MASK = 4           # gateway sits on the top row, door faces south

# (dy, dx, own door bit, facing bit of the neighbour)
DIRS = ((-1, 0, 0, 2), (1, 0, 2, 0), (0, 1, 1, 3), (0, -1, 3, 1))


def default_type_masks(seed: int = 0) -> np.ndarray:
    """Fixed entrance/exit masks per type for the stress test: every type has
    at least one door; door count is 1..4 with probabilities 0.15/0.45/0.3/0.1."""
    rng = np.random.default_rng(seed)
    masks = np.zeros(N_TYPES, np.int32)
    for t in range(N_TYPES):
        k = rng.choice([1, 2, 3, 4], p=[0.15, 0.45, 0.3, 0.1])
        bits = rng.choice(4, size=k, replace=False)
        masks[t] = int(sum(1 << b for b in bits))
    return masks


# masks per tile index, including wall (0) and gateway (south); replace with
# set_type_masks() for a designed tile set
TILE_MASK = jnp.asarray(np.concatenate([default_type_masks(), [0, GATE_MASK]]), jnp.int32)
BIT = jnp.stack([(TILE_MASK >> b) & 1 for b in range(4)]).astype(jnp.float32)  # (4, 52)
POP = jnp.asarray([bin(int(m)).count("1") for m in np.asarray(TILE_MASK)], jnp.float32)


def set_type_masks(masks):
    global TILE_MASK, BIT, POP
    TILE_MASK = jnp.asarray(np.concatenate([np.asarray(masks, np.int32), [0, GATE_MASK]]), jnp.int32)
    BIT = jnp.stack([(TILE_MASK >> b) & 1 for b in range(4)]).astype(jnp.float32)
    POP = jnp.asarray([bin(int(m)).count("1") for m in np.asarray(TILE_MASK)], jnp.float32)


def split_tile(tile):
    """tile int32 array -> (type, mask, is_room, is_wall, is_gate)."""
    is_room = tile < N_TYPES
    is_wall = tile == WALL
    is_gate = tile == GATE
    typ = jnp.where(is_room, tile, 0)
    mask = TILE_MASK[tile]
    return typ, mask, is_room, is_wall, is_gate


# --------------------------------------------------------------------------
# Noise: murmur3 finaliser on (castle, level, step, colour, y, x, slot).
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
    idx = jnp.nonzero(keep, size=size * size // 2)[0]
    return jnp.stack([ys[idx], xs[idx]], axis=1)
