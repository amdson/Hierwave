"""Legacy (G0, JAX castle sampler era); superseded by castlegen/channels. See the original README.md (git history).

Grid constants and coordinate-hashed noise.

Tiles are signature ids of a compiled tile set (castlegen.legacy.tileset); this
module holds only what does not depend on the tile set.  d uses INF = 32767.
"""
from __future__ import annotations

import jax.numpy as jnp

INF = jnp.int16(32767)

# (dy, dx) in the order N, E, S, W
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


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
    # top 24 bits: exact in float32 and strictly inside (0, 1); (h + 0.5) / 2^32
    # rounds to 1.0 for the largest h, which makes the Gumbel variate +inf
    return ((h >> 8).astype(jnp.float32) + 0.5) / 16777216.0


def gumbel(u):
    return -jnp.log(-jnp.log(u))


def checkerboard_coords(size: int, colour: int):
    """Coordinates (n, 2) of cells with (y + x) % 2 == colour, fixed order."""
    ys, xs = jnp.meshgrid(jnp.arange(size), jnp.arange(size), indexing="ij")
    ys, xs = ys.reshape(-1), xs.reshape(-1)
    keep = (ys + xs) % 2 == colour
    idx = jnp.nonzero(keep, size=size * size // 2)[0]
    return jnp.stack([ys[idx], xs[idx]], axis=1)
