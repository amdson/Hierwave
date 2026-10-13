"""Counter-based noise keyed by absolute coordinates (notes/bench.md).

FROZEN.  Every random number a variant uses at generation time should come
from here, keyed by (seed, tag, level, Y, X[, value]), so that the value at a
site does not depend on which region is being generated.  The same code runs
on numpy (xp = np) and jax.numpy (xp = jnp) and gives bit-identical uint32
hashes; uniforms agree to float32 rounding.

On a periodic strip, pass X modulo the level's circumference."""
from __future__ import annotations

import numpy as np

_M1, _M2 = 0x7FEB352D, 0x846CA68B
_GOLD = 0x9E3779B9


def _u32(xp, v):
    if xp is np:
        return np.asarray(v).astype(np.int64).astype(np.uint32)
    return xp.asarray(v).astype(xp.int32).astype(xp.uint32)


def mix(xp, h):
    """lowbias32 (Wellons): a bijective avalanche on uint32."""
    h = h ^ (h >> 16)
    h = h * xp.uint32(_M1)
    h = h ^ (h >> 15)
    h = h * xp.uint32(_M2)
    h = h ^ (h >> 16)
    return h


def hash32(xp, seed, tag, level, Y, X, v=0):
    """uint32 hash of the key, broadcast over array arguments."""
    with np.errstate(over="ignore"):
        h = mix(xp, _u32(xp, seed) ^ xp.uint32(_GOLD))
        for k in (tag, level, Y, X, v):
            h = mix(xp, h ^ _u32(xp, k) + xp.uint32(_GOLD))
        return h


def uniform(xp, seed, tag, level, Y, X, v=0):
    """Uniform in (0, 1), strictly inside."""
    h = hash32(xp, seed, tag, level, Y, X, v)
    return ((h >> 8).astype(xp.float32) + 0.5) * (1.0 / (1 << 24))


def gumbel(xp, seed, tag, level, Y, X, D):
    """(..., D) standard Gumbel noise, one per domain value (indexed by value,
    not by candidate position, so masks can change without reshuffling)."""
    Y = xp.asarray(Y)[..., None]
    X = xp.asarray(X)[..., None]
    v = xp.arange(D)
    u = uniform(xp, seed, tag, level, Y, X, v)
    return -xp.log(-xp.log(u))


def tag_of(name: str) -> int:
    """A stable int tag for a string (channel names, purposes)."""
    h = 2166136261
    for ch in name.encode():
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h & 0x7FFFFFFF
