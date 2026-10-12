"""ctypes wrapper of the C infinite-world sampler (castlegen/csrc/hwsampler.c,
built on demand into cache/libhwsampler.dylib).  Output equals
castlegen.infinite.generate bit for bit for models written by
castlegen/export_models.py.

    m = Model.load("luanti/games/hierwave/mods/hierwave_gen/data/biomes.hwm")
    g = m.tiles(seed, y0, x0, H, W)            # (H, W) int32
    w = World(m, seed, C=32); w.region(y0, x0, H, W)

Layered 3D models (kind 2, TileModel3D) return (L, H, W): every layer of
the requested columns, as castlegen.infinite.generate does."""
from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

SRC = os.path.join(os.path.dirname(__file__), "csrc", "hwsampler.c")
LIB = os.path.join(os.path.dirname(__file__), "..", "cache", "libhwsampler.dylib")
CFLAGS = ["-O3", "-std=c11", "-ffp-contract=off", "-shared", "-fPIC"]

_L = None


def lib():
    global _L
    if _L is None:
        if not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(SRC):
            os.makedirs(os.path.dirname(LIB), exist_ok=True)
            tmp = LIB + f".{os.getpid()}.tmp"
            subprocess.check_call(["cc", *CFLAGS, SRC, "-o", tmp, "-lm"])
            os.replace(tmp, LIB)
        L = ctypes.CDLL(LIB)
        P, I, U, I64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64, ctypes.c_int64
        L.hw_model_load.argtypes = [ctypes.c_char_p]
        L.hw_model_load.restype = P
        L.hw_model_free.argtypes = [P]
        for f in ("hw_model_T", "hw_model_S", "hw_model_B", "hw_model_pad", "hw_model_L", "hw_model_kind"):
            getattr(L, f).argtypes = [P]
            getattr(L, f).restype = I
        L.hw_model_set_fast.argtypes = [P, I]
        L.hw_tiles.argtypes = [P, U, I64, I64, I, I, P]
        L.hw_tiles.restype = I
        L.hw_world_new.argtypes = [P, U, I]
        L.hw_world_new.restype = P
        L.hw_world_region.argtypes = [P, I64, I64, I, I, P]
        L.hw_world_region.restype = I
        L.hw_world_free.argtypes = [P]
        L.hw_world_chunks.argtypes = [P]
        L.hw_world_chunks.restype = I
        L.hw_u01.argtypes = [U, I64, I64, I64]
        L.hw_u01.restype = ctypes.c_double
        _L = L
    return _L


M64 = (1 << 64) - 1


class Model:
    def __init__(self, ptr, path):
        self._p, self.path = ptr, path

    @classmethod
    def load(cls, path):
        p = lib().hw_model_load(os.fsencode(path))
        if not p:
            raise OSError(f"hw_model_load failed: {path}")
        return cls(p, path)

    def __del__(self):
        if getattr(self, "_p", None) and _L is not None:
            _L.hw_model_free(self._p)
            self._p = None

    @property
    def T(self):
        return lib().hw_model_T(self._p)

    @property
    def S(self):
        """schedule rows: sweeps (flat) or Sb K (block)"""
        return lib().hw_model_S(self._p)

    @property
    def B(self):
        """block size of a block schedule, 0 for a flat one"""
        return lib().hw_model_B(self._p)

    @property
    def L(self):
        """layers (1 for 2D models)"""
        return lib().hw_model_L(self._p)

    @property
    def kind(self):
        """0 pair tables, 1 sockets, 2 layered 3D"""
        return lib().hw_model_kind(self._p)

    def _shape(self, H, W):
        return (self.L, H, W) if self.kind == 2 else (H, W)

    @property
    def pad(self):
        """context tiles per side (block schedules add up to B - 1 for alignment)"""
        return lib().hw_model_pad(self._p)

    def set_fast(self, on=True):
        """Fast finite-gamma socket draws: deterministic, same distribution,
        NOT bit-equal to castlegen.infinite."""
        lib().hw_model_set_fast(self._p, int(bool(on)))

    def tiles(self, seed, y0, x0, H, W):
        """(H, W) int32, or (L, H, W) for a 3D model"""
        out = np.empty(self._shape(H, W), np.int32)
        if lib().hw_tiles(self._p, seed & M64, y0, x0, H, W, out.ctypes.data_as(ctypes.c_void_p)):
            raise MemoryError("hw_tiles")
        return out


class World:
    """Chunk cache of C x C cells (castlegen.infinite.World in C)."""

    def __init__(self, model: Model, seed=0, C=32):
        self.model = model                       # keep the model alive
        self._p = lib().hw_world_new(model._p, seed & M64, C)
        if not self._p:
            raise MemoryError("hw_world_new")

    def __del__(self):
        if getattr(self, "_p", None) and _L is not None:
            _L.hw_world_free(self._p)
            self._p = None

    @property
    def n_chunks(self):
        return lib().hw_world_chunks(self._p)

    def region(self, y0, x0, H, W):
        out = np.empty(self.model._shape(H, W), np.int32)
        if lib().hw_world_region(self._p, y0, x0, H, W, out.ctypes.data_as(ctypes.c_void_p)):
            raise MemoryError("hw_world_region")
        return out


def u01(seed, y, x, h):
    return lib().hw_u01(seed & M64, y, x, h)
