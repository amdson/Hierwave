"""The benchmark problem suite (notes/bench.md).  FROZEN.

Each module defines build_<name>() -> Problem; SUITE maps names to builders.
Problems change only by a version bump approved by the user."""
from __future__ import annotations

import importlib

SUITE = {
    "toy_support": ("toys", "build_toy_support"),
    "toy_modes": ("toys", "build_toy_modes"),
    "ground": ("ground", "build_ground"),
    "dimers": ("dimers", "build_dimers"),
    "columns": ("columns", "build_columns"),
}

_cache = {}


def get(name):
    if name not in _cache:
        mod, fn = SUITE[name]
        _cache[name] = getattr(importlib.import_module(f"castlegen.bench.problems.{mod}"), fn)()
    return _cache[name]
