"""The interface a sampler variant implements (notes/bench.md).

FROZEN.  A variant lives in castlegen/variants/<name>.py and registers its
class with @register.  The bench constructs it once per (problem, training
seed), calls train once, then only generate and score.

Rules the bench checks (constitution, hard constraints 2 and 3):
- generate(seed, region) is a deterministic function of (parameters, seed,
  region) and returns every sampled channel over the region.  On an open
  window the values must equal those of any other window covering the same
  cells (finite support): pad internally by your dependency radius, and key
  all noise by absolute coordinates (castlegen.bench.noise).
- "raw" is the draw before repair, "final" after.  Without repair, return the
  same dict for both.
- score(x, region) returns log q(raw draw = x) on a periodic region, or None
  if the variant cannot score (probe and free-energy metrics are then
  reported as missing, which fails acceptance).
- score_sites(x, region) (optional, version 2) returns the per-draw log q terms by cell.
- train must stop by its wall-clock budget; the bench measures it.
"""
from __future__ import annotations

import importlib
import pkgutil

from .problem import Problem, Region

VARIANTS = {}


def register(cls):
    VARIANTS[cls.name] = cls
    return cls


class Variant:
    name = "base"

    def __init__(self, problem: Problem, seed: int, workdir: str):
        self.problem, self.seed, self.workdir = problem, seed, workdir

    def train(self, budget_s: float, log) -> None:
        """Train for at most budget_s seconds of wall clock.  log(dict) records
        per-step metrics (written to the run's log file)."""

    def generate(self, seed: int, region: Region) -> dict:
        """{"raw": sample, "final": sample}; samples hold the sampled channels
        (designed ones may be omitted)."""
        raise NotImplementedError

    def score(self, x: dict, region: Region):
        """log q of a complete raw sample on a periodic region, or None."""
        return None

    def score_sites(self, x: dict, region: Region):
        """Bench version 2 (block scores).  The per-draw terms of log q: {channel: float array
        (h, w) at the channel's level} whose sum over all channels is score(x), or None if the
        variant cannot split its log q (the block check is then reported n/a)."""
        return None

    def info(self) -> dict:
        """Anything worth recording with the result (parameter count, radius)."""
        return {}


def load_variants():
    import castlegen.variants as pkg
    for m in pkgutil.iter_modules(pkg.__path__):
        if not m.name.startswith("_"):
            importlib.import_module(f"castlegen.variants.{m.name}")
    return VARIANTS
