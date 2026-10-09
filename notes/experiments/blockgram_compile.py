"""Compile the tower grammar (castlegen/blockgram/tower_spec.py) into
per-level channels with castlegen/blockgram/compile.py, with and without
bridges, and print the channel table, the decisions, read extents and the
honourability check.

    PYTHONPATH=. python notes/experiments/blockgram_compile.py
"""
import os, time

from castlegen.blockgram.compile import Compiled
from castlegen.blockgram.tower_spec import TowerSpec

N = int(os.environ.get("N", 24))
for bridges in (True, False):
    t0 = time.time()
    c = Compiled(TowerSpec(bridges=bridges), ncontexts=N)
    print(f"\n## tower grammar, bridges={bridges} ({N} contexts, {time.time() - t0:.0f}s)\n")
    print(c.report())
