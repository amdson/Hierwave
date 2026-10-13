"""Probe: generation on periodic and open windows (compile time, margin)."""
import sys
import time

from castlegen.bench import problems
from castlegen.bench.problem import Region
from castlegen.variants.cconv import CConv

prob = problems.get(sys.argv[1] if len(sys.argv) > 1 else "ground")
v = CConv(prob, 0, None, r=3, cw=16, hc=8, B=2)
print("margin", v.margin(), flush=True)
u = 1 << (prob.T + 2)
t = time.time()
v.generate(77, Region(prob.W_train, periodic=True))
print("periodic ok", time.time() - t, flush=True)
t = time.time()
v.generate(77, Region(4 * u, periodic=False, x0=0))
print("open ok", time.time() - t, flush=True)
