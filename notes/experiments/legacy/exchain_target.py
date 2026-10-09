"""What does the exemplar-defined target look like?  Long ExChain runs from
the exemplar's own terrain and from bare ground, per lam.
    .venv/bin/python notes/experiments/legacy/exchain_target.py LAM OUTDIR [SWEEPS]"""
import glob
import sys
import time

import numpy as np

from castlegen.legacy import exchain as XC
from castlegen.legacy import exemplar as ex
from castlegen.legacy import refheights as RH
from castlegen.legacy import tileset

lam, out = float(sys.argv[1]), sys.argv[2]
sweeps = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
ts = tileset.load("cliffs")
E = np.load(sorted(glob.glob("cache/exemplar-cliffs-*.npy"))[0])
Hex = RH.heights(ts, E, 8)
starts = {"exemplar": RH.from_heights(ts, np.tile(Hex, 2), 128, 8, "stone", "stone"),
          "band": RH.band_map(ts, 128, 8)}
print(f"exemplar heights: mean {Hex.mean():.1f} std {Hex.std():.1f} |slope| {np.abs(np.diff(Hex)).mean():.2f}")
for name, x in starts.items():
    ch = XC.ExChain(ts, x, 8, E, T=0.9, lam=lam, seed=1)
    t0 = time.time()
    for i in range(sweeps + 1):
        if i % 250 == 0:
            H = RH.heights(ts, ch.t, 8)
            print(f"lam={lam} start={name:8s} sweep {i:5d}: H mean {H.mean():5.1f} std {H.std():5.1f} "
                  f"|slope| {np.abs(np.roll(H, -1) - H).mean():.2f}  E_ex/cell {ch.energy_ex() / 128 ** 2:.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if i in (0, sweeps):
                ex.to_png(ts, ch.t, f"{out}/xt_lam{lam:g}_{name}_{i}.png", 3)
        if i < sweeps:
            ch.sweep()
