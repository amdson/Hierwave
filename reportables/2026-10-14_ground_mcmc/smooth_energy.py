"""Energy of hand-designed smooth ground scenes against the annealed MCMC samples
(same 8 x 8 patch).  Every coarse channel is lifted exactly from the tiles
(rho = counts, cert = full / occupied columns)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mcmc import H, SURFACE, W, build, lift  # noqa: E402

from castlegen.bench import energy as en  # noqa: E402
from castlegen.bench.problem import Region  # noqa: E402

p = build()
region = Region(W)
des = p.designed_values(0, region)


def E(t):
    x = {**des, **lift(np.asarray(t, np.int64))}
    e = en.energy(p, x, region)
    return e.soft, sum(e.viol.values()), {k: v[0] for k, v in e.per_factor.items() if v[0]}


def from_surface(sr):
    return (np.arange(H)[:, None] >= np.asarray(sr)[None, :]).astype(np.int64)


designs = {
    "designed line (3,3,3,3,2,2,2,2)": SURFACE,
    "flat at row 3": [3] * 8,
    "flat at row 2": [2] * 8,
    "ramp 3,3,3,2,2,2,2,3": [3, 3, 3, 2, 2, 2, 2, 3],
    "line with one +1 bump": [3, 3, 2, 3, 2, 2, 2, 2],
    "all dirt": [0] * 8,
}
S = np.load(Path(__file__).parent / "samples.npy")
samp = np.array([E(t)[0] for t in S])
print(f"MCMC samples (lifted): soft energy min {samp.min():.2f}, median {np.median(samp):.2f}, "
      f"max {samp.max():.2f}")
for name, sr in designs.items():
    s, v, per = E(from_surface(sr))
    rank = (samp < s).mean()
    print(f"{name:34s} S = {s:6.2f}  viol {v}  lower than {100 * (1 - rank):3.0f} % of samples   "
          + ", ".join(f"{k} {val:.1f}" for k, val in sorted(per.items())))
i = int(np.argsort(samp)[len(samp) // 2])
_, _, per = E(S[i])
print("median sample breakdown: " + ", ".join(f"{k} {val:.1f}" for k, val in sorted(per.items())))
