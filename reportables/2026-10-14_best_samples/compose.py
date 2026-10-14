"""One figure of the best results so far (bench images: one final world per seed)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(__file__).parent
R = D.parent
panels = [
    (D / "raw/cfast_ground_s0.png", "ground · cfast (baseline): all-dirt collapse"),
    (R / "2026-10-14_wave5_pop/runs/A_rkl4w3/eval_s0.png", "ground · ckpop (own draws only, in the rules): wave5 arm A"),
    (R / "2026-10-14_wave6_pop/runs/D_rkl123x2_cg40/eval_s0.png", "ground · ckpop (own draws only): wave6 arm D"),
    (D / "raw/ckernel_rkl4w3_ground_s2.png", "ground · ckernel_rkl4w3 seed 2 (probe contexts): ACCEPTED"),
    (D / "raw/ckernel_rkl4w3f_l2x2_ground_s1.png", "ground · ckernel_rkl4w3f_l2x2 seed 1 (probe contexts): ACCEPTED"),
    (D / "raw/cbig256_toy_support_s0.png", "toy_support · cbig256 (at F*)"),
    (D / "raw/cbig256_toy_modes_s0.png", "toy_modes · cbig256 (within 0.002 of F*)"),
]
panels = [(p, t) for p, t in panels if p.exists()]
fig, ax = plt.subplots(len(panels), 1, figsize=(9, 2.3 * len(panels)))
for a, (p, t) in zip(ax, panels):
    a.imshow(plt.imread(p), interpolation="nearest")
    a.set_title(t, fontsize=9, loc="left")
    a.axis("off")
fig.tight_layout()
fig.savefig(D / "best_samples.png", dpi=120)
print([t for _, t in panels])
