"""Cone sizes per level on ground: later draws inside a draw's cone, and the hull window."""
import sys, numpy as np
from castlegen.bench import problems
from castlegen.bench.problem import Region
from castlegen.variants.creplay import CReplay
from castlegen.variants import _cpass_core as core
name = sys.argv[1] if len(sys.argv) > 1 else "ground"
p = problems.get(name)
var = CReplay(p, 0, "/tmp/cfine_x")
W = var.cfg["W_train"] or p.W_train
lay = var.layout(Region(W))
print(name, "T", p.T, "H", p.H, "W", W, "nsteps", lay.nsteps, "hw", lay.hw, "nyx", lay.nyx)
fn = var.train_fns(lay)
cone = fn["cone"]
tot = {l: sum(lay.nsite(l) * len(lay.chans[l]) * core.NCOL for _ in [0]) for l in lay.levels}
for l in lay.levels:
    for ci, c in enumerate(lay.chans[l]):
        key = core.key_of(l, c.name)
        bx = cone.boxes[key]  # (16, n, S, 4)
        n = (np.maximum(bx[..., 1] - bx[..., 0] + 1, 0) * np.maximum(bx[..., 3] - bx[..., 2] + 1, 0)).sum(-1)
        # fine-cell extent of the last step's hull: approx by box at level 0 steps
        print(key, "sites", bx.shape[1], "cone draws mean", n.mean(), "max", n.max(), "of later", sum(tot[m] for m in lay.levels if m <= l))
        print("   reach", cone.reach[lay.step(l, 0, ci)])
