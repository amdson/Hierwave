"""Cone sizes per level (fraction of later draws inside the cone) and the
build time of the tables."""
import sys, time
import numpy as np
from castlegen.bench import problems
from castlegen.variants import _cpass_core as core
from castlegen.variants._cfast_cone import Cone

for name in sys.argv[1:]:
    prob = problems.get(name)
    lay = core.Layout(prob, prob.W_train, True)
    t = time.time()
    cone = Cone(lay, core.ACTOR_WIN)
    tb = time.time() - t
    nsite = {}
    for s, (l, k, ci, cn) in enumerate(cone.steps):
        nsite[s] = int(np.asarray(lay.sites(l, k)[2]).sum())
    print(f"{name}: build {tb:.2f}s, steps {lay.nsteps}, reach {sorted(set(cone.reach))}")
    for key, bx in cone.boxes.items():
        l = int(key[1:key.index('_')])
        sizes = np.maximum(bx[..., 1] - bx[..., 0] + 1, 0) * np.maximum(bx[..., 3] - bx[..., 2] + 1, 0)
        cone_n = sizes.sum(-1)                                  # (16, n)
        later = np.zeros((core.NCOL, lay.nsite(l)))
        for k in range(core.NCOL):
            ci = [c.name for c in lay.chans[l]].index(key[key.index('_') + 1:])
            s0 = lay.step(l, k, ci)
            later[k] = sum(nsite[s] for s in range(s0 + 1, lay.nsteps))
        fr = cone_n / np.maximum(later, 1)
        print(f"  {key}: mean cone {cone_n.mean():.0f} draws, later {later.mean():.0f}, fraction {fr.mean():.3f}"
              f" (min {fr.min():.3f})")
