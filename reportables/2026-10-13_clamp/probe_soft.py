"""Soft energy per level of the probe states themselves (energy.py's level
attribution = the pass's: a factor is paid at the level of its first view)."""
import json, sys, numpy as np
from castlegen.bench import energy as en
from castlegen.bench.problem import Region
from castlegen.bench.problems import get as get_problem
p = get_problem("ground"); reg = Region(p.W_train); cells = p.H * reg.W
seeds = np.arange(3_000_000, 3_000_000 + 32)
S = np.zeros((32, 4, p.T + 1))
for i, s in enumerate(seeds):
    des = p.designed_values(int(s), reg)
    for j, y in enumerate(p.probes(int(s), reg)):
        x = {**des, **{k: np.asarray(v) for k, v in y.items()}}
        E = en.energy(p, x, reg)
        assert sum(E.viol.values()) == 0
        for f in p.factors:
            s_, v_, n_ = E.per_factor[f.name]
            S[i, j, p.chan[f.a[0]].level] += s_
S /= cells
out = {"per_level_mean": S.mean((0, 1)).tolist(),
       "per_jitter_per_level": S.mean(0).tolist()}
json.dump(out, open("probe_soft.json", "w"), indent=1); print(json.dumps(out))
