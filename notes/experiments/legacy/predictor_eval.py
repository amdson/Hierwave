"""Held-out column NLL of count tables vs factored tables vs the linear
predictor, on the mu = 0 and mu = 0.11 reference corpora, plus the mountain
clamp profile.   .venv/bin/python notes/experiments/legacy/predictor_eval.py"""
import sys
import numpy as np
sys.path.insert(0, "notes/experiments/legacy")
from castlegen.legacy import envpredict as EP
from castlegen.legacy import refheights as RH
from castlegen.legacy import tileset
from castlegen.legacy.promise import Tables
from castlegen.legacy.quantities import envelope as EN

G, K = 8, 16
ts = tileset.load("cliffs")
lg = EN.lang()
corpora = {"mu=0": RH.corpus(ts, log=lambda *a: None),
           "mu=0.11": RH.corpus(ts, mu=0.11, init="flat", sweeps=1500, log=lambda *a: None)}
for name, maps in corpora.items():
    tr, te = list(maps[:18]), list(maps[18:])
    gte = RH.mine(ts, te, K, G, 64, lg)
    fits = {"counts": RH.tables(ts, tr, K, G, 64, lg=lg),
            "factored": RH.factored_tables(ts, tr, K, G, 64, lg=lg),
            "linear": EP.tables(ts, tr, K, G, 64, lg=lg, log=lambda *a: None)}
    print(f"== {name}: held-out NLL per column (6 maps)")
    for k, t in fits.items():
        print(f"  {k:9s}", "  ".join(f"h{h}: {EP.column_nll(t, gte, h, lg, G):7.3f}" for h in (16, 32, 64)))
