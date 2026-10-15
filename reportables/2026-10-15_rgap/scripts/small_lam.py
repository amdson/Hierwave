import sys, numpy as np
sys.path.insert(0, "reportables/2026-10-15_rgap/scripts")
from castlegen.variants import _rgap
_rgap.LAMS = (0.0, 1.0, 2.0, 5.0)
import run_gap as rg
import castlegen.variants._rgap as R
orig = R.enumerate_with_logr
rg.enumerate_with_logr = lambda sch, rho, ms: orig(sch, rho, ms, lam_rs=(0.0, 1.0, 2.0, 5.0))
for seed in (3, 1):
    var, ex = rg.build("flat", seed)
    r = ex.parts(var.params)
    print("flat", seed, "uniform gap %.3f" % r["uniform"]["gap"], {k: round(v["gap"], 3) for k, v in r.items() if k.startswith("des")}, "keys", list(ex.nlogr))
