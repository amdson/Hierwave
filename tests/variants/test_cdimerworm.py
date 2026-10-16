"""Smoke: cdimerworm and cdimerworm_stage register, train a few iterations on dimers (worm refresh keeps the
population valid) and the staged variant is flagged staged."""
import numpy as np
from castlegen.bench.problems import get
from castlegen.bench.variant import load_variants


def test_cdimerworm_smoke(tmp_path):
    V = load_variants()
    assert V["cdimerworm_stage"].staged is True
    v = V["cdimerworm"](get("dimers"), seed=0, workdir=str(tmp_path), max_iters=2)
    logs = []
    v.train(1e9, logs.append)
    its = [r for r in logs if "iter" in r]
    assert len(its) == 2 and its[-1]["pop"]["valid"] == 1.0
    assert v.updates > 0
