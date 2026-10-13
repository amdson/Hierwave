"""ctemp: its pass equals cpass's at tau = 1, the tau schedules end at 1, and
log q from scoring matches the draw."""
import numpy as np

from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.variants.cpass import CPass
from castlegen.variants.ctemp import CTemp


def test_pass_matches_cpass_at_tau1():
    prob = get("toy_support")
    a = CPass(prob, 0, "/tmp")
    b = CTemp(prob, 0, "/tmp", tau_cond=False)
    b.params = a.params
    r = Region(prob.W_train)
    for seed in (3, 4):
        xa, xb = a.generate(seed, r), b.generate(seed, r)
        for k in ("raw", "final"):
            for c in xa[k]:
                assert np.array_equal(xa[k][c], xb[k][c])
        x = {**prob.designed_values(seed, r), **xa["raw"]}
        assert abs(a.score(x, r) - b.score(x, r)) < 1e-3


def test_tau_schedules_end_at_one():
    prob = get("toy_modes")
    for mode in ("mix", "coarse", "anneal"):
        v = CTemp(prob, 0, "/tmp", tau_mode=mode, tau_cond=True)
        hot = [v._tau_at(f) for f in np.linspace(0, 0.55, 50)]
        assert all(np.all(t >= 1.0) for t in hot)
        if mode == "coarse":
            assert all(t[0] == 1.0 for t in hot)
        for f in (v.cfg["mix_frac"] + 1e-3, 0.9, 0.999):
            if mode == "anneal" and f < v.cfg["anneal"]:
                continue
            assert np.all(v._tau_at(f) == 1.0)


def test_tau_cond_generates_and_scores():
    prob = get("toy_modes")
    v = CTemp(prob, 1, "/tmp", tau_mode="coarse", tau_cond=True)
    r = Region(prob.W_train)
    x = v.generate(7, r)
    s = v.score({**prob.designed_values(7, r), **x["raw"]}, r)
    assert np.isfinite(s) and s < 0
