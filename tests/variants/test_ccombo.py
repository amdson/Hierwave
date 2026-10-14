"""ccombo: schedule logic (pure) and the start-depth draw."""
from __future__ import annotations

import numpy as np

from castlegen.bench import problems
from castlegen.variants.ccombo import COMBO, CCombo, schedule


def _cfg(**over):
    return {**COMBO, "tau0": 8.0, "tau1": 1.0, "anneal": 0.8, **over}


def test_schedule_shape():
    c = _cfg()
    s = schedule(c, 0.0)
    assert s["frac"] == 1.0 and s["Ls"] == (3,) and abs(s["tau"] - 8.0) < 1e-9 and s["lam_r"] == 0.0
    s = schedule(c, 0.3)
    assert s["frac"] == 1.0 and s["Ls"] == (3,)
    s = schedule(c, 0.575)                          # half-way through the decay
    assert abs(s["frac"] - 0.5) < 1e-9 and s["Ls"] == (3, 4)
    s = schedule(c, 0.4)                            # geometric tau: 8 * (1/8)^(0.5)
    assert abs(s["tau"] - 8.0 ** 0.5) < 1e-9
    s = schedule(c, 0.9)
    assert s["frac"] == 0.0 and s["tau"] == 1.0 and s["lam_r"] is None
    lin = schedule(_cfg(tau_geom=False), 0.4)
    assert abs(lin["tau"] - 4.5) < 1e-9
    taus = [schedule(c, f)["tau"] for f in np.linspace(0, 1, 50)]
    assert all(a >= b for a, b in zip(taus, taus[1:]))


def test_draw_L_follows_schedule():
    prob = problems.get("ground")
    v = CCombo(prob, 0, "/tmp/ccombo_t", W_train=64, B=8)
    T = prob.T
    v._sched = schedule(v.cfg, 0.1)
    assert (v.draw_L(8) == 3).all()
    v._sched = schedule(v.cfg, 0.575)
    L = v.draw_L(8)
    assert (L == T + 1).sum() == 4 and set(L[L <= T]) <= {3, 4}
    v._sched = schedule(v.cfg, 0.95)
    assert (v.draw_L(8) == T + 1).all()
