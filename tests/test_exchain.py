import glob

import numpy as np
import pytest

from castlegen import exchain as XC
from castlegen import refheights as RH
from castlegen import tileset


@pytest.fixture(scope="module")
def setup():
    ts = tileset.load("cliffs")
    E = np.load(sorted(glob.glob("cache/exemplar-cliffs-*.npy"))[0])
    rng = np.random.default_rng(0)
    H = np.clip(np.cumsum(rng.integers(-2, 3, 64)) + 30, 9, 60)
    x = RH.from_heights(ts, H, 64, 8, "stone", "stone")
    return ts, E, x


def test_incremental_energy_matches_recompute(setup):
    ts, E, x = setup
    ch = XC.ExChain(ts, x, 8, E, T=0.9, lam=2.0, seed=1)
    for _ in range(3):
        before = ch.energy_ex()
        B = {h: c["B"].copy() for h, c in ch.sc.items()}
        cs = np.array([int(ch.rng.integers(64))])
        old = ch.t[:, cs].copy()
        new = old.copy()
        Hc = RH.heights(ts, ch.t, 8)[cs[0]]
        new[64 - Hc - 1, 0] = RH._sig(ts, "stone")                       # grow by one
        d = -ch._extra_logw(cs, old, new, 1.0)[0]
        ch.t[:, cs] = new
        ch._on_accept(cs, old, new)
        after = ch.energy_ex()
        assert d == pytest.approx(after - before, rel=1e-6, abs=1e-6)
        ch._refresh()
        for h, c in ch.sc.items():
            assert np.allclose(c["B"], ch.sc[h]["B"])


def test_chain_keeps_support(setup):
    ts, E, x = setup
    ch = XC.ExChain(ts, x, 8, E, T=0.9, lam=2.0, seed=2, s_every=2)
    for _ in range(6):
        ch.sweep()
        assert RH.violations(ts, ch.t, 8) == 0
    B = {h: c["B"].copy() for h, c in ch.sc.items()}
    ch._refresh()
    assert all(np.allclose(B[h], ch.sc[h]["B"]) for h in B)
