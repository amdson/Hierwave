import numpy as np
import pytest

from castlegen.bench.energy import energy
from castlegen.bench.problem import Region
from castlegen.bench.problems.dimers import build_dimers
from castlegen.bench.problems.ground import build_ground, cert_domain


@pytest.fixture(scope="module", params=["ground", "dimers"])
def prob(request):
    return {"ground": build_ground, "dimers": build_dimers}[request.param]()


def _regions(P):
    return [Region(P.W_eval), Region(P.W_eval, periodic=False, x0=P.W_eval)]


def test_designed_deterministic(prob):
    for reg in _regions(prob):
        a = prob.designed_values(5, reg)
        b = prob.designed_values(5, reg)
        assert a.keys() == b.keys() and a
        for k in a:
            assert np.array_equal(a[k], b[k])
            assert a[k].shape == prob.shape(k, reg)


def test_probes_valid(prob):
    for reg in _regions(prob):
        ps = prob.probes(7, reg)
        assert len(ps) >= 3
        d = prob.designed_values(7, reg)
        for x in ps:
            for k in d:
                assert np.array_equal(x[k], d[k])
            E = energy(prob, x, reg)
            assert all(v == 0 for v in E.viol.values()), E.viol
            assert all(E.nhard[l] > 0 for l in E.nhard)
            st = prob.stats(x)
            assert set(st) <= set(prob.tol) | set(prob.exact) and all(np.isfinite(list(st.values())))
            img = prob.render(x)
            assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3


def test_violations_detected(prob):
    reg = Region(prob.W_eval)
    x = {k: v.copy() for k, v in prob.probes(1, reg)[0].items()}
    if prob.name == "dimers":
        x["p"][5, 5] = (x["p"][5, 5] + 1) % 6
        assert energy(prob, x, reg).viol[0] > 0
    else:
        x["t"][:, 3] = 0
        x["t"][0, 3] = 1                                # floating dirt in the top row
        assert energy(prob, x, reg).viol[0] > 0


def test_ground_hand_energy():
    P = build_ground()
    reg = Region(P.W_eval)
    x = {k: v.copy() for k, v in P.probes(2, reg)[1].items()}
    x["rho1"][3, 4] = max(x["rho1"][3, 4] - 1, 0) if x["rho1"][3, 4] > 0 else 1
    E = energy(P, x, reg)
    assert all(v == 0 for v in E.viol.values())
    r4, f = x["rho4"].astype(float), x["fill"] / 16
    assert np.isclose(E.per_factor["noise4"][0], (4.0 * (r4 - 256 * f) ** 2 / 256).sum())
    assert np.isclose(E.per_factor["smooth4"][0], (0.5 * np.abs(r4 - np.roll(r4, -1, 1)) / 16).sum())
    t = x["t"].astype(float)
    assert np.isclose(E.per_factor["smooth0"][0], (0.5 * np.abs(t - np.roll(t, -1, 1))).sum())
    s = t[0::2, 0::2] + t[0::2, 1::2] + t[1::2, 0::2] + t[1::2, 1::2]
    want = (1.0 * (s - x["rho1"]) ** 2 / 1).sum()
    assert want > 0 and np.isclose(E.per_factor["prior1"][0], want)
    r1 = x["rho1"].astype(float)
    s2 = r1[0::2, 0::2] + r1[0::2, 1::2] + r1[1::2, 0::2] + r1[1::2, 1::2]
    assert np.isclose(E.per_factor["prior2"][0], ((s2 - x["rho2"]) ** 2 / 4).sum())
    assert np.isclose(E.soft, sum(v[0] for v in E.per_factor.values()))


def test_ground_cert_domain():
    for h in (2, 4, 16):
        A, B = cert_domain(h)
        assert A.size == (h + 1) * (h + 2) // 2 and (A <= B).all() and B.max() == h
