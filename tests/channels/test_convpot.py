"""Factor kind CONVPOT (notes/convpot_test.md): the kernel's total energy and
one-site conditionals against the numpy reference convref.convpot_energy,
a sweep, and the one-hot encoding of a pair table."""
import numpy as np
import pytest

from castlegen.channels.convref import convpot_energy
from castlegen.channels.core import Channel, Factor, Model

D = 5


def params(rng, V, k, m, scale=0.5):
    R = lambda *s: rng.normal(scale=scale, size=s)
    return dict(E=R(V, k), a=R(k), A=R(9, k, k), W=R(9, k, m), b=R(m), v=R(m))


def conv_model(rng, rows, cols, k, m, pad, view=False):
    c = Channel("c", 1, D).add_view("id", np.arange(D))
    vname = "id"
    V = D
    if view:                                            # a coarser view: 0 1 1 2 0
        c.add_view("w", [0, 1, 1, 2, 0])
        vname, V = "w", 3
    P = params(rng, V, k, m)
    f = Factor.convpot(("c", vname), P["E"], P["a"], P["A"], P["W"], P["b"], P["v"], pad=pad, name="cv")
    mdl = Model(rows, cols, [c], [f])
    c.grid[:] = rng.integers(0, D, c.grid.shape)
    vw = c.views[vname]
    return mdl, P, (lambda g: convpot_energy(vw[g], P["E"], P["a"], P["A"], P["W"], P["b"], P["v"], pad))


@pytest.mark.parametrize("pad", [-1, 2])
@pytest.mark.parametrize("m", [0, 8])
@pytest.mark.parametrize("view", [False, True])
def test_total_energy_matches_reference(pad, m, view):
    rng = np.random.default_rng(10 + m + pad)
    for shape in [(5, 7), (1, 4), (3, 3)]:
        mdl, P, ref = conv_model(rng, *shape, k=4, m=m, pad=pad, view=view)
        tot, nv = mdl.energy("c")
        assert nv == 0
        np.testing.assert_allclose(tot, ref(mdl.chan("c").grid), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("pad", [-1, 1])
@pytest.mark.parametrize("m", [0, 8])
@pytest.mark.parametrize("view", [False, True])
def test_site_energy_differences(pad, m, view):
    rng = np.random.default_rng(20 + m + pad)
    mdl, P, ref = conv_model(rng, 5, 6, k=4, m=m, pad=pad, view=view)
    g = mdl.chan("c").grid
    for (y, x) in [(0, 0), (0, 5), (4, 0), (4, 5), (2, 3), (1, 1), (0, 2), (3, 5)]:
        e = mdl.site_energies("c", y, x)
        old = g[y, x]
        r = np.empty(D)
        for t in range(D):
            g[y, x] = t
            r[t] = ref(g)
        g[y, x] = old
        np.testing.assert_allclose(e - e[0], r - r[0], rtol=1e-9, atol=1e-9)


def test_sweep_runs():
    rng = np.random.default_rng(3)
    mdl, P, ref = conv_model(rng, 12, 12, k=4, m=8, pad=0)
    P_ = mdl.compile("c")
    assert P_.radius == 2 and P_.ncol == 5
    assert "conv" in mdl.describe("c")
    for s in range(3):
        bad = mdl.sweep("c", 2, seed=s)
        assert bad == 0
        g = mdl.chan("c").grid
        assert g.min() >= 0 and g.max() < D
    tot, _ = mdl.energy("c")
    np.testing.assert_allclose(tot, ref(mdl.chan("c").grid), rtol=1e-10)


def test_kinds_unchanged_without_convpot():
    """A pair model packs no convs and its rows are unchanged by the kind."""
    c = Channel("c", 1, D).add_view("id", np.arange(D))
    T = np.random.default_rng(0).normal(size=(D, D))
    mdl = Model(4, 4, [c], [Factor.pair(("c", "id"), ("c", "id"), (0, 1), T, name="p")])
    P = mdl.compile("c")
    assert P.fac.shape == (2, 8) and len(P.convs) == 1 and P.convs[0].shape == (1,)


@pytest.mark.parametrize("pad", [-1, 3])
def test_onehot_encodes_pair_table(pad):
    rng = np.random.default_rng(4)
    T = rng.normal(size=(D, D))
    A = np.zeros((9, D, D))
    A[5] = T                                            # d = (0, 1), east
    cv = Factor.convpot(("c", "id"), np.eye(D), np.zeros(D), A, np.zeros((9, D, 0)), np.zeros(0), np.zeros(0),
                        pad=pad, name="cv")
    pr = Factor.pair(("c", "id"), ("c", "id"), (0, 1), T, pad_b=pad, pad_a=-1, name="p")
    grid = rng.integers(0, D, (6, 7))
    mods = []
    for f in (cv, pr):
        c = Channel("c", 1, D).add_view("id", np.arange(D))
        c.grid = grid.astype(np.int32).copy()
        mods.append(Model(6, 7, [c], [f]))
    np.testing.assert_allclose(mods[0].energy("c")[0], mods[1].energy("c")[0], rtol=1e-12)
    for y in range(6):
        for x in range(7):
            e0 = mods[0].site_energies("c", y, x)
            e1 = mods[1].site_energies("c", y, x)
            np.testing.assert_allclose(e0 - e0[0], e1 - e1[0], atol=1e-12)
