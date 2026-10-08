"""castlegen.channels.convfit: the JAX conv-potential fit (notes/convpot_test.md)."""
import numpy as np
import pytest

from castlegen.channels import convfit as cf
from castlegen.channels.convref import convpot_energy

D, PAD = 5, 0


def _dc(A):
    return A - A.mean(1, keepdims=True) - A.mean(0, keepdims=True) + A.mean()


def _random_full(rng, D, k, m):
    return dict(a=rng.standard_normal(k), A=rng.standard_normal((9, k, k)),
                W=rng.standard_normal((9, k, m)), b=rng.standard_normal(m), v=rng.standard_normal(m))


def _targets(params, Et, w, pad, rng=None, noise=0.0):
    R, C = w.shape[1:]
    t = cf.energy_window(params, Et, w, pad) - cf.energy_window(params, Et, np.full((1, R, C), pad), pad)[0]
    if noise:
        t = t + noise * rng.standard_normal(len(t))
    return t


def _bump_head(D, m=8, s=2.0, vs=4.0):
    """A genuinely non-additive head: with c the number of non-pad cells in the
    3 x 3 neighbourhood, groups of 4 units v . softplus(s (c - o)) with o =
    .5, 1.5, 2.5, 3.5 and weights vs (1, -2, 1, 0): a bump over c = 1..3."""
    W, b, v = np.zeros((9, D, m)), np.zeros(m), np.zeros(m)
    r = (np.arange(D) != PAD).astype(float)
    for g in range(m // 4):
        for j, c in enumerate((1, -2, 1, 0)):
            W[:, :, 4 * g + j] = s * r[None, :]
            b[4 * g + j] = -s * (0.5 + j)
            v[4 * g + j] = vs * c
    return dict(a=np.zeros(D), A=np.zeros((4, D, D)), W=W, b=b, v=v)


@pytest.mark.parametrize("m", [0, 3])
@pytest.mark.parametrize("pad", [-1, 2])
def test_energy_matches_reference(m, pad):
    rng = np.random.default_rng(10 * m + pad + 1)
    k = 4
    E = rng.standard_normal((D, k))
    p = _random_full(rng, D, k, m)
    w = rng.integers(0, D, (8, 3, 4))
    e = cf.energy_window(p, E, w, pad)
    r = np.array([convpot_energy(x, E, p["a"], p["A"], p["W"], p["b"], p["v"], pad) for x in w])
    assert np.abs(e - r).max() < 1e-6
    # tied params expand to the same energy
    pt = cf.init_params(D, k, m, seed=3, scale=0.7, tie=True)
    pf = cf.expand(pt)
    assert np.allclose(pf["A"][2], pf["A"][6].T) and np.all(pf["A"][4] == 0)
    e = cf.energy_window(pt, E, w, pad)
    r = np.array([convpot_energy(x, E, pf["a"], pf["A"], pf["W"], pf["b"], pf["v"], pad) for x in w])
    assert np.abs(e - r).max() < 1e-6


def test_ls_equals_adam_small():
    rng = np.random.default_rng(1)
    Dk = 3
    E = np.linalg.qr(rng.standard_normal((Dk, Dk)))[0]
    w = rng.integers(0, Dk, (300, 2, 3))
    t = rng.standard_normal(300)                       # arbitrary targets: compare the minimisers' losses
    l2 = 1e-3
    ls = cf.fit_bilinear_ls(w, t, E, PAD, l2=l2)
    ad = cf.fit(w, t, E, PAD, m=0, steps=6000, lr=0.05, lr_end=5e-5, l2=l2)
    assert ad["curve"][-1] <= ls["loss"] * (1 + 1e-3) + 1e-8
    assert ad["curve"][-1] >= ls["loss"] - 1e-8      # ls is the minimiser
    assert abs(ad["resid_rms"] - ls["resid_rms"]) < 1e-3


def test_materialise_planted_east():
    rng = np.random.default_rng(2)
    T = rng.standard_normal((D, D))
    p = dict(a=rng.standard_normal(D), A=np.zeros((9, D, D)), W=np.zeros((9, D, 0)), b=np.zeros(0), v=np.zeros(0))
    p["A"][5] = T                                      # east
    u, g = cf.materialise_pairs(p, np.eye(D), PAD)
    assert np.allclose(_dc(g[5]), _dc(T), atol=1e-9)
    assert np.allclose(g[3], g[5].T, atol=1e-9)       # west is the same pairs seen from the other cell
    for i in (0, 2, 6, 8, 1, 7, 4):
        assert np.abs(g[i]).max() < 1e-9
    a = p["a"]                                         # u carries the pad-neighbour pair terms
    assert np.allclose(u, a - a[PAD] + T[:, PAD] + T[PAD, :] - 2 * T[PAD, PAD], atol=1e-9)


def test_end_to_end_bilinear_recovery():
    rng = np.random.default_rng(3)
    planted = dict(a=rng.standard_normal(D), A=rng.standard_normal((4, D, D)),
                   W=np.zeros((9, D, 0)), b=np.zeros(0), v=np.zeros(0))
    w = rng.integers(0, D, (2000, 3, 3))
    t = _targets(planted, np.eye(D), w, PAD, rng, noise=0.01)
    _, g0 = cf.materialise_pairs(planted, np.eye(D), PAD)
    E = np.linalg.qr(rng.standard_normal((D, D)))[0]  # a random (well-conditioned) embedding, k = D
    ls = cf.fit_bilinear_ls(w, t, E, PAD, l2=1e-8)
    ad = cf.fit(w, t, E, PAD, m=0, steps=6000, lr=0.1, lr_end=1e-4, l2=1e-8)
    for r in (ls, ad):
        _, g = cf.materialise_pairs(r["params"], E, PAD)
        err = max(np.abs(_dc(g[i]) - _dc(g0[i])).max() for i in range(9))
        assert err < 0.05, err


def test_end_to_end_head():
    rng = np.random.default_rng(4)
    planted = _bump_head(D)
    w = rng.integers(0, D, (2000, 3, 3))
    t = _targets(planted, np.eye(D), w, PAD)
    E = np.linalg.qr(rng.standard_normal((D, D)))[0]
    ls = cf.fit_bilinear_ls(w, t, E, PAD, l2=1e-8)
    assert ls["rel"] > 0.3, ls["rel"]                  # genuinely non-additive (best m = 0 fit)
    ad0 = cf.fit(w, t, E, PAD, m=0, steps=2000, lr=0.05, lr_end=5e-4, l2=1e-8)
    assert ad0["rel"] > 0.3
    ad = cf.fit(w, t, E, PAD, m=8, steps=4000, lr=0.05, lr_end=5e-4, l2=1e-8, seed=0, scale=0.1)
    assert ad["rel"] < 0.1, ad["rel"]


def test_to_factor():
    rng = np.random.default_rng(5)
    p = cf.init_params(D, 4, 2, seed=0, tie=True)
    E = rng.standard_normal((D, 4))
    f = cf.to_factor(p, E, ("mid", "self"), PAD, "cp")
    if isinstance(f, dict):
        assert f["A"].shape == (9, 4, 4)
    else:
        assert f.name == "cp"
