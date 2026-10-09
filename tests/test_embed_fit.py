import numpy as np

from castlegen.channels import embed_fit as ef

DIRS4 = [(0, True), (1, True), (0, False), (1, False)]       # +h, +v, -h, -v


def _data(D, n, rng, nf=0, full=False, G=None, u=None):
    recs = []
    for _ in range(n):
        cand = np.arange(D) if full else np.sort(rng.choice(D, rng.integers(2, D + 1), replace=False))
        nb = rng.integers(-1, D, 4)
        X = rng.normal(size=(len(cand), nf)) if nf else None
        recs.append([cand, rng.normal(size=len(cand)), np.ones(len(cand)) / len(cand), nb, X])
    d = ef.Data(recs, D)
    if G is not None:                                         # exact conditionals of the pairwise model
        a = np.where(d.MF, -(d.EF + ef.tables_energy(G, u, d, DIRS4)), -np.inf)
        d.PIF = np.where(d.MF, np.exp(a - a.max(1, keepdims=True)), 0.0)
    else:
        d.PIF = np.where(d.MF, rng.random(d.MF.shape), 0.0)
    d.PIF /= d.PIF.sum(1, keepdims=True)
    return d


def _fd(p, d, fix0=True):
    L = lambda q: ef.loss(q, d, DIRS4, 1e-3, fix0)
    v, g = L(p)
    x = ef.pack(p)
    ga = ef.pack(g)
    rng = np.random.default_rng(1)
    for i in rng.choice(len(x), 25, replace=False):
        if ga[i] == 0 and abs(x[i]) == 0:                     # fixed gauge entries (E[0], u[0])
            continue
        h = 1e-6
        xp, xm = x.copy(), x.copy()
        xp[i] += h; xm[i] -= h
        num = (L(ef.unpack(xp, p))[0] - L(ef.unpack(xm, p))[0]) / (2 * h)
        assert abs(num - ga[i]) < 1e-6 * max(1, abs(num)), (i, num, ga[i])


def test_stamp_part_full():
    rng = np.random.default_rng(5)
    d = _data(6, 10, rng, nf=3)
    p = ef.init(6, 2, 2, rng, nf=3)
    p["theta"] = rng.normal(size=3)
    P0 = ef.psi(dict(p, theta=np.zeros(3)), d, DIRS4)
    S = d.XS @ p["theta"]
    assert np.allclose((ef.psi(p, d, DIRS4) - P0)[d.R, d.Cf], S[d.R, d.Cc])


def test_loss_gradient():
    rng = np.random.default_rng(0)
    d = _data(6, 40, rng, nf=3)
    _fd(ef.init(6, 3, 2, rng, scale=0.5, nf=3), d)
    p = ef.init(6, 3, 2, rng, scale=0.5, mlp=5, nd=4)
    p["b1"] += 0.1
    _fd(p, d)
    p = ef.init(6, 3, 2, rng, scale=0.5, fix0=False)
    p["u"] = rng.normal(size=6)
    _fd(p, d, fix0=False)


def test_materialise_matches_direct():
    rng = np.random.default_rng(2)
    d = _data(7, 50, rng)
    p = ef.init(7, 3, 2, rng, scale=0.7)
    p["u"][1:] = rng.normal(size=6)
    G, u = ef.tables(p)
    P = ef.psi(p, d, DIRS4)
    assert np.allclose(ef.tables_energy(G, u, d, DIRS4)[d.MF], P[d.MF], atol=1e-12)
    for n in range(5):                                        # direct: u[c] + sum_j g(c, nb_j) by loops
        for c in d.CAND[n][d.M[n]]:
            v = p["u"][c] + sum((G[o][c, b] if f else G[o][b, c]) for (o, f), b in zip(DIRS4, d.NB[n]) if b >= 0)
            assert abs(v - P[n, c]) < 1e-12
    assert np.all(G[:, 0, :] == 0) and np.all(G[:, :, 0] == 0)


def test_recover_rank2():
    rng = np.random.default_rng(3)
    true = ef.init(5, 2, 2, rng, scale=1.0)
    true["u"][1:] = rng.normal(size=4)
    G, u = ef.tables(true)
    d = _data(5, 400, rng, full=True, G=G, u=u)
    p, info = ef.fit_embed(d, 5, 2, DIRS4, 2, l2=0.0, restarts=3, steps=1500, seed=4)
    Gf, uf = ef.tables(p)
    assert np.abs(Gf - G).max() < 0.05 and np.abs(uf - u).max() < 0.05, (np.abs(Gf - G).max(), np.abs(uf - u).max())
