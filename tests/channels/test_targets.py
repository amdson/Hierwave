"""Targets and the bootstrap fit (train.fit) on tiny Potts models."""
import numpy as np

from castlegen.channels.potts import Potts, Oracle, dist
from castlegen.channels.core import Model
from castlegen.channels.targets import ExactTargets, SampledTargets
from castlegen.channels import train

MID = ["mid_h", "mid_v", "mid_u"]


def _softmax(e):
    w = np.exp(-(e - e.min()))
    return w / w.sum()


def _setup(P, seed=0):
    """Oracle + designed model + tabular feats sharing one set of channels."""
    orc = Oracle(P, seed=seed)
    feats = train.tabular_feats(MID)
    de = train.designed_energies(orc.model, "mid")
    shapes = {n: P.theta0()[n].shape for n in MID}

    def tables(th):
        t = P.theta0()
        if th is not None:
            t.update(train.unpack(th, shapes))
        return t
    return orc, feats, de, shapes, tables


def test_exact_targets_restrict():
    T = ExactTargets(lambda m, h, y, x: np.array([0.1, 0.2, 0.3, 0.4]))
    assert np.allclose(T.at(None, "mid", 0, 0, [0, 1, 2, 3]), [0.1, 0.2, 0.3, 0.4])
    p = T.at(None, "mid", 0, 0, np.array([1, 3]))
    assert np.allclose(p, [1 / 3, 2 / 3]) and abs(p.sum() - 1) < 1e-15


def test_sampled_targets_onehot():
    P = Potts(1, 1, BM=2)
    orc = Oracle(P, seed=1)
    orc.mid.grid[0, 1] = 2
    assert np.array_equal(SampledTargets().at(orc.model, "mid", 0, 1, np.array([0, 2, 3])), [0, 1, 0])


def test_tabular_feats_match_counts():
    P = Potts(2, 2, BM=2)
    orc, feats, *_ = _setup(P, 3)
    m = P.model(orc.top, orc.mid, orc.tile, P.theta0())
    X = feats(m, "mid", 1, 2, np.arange(4))
    g = orc.mid.grid
    cur = g[1, 2]
    n0 = train.pack(train.counts(m, MID), MID)
    for t in range(4):
        g[1, 2] = t
        assert np.allclose(X[t], train.pack(train.counts(m, MID), MID) - n0)
    g[1, 2] = cur


def test_kl_gradient():
    rng = np.random.default_rng(0)
    N, Dm, nf = 30, 4, 7
    X = rng.normal(size=(N, Dm, nf)); E = rng.normal(size=(N, Dm))
    M = rng.random((N, Dm)) < 0.8
    M[:, 0] = True
    PI = np.where(M, rng.random((N, Dm)), 0); PI /= PI.sum(1, keepdims=True)
    th = rng.normal(size=nf)
    f, g = train.kl_loss(th, X, E, PI, M, l2=0.01)
    num = np.array([(train.kl_loss(th + h, X, E, PI, M, 0.01)[0] - train.kl_loss(th - h, X, E, PI, M, 0.01)[0]) / 2e-6
                    for h in np.eye(nf) * 1e-6])
    assert np.allclose(g, num, atol=1e-6), (g, num)
    assert train.kl_loss(np.zeros(nf), X, E, _softmax_rows(E, M), M)[0] < 1e-12


def _softmax_rows(E, M):
    a = np.where(M, -E, -np.inf)
    w = np.exp(a - a.max(1, keepdims=True))
    return w / w.sum(1, keepdims=True)


def test_fit_recovers_known_tables_and_consistency():
    """Targets = the conditionals of a joint with known mid tables: the fit
    recovers them (double-centred) and both consistency violations are ~0;
    a one-sided (non-joint) conditional shows a violation."""
    P = Potts(2, 2, BM=2, J=0.3)
    orc, feats, de, shapes, tables = _setup(P, 4)
    rng = np.random.default_rng(9)
    true = P.theta0()
    true["mid_h"] = rng.normal(size=(4, 4)); true["mid_v"] = rng.normal(size=(4, 4)); true["mid_u"] = rng.normal(size=4)
    m_true = P.model(orc.top, orc.mid, orc.tile, true)
    T = ExactTargets(lambda m, h, y, x: _softmax(m_true.site_energies(h, y, x)))

    def contexts(th, n, r):
        for _ in range(n):
            for c in (orc.top, orc.mid, orc.tile):
                c.grid[:] = r.integers(c.D, size=c.grid.shape)
            yield P.model(orc.top, orc.mid, orc.tile, tables(th))

    th, rep = train.fit(contexts, "mid", T, feats, de, iters=2, n_contexts=40, K=1, l2=1e-8, holdout=0.1)
    fitted = train.unpack(th, shapes)
    for k in ("mid_h", "mid_v"):
        assert np.abs(train.double_centre(fitted[k]) - train.double_centre(true[k])).max() < 1e-3, k
    assert rep["kl_held"][-1] < 1e-6 and rep["viol_targets"][-1] < 1e-9 and rep["viol_fit"][-1] < 1e-9

    A = rng.normal(size=(4, 4)) * 2
    one = ExactTargets(lambda m, h, y, x: _softmax(A[m.chan(h).grid[y, x - 1]] if x > 0 else np.zeros(4)))
    _, rep1 = train.fit(contexts, "mid", one, feats, de, iters=1, n_contexts=10, K=0)
    assert rep1["viol_targets"][0] > 0.1 and rep1["viol_fit"][0] < 1e-9


def test_fit_potts_mid_probs():
    """Oracle contexts, ExactTargets(Oracle.mid_probs), K = 0: at kappa 8 the
    implied mid table is BM J d(m, m'); the fit matches it double-centred."""
    P = Potts(1, 1, BM=2, J=0.5, kappa=8.0)
    orc, feats, de, shapes, tables = _setup(P, 2)
    orc.sweep(20)

    def contexts(th, n, r):
        for _ in range(n):
            orc.sweep(2)
            yield P.model(orc.top, orc.mid, orc.tile, tables(th))

    T = ExactTargets(lambda m, h, y, x: orc.mid_probs(y, x))
    th, rep = train.fit(contexts, "mid", T, feats, de, iters=2, n_contexts=300, K=0, l2=1e-6)
    fitted = train.unpack(th, shapes)
    cc = np.arange(4)
    ref = train.double_centre(P.BM * P.J * dist(cc[:, None], cc[None, :]))
    for k in ("mid_h", "mid_v"):
        err = np.abs(train.double_centre(fitted[k]) - ref).max()
        assert err < 0.05, (k, err, train.double_centre(fitted[k]))
    assert rep["viol_targets"][-1] < 1e-9 and rep["viol_fit"][-1] < 1e-9


def test_autocorr_clamped():
    P = Potts(1, 1, BM=2, J=0.0, kappa=0.0)
    orc = Oracle(P, seed=0)
    assert np.isclose(train.iat(np.ones(50)), 1.0)
    rng = np.random.default_rng(0)
    assert abs(train.iat(rng.normal(size=20000)) - 1) < 0.15
    ar = np.zeros(40000)
    for i in range(1, len(ar)):
        ar[i] = 0.9 * ar[i - 1] + rng.normal()
    assert abs(train.iat(ar) - 19) < 4                              # AR(1): (1 + a) / (1 - a)
    tau = train.autocorr_clamped(orc.model, "tile", ["mid"], sweeps=500, burn=10)
    assert tau < 2                                                   # J = kappa = 0: independent tiles
