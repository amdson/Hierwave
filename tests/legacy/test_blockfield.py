import numpy as np

from castlegen.legacy import blockconn as BC, blockfield as BF, tileset


def _setup(seed):
    ts = tileset.load("demo")
    tabs = BC.tables(ts)
    E = np.load("cache/castle_ex.npy").astype(np.int64)
    rng = np.random.default_rng(seed)
    y, x = rng.integers(0, 64, 2)
    T = np.ascontiguousarray(np.roll(E, (-y, -x), (0, 1))[:16, :16])
    lab = np.zeros(8, np.int64)
    lab[rng.choice(8, 2, replace=False)] = 1
    lab[(lab == 0) & (rng.random(8) < 0.5)] = BC.OPEN
    G, D = 4, 20
    Gf, Df = BF.init_fields(T, lab, tabs, G, D)
    # perturb: random fields and tiles, so sites see every kind of neighbour
    m = rng.random((16, 16)) < 0.3
    Gf[m] = rng.integers(0, G + 2, m.sum())
    Df[m] = rng.integers(0, D + 1, m.sum())
    T[rng.random((16, 16)) < 0.1] = rng.integers(0, ts.n_sig - 1)
    t = ts.np_tables
    Eh, Ev, logz = (np.ascontiguousarray(t[k], np.float64) for k in ("Eh", "Ev", "logz"))
    allowed = np.ones(ts.n_sig, bool)
    allowed[ts.GATE] = False
    pc = rng.random((16, 16, ts.n_sig))
    return ts, tabs, T, Gf, Df, lab, Eh, Ev, logz, allowed, pc, G, D, rng


def test_closed_form_matches_enumeration():
    for seed in range(6):
        ts, tabs, T, Gf, Df, lab, Eh, Ev, logz, allowed, pc, G, D, rng = _setup(seed)
        for _ in range(8):
            y, x = rng.integers(0, 16, 2)
            for lam, LV, DELTA in ((4.0, 1000.0, 0.05), (4.0, 3.0, 0.3), (0.0, 2.0, 0.0), (1.5, 5.0, 0.0)):
                args = (Eh, Ev, logz, allowed, pc, lab, *tabs, lam, 1.0, LV, 0.1, DELTA, G, D, ts.WALL)
                full = BF.site_logw_enum(y, x, T, Gf, Df, *args)
                marg = BF.site_logw_closed(y, x, T, Gf, Df, *args)
                with np.errstate(divide="ignore"):
                    ref = np.logaddexp.reduce(full.reshape(len(full), -1), axis=1)
                np.testing.assert_allclose(marg, ref, rtol=1e-9, atol=1e-9)
                for t in rng.choice(np.flatnonzero(allowed), 6):
                    p = BF.gd_probs_closed(y, x, t, T, Gf, Df, lab, *tabs, lam, 1.0, LV, 0.1, DELTA, G, D)
                    w = np.exp(full[t] - full[t].max())
                    np.testing.assert_allclose(p, w / w.sum(), atol=1e-12)
