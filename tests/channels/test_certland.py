import numpy as np

from castlegen.channels import certland as cl, certland_fit as cf


def _world(m=4, seed=0):
    rng = np.random.default_rng(seed)
    w = cl.World(64, 64)
    cf.set_noise(w, 1)
    cl.generate(w, sweeps=2, seed=0)
    for i in range(4):
        p = cl.zero_params(m)
        for k in "aSAWbv":
            p[k] = rng.normal(0, 0.3, np.shape(p[k]))
        w.params[i] = cl.pack_params(p)
    return w, rng


def test_generate_valid():
    w, _ = _world()
    bad = cl.generate(w, sweeps=3, seed=1)
    assert sum(bad) == 0
    assert cl.overhangs(w.rho[cl.NL - 1]) == 0


def test_site_matches_window():
    w, rng = _world()
    for i in (1, 2, 3):
        a = w.args(i)
        h = cl.HS[i]
        rows, cols = w.rho[i].shape
        for _ in range(5):
            by, bx = rng.integers(rows // 2), rng.integers(cols // 2)
            y, x = 2 * by + rng.integers(2), 2 * bx + rng.integers(2)
            for ch in (0, 1):
                D = h * h + 1 if ch == 0 else len(w.Aof[i])
                soft, viol = np.zeros(D), np.zeros(D, np.int64)
                cl.site_terms(*a, y, x, ch, soft, viol)
                g = w.rho[i] if ch == 0 else w.cert[i]
                cur = g[y, x]
                ws = []
                for t in range(D):
                    g[y, x] = t
                    ws.append(cl.window_terms(*a, by, bx))
                g[y, x] = cur
                s = np.array([u for u, _ in ws])
                v = np.array([u for _, u in ws])
                assert np.allclose(s - s[0], soft - soft[0], atol=1e-8)
                assert np.array_equal(v - v[0], viol - viol[0])


def test_jax_matches_kernel():
    w, rng = _world()
    i, h, m = 2, cl.HS[2], 4
    th, _ = w.params[i]
    a_, S, A, W, b, v = cl._unpack(th, m)
    P = dict(a=a_, S=S, A=A, W=W, b=b, v=v)
    a = w.args(i)
    for y, x in [(0, 0), (3, 5), (7, 7)]:
        for ch in (0, 1):
            D = h * h + 1 if ch == 0 else len(w.Aof[i])
            s1, v1 = np.zeros(D), np.zeros(D, np.int64)
            s0, v0 = np.zeros(D), np.zeros(D, np.int64)
            cl.site_terms(*a, y, x, ch, s1, v1)
            cl.site_terms(*a[:-1], 0, y, x, ch, s0, v0)
            lk = s1 - s0
            w0, mask = cf.window(cf.phi_grid(w, i), y, x)
            E = w.emb[i][ch]
            cur = (w.rho[i] if ch == 0 else w.cert[i])[y, x]
            wins = np.repeat(w0[None], D, 0)
            wins[:, 2, 2] = w0[2, 2] - E[cur] + E
            le = np.asarray(cf.local_energy(P, wins, np.repeat(mask[None], D, 0), m))
            assert np.allclose(le - le[0], lk - lk[0], atol=1e-9)
