"""biome_fit: materialise against the closed-form reference, the exact top
hook against brute force, Phi_top(none) = 0 from AISTargets."""
import numpy as np

from castlegen.channels import paintpot
from castlegen.channels.aistargets import AISTargets
from castlegen.channels.biome_fit import Bench, materialise, mid_tables
from castlegen.channels.circles_biome import OFFNAMES, OFFS, CirclesBiome


def ref_theta(C):
    """psi = sum_t fz[dem_t] (conflict 0: never painted by an admissible pair)."""
    th = np.zeros(paintpot.nfeat(4, paintpot.OFF8))
    th[:3] = C.fz[:3]
    return th


def test_materialise_reproduces_reference():
    C = CirclesBiome(1, 1)
    T = materialise(C, ref_theta(C))
    for n, d in zip(OFFNAMES, OFFS):
        fin = np.isfinite(C.PAIR[d])
        assert np.abs(T[n][fin] - C.PAIR[d][fin]).max() < 1e-9, n
    ref_u = C.reference(centred=False)["obj_u"] - C.pres_e
    assert np.abs(T["obj_u"] - ref_u).max() < 1e-9
    assert T["obj_u"][0] == 0 and all((T[n][0] == 0).all() and (T[n][:, 0] == 0).all() for n in OFFNAMES)


def test_top_hook_and_none():
    C = CirclesBiome(2, 2)
    B = Bench(C, seed=1)
    tab = mid_tables(C, ref_theta(C))
    B.fw.theta = tab
    B.fw.run(10, 10, 2)
    m = B.fw.model
    hook = B.top_hook(tab)
    A = AISTargets("obj", B.top_region, B.designed_e("biome"), B.top_dormancy, K=8, M=4, seed=0)
    st = [(c.grid.copy(), c.fixed.copy()) for c in m.channels]
    lz, se = A.log_z(m, "biome", 0, 1, np.arange(C.P))
    assert lz[0] == 0.0 and se[0] == 0.0                          # Phi_top(none) = 0 exactly
    assert all(np.array_equal(c.grid, g) and np.array_equal(c.fixed, f) for c, (g, f) in zip(m.channels, st))
    E = AISTargets("obj", B.top_region, B.designed_e("biome"), B.top_dormancy, hook=hook)
    lx, _ = E.log_z(m, "biome", 0, 1, np.arange(C.P))
    assert lx[0] == 0.0
    # brute force at T = discs over its four slots (17^4), through the model's own energy
    B.biome.grid[0, 1] = 1
    B.top_dormancy(m, "biome", 0, 1)
    sl = [(a, b) for a in range(2) for b in range(2, 4)]
    vals = np.flatnonzero(C.FAM <= 1)
    old = B.obj.grid.copy()
    en = lambda: (lambda r: r[0] if r[1] == 0 else np.inf)(m.energy("obj"))
    es = []
    for v in np.array(np.meshgrid(*[vals] * 4, indexing="ij")).reshape(4, -1).T:
        for (a, b), t in zip(sl, v):
            B.obj.grid[a, b] = t
        es.append(en())
    B.obj.grid[:] = old
    es = np.array(es)
    fin = np.isfinite(es)
    mn = es[fin].min()
    for (a, b) in sl:
        B.obj.grid[a, b] = 0
    e0 = en()
    B.obj.grid[:] = old
    brute = -(mn - e0) + np.log(np.exp(-(es[fin] - mn)).sum())
    assert abs(hook(m, "biome", 0, 1) - brute) < 1e-8
