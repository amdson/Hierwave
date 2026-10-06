"""The generic channel kernel: a site's conditional energies agree with
differences of the total energy (reflections, pads, counts, unaries), and
the ground set keeps the support rule."""
import numpy as np

from castlegen.channels import Model, ground


def _model(H=16, W=24, seed=1):
    tile = ground.tile_channel(ground.KINDS)
    surf = ground.surf_channel()
    m = Model(H, W, [tile, surf], ground.factors())
    rng = np.random.default_rng(seed)
    tile.grid[:] = rng.integers(0, 3, tile.grid.shape)
    surf.grid[:] = rng.integers(0, 9, surf.grid.shape)
    return m, tile, surf


def test_conditional_matches_total():
    m, tile, surf = _model()
    rng = np.random.default_rng(0)
    for _ in range(40):
        y, x = int(rng.integers(m.H)), int(rng.integers(m.W))
        e = m.site_energies("tile", y, x)
        tot = []
        for t in range(tile.D):
            tile.grid[y, x] = t
            tot.append(m.energy("tile"))
        fin = [t for t in range(tile.D) if np.isfinite(e[t])]
        for t in range(tile.D):
            assert np.isfinite(e[t]) == (tot[t][1] == min(v[1] for v in tot)) or not np.isfinite(e[t]), (y, x, t)
        for a in fin:
            for b in fin:
                assert abs((e[a] - e[b]) - (tot[a][0] - tot[b][0])) < 1e-9, (y, x, a, b)


def test_ground_support_and_count():
    m, tile, surf = _model(32, 32)
    tile.grid[:] = 0
    ground.sample_surface(surf, m.H, m.W, seed=3)
    bad = m.sweep("tile", 30, seed=0)
    assert bad == 0
    met = ground.metrics(tile, surf)
    assert met["unsupported"] == 0
    assert met["count_mae"] < 2.0, met
