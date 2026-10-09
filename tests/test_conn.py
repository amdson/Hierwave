import numpy as np
import pytest

from castlegen import conn, tileset


@pytest.fixture(scope="module")
def ts():
    return tileset.load("demo")


def _random_tiles(ts, rng, H, W, density=0.75):
    sigs = np.arange(ts.n_sig - 1)                       # anything but the gate
    tiles = np.where(rng.random((H, W)) < density, rng.choice(sigs, (H, W)), ts.WALL).astype(np.int32)
    tiles[0, W // 2] = ts.GATE
    return tiles


@pytest.mark.parametrize("H,W", [(4, 4), (16, 16), (20, 20), (37, 23)])
def test_full_build_matches_bfs(ts, H, W):
    rng = np.random.default_rng(H * 100 + W)
    for _ in range(5):
        tiles = _random_tiles(ts, rng, H, W)
        c = conn.Connectivity(ts, tiles)
        assert (c.components(), c.unreached_components()) == conn.components_bfs(ts, tiles)


def test_incremental_edits_match_bfs(ts):
    rng = np.random.default_rng(0)
    tiles = _random_tiles(ts, rng, 20, 20, density=0.5)
    c = conn.Connectivity(ts, tiles)
    sigs = np.arange(ts.n_sig - 1)
    for _ in range(400):
        y, x = rng.integers(0, 20, 2)
        if tiles[y, x] == ts.GATE:
            continue
        tiles[y, x] = rng.choice(sigs)
        c.set(y, x, tiles[y, x])
        assert (c.components(), c.unreached_components()) == conn.components_bfs(ts, tiles)
    assert np.array_equal(c.tiles, tiles)


def test_open_tiles_everywhere_is_one_component(ts):
    cross = next(s for s in range(ts.n_sig) if ts.sig_name(s) == "hall_cross@0")
    tiles = np.full((20, 20), cross, np.int32)
    tiles[0, 10] = ts.GATE
    c = conn.Connectivity(ts, tiles)
    assert c.components() == 1 and c.unreached_components() == 0
    c.set(0, 11, ts.WALL)                                # still connected around it
    assert c.components() == 1
