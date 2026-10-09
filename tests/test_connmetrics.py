import numpy as np

from castlegen import conn, connmetrics as cm
from castlegen import exemplar as ex
from castlegen import tileset
from test_exemplar import LAYOUT


def test_labels_match_bfs_and_wrap():
    ts = tileset.load("demo")
    _, t, _ = ex.build(LAYOUT, ts)
    assert cm.labels(ts, t, torus=False)[1] == conn.components_bfs(ts, t)[0]
    g = np.full((4, 4), ts.WALL, np.int32)
    g[1, 0] = g[1, 3] = ex.sig_by_doors(ts, "hallway", {1, 3})     # east-west straights at the two edges
    assert cm.labels(ts, g, torus=False)[1] == 2
    assert cm.labels(ts, g, torus=True)[1] == 1                    # joined across the wrap


def test_connect_joins_everything_without_new_violations():
    ts = tileset.load("demo")
    _, t, clamp = ex.build(LAYOUT, ts)
    assert cm.global_stats(ts, t)["components"] > 1
    c, left = ex.connect(ts, t, clamp)
    assert left == 0 and cm.global_stats(ts, c)["components"] == 1
    assert ex.bad_cells(ts, c, 1.9).sum() <= ex.bad_cells(ts, t, 1.9).sum()
    cost, _ = cm.repair_cost(ts, c)
    assert len(cost) == 0


def test_local_connect_adds_no_components_or_violations():
    ts = tileset.load("demo")
    _, t, clamp = ex.build(dict(LAYOUT, size=[32, 32]), ts)
    out, st = ex.local_connect(ts, t, K=8, r=2, max_cost=4, clamp=clamp)
    assert st["windows"] == 16
    assert cm.global_stats(ts, out)["components"] <= cm.global_stats(ts, t)["components"]
    assert ex.bad_cells(ts, out, 1.9).sum() <= ex.bad_cells(ts, t, 1.9).sum()
