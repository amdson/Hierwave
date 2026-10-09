import numpy as np
import pytest

from castlegen.legacy import hier, pipeline
from castlegen.legacy.promise import Tables
from castlegen.legacy.quantities import connectivity as C


def _spec(stages):
    return {"name": "t", "tileset": "demo", "exemplar": "castle", "size": 64, "stages": stages}


HIER = {"stage": "hier", "K": 16, "h_top": 64, "tables": "flat", "lam": 1.0, "sweeps": 2, "top_sweeps": 5,
        "n_pca": 8}
FULL = [HIER, {"stage": "tiles"}, {"stage": "fulfil", "protect_nodes": True},
        {"stage": "repair", "rounds": 5, "T": 0.4, "tau": 1.9, "no_new_nodes": True}, {"stage": "fulfil"}]


@pytest.fixture(scope="module")
def full_run():
    ctxs = []
    ts, tiles, report = pipeline.run(_spec(FULL), seed=4, log=lambda *a: None, ctx_out=ctxs)
    return ts, tiles, report, ctxs[0]


def test_tables_fit_and_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    grids = {16: [rng.integers(0, 16, (8, 8)) for _ in range(3)], 32: [rng.integers(0, 16, (4, 4)) for _ in range(3)]}
    tab = Tables.fit(16, grids, alpha=0.5)
    assert tab.u[16].shape == (16,) and tab.pair[16].shape == (2, 16, 16) and tab.parent[16].shape == (4, 16, 16)
    assert all(np.isfinite(a).all() for a in (tab.u[16], tab.pair[16], tab.parent[16], tab.u[32]))
    tab.save(str(tmp_path / "t.npz"))
    back = Tables.load(str(tmp_path / "t.npz"))
    assert np.allclose(back.pair[16], tab.pair[16]) and np.allclose(back.parent[16], tab.parent[16])
    assert np.array_equal(back.counts["u16"], tab.counts["u16"])


def test_group_sampling_prefers_low_energy():
    """With a unary that makes the value 15 cheap, the per-parent sweep moves
    most blocks there (the sampler actually uses the tables)."""
    tab = Tables.flat(16, [16, 32, 64])
    for h in (16, 32, 64):
        tab.u[h] = np.full(16, 4.0)
        tab.u[h][15] = 0.0
    v = C.ConnectivityPromise(K=16, h_top=64, tables=tab, T=1.0, sweeps=3, top_sweeps=5)
    ctx = hier.Ctx(seed=0)
    hier.run([v], [], 128, ctx)
    O = next(l.vars["conn"] for l in ctx.levels if l.h == 16)
    assert C.consistent(O)
    assert (O == 15).mean() > 0.6


def test_end_to_end(full_run):
    ts, tiles, report, ctx = full_run
    rows = {r["stage"]: r for r in report}
    fulfils = [r for r in report if r["stage"] == "fulfil"]
    for r in fulfils + [rows["repair"]]:          # a node-protected repair keeps the network
        assert r["main_share"] == 1.0
        assert r["comps_per_1k"] == pytest.approx(1000.0 / 64 ** 2)
    assert "max_edits" in fulfils[0] and fulfils[0]["max_edits"] <= 16 * 16
    assert rows["hier"]["sat"].keys() == {"conn16", "conn32", "conn64"}


def test_bottom_up(full_run):
    """Every level's promise is kept by the exact state recomputed from the
    final tiles, and every level is a consistent refinement of the next."""
    ts, tiles, report, ctx = full_run
    v = ctx.cache["promises"][0]
    levels = {lv.h: lv.vars["conn"] for lv in ctx.cache["levels"] if "conn" in lv.vars}
    assert sorted(levels) == [16, 32, 64]
    for h, O in levels.items():
        assert C.consistent(O, levels.get(2 * h))
        S = v.level_states(ts, tiles, h)
        for y, x in np.ndindex(*O.shape):
            assert C.satisfied(int(O[y, x]), S[y][x])
            assert int(O[y, x]) & ~C.abstract(S[y][x]) == 0
    assert all(f == 1.0 for f in report[-1]["sat"].values())


def test_determinism():
    spec = _spec([HIER, {"stage": "tiles"}, {"stage": "fulfil"}])
    a = pipeline.run(spec, seed=7, log=lambda *a: None)[1]
    b = pipeline.run(spec, seed=7, log=lambda *a: None)[1]
    assert np.array_equal(a, b)
