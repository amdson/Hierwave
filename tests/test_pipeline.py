import numpy as np
import pytest

from castlegen import exemplar as ex
from castlegen import pipeline as pl
from castlegen import tileset

CASTLE = {"name": "castle-mini", "tileset": "demo", "size": 32, "seed": 1,
          "exemplar": {"tileset": "demo", "size": [16, 16], "sweeps": 20, "T": 0.6, "clean": True, "connect": True,
                       "place": [{"room": "courtyard", "at": [1, 1], "size": [5, 6], "gates": [["E", 2]]},
                                 {"path": "hallway", "points": [[3, 7], [3, 12], [8, 12]]}]},
          "stages": [{"stage": "synthesize", "r": [1, 0, 0, 0, 0]},
                     {"stage": "repair", "rounds": 5},
                     {"stage": "connect", "K": 8, "r": 2, "tag": "hallway"}]}

WILDS = {"name": "wilds-mini", "tileset": "wilds", "size": 32, "seed": 1,
         "exemplar": {"tileset": "wilds", "size": [16, 16], "sweeps": 20, "T": 0.6, "clean": True, "flow": True,
                      "connect": {"tag": "road", "void": "lake"},
                      "place": [{"stroke": "river", "points": [[8, 0], [8, 15]], "width": 2},
                                {"path": "bridge", "points": [[7, 4], [8, 4]]},
                                {"cell": "landing@180", "at": [6, 4]}, {"cell": "landing@0", "at": [9, 4]}]},
         "stages": [{"stage": "synthesize", "r": [1, 0, 0, 0, 0]},
                    {"stage": "repair", "rounds": 5},
                    {"stage": "connect", "K": 8, "r": 2, "tag": "road"}]}


@pytest.mark.parametrize("spec", [CASTLE, WILDS], ids=["castle", "wilds"])
def test_pipeline_runs_every_stage(spec, tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "CACHE_DIR", str(tmp_path))
    seen = []
    ts, tiles, report = pl.run(spec, log=lambda *a: None, on_stage=lambda name, t: seen.append(name))
    assert tiles.shape == (32, 32)
    assert [r["stage"] for r in report] == ["exemplar", "synthesize", "repair", "connect"]
    assert seen == ["synthesize", "repair", "connect"]
    assert report[-1]["bad"] <= report[1]["bad"]                      # repairs never add violations
    assert report[-1]["comps_per_1k"] <= report[2]["comps_per_1k"]
    assert len(list(tmp_path.glob("exemplar-*.npy"))) == 1           # exemplar cached
    _, again, _ = pl.run(spec, log=lambda *a: None)
    assert np.array_equal(tiles, again)                               # deterministic for a seed


def test_wilds_placement():
    ts = tileset.load("wilds")
    tiles, clamp, _ = ex.place(ts, WILDS["exemplar"])
    river = ts.water[tiles[7:9, :]] & ~np.asarray(ts.np_tables["is_room"])[tiles[7:9, :]]
    assert river.sum() == 32 - 2                                       # the river, minus the bridge
    assert (ts.flow_dir[tiles[7:9, :]] == 1).all()                    # laid flowing east, bridges included
    assert ts.kinds[ts.sig_kind[tiles[8, 4]]].name == "bridge_straight"
    road = ex.sig_by_doors(ts, "road", {0, 2})                         # land on the other sides
    assert ts.kinds[ts.sig_kind[road]].name == "road_straight"
