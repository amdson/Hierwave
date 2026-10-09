import numpy as np

from castlegen.legacy import exemplar as ex
from castlegen.legacy import texsyn, tileset

LAYOUT = {"size": [16, 16], "sweeps": 20, "place": [
    {"room": "courtyard", "at": [1, 1], "size": [5, 6], "gates": [["E", 2]], "interior": "clamp"},
    {"structure": "greenhouse", "at": [9, 2]},
    {"path": "hallway", "points": [[3, 7], [3, 12], [8, 12]]},
]}


def test_placement_is_valid_and_clamped():
    ts = tileset.load("demo")
    tiles, clamp, _ = ex.place(ts, LAYOUT)
    assert tileset.structure_census(ts, tiles)["greenhouse"] == (1, 0)
    ring = ex.bad_cells(ts, np.where(clamp, tiles, ts.WALL))
    assert not ring[1:6, 1:7].any()                      # the walled courtyard ring is consistent
    ts, built, clamp2 = ex.build(LAYOUT, ts)
    assert np.array_equal(built[clamp], tiles[clamp])     # infill never touches placed cells


def test_zero_jitter_tiles_the_exemplar():
    ts = tileset.load("demo")
    _, E, _ = ex.build(LAYOUT, ts)
    an = texsyn.Analysis(ts, E, n_pca=8)
    S = texsyn.synthesize(an, 32, 0.0)
    assert not texsyn.seam_mask(S, 16).any()
    y0, x0 = S[0, 0]                                     # a seamless tiling, offset by S[0, 0]
    assert np.array_equal(E[S[..., 0], S[..., 1]], np.roll(np.tile(E, (2, 2)), (-y0, -x0), (0, 1)))
