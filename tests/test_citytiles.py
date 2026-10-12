import numpy as np
import pytest

from castlegen import citytiles as ct


@pytest.fixture(scope="module")
def ts():
    return ct.TileSet()


def _id(ts, corners, edges):
    return ts.tiles.index((tuple(corners), tuple(edges)))


def test_every_socket_has_a_partner(ts):
    """No tile is dead on arrival: each of its sockets is offered by some
    tile on the facing side."""
    assert set(ts.sE) <= set(ts.sW) and set(ts.sW) <= set(ts.sE)
    assert set(ts.sS) <= set(ts.sN) and set(ts.sN) <= set(ts.sS)


@pytest.mark.parametrize("corners, edges", [
    ("PPPP", "0000"),        # parkland
    ("WWWW", "0000"),        # open water
    ("RRRR", "ssss"),        # residential street grid
    ("DDDD", "ssss"),        # downtown grid
    ("IIII", "hhhh"),        # industrial highway grid
])
def test_periodic_tilings(ts, corners, edges):
    """A single tile that tiles the plane: the set has trivial infinite tilings."""
    t = _id(ts, corners, edges)
    assert ts.sE[t] == ts.sW[t] and ts.sS[t] == ts.sN[t]


def test_rules(ts):
    tiles = set(ts.tiles)
    assert (tuple("RRRR"), tuple("s0s0")) in tiles
    assert (tuple("RRRR"), tuple("h0h0")) not in tiles          # no housing on a highway
    assert (tuple("RRII"), tuple("ssss")) not in tiles          # I never meets R
    assert (tuple("RRCC"), tuple("0000")) not in tiles          # district boundary needs a road
    assert (tuple("PRCI"), tuple("ssss")) not in tiles          # three land zones never meet
    assert (tuple("WWWW"), tuple("s0s0")) in tiles              # a bridge
    assert (tuple("WWWW"), tuple("ssss")) not in tiles          # bridges do not cross
    assert (tuple("RRWW"), tuple("0s0s")) not in tiles          # nothing along a shore
    assert (("H00", "H01", "H11", "H10"), tuple("0000")) in tiles   # airport interior
    assert (("H00", "H01", "H11", "H10"), tuple("s000")) not in tiles
    assert not any(c[0] == "H01" and c[1] == "H00" for c, _ in ts.tiles)  # footprint order


def test_highways_are_straight(ts):
    h = ct.F["h"]
    for f in ts.feats:
        k = np.flatnonzero(f == h)
        assert len(k) in (0, 2, 4)
        if len(k) == 2:
            assert k[1] - k[0] == 2
