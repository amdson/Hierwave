import numpy as np
import pytest

from castlegen import preview, render, tileset


@pytest.fixture(scope="module")
def ts():
    return tileset.load("demo")


def sig(ts, name, deg=0):
    return next(s for s in range(ts.n_sig) if ts.sig_name(s) == f"{name}@{deg}")


def test_rotated_sockets_make_doors(ts):
    Dh, Dv = ts.np_tables["Dh"], ts.np_tables["Dv"]
    ns, ew = sig(ts, "hall_straight", 0), sig(ts, "hall_straight", 90)
    assert Dv[ns, ns] and not Dh[ns, ns]
    assert Dh[ew, ew] and not Dv[ew, ew]
    # └ opens N and E: joins a │ above it and a ─ to its right, nothing below or left
    corner = sig(ts, "hall_corner", 0)
    assert Dv[ns, corner] and Dh[corner, ew]
    assert not Dv[corner, ns] and not Dh[ew, corner]
    # the gate's south door joins a chamber whose door faces north
    assert Dv[ts.GATE, sig(ts, "chamber", 0)] and not Dv[ts.GATE, sig(ts, "chamber", 180)]


def test_pair_energies(ts):
    Eh, Ev = ts.np_tables["Eh"], ts.np_tables["Ev"]
    ns, court, tree = sig(ts, "hall_straight"), sig(ts, "courtyard"), sig(ts, "tree")
    assert Ev[ts.WALL, ns] == pytest.approx(2.0)                  # north door into a wall dangles
    assert Eh[ts.WALL, ns] == 0.0                                  # its wall side does not
    assert Eh[tree, court] == Eh[court, tree] == pytest.approx(-1.3)
    assert Eh[tree, ts.WALL] == Ev[ts.WALL, tree] == pytest.approx(3.0)    # its court sides dangle
    assert Ev[court, court] == pytest.approx(-1.4)
    # walled courtyard: wall side against a wall is free, court side against it dangles
    edge_n = sig(ts, "courtyard_wall", 0)
    assert Ev[ts.WALL, edge_n] == 0.0 and Eh[ts.WALL, edge_n] == pytest.approx(3.0)
    # the courtyard does not connect to a hallway; its gateway does
    assert not ts.np_tables["Dv"][ns, court] and ts.np_tables["Dv"][ns, sig(ts, "courtyard_gateway", 0)]
    # chamber door onto a hallway: coupling, no dangling
    assert Ev[ns, sig(ts, "chamber", 0)] == pytest.approx(-2.0)


def test_unknown_tag_is_an_error():
    spec = {"sockets": ["wall", "door"], "connects": [["door", "door"]],
            "kinds": [{"name": "a", "sides": {"N": "door", "E": "wall", "S": "wall", "W": "wall"}}],
            "rules": [{"a": "a", "b": "typo", "energy": 1.0}]}
    with pytest.raises(ValueError, match="typo"):
        tileset.compile_spec(spec)


def _million_spec():
    axis = lambda n: [f"v{i}" for i in range(n)]
    return {"sockets": ["wall", "door", "open"], "connects": [["door", "door"], ["door", "open"], ["open", "open"]],
            "kinds": [
                {"name": "room", "sides": {"N": "door", "E": "wall", "S": "wall", "W": "wall"}, "rotations": "all",
                 "variants": {"style": axis(100), "decor": axis(50),
                              "size": [{"name": "small", "weight": 3.0}] + axis(49)}},
                {"name": "yard", "sides": {"N": "open", "E": "open", "S": "open", "W": "open"},
                 "variants": {"flora": axis(1000)}}]}


def test_a_million_types_keep_a_small_signature_set():
    ts = tileset.compile_spec(_million_spec())
    assert ts.n_types == 4 * 100 * 50 * 50 + 1000 + 2
    assert ts.n_sig == 4 + 1 + 2
    rng = np.random.default_rng(0)
    grid = rng.integers(0, ts.n_sig - 2, (64, 64)).astype(np.int32)
    types = np.asarray(tileset.decorate(ts, 5, grid))
    assert types.min() >= 0 and types.max() < ts.n_types
    assert np.array_equal(ts.sig_of_type(types), grid)             # the draw never leaves the signature
    k, _, vals = ts.decode(types)
    small = (vals[2][k == 0] == 0).mean()                           # size axis: weight 3 of 52
    assert abs(small - 3 / 52) < 0.02
    room_types = types[k == 0]                                      # 10^6 room types: nearly all distinct
    assert len(np.unique(room_types)) > 0.95 * room_types.size
    name = ts.type_name(int(types[0, 0]))
    assert name.startswith(("room@", "yard@"))


def test_render_and_preview(ts):
    tiles, types = preview.sample_local(ts, size=16, sweeps=20)
    img = render.image(ts, types, px=6)
    assert img.shape == (96, 96, 3) and img.dtype == np.uint8
    assert np.array_equal(img, render.image(ts, types, px=6))      # stable colours
    txt = render.ascii_grid(ts, types).splitlines()
    assert len(txt) == 16 and all(len(r) == 16 for r in txt)
    tree = int(ts.kind_offset[[k.name for k in ts.kinds].index("tree")])
    assert render.sprite(ts, tree, 8).std() > 0                    # the image tile, not a flat colour


def test_structure_expansion_and_census(ts):
    name, grid = ts.structures[0]
    assert name == "greenhouse" and grid.shape == (3, 4)
    sig = {k: int(np.nonzero(ts.sig_kind == k)[0][0]) for k in grid.ravel()}
    t = np.full((8, 8), ts.WALL, np.int32)
    t[2:5, 1:5] = np.vectorize(sig.get)(grid)
    assert tileset.structure_census(ts, t)["greenhouse"] == (1, 0)
    Eh = ts.np_tables["Eh"]
    # a seam joins only its partner: r0c0 | r0c1 bonds; r0c1 | r0c0 leaves r0c1's east seam
    # facing r0c0's west wall (r0c0 is on the west edge), so one dangle
    assert Eh[sig[grid[0, 0]], sig[grid[0, 1]]] == pytest.approx(-1.5)
    assert Eh[sig[grid[0, 1]], sig[grid[0, 0]]] == pytest.approx(4.5)
    t[2, 1] = ts.WALL
    assert tileset.structure_census(ts, t)["greenhouse"] == (0, 11)
