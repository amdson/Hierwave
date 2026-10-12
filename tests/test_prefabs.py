import numpy as np
import pytest

from castlegen import citytiles as ct
from castlegen import prefabs as pf

P = pf.PAL


def write(path, text):
    path.write_text(text)
    return str(path)


def test_parse_inline(tmp_path):
    p = write(tmp_path / "x.txt", """name: x
size: 2 3   # rows cols
legend: g=grass w=house_wall
layer 0-2: g
layer 3:
gwg
_.w
# comment
layer 5:
www
www
""")
    d = pf.parse_prefab(p)
    assert d["name"] == "x" and d["size"] == (2, 3)
    a = d["layers"]
    assert a.shape == (24, 2, 3)
    assert (a[1] == P["grass"]).all()
    assert a[3, 0, 1] == P["house_wall"] and a[3, 1, 0] == -1 and a[3, 1, 1] == P["air"]
    assert (a[4] == -1).all() and (a[5] == P["house_wall"]).all()


@pytest.mark.parametrize("body,msg", [
    ("size: 1 3\nlegend: g=grass\nlayer 3:\ngg\n", "t.txt:4: row has length 2"),
    ("size: 1 2\nlegend: g=grass\nlayer 3:\ngq\n", "t.txt:4: unknown legend char 'q'"),
    ("size: 1 2\nlegend: g=nothing\n", "t.txt:2: unknown palette name 'nothing'"),
    ("size: 1 2\nlegend: g=grass\nlayer 3:\n", "incomplete"),
    ("size: 1 2\nlayer 30: _\n", "t.txt:2: layer range"),
])
def test_parse_errors(tmp_path, body, msg):
    p = write(tmp_path / "t.txt", body)
    with pytest.raises(pf.PrefabError) as e:
        pf.parse_prefab(p)
    assert msg in str(e.value).replace(str(tmp_path) + "/", "")


def test_transforms():
    rng = np.random.RandomState(0)
    a = rng.randint(0, 9, (3, 4, 2)).astype(np.int16)
    b = a
    for _ in range(4):
        b = pf.rot90(b)
    assert (b == a).all()
    assert (pf.mirror_j(pf.mirror_j(a)) == a).all()
    assert (pf.mirror_i(pf.mirror_i(a)) == a).all()
    assert (pf.rot90(a, 2) == pf.mirror_i(pf.mirror_j(a))).all()
    # north strip -> east: outer row 0 goes to the last column
    m = np.zeros((1, 4, 2), np.int16)
    m[0, 0, :] = 5
    r = pf.rot90(m)
    assert r.shape == (1, 2, 4) and (r[0, :, 3] == 5).all() and r.sum() == 10


def fake_set(d):
    d.mkdir(exist_ok=True)
    mk = {"P": "leaves", "R": "house_wall", "C": "shop_wall", "D": "tower_wall",
          "I": "shed_wall", "W": "water", "A": "tarmac"}
    for z, nm in mk.items():
        write(d / f"zone_{z}.txt", f"size: 4 4\nlegend: x={nm}\nlayer 4: x\n")
    pal = {"street": "asphalt", "avenue": "concrete", "rail": "rail", "highway": "deck"}
    for f, nm in pal.items():
        write(d / f"edge_{f}.txt", f"size: 4 2\nlegend: x={nm}\nlayer 3: x\n")
        write(d / f"bridge_{f}.txt", f"size: 4 2\nlegend: x={nm}\nlayer 6: x\n")
    write(d / "wildcard.txt", "size: 8 8\nlegend: x=plaza\nlayer 3: x\n")
    return str(d)


class Tiny:
    """Tile 0: R everywhere, street N.  Tile 1: W everywhere, highway E, street W
    (bridges).  Tile 2: P P / W W (SE, SW water), no features."""
    T = 3
    corners = np.array([[ct.Z["R"]] * 4, [ct.Z["W"]] * 4,
                        [ct.Z["P"], ct.Z["P"], ct.Z["W"], ct.Z["W"]]])
    feats = np.array([[1, 0, 0, 0], [0, 4, 0, 1], [0, 0, 0, 0]])


def test_assemble_city(tmp_path):
    d = fake_set(tmp_path / "city")
    out = pf.assemble_city(Tiny, d)
    assert out.shape == (4, 24, 8, 8) and out.dtype == np.uint8
    t0 = out[0]
    assert t0[3, 0, 3] == P["asphalt"] and t0[3, 0, 4] == P["asphalt"]
    assert t0[3, 7, 3] == P["grass"]                                   # no street south
    assert t0[3, 3, 3] == P["asphalt"] and t0[3, 4, 4] == P["asphalt"]  # centre
    assert t0[4, 0, 0] == P["house_wall"] and t0[4, 7, 7] == P["house_wall"]
    assert t0[0, 0, 0] == P["dirt"]
    t1 = out[1]
    assert (t1[6, 3:5, 0:3] == P["asphalt"]).all()     # W street bridge
    assert (t1[6, 3:5, 5:8] == P["deck"]).all()        # E highway bridge
    assert t1[6, 3, 3] == P["deck"]                    # highway outranks street in the centre
    assert t1[3, 0, 0] == P["water"] or t1[3, 0, 0] != P["asphalt"]
    t2 = out[2]
    assert t2[4, 0, 0] == P["leaves"] and t2[4, 7, 0] == P["water"]
    assert (out[3, 3] == P["plaza"]).all()


def test_roads_on_right_sides(tmp_path):
    d = fake_set(tmp_path / "city")

    class T4:
        T = 4
        corners = np.full((4, 4), ct.Z["R"])
        feats = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    out = pf.assemble_city(T4, d)
    cells = [(0, 3), (3, 7), (7, 4), (4, 0)]    # outer cell of the N, E, S, W strip
    for t, (i, j) in enumerate(cells):
        assert out[t, 3, i, j] == P["asphalt"]
        assert out[t, 3, 7 - i, 7 - j] != P["asphalt"]


def test_missing(tmp_path):
    (tmp_path / "e").mkdir()
    with pytest.raises(FileNotFoundError):
        pf.assemble_city(Tiny, str(tmp_path / "e"))
    out = pf.assemble_city(Tiny, str(tmp_path / "e"), allow_missing=True)
    assert out.shape == (4, 24, 8, 8)


def test_biomes(tmp_path):
    d = tmp_path / "b"
    d.mkdir()
    from castlegen.infinite import biomes
    for i, n in enumerate(biomes().names):
        write(d / f"band_{n}.txt", f"size: 8 8\nlegend: x=sand\nlayer {i % 4}: x\n")
    out = pf.assemble_biomes(str(d))
    assert out.shape == (8, 24, 8, 8)
    assert out[2, 2, 0, 0] == P["sand"] and out[2, 3, 0, 0] == P["grass"]


def test_hwt_roundtrip(tmp_path):
    a = np.random.RandomState(1).randint(0, 40, (5, 24, 8, 8)).astype(np.uint8)
    p = str(tmp_path / "sub" / "a.hwt")
    pf.write_hwt(p, a)
    assert open(p, "rb").read(4) == b"HWT1"
    assert (pf.read_hwt(p) == a).all()
