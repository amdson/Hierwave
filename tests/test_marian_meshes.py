"""Marian block meshes -> voxel templates (castlegen.marian_meshes)."""

import os

import numpy as np
import pytest

from castlegen import marian_meshes as mm

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(mm.ASSETS, "blocks.fbx")),
    reason="cache/marian not downloaded")


@pytest.fixture(scope="module")
def meshes():
    return mm.read_fbx_meshes()


@pytest.fixture(scope="module")
def mods():
    return mm.load_modules()


@pytest.fixture(scope="module")
def tem():
    return mm.module_templates(8)


def test_fbx_meshes(meshes):
    assert len(meshes) == 151
    for name in ("Floor", "Wall", "Roof 2", "Stairs"):
        V, F = meshes[name]
        U = mm.fbx_to_unity(V)
        # blocks are 2 x 2 x 2 Unity units centred on the origin
        assert U.min() >= -1.1 and U.max() <= 1.1
        assert 0.95 < np.abs(U[:, [0, 2]]).max() <= 1.1
        assert F.max() < len(V) and F.min() >= 0


def test_modules(mods):
    protos, modules = mods
    assert len(modules) == 454 and len(protos) == 121
    assert modules[0]["name"] == "Empty R0" and modules[1]["name"] == "Solid R0"
    assert all(m["neighbors"].shape == (6, 454) for m in modules[:3])


def test_template_shape(tem):
    assert tem.shape == (455, 8, 8, 8) and tem.dtype == np.uint8
    assert tem.max() < len(mm.palette())
    assert (tem[454] == mm.palette()["concrete"]).all()


def test_empty_and_solid(tem):
    assert (tem[0] == 0).all()
    assert (tem[1] > 0).all()


def test_rotation_identity():
    v = np.random.default_rng(0).standard_normal((20, 8, 8))[:8]
    four = mm.rotate_xyz(v, 1)
    for _ in range(3):
        four = mm.rotate_xyz(four, 1)
    assert (four == v).all()
    assert (mm.rotate_xyz(v, 2) == mm.rotate_xyz(mm.rotate_xyz(v, 1), 1)).all()


def test_rotation_sense():
    # a marker at Unity (-z) must go to -x for Rotation 1 (Euler(0, 90, 0))
    k = 4
    v = np.zeros((k, k, k), np.uint8)
    v[3, 0, 0] = 1          # south-east corner (x east, z south)
    r = mm.rotate_xyz(v, 1)
    # south-east corner -> south-west for a clockwise turn seen from above
    assert r[0, 0, 0] == 1 and r.sum() == 1
    p = mm.rotate_unity_y(np.array([[0.0, 0.0, -1.0]]), 1)
    assert np.allclose(p, [[-1.0, 0.0, 0.0]])
    # the same corner point through the mesh path: SE (1, -1) -> SW (-1, -1)
    q = mm.rotate_unity_y(np.array([[1.0, 0.0, -1.0]]), 1)
    assert np.allclose(q, [[-1.0, 0.0, -1.0]])


def test_templates_rotate_consistently(tem, mods):
    protos, modules = mods
    by = {}
    for i, m in enumerate(modules):
        by.setdefault(m["proto"], []).append(i)
    # Rotation r equals rotation 0 turned r times, in template axes
    for p, idx in list(by.items())[:20]:
        base = tem[idx[0]]
        for i in idx[1:]:
            r = modules[i]["rotation"]
            want = mm.to_template(mm.rotate_xyz(mm.from_template(base), r))
            assert (tem[i] == want).all()
    # and a stairs-like prototype is not rotation invariant
    st = by[[i for i, p in enumerate(protos) if p["name"] == "Stairs"][0]]
    assert not (tem[st[0]] == tem[st[1]]).all()


def test_neighbours_agree_better_than_chance(tem, mods):
    protos, modules = mods
    nb = np.stack([m["neighbors"] for m in modules])

    def face(t, d):
        k = t.shape[0]
        return {0: t[:, :, 0], 3: t[:, :, k - 1], 2: t[:, k - 1, :],
                5: t[:, 0, :]}[d] > 0

    def agree(a, b):
        u = (a | b).sum()
        return 1.0 if u == 0 else (a & b).sum() / u

    rng = np.random.default_rng(1)
    good, rnd = [], []
    for a in range(0, 454, 4):
        for d in (0, 2, 3, 5):
            bs = np.nonzero(nb[a, d])[0]
            if len(bs):
                b = rng.choice(bs)
                good.append(agree(face(tem[a], d), face(tem[b], (d + 3) % 6)))
                rnd.append(agree(face(tem[a], d), face(tem[rng.integers(454)], (d + 3) % 6)))
    assert np.mean(good) > np.mean(rnd) + 0.3


def test_hwt_roundtrip(tmp_path, tem):
    from castlegen.prefabs import read_hwt, write_hwt
    write_hwt(str(tmp_path / "m.hwt"), tem)
    assert (read_hwt(str(tmp_path / "m.hwt")) == tem).all()
