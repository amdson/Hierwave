import numpy as np

from castlegen.channels import limbs, train
from castlegen.channels.aistargets import AISTargets
from castlegen.channels.treeland import CH, TREE_LEARNED, Bench, Forward, Treeland


def _valid(T, chans):
    c = T.unpack(chans)
    tile, tree, surf, ground, tdem = c["tile"], c["tree"], c["surf"], c["ground"], c["tdem"]
    rv = tile.views["root"][tile.grid]
    # soil iff under the surface; trunk iff demanded; limbs never on soil
    assert ((rv == limbs.SOIL) == (ground.grid == 1)).all()
    assert ((rv == limbs.TRUNK) == (tdem.grid == 1)).all()
    # every present slot's trunk sits on the surface
    g = tree.grid[T.brow]
    sr = T.surface_rows(surf)
    for j in np.flatnonzero(g > 0):
        t = g[j]
        y, x = T.R0 + T.rec_r[t], j * CH + T.rec_c[t]
        assert sr[j] == y + 1 and tdem.grid[y, x] == 1


def test_limb_kinds_and_views():
    k = limbs.KINDS
    assert k.names[:3] == ["sky", "soil", "trunk"] and len(k) == 3 + 45
    tile = limbs.tile_channel()
    v = tile.views
    assert v["root"][k.index("trunk")] == limbs.TRUNK and v["pN"][k.index("trunk")] == 1
    assert v["root"][k.index("limb2_NS")] == limbs.LIMB and v["mass"][k.index("limb2_NS")] == 2
    assert v["root"][k.index("soil")] == limbs.SOIL and v["root"][k.index("sky")] == limbs.SKY


def test_forward_valid_and_attached():
    T = Treeland(8, margin=1, limb_kw=dict(grow=3.0))
    fw = Forward(T, seed=0)
    st = fw.run(20, 10, 80)
    _valid(T, fw.chans)
    m = limbs.metrics(fw.c["tile"], fw.c["cert"])
    assert m["rule_violations"] == 0
    assert m["attached_frac"] == 1.0
    assert m["trunks"] == int((fw.c["tree"].grid > 0).sum())
    assert set(st) >= {"present", "limb_cells", "attached_frac", "present_rel0", "limb_uphill_minus_downhill"}


def test_components_counts_detached():
    tile = limbs.tile_channel()
    k = limbs.KINDS
    tile.grid = np.full((6, 6), k.index("sky"), np.int32)
    tile.grid[5] = k.index("soil")
    tile.grid[4, 1] = k.index("trunk")
    tile.grid[3, 1] = k.index("limb2_S")                     # attached to the trunk
    tile.grid[1, 4] = k.index("limb1_E")                     # a floating dimer
    tile.grid[1, 5] = k.index("limb1_W")
    total, attached = limbs.components(tile)
    assert (total, attached) == (3, 1)


def test_paint_fine_partial_matches_full():
    T = Treeland(6, margin=1)
    fw = Forward(T, seed=2)
    fw.run(10, 10, 20)
    chans = fw.chans
    c = T.unpack(chans)
    full = {n: c[n].grid.copy() for n in ("ground", "tdem", "tile")}
    # move one slot's tree, repaint its columns only, compare with a full repaint on a copy
    J = T.J0 + 2
    c["tree"].grid[T.brow, J] = 1 + (CH - 1 - c["surf"].grid[T.brow, J]) * CH + 3
    T.paint_fine(chans, reset=False, cols=(J * CH, (J + 1) * CH))
    part = {n: c[n].grid.copy() for n in ("ground", "tdem", "tile")}
    for n in full:
        c[n].grid[:] = full[n]
    T.paint_fine(chans, reset=False)
    for n in part:
        assert (c[n].grid == part[n]).all(), n


def test_tree_ais_targets_finite():
    T = Treeland(6, margin=1, win=(6, 4), limb_kw=dict(grow=3.0))
    B = Bench(T, seed=0, S=(10, 10, 20))
    B.fw.tables = B.tables_of("tree", None)
    B.fw.run(10, 10, 20)
    model = B.soft_model()
    J = T.J0 + 2
    h = B.c["surf"].grid[T.brow, J]
    cand = np.array([0] + [1 + (CH - 1 - h) * CH + cc for cc in (0, 4)])
    tg = AISTargets("tile", B.tree_region, B.tree_designed_e, B.tree_repaint, K=8, M=4, L=30.0, seed=1)
    pi = tg.at(model, "tree", T.brow, J, cand)
    assert np.isfinite(pi).all() and abs(pi.sum() - 1) < 1e-9
    _valid(T, B.chans)                                          # the state is restored
    X = B.tree_feats(model, "tree", T.brow, J, cand)
    assert X.shape == (3, sum(int(np.prod(s)) for s in T.shapes("tree").values()))
