import numpy as np

from castlegen.channels import coord
from castlegen.channels.treecopy import KINDS, TreeCopy, parse_tree, procedural_tree


def test_exemplar_parse_satisfies_certificate():
    A = procedural_tree()
    alpha, mass, mask, trunks, n_ground = parse_tree(KINDS, A)
    assert coord.check_exemplar(KINDS, alpha, np.where(mass > 0, mass, 0)) == 0
    (ty, tx), = trunks
    assert alpha[ty, tx] == 2 + KINDS.index("trunk") and n_ground == ty
    assert (alpha[:n_ground] == coord.FREE).all()                      # the ground is the ground's
    depth = coord.exemplar_depth(KINDS, alpha)
    assert ((depth >= 0) == (mass > 0)).all()                           # every tree cell on the port tree
    # mass non-increasing along the tree: a child's mass <= its parent's
    v = KINDS
    tv = __import__("castlegen.channels.treecopy", fromlist=["tile_channel"]).tile_channel().views
    for y, x in zip(*np.nonzero(depth > 0)):
        t = alpha[y, x] - 2
        par = [(y + dy, x + dx) for i, (dy, dx) in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1)))
               if tv["p" + "NESW"[i]][t] and 0 <= y + dy < alpha.shape[0] and 0 <= x + dx < alpha.shape[1]
               and depth[y + dy, x + dx] == depth[y, x] - 1]
        assert par and all(tv["mass"][alpha[p] - 2] >= tv["mass"][t] for p in par)


def test_guarantees_hold_after_copy():
    tc = TreeCopy(128, [40, 104], seed=0)
    nwin, pruned, bad = tc.seed_coarse(40)
    assert nwin >= 2 and tc.metrics()["rule_violations"] == 0
    tc.sweep(10, mu=2.0)
    m = tc.metrics()
    assert m["rule_violations"] == 0 and m["multi_parent"] == 0
    assert m["attached_frac"] == 1.0
    assert m["trunks"] == 2 and m["in_ground"] == 0
    assert m["tree_cells"] > 200 and m["leaves"] > 20
    g = tc.tile.grid
    assert (g[tc.n_ground, [40, 104]] == KINDS.index("trunk")).all()
    assert (g[:tc.n_ground - 1] == KINDS.index("soil")).all()


def test_sequential_growth_valid():
    tc = TreeCopy(96, [40], seed=1)
    tc.sweep(30)
    m = tc.metrics()
    assert m["rule_violations"] == 0 and m["attached_frac"] == 1.0 and m["trunks"] == 1 and m["in_ground"] == 0


def test_sheet_parse_and_copy():
    from castlegen.channels.treecopy import sheet
    A = sheet(seeds=(0, 1), mirror=True)
    alpha, mass, mask, trunks, n_ground = parse_tree(KINDS, A)
    assert len(trunks) == 4 and all(t[0] == n_ground for t in trunks)
    assert coord.check_exemplar(KINDS, alpha, np.where(mass > 0, mass, 0)) == 0
    tc = TreeCopy(128, [40, 104], A=A, seed=3)
    assert len(tc.trunk_win) == 4 * 8                                   # every trunk at every column of a chunk
    tc.seed_coarse(40)
    tc.sweep(5, mu=2.0)
    m = tc.metrics()
    assert m["rule_violations"] == 0 and m["attached_frac"] == 1.0 and m["trunks"] == 2 and m["in_ground"] == 0
