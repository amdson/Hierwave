"""Stage 8 (notes/circles_biome_test.md): the real channel sets (roots,
ground, coord) through the ChannelSampler interface, bit-identical to their
existing drivers, and generate() on the combined roots + ground model."""
import numpy as np
import pytest

from castlegen.channels import Channel, Factor, Kinds, Model, coord, ground, roots
from castlegen.channels.sampler import ChannelSampler, Sampler, generate

INF = np.inf
KINDS = Kinds.concat(ground.KINDS, roots.KINDS)


def roots_model(H=32, W=48, seed=0, tree=False, allow=None):
    """chan_combined.py's set (ground + roots on one tile kernel, one
    certificate), surface flat at row 8.  allow: optional (H/8, W/8) grid of
    a painted parent; 0 admits only soil in that chunk."""
    tile = roots.tile_views(ground.tile_channel(KINDS), KINDS)
    surf, trees, cert = ground.surf_channel(), roots.trees_channel(), roots.cert_channel(roots.certificate().Dmax)
    chans, fac = [tile, surf, trees, cert], ground.factors() + roots.factors(KINDS, grow=1.5)
    if allow is not None:
        a = Channel("allow", 8, 2).add_view("v", np.arange(2))
        tab = np.zeros((len(KINDS), 2))
        tab[:, 0] = INF
        tab[KINDS.index("soil"), 0] = 0.0
        fac.append(Factor.pair(("tile", "self"), ("allow", "v"), (0, 0), tab, name="allow"))
        chans.append(a)
    m = Model(H, W, chans, fac, [roots.certificate(tree=tree)])
    if allow is not None:
        a.grid[:] = allow
    surf.grid[:] = 8
    surf.grid[0] = 0
    ground.init_tiles(tile, surf, KINDS)
    cert.grid[:] = cert.D - 1
    trees.grid[1, ::2] = 1
    return m, tile, cert, trees


@pytest.mark.parametrize("tree", [False, True])
def test_roots_identity(tree):
    m1, t1, c1, _ = roots_model(tree=tree)
    m2, t2, c2, _ = roots_model(tree=tree)
    m1.sweep("tile", 12, seed=5)
    S = Sampler(m2, "tile")
    assert S.init() == 0.0
    S.sweep(12, seed=5)
    assert np.array_equal(t1.grid, t2.grid) and np.array_equal(c1.grid, c2.grid)
    assert (t1.views["root"][t1.grid] == roots.ROOT).any()
    assert m1.energy("tile") == S.energy()


def test_roots_dormant_certificate():
    allow = np.ones((4, 6), np.int32)
    allow[2:, 1:3] = 0                                     # two chunk columns under the surface: soil only
    m, tile, cert, trees = roots_model(tree=True, allow=allow)
    S = Sampler(m, "tile", hb=8)
    frac = S.init()
    dz = np.kron(allow == 0, np.ones((8, 8), bool))
    assert frac == dz.mean() and np.array_equal(S.dormant, dz)
    assert np.array_equal(S.active, allow == 1)
    assert (cert.grid[dz] == cert.D - 1).all()
    S.sweep(40, seed=1)
    soil = KINDS.index("soil")
    assert (tile.grid[dz] == soil).all() and (cert.grid[dz] == cert.D - 1).all()
    met = roots.metrics(tile, cert, trees)
    assert met["rule_violations"] == 0 and met["root_cells"] > 5, met   # the certificate holds around the dead ends
    assert S.energy()[1] == 0


def test_ground_identity():
    def build():
        tile, surf = ground.tile_channel(ground.KINDS), ground.surf_channel()
        m = Model(32, 32, [tile, surf], ground.factors())
        ground.sample_surface(surf, 32, 32, seed=3)
        return m, tile
    m1, t1 = build()
    m2, t2 = build()
    m1.sweep("tile", 20, seed=0)
    S = Sampler(m2, "tile")
    assert S.init() == 0.0                                 # support and honour read home: no parent rows
    assert S.sweep(20, seed=0) == 0
    assert np.array_equal(t1.grid, t2.grid)
    assert ground.metrics(t2, m2.chan("surf"))["unsupported"] == 0


def coord_set(H=24, W=40, seed=0, K=3, Kt=2):
    tile = roots.tile_views(ground.tile_channel(KINDS), KINDS)
    alpha, _, _, _ = coord.parse_exemplar(KINDS, rows=coord.EXEMPLAR, tree=True)
    u = coord.coord_channel(alpha)
    surf, cert = ground.surf_channel(), roots.cert_channel(roots.certificate().Dmax)
    rs = [f for f in roots.factors(KINDS, counted=False) if f.name != "trunk_count"]
    m = Model(H, W, [tile, u, surf, cert], ground.factors() + rs, [roots.certificate(tree=True)])
    _, coup = coord.coupling(KINDS, tile, u, nu=6.0)
    r = np.random.default_rng(seed)
    surf.grid[:] = 8
    surf.grid[0] = 0
    ground.init_tiles(tile, surf, KINDS)
    cert.grid[:] = cert.D - 1
    u.grid[:] = r.integers(0, u.D, u.grid.shape)
    uref = np.where(r.random(u.grid.shape) < 0.3, r.integers(0, u.D, u.grid.shape), -1)
    return m, coup, coord.CoordKernel(u, tile, alpha, K=K, Kt=Kt, w=0.5), uref


def test_coord_identity():
    for joint in (False, True):
        m1, coup1, ck1, uref = coord_set()
        m2, coup2, ck2, _ = coord_set()
        cs = coord.CoordSampler(ck2, joint=(m2, coup2) if joint else None, mu=2.0)
        assert isinstance(cs, ChannelSampler) and cs.init() == 0.0
        if joint:
            ck1.sweep_joint(m1, coup1, 4, 9, uref=uref, mu=2.0)
            cs.uref = uref
        else:
            ck1.sweep(4, 9)
        cs.sweep(4, seed=9)
        assert np.array_equal(ck1.u.grid, ck2.u.grid) and np.array_equal(ck1.tile.grid, ck2.tile.grid)
        assert np.array_equal(m1.chan("cert").grid, m2.chan("cert").grid)
        with pytest.raises(NotImplementedError):
            cs.relax(1)


def test_coord_candidates_and_energy():
    m, coup, ck, uref = coord_set(K=0, Kt=0)
    cs = coord.CoordSampler(ck)
    U, mx = ck.u.grid, ck.alpha.shape[1]
    y, x = 5, 7
    cont = set()
    for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1)):
        uq = U[y + dy, x + dx]
        cy, cx = uq // mx - dy, uq % mx - dx
        if 0 <= cy < ck.alpha.shape[0] and 0 <= cx < mx:
            cont.add(int(cy * mx + cx))
    c = cs.candidates(y, x)
    assert c[0] == U[y, x] and set(c[1:].tolist()) == cont
    ck.K, ck.Kt = 3, 2
    assert len(cs.candidates(y, x)) <= 1 + 4 + 3 + 2
    for joint in (None, (m, coup)):                        # a one-site change moves energy() by the site energy change
        cs = coord.CoordSampler(ck, joint=joint, mu=2.0)
        cs.uref = uref
        r = np.random.default_rng(1)
        for _ in range(10):
            y, x = int(r.integers(U.shape[0])), int(r.integers(U.shape[1]))
            a, b = int(U[y, x]), int(r.integers(ck.u.D))
            e0 = cs.energy()[0]
            U[y, x] = b
            e1 = cs.energy()[0]
            U[y, x] = a
            my = ck.alpha.shape[0]
            H, W = U.shape
            ea = coord._energy_u(a, y, x, U, ck.alpha, my, mx, H, W, ck.lam)
            eb = coord._energy_u(b, y, x, U, ck.alpha, my, mx, H, W, ck.lam)
            if joint is None:
                ea = coord._energy(a, y, x, U, ck.tile.grid, ck.alpha, ck.rootv, my, mx, H, W, ck.lam, ck.w, ck.nu,
                                   ck.radius)
                eb = coord._energy(b, y, x, U, ck.tile.grid, ck.alpha, ck.rootv, my, mx, H, W, ck.lam, ck.w, ck.nu,
                                   ck.radius)
            else:
                t, al = ck.tile.grid[y, x], ck.alpha.ravel()
                ea += coup[t, al[a]] + 2.0 * (uref[y, x] >= 0 and a != uref[y, x])
                eb += coup[t, al[b]] + 2.0 * (uref[y, x] >= 0 and b != uref[y, x])
            assert abs((e1 - e0) - (eb - ea)) < 1e-9


def test_generate_combined():
    """migrate_combined.py's levels against chan_combined3's direct calls, small."""
    import importlib.util, pathlib
    p = pathlib.Path(__file__).parents[1] / "notes/experiments/migrate_combined.py"
    spec = importlib.util.spec_from_file_location("migrate_combined", p)
    mc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mc)
    H, W, seed = 48, 128, 2
    levels, painters, st = mc.build(H, W, seed, "small")
    generate(levels, mc.schedule(seed, 10, 20), painters=painters)
    levels2, painters2, st2 = mc.build(H, W, seed, "small")    # by hand, as chan_combined3 does
    painters2[0]()
    S8 = levels2[0][0]
    S8.model.sweep("u8", 10, seed=seed)
    painters2[1]()
    cs = levels2[1][0]
    done = 0
    for k in (5, 15, 20):
        cs.ck.sweep_joint(cs.joint[0], cs.joint[1], k - done, seed + k, uref=cs.uref, mu=cs.mu)
        done = k
    for name in ("u8", "d8", "tile", "u", "cert"):
        assert np.array_equal(st[name].grid, st2[name].grid), name
    assert (st["u8"].grid > 0).any() and roots.metrics(st["tile"], st["cert"], st["trees"])["rule_violations"] == 0
