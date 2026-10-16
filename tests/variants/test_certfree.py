"""certfree: the derived certificates are valid by construction (ideas #22 target B)."""
import numpy as np
import jax.numpy as jnp

from castlegen.bench import energy as en
from castlegen.bench.problem import Region
from castlegen.bench.problems import ground as G
from castlegen.variants._certfree_problem import derive_certs, make_ground_cf

P = G.build_ground()


def random_terrain(rng, W):
    sr = np.clip(np.cumsum(rng.integers(-3, 4, W)) + rng.integers(5, 55), 0, G.H)
    sr = np.where(rng.random(W) < 0.1, rng.integers(0, G.H + 1, W), sr)
    return (np.arange(G.H)[:, None] >= sr[None, :]).astype(np.int64)


def test_cf_problem():
    cf = make_ground_cf(P)
    assert not any(c.name.startswith("cert") for c in cf.chans)
    assert [f.name for f in cf.factors if f.hard] == ["support0"]


def test_derived_equals_lift_and_valid():
    rng = np.random.default_rng(0)
    reg = Region(128)
    for _ in range(25):
        t = random_terrain(rng, 128)
        x = G.lift(t)
        d = derive_certs(P, t)
        for l in range(1, 5):
            assert (d[f"cert{l}"] == x[f"cert{l}"]).all()
        x.update(d)
        x["fill"] = G.fill_designed(1, *np.meshgrid(np.arange(4), np.arange(8), indexing="ij"), 8)
        E = en.energy(P, x, reg)
        assert sum(E.viol.values()) == 0, E.viol


def test_derived_valid_on_arbitrary_tiles():
    """Even on tiles that break support0: A <= B (every cert index exists) and honour holds."""
    rng = np.random.default_rng(1)
    reg = Region(128)
    for _ in range(10):
        t = (rng.random((G.H, 128)) < rng.random()).astype(np.int64)
        d = derive_certs(P, t)
        x = G.lift(t); x.update(d)
        x["fill"] = np.zeros((4, 8), np.int64)
        E = en.energy(P, x, reg)
        for k, (_, v, _) in E.per_factor.items():
            if k.startswith("honour"):
                assert v == 0, k


def test_jnp_matches_np():
    rng = np.random.default_rng(2)
    t = np.stack([random_terrain(rng, 128) for _ in range(3)])
    a = derive_certs(P, t)
    b = derive_certs(P, jnp.asarray(t), xp=jnp)
    for k in a:
        assert (a[k] == np.asarray(b[k])).all()


def test_variant_zero_stage_output():
    """ccertfree with no stage: raw = final = pass tiles / rho + derived certificates; honour factors and A <= B hold,
    every other violation is the untrained pass's own (support of rho-free tiles), measured by the bench energy on ground."""
    from castlegen.bench.variant import load_variants
    v = load_variants()["ccertfree"](P, 0, "/tmp/test_certfree", stage_R=0, stage_F=0)
    reg = Region(128)
    g = v.generate(7000, reg)
    x = dict(g["final"]); x["fill"] = P.designed_values(7000, reg)["fill"]
    assert set(g["final"]) == {c.name for c in P.sampled()}
    d = derive_certs(P, x["t"])
    for l in range(1, 5):
        assert (d[f"cert{l}"] == x[f"cert{l}"]).all()
    E = en.energy(P, x, reg)
    assert all(v_ == 0 for k, (_, v_, _) in E.per_factor.items() if k.startswith("honour"))
