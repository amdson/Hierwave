import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench.problems import get
from castlegen.variants import _cordinal_core as core


def _chan(problem, name):
    return next(c for c in problem.chans if c.name == name)


def test_embedding_sizes():
    p = get("ground")
    assert core.channel_embedding(_chan(p, "rho4")).shape == (257, 16)            # thermometer only
    assert core.channel_embedding(_chan(p, "cert4")).shape == (153, 2 * (16 + 17) + 16)
    assert core.channel_embedding(_chan(p, "rho1")).shape == (5, 5)               # one-hot
    e = core.channel_embedding(_chan(p, "rho4"))
    assert len({tuple(r) for r in e}) == 257                                       # injective


def test_specs_generic():
    p = get("ground")
    assert core.head_spec(_chan(p, "rho1")) is None
    sp = core.head_spec(_chan(p, "cert4"))
    assert len(sp.views) == 2 and sp.dout == 153 + 16                              # views A and B, not id
    assert len(core.head_spec(_chan(p, "rho4")).views) == 1                         # id view


def test_head_matches_numpy():
    p = get("ground")
    c = _chan(p, "cert3")
    sp = core.head_spec(c, K=8, rscale=0.1, rel=True)
    prm = core.mlp_init(jax.random.PRNGKey(0), 5, sp.dout)
    prm["W3"] = jax.random.normal(jax.random.PRNGKey(1), prm["W3"].shape)
    x = jax.random.normal(jax.random.PRNGKey(2), (3, 5))
    rng = np.random.default_rng(0)
    mask = rng.random((3, c.D)) < 0.3
    mask[:, 7] = True
    b = np.asarray(core.head(sp, prm, x, jnp.asarray(mask)))
    out = np.asarray(core.mlp(prm, x))
    ref = 0.1 * out[:, :c.D]
    for i, (view, n) in enumerate(sp.views):
        for s in range(3):
            lo, hi = view[mask[s]].min(), view[mask[s]].max()
            u = np.clip((view - lo) / max(hi - lo, 1), 0, 1)
            for k in range(8):
                ref[s] += out[s, c.D + 8 * i + k] * np.cos(np.pi * (k + 1) * u)
    assert np.allclose(b, ref, atol=1e-4)


def test_variant_trains_and_generates(tmp_path):
    from castlegen.bench.problem import Region
    from castlegen.variants.cordinal import COrdinal
    p = get("toy_support")
    v = COrdinal(p, 0, str(tmp_path), B=2)
    logs = []
    v.train(3.0, logs.append)
    assert v.updates >= 1
    x = v.generate(1_000_000, Region(p.W_train))
    assert set(x) == {"raw", "final"}
    assert np.isfinite(v.score(x["raw"], Region(p.W_train)))
