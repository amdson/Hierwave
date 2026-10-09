import time

import numpy as np
import pytest

from castlegen.legacy import core, hier, texsyn, tileset


def _exemplar(m, seed=0):
    ts = tileset.load("demo")
    return ts, np.random.default_rng(seed).integers(0, ts.n_sig, (m, m))


@pytest.fixture(scope="module")
def an32():
    ts, E = _exemplar(32)
    return texsyn.Analysis(ts, E, n_pca=8)


def _jitters(an, n, r, seed):
    """Per texsyn level l: an int array (n/h, n/h, 2), drawn once."""
    rng = np.random.default_rng(seed)
    r = np.broadcast_to(np.asarray(r, float), (an.L + 1,))
    out = []
    for l in range(an.L + 1):
        h = 2 ** (an.L - l)
        out.append(np.floor(h * r[l] * rng.uniform(-1, 1, (n // h, n // h, 2)) + 0.5).astype(np.int64))
    return out


SETTINGS = [dict(r=[1, 1, 0, 0, 0, 0], corrections=2, kappa=1.5),
            dict(r=0.5, corrections=1, kappa=4.0, first_corrected=1, final_corrections=3, beta=0.5),
            dict(r=[2, 1, 1, 0.5, 0.25, 0], corrections=3, kappa=2.0, first_corrected=0, last_corrected=4)]


@pytest.mark.parametrize("kw", SETTINGS, ids=["default", "beta", "all-levels"])
def test_equivalence_with_legacy(an32, kw):
    n = 64
    J = _jitters(an32, n, kw["r"], seed=3)
    legacy_levels = []
    S0 = texsyn._synthesize_legacy(an32, n, seed=0, jitter=J, levels_out=legacy_levels, **kw)
    ctx = hier.Ctx(seed=0)
    S1 = texsyn.synthesize(an32, n, seed=0, jitter=J, ctx=ctx, **kw)
    assert np.array_equal(S0, S1)
    assert [lv.h for lv in ctx.levels] == [2 ** (an32.L - l) for l in range(an32.L + 1)]
    for a, lv in zip(legacy_levels, ctx.levels):
        assert np.array_equal(a, lv.vars["coord"])


def test_noise_matches_core():
    y, x = np.meshgrid(np.arange(17), np.arange(9), indexing="ij")
    for args in [(0, 0, 0, 0), (123456789, 5, 7, 3), (2 ** 32 - 1, 31, hier.INIT_STEP, 1)]:
        a = hier.noise(*args, y[..., None], x[..., None], np.arange(4))
        b = np.asarray(core.noise(*args, y[..., None], x[..., None], np.arange(4)))
        assert a.dtype == np.float32 and np.array_equal(a, b)
        assert (a > 0).all() and (a < 1).all()


def test_determinism(an32):
    r = [1, 1, 0.5, 0, 0, 0]
    a = texsyn.synthesize(an32, 64, r, seed=5)
    b = texsyn.synthesize(an32, 64, r, seed=5)
    c = texsyn.synthesize(an32, 64, r, seed=6)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_sampling_at_positive_T(an32):
    r = [1, 1, 0, 0, 0, 0]
    S = texsyn.synthesize(an32, 64, r, seed=2, T=0.5)
    assert S.shape == (64, 64, 2) and S.dtype == np.int64
    assert S.min() >= 0 and S.max() < an32.m
    assert np.array_equal(S, texsyn.synthesize(an32, 64, r, seed=2, T=0.5))
    assert not np.array_equal(S, texsyn.synthesize(an32, 64, r, seed=2, T=0.0))


def test_timing_against_legacy():
    ts, E = _exemplar(64, seed=1)
    an = texsyn.Analysis(ts, E)
    r, n = [1, 1, 0, 0, 0, 0, 0], 128
    J = _jitters(an, n, r, seed=0)
    best = {}
    for name, fn in [("legacy", texsyn._synthesize_legacy), ("hier", texsyn.synthesize)]:
        ts_ = []
        for _ in range(3):
            t0 = time.perf_counter()
            fn(an, n, r, kappa=4.0, jitter=J)
            ts_.append(time.perf_counter() - t0)
        best[name] = min(ts_)
    print(f"\nsynthesize 64 -> 128: legacy {best['legacy']:.3f}s  hier {best['hier']:.3f}s  "
          f"ratio {best['hier'] / best['legacy']:.2f}")
    assert best["hier"] < 1.5 * best["legacy"]
