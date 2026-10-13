"""cbig: batched, cached generation equals the single-world pass; cached
log q equals forced scoring; a short training run works at large B."""
import numpy as np
import pytest

from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.variants.cbig import CBig
from castlegen.variants.cpass import CPass


@pytest.mark.parametrize("pname", ["toy_support", "dimers"])
def test_batched_generation_matches_single(pname, tmp_path):
    p = get(pname)
    v = CBig(p, seed=0, workdir=str(tmp_path), B=16)
    region = Region(p.W_eval)
    for s in (1_000_123, 1_000_124, 1_000_140):
        g = v.generate(s, region)
        ref = CPass.generate(v, s, region)
        for kind in ("raw", "final"):
            for c in p.sampled():
                assert np.array_equal(g[kind][c.name], ref[kind][c.name]), (s, kind, c.name)
        x = {**p.designed_values(s, region), **g["raw"]}
        lq = v.score(x, region)
        ref_lq = CPass.score(v, x, region)
        assert abs(lq - ref_lq) <= 1e-3 * max(1.0, abs(ref_lq)), (lq, ref_lq)


def test_short_training(tmp_path):
    p = get("toy_support")
    v = CBig(p, seed=0, workdir=str(tmp_path), B=64)
    logs = []
    v.train(150.0, logs.append)
    assert v.updates > 5
    assert np.isfinite([d["loss"] for d in logs if "loss" in d]).all()


@pytest.mark.parametrize("pname", ["ground", "dimers"])
def test_vectorised_designed(pname, tmp_path):
    p = get(pname)
    v = CBig(p, seed=0, workdir=str(tmp_path), B=8)
    for region in (Region(p.W_train), Region(p.W_eval, periodic=False, x0=64)):
        seeds = np.arange(77, 90)
        d = v._batch_np(seeds, region)
        for i, s in enumerate(seeds):
            ref = p.designed_values(int(s), region)
            for k in ref:
                assert np.array_equal(d[k][i], ref[k])
