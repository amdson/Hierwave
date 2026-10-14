"""Graded distance to feasibility (castlegen/graded.py, ideas.md #21).

d_f = 0 <=> H_f = 0 on every suite problem (probes, random states, sampler draws), d_f integer and
non-negative, counts identical to energy.py, and hand-checked cases: one unsupported column -> 1,
three -> 3; a count one off -> 1, two off -> 2; a dimers rhombus conflict -> one per violated
instance (edit distance 1 each).  Run on CPU (JAX_PLATFORMS=cpu)."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from castlegen import graded as gr
from castlegen.bench import energy as en
from castlegen.bench.problem import INF, Chan, Problem, Region, count, pair, unary
from castlegen.bench.problems import get
from castlegen.bench.problems import ground as gmod
from castlegen.bench.problems import toys

SUITE = ["toy_support", "toy_modes", "ground", "dimers", "columns"]


def _full(p, x, seed, region):
    return {**p.designed_values(seed, region), **{k: np.asarray(v) for k, v in x.items()}}


def _check_equiv(p, x, region):
    E = en.energy(p, x, region)
    G = gr.graded(p, x, region)
    assert G.viol == E.viol and G.nhard == E.nhard
    for name, (s, v, n) in E.per_factor.items():
        if name in G.per_factor:
            gv, gm, gn = G.per_factor[name]
            assert (gv, gn) == (v, n)
            assert gm >= gv and (gm == 0) == (gv == 0), name
    for name, d in G.inst.items():
        assert d.dtype.kind == "i" and (d >= 0).all()
    return E, G


# ------------------------------------------------------------- d = 0 <=> H = 0
@pytest.mark.parametrize("name", SUITE)
def test_zero_iff_satisfied_probes_and_random(name):
    p = get(name)
    region = Region(p.W_eval)
    rng = np.random.default_rng(1)
    for seed in (5, 6):
        for y in p.probes(seed, region):
            E, G = _check_equiv(p, _full(p, y, seed, region), region)
            assert sum(E.viol.values()) == 0 and sum(G.mass.values()) == 0
    for _ in range(3):
        x = {c.name: rng.integers(0, c.D, p.shape(c.name, region)) for c in p.sampled()}
        E, G = _check_equiv(p, _full(p, x, 7, region), region)
        assert sum(E.viol.values()) > 0                      # random states violate
        for l in range(p.T + 1):
            assert G.mass[l] >= E.viol[l]


@pytest.mark.parametrize("name", ["toy_support", "toy_modes"])
def test_exact_toy_samples_have_zero_mass(name):
    p = get(name)
    region = Region(p.W_eval)
    energy_fn = toys.support_block_energy if name == "toy_support" else toys.modes_block_energy
    prob = toys._probs(energy_fn(toys.all_blocks()))
    rng = np.random.default_rng(3)
    for _ in range(4):
        x = toys.lift(toys.sample_t(rng, prob, toys.H, region.W))
        E, G = _check_equiv(p, x, region)
        assert sum(E.viol.values()) == 0 and sum(G.mass.values()) == 0


@pytest.mark.parametrize("name", ["toy_support", "dimers"])
def test_sampler_draws(name):
    """Untrained cfast draws (raw and repaired) on a small periodic region."""
    from castlegen.bench.variant import load_variants
    p = get(name)
    V = load_variants()
    v = V["cfast"](p, 0, "/tmp/graded_test")
    region = Region(32)
    for i in range(2):
        g = v.generate(1_000_000 + i, region)
        for kind in ("raw", "final"):
            E, G = _check_equiv(p, _full(p, g[kind], 1_000_000 + i, region), region)
            if kind == "final":
                for l in range(p.T + 1):
                    assert G.mass[l] >= E.viol[l]


def test_open_window_matches_periodic_count():
    """Instances on an open window (pads, cut edges) are enumerated like energy.py's."""
    for name in ("ground", "columns", "dimers"):
        p = get(name)
        region = Region(p.W_eval, periodic=False, x0=64)
        rng = np.random.default_rng(2)
        x = {c.name: rng.integers(0, c.D, p.shape(c.name, region)) for c in p.sampled()}
        _check_equiv(p, _full(p, x, 9, region), region)


# -------------------------------------------------------------- hand-checked
def test_ground_unsupported_columns():
    """Lower a probe's cert A below a block: support_l d = number of unsupported columns."""
    p = get("ground")
    region = Region(p.W_eval)
    y = _full(p, p.probes(11, region)[0], 11, region)
    E0, G0 = _check_equiv(p, y, region)
    assert sum(E0.viol.values()) == 0
    A4, B4 = gmod.cert_domain(16)
    idx = gmod.cert_index(16)
    c = np.asarray(y["cert4"]).copy()
    # find a top-level block p with B(p) >= 3 whose block below has A(below) = B(p) (fully supported)
    Y, X = np.nonzero((B4[c] >= 3)[:-1] & (A4[c][1:] == B4[c][:-1]))
    assert Y.size
    yy, xx = int(Y[0]), int(X[0])
    b_above = int(B4[c[yy, xx]])
    for k in (1, 3):
        c2 = c.copy()
        below = c[yy + 1, xx]
        a_new, b_new = int(A4[below]) - k, int(B4[below])
        c2[yy + 1, xx] = idx[a_new, b_new]                 # drop k full columns of the block below
        x = {**y, "cert4": c2}
        E, G = _check_equiv(p, x, region)
        v, m, _ = G.per_factor["support4"]
        assert (v, m) == (1, k), (k, v, m)
        assert b_above - a_new == k


def test_ground_row_honour_shortfall():
    """Raise a parent's A by k: honourA d per child row = max(0, A_new - sum of the row's children A)."""
    p = get("ground")
    region = Region(p.W_eval)
    y = _full(p, p.probes(12, region)[1], 12, region)
    A4, B4 = gmod.cert_domain(16)
    A3, _ = gmod.cert_domain(8)
    idx = gmod.cert_index(16)
    c = np.asarray(y["cert4"]).copy()
    c3 = np.asarray(y["cert3"])
    Y, X = np.nonzero(A4[c] + 2 <= B4[c])
    assert Y.size
    yy, xx = int(Y[0]), int(X[0])
    base = (int(A4[c[yy, xx]]), int(B4[c[yy, xx]]))
    rows = {"top": int(A3[c3[2 * yy, 2 * xx]] + A3[c3[2 * yy, 2 * xx + 1]]),
            "bot": int(A3[c3[2 * yy + 1, 2 * xx]] + A3[c3[2 * yy + 1, 2 * xx + 1]])}
    assert rows["top"] == base[0]                      # monotone columns: the top row sums exactly to A
    for k in (1, 2):
        c2 = c.copy()
        c2[yy, xx] = idx[base[0] + k, base[1]]
        G = gr.graded(p, {**y, "cert4": c2}, region)
        for r, s_r in rows.items():
            d_r = max(0, base[0] + k - s_r)
            assert G.per_factor[f"honourA4{r}"][:2] == (int(d_r > 0), d_r), (k, r)
        assert G.per_factor["honourA4top"][:2] == (1, k)


def test_toy_count_off_by_k():
    p = get("toy_support")
    region = Region(p.W_eval)
    x = toys.lift(np.zeros((toys.H, region.W), np.int64))
    E, G = _check_equiv(p, x, region)
    assert sum(G.mass.values()) == 0
    for k in (1, 2):
        r1 = x["rho1"].copy()
        r1[0, 0] = k
        G = gr.graded(p, {**x, "rho1": r1}, region)
        assert G.per_factor["rho1=sum t"][:2] == (1, k)
        assert G.per_factor["rho2=sum rho1"][:2] == (1, k)


def test_toy_modes_unary_distance():
    p = get("toy_modes")
    region = Region(p.W_eval)
    x = toys.lift(np.zeros((toys.H, region.W), np.int64))
    G = gr.graded(p, x, region)
    nb = (toys.H // 4) * (region.W // 4)
    assert G.per_factor["rho2=8"][:2] == (nb, 8 * nb)            # every block is 8 short of 8


def test_dimers_rhombus_conflict():
    """Two horizontal dimers on a 2 x 2 square; redirect one cell to S: the cell's old partner and
    its new claim are both broken, each a distance-1 instance (edit distance 1 per instance)."""
    p = get("dimers")
    region = Region(p.W_eval)
    y = _full(p, p.probes(4, region)[0], 4, region)       # squares tiling
    pp = np.asarray(y["p"]).copy()
    Y, X = np.nonzero((pp[:-1] == 0) & (pp[1:] == 0))     # E above E: a square of two horizontals
    yy, xx = int(Y[0]), int(X[0])
    pp[yy, xx] = 2                                       # S: claims (yy + 1, xx), which still says E
    E, G = _check_equiv(p, {**y, "p": pp}, region)
    assert sum(E.viol.values()) == 2 and sum(G.mass.values()) == 2
    assert G.hist() == {1: 2}


# ------------------------------------------------------------- band detection
def test_band_test():
    v = np.arange(5)
    assert gr.axis_is_band(v[:, None] <= v[None, :], 0) and gr.axis_is_band(v[:, None] <= v[None, :], 1)
    assert gr.axis_is_band(np.eye(5, dtype=bool), 0)
    F = np.ones((5, 5), bool)
    F[2, :] = False                                      # a hole in the middle: not a band
    assert not gr.axis_is_band(F, 0) and gr.axis_is_band(F, 1)
    assert not gr.axis_is_band(np.zeros((3, 3), bool), 0)


def test_fixed_designed_side_and_unordered_fallback():
    """A hard pair against a designed channel: only the sampled side moves; an unordered table gives 1."""
    def des(seed, Y, X, period):
        return np.asarray(X, np.int64) % 3
    tab = np.zeros((4, 3))
    tab[[0, 2], 1] = INF                                 # values 0 and 2 forbidden where d = 1 (not an interval)
    p = Problem("tiny", 0, 2, [Chan("d", 0, 3, designed=des), Chan("a", 0, 4)],
                [pair(("a", "id"), ("d", "id"), (0, 0), tab, name="f")], W_train=8, W_eval=8)
    region = Region(8)
    x = {"a": np.zeros((2, 8), np.int64), **p.designed_values(0, region)}
    E, G = _check_equiv(p, x, region)
    assert E.viol[0] == G.mass[0] == int((x["d"] == 1).sum())
    assert gr.distances(p)[0].ordered == (False, False)
