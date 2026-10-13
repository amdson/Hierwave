"""toy_support, toy_modes, columns: declarations, probes and exact answers,
cross-checked against the frozen reference evaluator (castlegen.bench.energy)."""
import numpy as np
import pytest

from castlegen.bench.energy import energy
from castlegen.bench.problem import Region
from castlegen.bench.problems import columns, get, toys

NAMES = ["toy_support", "toy_modes", "columns"]


def _valid(p, x, region):
    E = energy(p, x, region)
    for l in range(p.T + 1):
        assert E.viol[l] == 0, (p.name, l, E.per_factor)
    return E


@pytest.mark.parametrize("name", NAMES)
def test_probes_valid(name):
    p = get(name)
    assert set(p.exact) == set(p.tol)
    for region in (Region(p.W_eval), Region(p.W_eval, periodic=False, x0=3 << p.T)):
        xs = p.probes(7, region)
        assert len(xs) >= 4
        for x in xs:
            x = {**x, **p.designed_values(7, region)}
            p.check_sample(x, region)
            E = _valid(p, x, region)
            assert all(E.nhard[l] > 0 for l in range(p.T))  # level T homes no factor here
            st = p.stats(x)
            assert set(p.exact) <= set(st)
            img = p.render(x)
            assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3


@pytest.mark.parametrize("name", NAMES)
def test_probes_deterministic(name):
    p = get(name)
    a, b = p.probes(3, Region(p.W_eval)), p.probes(3, Region(p.W_eval))
    for xa, xb in zip(a, b):
        for c in xa:
            assert np.array_equal(xa[c], xb[c])


# ------------------------------------------------------------------ toys
@pytest.mark.parametrize("name,efn", [("toy_support", toys.support_block_energy),
                                      ("toy_modes", toys.modes_block_energy)])
def test_toy_block_energy_matches_reference(name, efn):
    """The enumeration's block energies equal energy.energy on tiled worlds."""
    p = get(name)
    region = Region(p.W_eval)
    rng = np.random.default_rng(0)
    allb = toys.all_blocks()
    E = efn(allb)
    valid = np.flatnonzero(np.isfinite(E))
    invalid = np.flatnonzero(~np.isfinite(E))
    nb = (p.H // 4) * (p.W_eval // 4)
    for trial in range(20):
        idx = rng.choice(valid, nb)
        if trial % 2:
            idx[rng.integers(nb)] = rng.choice(invalid)
        t = toys.tile_blocks(allb[idx], p.H, p.W_eval)
        assert np.array_equal(toys.blocks_of(t), allb[idx])
        R = energy(p, toys.lift(t), region)
        nviol = sum(R.viol.values())
        if trial % 2:
            assert nviol > 0
        else:
            assert nviol == 0
            assert np.isclose(R.soft, E[idx].sum())


@pytest.mark.parametrize("name,sfn", [("toy_support", toys.support_block_stats),
                                      ("toy_modes", toys.modes_block_stats)])
def test_toy_exact_vs_mc(name, sfn):
    p = get(name)
    allb = toys.all_blocks()
    efn = toys.support_block_energy if name == "toy_support" else toys.modes_block_energy
    pr = toys._probs(efn(allb))
    rng = np.random.default_rng(1)
    n = 3000
    t = toys.sample_t(rng, pr, p.H, p.W_eval * n)       # n worlds side by side
    acc = {k: [] for k in p.exact}
    for i in range(n):
        st = p.stats({"t": t[:, i * p.W_eval:(i + 1) * p.W_eval]})
        for k in acc:
            acc[k].append(st[k])
    for k, v in acc.items():
        v = np.asarray(v)
        se = v.std() / np.sqrt(n)
        assert abs(v.mean() - p.exact[k]) < max(4 * se, 1e-9), (k, v.mean(), p.exact[k], se)
        assert 4 * se < p.tol[k], (k, se)


def test_toy_modes_symmetry():
    ex = get("toy_modes").exact
    for m in "RTB":
        assert np.isclose(ex["mode_" + m], ex["mode_L"])
    assert np.isclose(sum(ex["mode_" + m] for m in "LRTB") + ex["mode_other"], 1.0)


# --------------------------------------------------------------- columns
def _chain_ok(c, A):
    """A fine column is valid: allowed transitions with air pads above and below."""
    s = np.concatenate([[0], c, [0]])
    return bool(A[s[:-1], s[1:]].all())


def test_columns_dp_vs_bruteforce():
    """forward_backward on a short strip against enumerating every valid chain."""
    p = get("columns")
    parts = columns.chain_parts(p.factors[0].table, p.factors[1].table)
    M = parts[0] > 0
    h = 30
    chains = []

    def rec(prefix):
        if len(prefix) == h:
            if M[prefix[-1], 0]:
                chains.append(prefix)
            return
        for b in np.flatnonzero(M[prefix[-1] if prefix else 0]):
            rec(prefix + [int(b)])

    rec([])
    C = np.array(chains)
    lw = columns.MU * (C > 0).sum(1)
    w = np.exp(lw - lw.max())
    w /= w.sum()
    marg, _, logz = columns.forward_backward(*parts, h=h)
    assert np.isclose(logz, np.log(np.exp(lw - lw.max()).sum()) + lw.max())
    dens = (w[:, None] * (C > 0)).sum(0)
    assert np.allclose(dens, marg[:, 1:].sum(1))
    # sampler agrees with the enumeration
    rng = np.random.default_rng(2)
    S = columns.sample_chains(rng, 20000, parts, h=h)
    assert abs((S > 0).mean() - dens.mean()) < 0.01
    assert all(_chain_ok(S[:, i], M) for i in range(50))


def test_columns_validity_matches_reference():
    """Single-cell mutations of a valid world: energy.energy flags a violation
    exactly when the 1-D chain is invalid; soft energy is -mu per column cell."""
    p = get("columns")
    region = Region(p.W_eval)
    M = columns.allowed_table()
    x = p.probes(5, region)[0]
    E = _valid(p, x, region)
    assert np.isclose(E.soft, -columns.MU * (x["k"] > 0).sum())
    rng = np.random.default_rng(3)
    for _ in range(40):
        k = x["k"].copy()
        y, xx = rng.integers(p.H), rng.integers(p.W_eval)
        k[y, xx] = rng.integers(columns.D)
        R = energy(p, columns.lift(k), region)
        assert (R.viol[0] == 0) == _chain_ok(k[:, xx], M)


def test_columns_exact_vs_mc():
    p = get("columns")
    parts = columns.chain_parts(p.factors[0].table, p.factors[1].table)
    rng = np.random.default_rng(4)
    n, W = 1000, p.W_eval
    ks = columns.sample_chains(rng, W * n, parts)
    acc = {k: [] for k in p.exact}
    for i in range(n):
        st = p.stats({"k": ks[:, i * W:(i + 1) * W]})
        for k in acc:
            acc[k].append(st[k])
    for k, v in acc.items():
        v = np.asarray(v)
        se = v.std() / np.sqrt(n)
        assert abs(v.mean() - p.exact[k]) < 4 * se + 0.005 * (k == "gap_len") + 1e-12, (k, v.mean(), p.exact[k])
        assert 4 * se < p.tol[k], (k, se)
