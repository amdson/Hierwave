"""colour_gibbs: colouring validity, energies, brute-force single-cell conditionals,
invariance on the enumerable toys (per level and from random starts, z-scores against
the exact penalised target), and AIS partition-function ratios.  GPU or CPU, a few minutes."""
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from castlegen.bench import energy as en
from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.bench.problems import toys
from castlegen.kernels import colour_gibbs as cg
from castlegen.variants import _cpass_core as core


def _lay(name, W=None):
    p = get(name)
    return p, core.Layout(p, W or (64 if name == "ground" else 16), True, 0)


def _random_state(p, lay, B, seed):
    rng = np.random.default_rng(seed)
    des = p.designed_values(seed, Region(lay.W))
    return {c.name: jnp.asarray(np.broadcast_to(des[c.name], (B,) + lay.hw[c.level])
                                if c.name in des else rng.integers(0, c.D, (B,) + lay.hw[c.level]), jnp.int32)
            for c in p.chans}


# ------------------------------------------------------------------ exact penalised toys
def toy_exact(name, lam, beta=1.0):
    """Block marginal of exp(-beta (soft + lam nv)) over (t, rho1 x4, rho2) of one 4x4 block."""
    blocks = toys.all_blocks()
    if name == "toy_modes":
        soft_t = toys.MODES_J * ((blocks[:, :, 1:] != blocks[:, :, :-1]).sum((1, 2))
                                 + (blocks[:, 1:, :] != blocks[:, :-1, :]).sum((1, 2))).astype(float)
        vt = np.zeros(len(blocks))
        r2pen = lambda r2: (r2 != toys.MODES_N).astype(float)
        stats = toys.modes_block_stats
    else:
        soft_t = toys.SUP_SMOOTH * (blocks[:, :, 1:] != blocks[:, :, :-1]).sum((1, 2)) \
            + toys.SUP_DIRT * blocks.sum((1, 2))
        vt = ((blocks[:, :-1, :] == 1) & (blocks[:, 1:, :] == 0)).sum((1, 2)).astype(float)
        r2pen = lambda r2: np.zeros(r2.shape)
        stats = toys.support_block_stats
    s4 = np.stack([blocks[:, 2 * a:2 * a + 2, 2 * b:2 * b + 2].sum((1, 2)) for a in (0, 1) for b in (0, 1)], 1)
    sidx = (s4 * 5 ** np.arange(4)).sum(1)
    S = np.array(list(itertools.product(range(5), repeat=4)))[:, ::-1]      # S[i] has code i (little endian)
    S = np.stack([(np.arange(625) // 5 ** j) % 5 for j in range(4)], 1)
    R1 = S                                                                    # same enumeration for rho1
    R2 = np.arange(17)
    # nvc[s, r1, r2]
    nvc = ((R1[None, :, :] != S[:, None, :]).sum(2)[:, :, None]
           + (R2[None, None, :] != R1.sum(1)[None, :, None])
           + r2pen(R2)[None, None, :])
    w = np.exp(-beta * lam * nvc)                                            # (625, 625, 17)
    Zs = w.sum((1, 2))
    wt = np.exp(-beta * (soft_t + lam * vt)) * Zs[sidx]
    Z = wt.sum()
    pt = wt / Z
    ex = {k: float(np.dot(pt, v)) for k, v in stats(blocks).items()}
    ex["rho2_mean"] = float(np.dot(pt, (w * R2[None, None, :]).sum((1, 2))[sidx] / Zs[sidx]))
    ex["nviol"] = float(np.dot(pt, vt + (w * nvc).sum((1, 2))[sidx] / Zs[sidx]))
    return dict(logZ=float(np.log(Z)), pt=pt, w=w, Zs=Zs, sidx=sidx, S=S, exact=ex, blocks=blocks,
                keys=list(ex), stats=stats, lam=lam, name=name)


def block_stats(X, ex):
    """Per-block stats of a batch of states: (B, nblocks, nstat), order ex['keys']."""
    t = np.asarray(X["t"]); r1 = np.asarray(X["rho1"]); r2 = np.asarray(X["rho2"])
    B = t.shape[0]
    out = []
    for b in range(B):
        tb = toys.blocks_of(t[b])
        st = ex["stats"](tb)
        r2b = r2[b].ravel()
        r1b = np.stack([r1[b][2 * (i // (t.shape[2] // 4)):2 * (i // (t.shape[2] // 4)) + 2,
                              2 * (i % (t.shape[2] // 4)):2 * (i % (t.shape[2] // 4)) + 2].ravel()
                        for i in range(len(tb))])
        s4 = np.stack([tb[:, 2 * a:2 * a + 2, 2 * c:2 * c + 2].sum((1, 2)) for a in (0, 1) for c in (0, 1)], 1)
        nvc = (r1b != s4).sum(1) + (r2b != r1b.sum(1))
        if ex["name"] == "toy_modes":
            nvc = nvc + (r2b != toys.MODES_N)
            vt = 0
        else:
            vt = ((tb[:, :-1, :] == 1) & (tb[:, 1:, :] == 0)).sum((1, 2))
        row = [st[k] for k in ex["keys"] if k in st] + [r2b.astype(float), nvc + vt]
        out.append(np.stack(row, 1))
    return np.stack(out)


def exact_samples(ex, rng, B, nblk=8, H=8, W=16):
    """Exact samples of the penalised toy target on a (8 x 16) region: iid blocks."""
    blocks = ex["blocks"]
    S, w, Zs, sidx = ex["S"], ex["w"], ex["Zs"], ex["sidx"]
    cdf = {}
    t = np.zeros((B, H, W), np.int64); r1 = np.zeros((B, H // 2, W // 2), np.int64); r2 = np.zeros((B, H // 4, W // 4), np.int64)
    idx = rng.choice(len(blocks), size=(B, nblk), p=ex["pt"])
    for b in range(B):
        for i in range(nblk):
            y, x = divmod(i, W // 4)
            tb = blocks[idx[b, i]]
            t[b, 4 * y:4 * y + 4, 4 * x:4 * x + 4] = tb
            s = sidx[idx[b, i]]
            if s not in cdf:
                cdf[s] = np.cumsum(w[s].ravel() / Zs[s])
            j = min(np.searchsorted(cdf[s], rng.random()), 625 * 17 - 1)
            a, c = divmod(j, 17)
            r2[b, y, x] = c
            r1[b, 2 * y:2 * y + 2, 2 * x:2 * x + 2] = S[a].reshape(2, 2)
    return {"t": jnp.asarray(t, jnp.int32), "rho1": jnp.asarray(r1, jnp.int32), "rho2": jnp.asarray(r2, jnp.int32)}


# ------------------------------------------------------------------ colouring / energies
@pytest.mark.parametrize("name", ["toy_modes", "toy_support", "ground"])
def test_colouring_valid(name):
    p, lay = _lay(name)
    assert cg.conflicts(lay) == []
    assert cg.get_plan(lay).ngroups == 1


@pytest.mark.parametrize("name", ["toy_modes", "toy_support", "ground"])
def test_state_energy_matches_reference(name):
    p, lay = _lay(name)
    reg = Region(lay.W)
    x = {**p.probes(5, reg)[1], **p.designed_values(5, reg)}
    E = en.energy(p, x, reg)
    st = {k: jnp.asarray(np.asarray(v)[None], jnp.int32) for k, v in x.items()}
    s, nv = cg.state_energy(lay, st)
    assert abs(float(s[0]) - E.soft) < 1e-2 and int(nv[0]) == sum(E.viol.values())


# ------------------------------------------------------------------ (b) brute-force conditionals
@pytest.mark.parametrize("name", ["toy_modes", "toy_support", "ground"])
def test_conditional_matches_brute_force(name):
    p, lay = _lay(name)
    st = _random_state(p, lay, 3, 11)
    rng = np.random.default_rng(0)
    nchecked = 0
    for l in lay.levels:
        for c in lay.chans[l]:
            k = int(rng.integers(0, 16))
            soft, nv, (Y, X, valid) = cg.conditional(lay, st, l, c.name, k)
            sites = np.flatnonzero(np.asarray(valid))
            for site in rng.choice(sites, min(2, sites.size), replace=False):
                y, x = int(Y[site]), int(X[site])
                b = int(rng.integers(0, 3))
                base = {n: np.asarray(a[b]).copy() for n, a in st.items()}
                batch = {n: jnp.broadcast_to(jnp.asarray(a), (c.D,) + a.shape) for n, a in base.items()}
                batch[c.name] = batch[c.name].at[:, y, x].set(jnp.arange(c.D, dtype=jnp.int32))
                s, nvv = cg.state_energy(lay, batch)
                Es = np.stack([np.asarray(s), np.asarray(nvv)], 1)
                # differences of the full energy equal differences of the conditional
                np.testing.assert_allclose(Es[:, 0] - Es[0, 0], np.asarray(soft[b, site]) - float(soft[b, site, 0]), atol=2e-3)
                np.testing.assert_array_equal(Es[:, 1] - Es[0, 1], np.asarray(nv[b, site]) - int(nv[b, site, 0]))
                nchecked += 1
    assert nchecked >= 2


@pytest.mark.parametrize("name,lam", [("toy_support", 3.0), ("ground", 3.0), ("ground", None)])
def test_sampled_conditional_matches_softmax(name, lam):
    """A single class sweep on many copies of one state draws each cell from the softmax."""
    p, lay = _lay(name)
    B = 20000
    st1 = _random_state(p, lay, 1, 5)
    st = {n: jnp.broadcast_to(a, (B,) + a.shape[1:]) for n, a in st1.items()}
    l = 1
    c = lay.chans[l][0]
    k = 5
    soft, nv, (Y, X, valid) = cg.conditional(lay, st1, l, c.name, k)
    new = cg.gibbs_sweep(lay, st, l, c.name, k, 1.0, lam, jax.random.PRNGKey(1))
    soft, nv = np.asarray(soft[0]), np.asarray(nv[0])
    lg = -(soft + (lam or 0) * nv) if lam is not None else np.where(nv > 0, -np.inf, -soft)
    pr = np.exp(lg - lg.max(1, keepdims=True))
    pr /= pr.sum(1, keepdims=True)
    zmax = 0
    for site in np.flatnonzero(np.asarray(valid))[:6]:
        y, x = int(Y[site]), int(X[site])
        cnt = np.bincount(np.asarray(new[c.name][:, y, x]), minlength=c.D)
        n = B
        z = (cnt - n * pr[site]) / np.sqrt(np.maximum(n * pr[site] * (1 - pr[site]), 1.0))
        zmax = max(zmax, np.abs(z[pr[site] > 1e-4]).max())
        assert cnt[pr[site] == 0].sum() == 0
    assert zmax < 4.5, zmax


# ------------------------------------------------------------------ (a) toy invariance
def _zscores(ex, X0, X1=None):
    """z of chain-vs-exact means: per-chain means over blocks, SE across chains."""
    s = block_stats(X0, ex).mean(1)                    # (B, nstat)
    est, se = s.mean(0), s.std(0, ddof=1) / np.sqrt(len(s))
    exact = np.array([ex["exact"][k] for k in ex["keys"]])
    return (est - exact) / np.maximum(se, 2e-3)


@pytest.mark.parametrize("name", ["toy_modes", "toy_support"])
@pytest.mark.parametrize("level", [0, 1, 2])
def test_level_invariance_from_exact(name, level):
    """Exact (penalised) samples stay exact under repeated sweeps of one level."""
    lam = 2.0
    p, lay = _lay(name)
    ex = toy_exact(name, lam)
    rng = np.random.default_rng(level)
    B = 256
    st = {**exact_samples(ex, rng, B)}
    z0 = _zscores(ex, st)
    key = jax.random.PRNGKey(level)
    for i in range(6):
        key, sub = jax.random.split(key)
        st = cg.sweep_level(lay, st, level, 1.0, lam, sub)
    z1 = _zscores(ex, st)
    assert np.abs(z0).max() < 4.5, ("start", z0)
    assert np.abs(z1).max() < 4.5, (name, level, dict(zip(ex["keys"], z1)))


@pytest.mark.parametrize("name", ["toy_modes", "toy_support"])
def test_mixing_from_random_start(name):
    """From uniform-random starts, all-level sweeps reach the exact penalised stats."""
    lam = 2.0
    p, lay = _lay(name)
    ex = toy_exact(name, lam)
    B = 256
    st = _random_state(p, lay, B, 3)
    key = jax.random.PRNGKey(0)
    traj = []
    for i in range(200):
        key, sub = jax.random.split(key)
        st = cg.sweep_all(lay, st, 1.0, lam, sub)
        if i >= 100 and i % 4 == 0:
            traj.append(block_stats(st, ex).mean(1))     # (B, nstat)
    s = np.mean(traj, 0)                                  # per-chain time means
    est, se = s.mean(0), s.std(0, ddof=1) / np.sqrt(B)
    exact = np.array([ex["exact"][k] for k in ex["keys"]])
    z = (est - exact) / np.maximum(se, 2e-3)
    assert np.abs(z).max() < 5, dict(zip(ex["keys"], z.round(2)))


# ------------------------------------------------------------------ (c) AIS
@pytest.mark.parametrize("name", ["toy_modes", "toy_support"])
def test_ais_partition_ratio(name):
    lam, tau_hi, n_steps, B = 2.0, 4.0, 64, 4000
    p, lay = _lay(name)
    ex0 = toy_exact(name, lam, 1.0 / tau_hi)
    ex1 = toy_exact(name, lam, 1.0)
    true = ex1["logZ"] - ex0["logZ"]
    rng = np.random.default_rng(0)
    st = exact_samples(ex0, rng, B)                       # block of cell (0, 0) is a pi_beta0 sample
    mask = cg.subtree_mask(lay, 2, np.zeros((B, 2), np.int32))
    st2, logw, info = cg.anneal_block(lay, st, mask, jax.random.PRNGKey(0), tau_hi, n_steps, lam)
    lw = np.asarray(logw, np.float64)
    est = np.log(np.mean(np.exp(lw - lw.max()))) + lw.max()
    # bootstrap standard error of the log-mean-exp
    bs = [np.log(np.mean(np.exp(lw[i] - lw.max()))) + lw.max() for i in rng.integers(0, B, (200, B))]
    se = np.std(bs)
    assert abs(est - true) < max(4 * se, 0.02), (est, true, se)
    # tracked energy equals recomputed energy
    s1, n1 = info["soft_end"], info["nv_end"]
    np.testing.assert_allclose(np.asarray(info["E_end"]), np.asarray(s1 + lam * n1), atol=5e-2)


# ------------------------------------------------------------------ part 2: lambda-annealed AIS
@pytest.mark.parametrize("name", ["toy_modes", "toy_support"])
@pytest.mark.parametrize("kind", ["ramp", "geo"])
def test_ais_lambda_ramp(name, kind):
    from castlegen.kernels import anneal as an
    lam_end, tau_hi, n_steps, B = 2.0, 4.0, 64, 4000
    p, lay = _lay(name)
    lam0 = float(an.lam_schedule(kind, n_steps, lam_end)[0])
    ex0 = toy_exact(name, lam0, 1.0 / tau_hi)
    ex1 = toy_exact(name, lam_end, 1.0)
    true = ex1["logZ"] - ex0["logZ"]
    rng = np.random.default_rng(0)
    st = exact_samples(ex0, rng, B)
    mask = cg.subtree_mask(lay, 2, np.zeros((B, 2), np.int32))
    st2, logw, valid, info = an.anneal_block_valid(lay, st, mask, jax.random.PRNGKey(0), tau_hi, n_steps,
                                                   kind, lam_end, final="lam")
    lw = np.asarray(logw, np.float64)
    m = lw.max()
    lme = lambda a: np.log(np.mean(np.exp(a - m))) + m
    se = np.std([lme(lw[i]) for i in rng.integers(0, B, (200, B))])
    assert abs(lme(lw) - true) < max(4 * se, 0.02), (lme(lw), true, se)
    np.testing.assert_array_equal(np.asarray(valid), np.asarray(info["N_end"]) == 0)


def test_hard_final_weights_inf_iff_violating():
    from castlegen.kernels import anneal as an
    p, lay = _lay("toy_modes")
    B = 64
    st = _random_state(p, lay, B, 1)
    mask = cg.subtree_mask(lay, 2, np.zeros((B, 2), np.int32))
    st2, logw, valid, info = an.anneal_block_valid(lay, st, mask, jax.random.PRNGKey(0), 4.0, 8, "ramp", 2.0,
                                                   final="hard", n_hard=1)
    s, nv = cg.state_energy(lay, st2, mask)
    np.testing.assert_array_equal(np.asarray(valid), np.asarray(nv) == 0)
    assert (np.isfinite(np.asarray(logw)) <= np.asarray(valid)).all()
