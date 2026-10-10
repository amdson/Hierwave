"""Conditional DC-SMC subtree move (castlegen/channels/certland_dc.py): the
intermediate targets against a brute term list, the LCA accounting of the
proposal offsets, the predictor's context reading only T(c) and the fixed
outside, numba bias vs JAX, validity of moves, and exactness by one-step
tests from exact draws of p* on an enumerable h = 2 root (480 states) and on
an enumerable h = 4 root (forced-empty certificates: 17 x 5^4 states)."""
import functools
import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from castlegen.channels import certland as cl, certland_fit as cf      # noqa: E402
from castlegen.channels import certland_chain as cc                    # noqa: E402
from castlegen.channels import certland_dc as dc                       # noqa: E402
from castlegen.channels import certland_pred as cp                     # noqa: E402
from castlegen.channels import certland_train as ct                    # noqa: E402

H, W = 32, 64


@functools.lru_cache(None)
def _world0():
    w = cl.World(H, W)
    cf.set_noise(w, 1)
    cl.generate(w, sweeps=3, seed=0)
    return w


def world():
    w0 = _world0()
    w = cl.World(H, W)
    w.fill[:] = w0.fill
    for i in range(cl.NL):
        w.rho[i][:] = w0.rho[i]
        if i < cl.NL - 1:
            w.cert[i][:] = w0.cert[i]
    return w


def random_up(levels=(0, 1, 2, 3), scale=0.3, seed=0, nh=16):
    rng = np.random.default_rng(seed)
    pu = dc.UpPreds(nh)
    for l in levels:
        for ch in (0, 1):
            p = cp.init_params(l, ch, nh, seed + 10 * l + ch, dc.NFU)
            pu.set(l, ch, {k: rng.normal(0, scale, np.shape(v)) for k, v in p.items()})
    return pu


# ------------------------------------------------------------------ brute term list
def _inb(l, y, x, bl, by, bx, bs):
    if l < bl:
        return False
    k = l - bl
    return (by << k) <= y < ((by + bs) << k) and (bx << k) <= x < ((bx + bs) << k)


def all_terms(w):
    """[(cells, soft, violations)] of every designed term of the world."""
    out = []
    for l, h in enumerate(cl.HS):
        r = w.rho[l]
        A, B = w.Aof[l][w.cert[l]], w.Bof[l][w.cert[l]]
        R, C = r.shape
        for y in range(R):
            for x in range(C):
                x2 = (x + 1) % C
                out.append((((l, y, x), (l, y, x2)), w.lam * abs(int(r[y, x]) - int(r[y, x2])) / h, 0))
                if y + 1 < R:
                    out.append((((l, y, x), (l, y + 1, x)), 0.0, int(B[y, x] > A[y + 1, x])))
                if l == 0:
                    out.append((((l, y, x),), w.beta * (r[y, x] - h * h * w.fill[y, x]) ** 2 / (h * h), 0))
        if l > 0:
            pA, pB = w.Aof[l - 1][w.cert[l - 1]], w.Bof[l - 1][w.cert[l - 1]]
            for py in range(R // 2):
                for px in range(C // 2):
                    ch = [(l, 2 * py + a, 2 * px + b) for a in (0, 1) for b in (0, 1)]
                    d = sum(int(r[c[1], c[2]]) for c in ch) - int(w.rho[l - 1][py, px])
                    out.append((tuple(ch) + ((l - 1, py, px),), w.kappa * d * d / (h * h), 0))
                    for a in (0, 1):
                        y = 2 * py + a
                        sa = A[y, 2 * px] + A[y, 2 * px + 1]
                        sb = B[y, 2 * px] + B[y, 2 * px + 1]
                        out.append((((l, y, 2 * px), (l, y, 2 * px + 1), (l - 1, py, px)), 0.0,
                                    int(sa < pA[py, px]) + int(sb > pB[py, px])))
    return out


def brute_gamma(terms, T, root):
    soft, nv = 0.0, 0
    for cells, s, v in terms:
        inT = [_inb(*c, *T) for c in cells]
        inS = [_inb(*c, *root) for c in cells]
        if any(inT) and all(t or not s_ for t, s_ in zip(inT, inS)):
            soft += s
            nv += v
    return soft, nv


def _randomise(w, i, ry, rx, rng):
    new = []
    for l, (r, c) in zip(range(i, cl.NL), cc.get_subtree(w, i, ry, rx)):
        h = cl.HS[l]
        if l == cl.NL - 1:
            t = rng.integers(0, 2, r.shape)
            new.append((t, t))
        else:
            new.append((rng.integers(0, h * h + 1, r.shape), rng.integers(0, len(w.Aof[l]), c.shape)))
    cc.set_subtree(w, i, ry, rx, new)


def _nodes(i, ry, rx, rng):
    """A few DC nodes (tl, ty, tx, ts) of the root: the root, a node per
    level below, a leaf."""
    out = [(i, ry, rx, 1)]
    for l in range(i + 1, cl.NL - 1):
        s = 1 << (l - i)
        out.append((l, ry * s + int(rng.integers(s)), rx * s + int(rng.integers(s)), 1))
    s3 = 1 << (3 - i)
    y3, x3 = ry * s3 + int(rng.integers(s3)), rx * s3 + int(rng.integers(s3))
    out.append((4, 2 * y3, 2 * x3, 2))
    return out


def test_gamma_matches_brute():
    w = world()
    rng = np.random.default_rng(0)
    for i in range(cl.NL - 1):
        rows, cols = w.rho[i].shape
        for ry, rx in [(0, 0), (rows - 1, cols - 1), (int(rng.integers(rows)), int(rng.integers(cols)))]:
            keep = cc.get_subtree(w, i, ry, rx)
            for trial in range(2):
                if trial:
                    _randomise(w, i, ry, rx, rng)
                terms = all_terms(w)
                for T in _nodes(i, ry, rx, rng):
                    a = dc.gamma(w, *T, i, ry, rx)
                    b = brute_gamma(terms, T, (i, ry, rx, 1))
                    assert np.isclose(a[0], b[0], atol=1e-8) and a[1] == b[1], (i, ry, rx, T, trial, a, b)
                # the root's gamma is p* given the outside (certland_chain.subtree_energy)
                s, nv = cc.subtree_energy(w, i, ry, rx)
                g = dc.gamma(w, i, ry, rx, 1, i, ry, rx)
                assert np.isclose(g[0], s) and g[1] == nv
            cc.set_subtree(w, i, ry, rx, keep)


def _children_gamma(w, l, y, x, i, ry, rx):
    if l == cl.NL - 2:
        return dc.gamma(w, 4, 2 * y, 2 * x, 2, i, ry, rx)
    s, nv = 0.0, 0
    for a in (0, 1):
        for b in (0, 1):
            u, v = dc.gamma(w, l + 1, 2 * y + a, 2 * x + b, 1, i, ry, rx)
            s += u
            nv += v
    return s, nv


def test_offsets_are_the_node_terms():
    """E_c - sum_k E_k - off(t) does not depend on c's value t: the proposal's
    offsets are exactly the terms with LCA c that involve c (rho: soft; cert:
    hard), the rest of dE_c only enters the weight."""
    w = world()
    rng = np.random.default_rng(1)
    rhos, certs, Aofs, Bofs = cc._tup(w)
    for i in range(cl.NL - 1):
        rows, cols = w.rho[i].shape
        for ry, rx in [(0, 0), (rows - 1, int(rng.integers(cols)))]:
            keep = cc.get_subtree(w, i, ry, rx)
            _randomise(w, i, ry, rx, rng)
            for (l, y, x, _) in _nodes(i, ry, rx, rng)[:-1]:
                Ek = _children_gamma(w, l, y, x, i, ry, rx)
                r0, c0 = w.rho[l][y, x], w.cert[l][y, x]
                off = dc.node_offsets(w, l, y, x, 0, i, ry, rx)
                diffs = []
                for t in range(len(off)):
                    w.rho[l][y, x] = t
                    diffs.append(dc.gamma(w, l, y, x, 1, i, ry, rx)[0] - Ek[0] - off[t])
                w.rho[l][y, x] = r0
                assert np.ptp(diffs) < 1e-8, (i, l, y, x, np.ptp(diffs))
                dv = []
                for t in range(len(w.Aof[l])):
                    w.cert[l][y, x] = t
                    dv.append(dc.gamma(w, l, y, x, 1, i, ry, rx)[1] - Ek[1]
                              - dc._cert_viol(l, y, x, t, i, ry, rx, rhos, certs, Aofs, Bofs))
                w.cert[l][y, x] = c0
                assert len(set(dv)) == 1, (i, l, y, x, dv)
            cc.set_subtree(w, i, ry, rx, keep)


def test_up_features_read_only_T_and_outside():
    w = world()
    rng = np.random.default_rng(2)
    for i in range(cl.NL - 1):
        rows, cols = w.rho[i].shape
        ry, rx = int(rng.integers(rows)), int(rng.integers(cols))
        keep = cc.get_subtree(w, i, ry, rx)
        for (l, y, x, _) in _nodes(i, ry, rx, rng)[:-1]:
            for ch in (0, 1):
                f0 = dc.features_up(w, l, y, x, ch, i, ry, rx)
                assert f0.shape == (dc.NFU,) and np.isfinite(f0).all()
                sub = cc.get_subtree(w, l, y, x)
                _randomise(w, i, ry, rx, rng)            # everything in S changes ...
                cc.set_subtree(w, l, y, x, sub)          # ... except T(c)
                if ch == 0:
                    w.cert[l][y, x] = int(rng.integers(len(w.Aof[l])))     # c's cert is not known when drawing rho
                assert np.array_equal(f0, dc.features_up(w, l, y, x, ch, i, ry, rx)), (i, l, y, x, ch)
                cc.set_subtree(w, i, ry, rx, keep)


def test_up_bias_matches_jax():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    w = world()
    pu = random_up()
    rng = np.random.default_rng(3)
    for i in range(cl.NL - 1):
        rows, cols = w.rho[i].shape
        ry, rx = int(rng.integers(rows)), int(rng.integers(cols))
        for (l, y, x, _) in _nodes(i, ry, rx, rng)[:-1]:
            for ch in (0, 1):
                f = dc.features_up(w, l, y, x, ch, i, ry, rx)
                p = {k: jnp.asarray(v) for k, v in pu.params(l, ch).items()}
                jb = np.asarray(cp.jax_bias(p, jnp.asarray(f[None])))[0]
                nb = dc.node_bias(w, pu, l, y, x, ch, i, ry, rx)
                assert np.abs(nb).max() > 1e-6 and np.allclose(nb, jb, atol=1e-9)


@pytest.mark.parametrize("withpred", [False, True])
def test_moves_valid(withpred):
    w = world()
    pu = random_up() if withpred else None
    rng = np.random.default_rng(4)
    for i in (3, 2, 1, 0):
        out = cc.move_level(w, pu, i, 3, rng, move=dc.dc_move)
        assert out["n"] == w.rho[i].size and 1 <= out["ess"] <= 4 + 1e-9 and 0 <= out["changed"] <= 1
        assert sum(ct.count_violations(w)) == 0, i
        assert cl.overhangs(w.rho[cl.NL - 1]) == 0
        out = dc.move_level_dc(w, pu, i, 3, rng)
        assert sum(ct.count_violations(w)) == 0, i
    # collecting: examples for every node and channel, values admissible
    ex = ct.Examples()
    col = dc.UpCollector(ex, 0)
    dc.dc_move(w, pu, 1, 0, 1, 2, rng, collect=col)
    for l in range(1, 4):
        for ch in (0, 1):
            assert ex.count(l, ch) == 4 ** (l - 1)


# ------------------------------------------------------------------ exactness
def tv_tol(p, n, z=4.0):
    """z times the expected TV of an iid histogram of n draws from p."""
    return z * 0.5 * np.sqrt(2 * p * (1 - p) / (np.pi * n)).sum()


def one_step(w, i, ry, rx, p, setter, getter, flat_index, N, n, pu, seed):
    """n independent one-step moves from exact draws s0 ~ p: the outputs are
    iid ~ p iff the kernel leaves p invariant.  Also the states of the fresh
    root particle 1 (its law is q-like: the test's power; -1 when dead)."""
    rng = np.random.default_rng(seed)
    keep = cc.get_subtree(w, i, ry, rx)
    s0 = rng.choice(len(p), n, p=p)
    out = np.empty(n, np.int64)
    fresh = np.full(n, -1, np.int64)
    for t in range(n):
        setter(w, int(s0[t]))
        r = dc.dc_move(w, pu, i, ry, rx, N, rng)
        out[t] = getter(w)
        if np.isfinite(r["logw"][1]):
            fresh[t] = flat_index(r["states"][1])
    cc.set_subtree(w, i, ry, rx, keep)
    return out, fresh


def _tv(hist, p):
    return 0.5 * np.abs(hist - p).sum()


def chi_p(idx, p):
    """Chi-square goodness-of-fit p-value of iid draws idx against p (bins
    with expected count < 5 pooled)."""
    from scipy.stats import chisquare
    o = np.bincount(idx, minlength=len(p)).astype(float)
    e = p * len(idx)
    keep = e >= 5
    o2, e2 = list(o[keep]), list(e[keep])
    if (~keep).any():
        o2[-1] += o[~keep].sum()
        e2[-1] += e[~keep].sum()
    return float(chisquare(o2, e2).pvalue)


# h = 2 root: rho (5) x cert (6) x tiles (16) = 480 states
def _set_h2(w, ry, rx, s):
    w.rho[3][ry, rx] = s // 96
    w.cert[3][ry, rx] = (s // 16) % 6
    for k in range(4):
        w.rho[4][2 * ry + k // 2, 2 * rx + k % 2] = (s % 16 >> k) & 1


def _get_h2(w, ry, rx):
    t = w.rho[4][2 * ry:2 * ry + 2, 2 * rx:2 * rx + 2].ravel()
    return (int(w.rho[3][ry, rx]) * 6 + int(w.cert[3][ry, rx])) * 16 + int(t @ (1 << np.arange(4)))


def _flat_h2(f):
    """Flat root state (rho, cert, 4 tiles row-major) -> index."""
    return (int(f[0]) * 6 + int(f[1])) * 16 + int(f[2] + 2 * f[3] + 4 * f[4] + 8 * f[5])


def exact_h2(w, ry, rx):
    keep = _get_h2(w, ry, rx)
    neg = np.full(480, -np.inf)
    for s in range(480):
        _set_h2(w, ry, rx, s)
        e, nv = cc.subtree_energy(w, 3, ry, rx)
        if nv == 0:
            neg[s] = -e
    _set_h2(w, ry, rx, keep)
    p = np.exp(neg - neg.max())
    return p / p.sum()


def pick_h2_root(w):
    ys, xs = np.nonzero((w.rho[3] > 0) & (w.rho[3] < 4))
    best = None
    for j in np.random.default_rng(0).permutation(len(ys))[:12]:
        p = exact_h2(w, int(ys[j]), int(xs[j]))
        ent = -(p[p > 0] * np.log(p[p > 0])).sum()
        if best is None or ent > best[0]:
            best = (ent, int(ys[j]), int(xs[j]), p)
    return best


@pytest.mark.parametrize("withpred", [False, True])
def test_dc_exact_h2(withpred):
    w = world()
    ent, ry, rx, p = pick_h2_root(w)
    assert ent > 0.5
    pu = random_up(levels=(3,), scale=0.3, seed=7) if withpred else None
    n = 30000
    out, fresh = one_step(w, 3, ry, rx, p, lambda w_, s: _set_h2(w_, ry, rx, s), lambda w_: _get_h2(w_, ry, rx),
                          _flat_h2, 2, n, pu, 11)
    tv = _tv(np.bincount(out, minlength=480) / n, p)
    fr = fresh[fresh >= 0]
    tv_q = _tv(np.bincount(fr, minlength=480) / len(fr), p)
    tol = tv_tol(p, n)
    pv, pv_q = chi_p(out, p), chi_p(fr, p)
    print(f"h2 withpred={withpred}: root ({ry},{rx}) H(p*)={ent:.2f} TV(one-step, p*)={tv:.4f} tol={tol:.4f} "
          f"chi2 p={pv:.3g}; fresh particle: TV {tv_q:.3f} chi2 p={pv_q:.3g} (alive {len(fr) / n:.2f})")
    assert tv_q > 3 * tol and pv_q < 1e-10, "the proposal is too close to p* for the test to have power"
    assert tv < tol and pv > 1e-4, (tv, tol, pv)


# h = 4 root whose certificates are all forced to (0, 0) (the fixed cell below has A = 0, so B_root = 0,
# so every descendant certificate and tile is 0): the state is the rhos, 17 x 5^4 states, enumerable.
# Its moves still exercise every merge (leaves propose solid tiles that die at the h = 2 or h = 4 merge).
def _set_h4(w, ry, rx, s):
    w.rho[2][ry, rx] = s // 625
    w.cert[2][ry, rx] = 0
    c = s % 625
    for k in range(4):
        y, x = 2 * ry + k // 2, 2 * rx + k % 2
        w.rho[3][y, x] = (c // 5 ** (3 - k)) % 5
        w.cert[3][y, x] = 0
    w.rho[4][4 * ry:4 * ry + 4, 4 * rx:4 * rx + 4] = 0


def _get_h4(w, ry, rx):
    r3 = w.rho[3][2 * ry:2 * ry + 2, 2 * rx:2 * rx + 2].ravel()
    assert w.cert[2][ry, rx] == 0 and not w.cert[3][2 * ry:2 * ry + 2, 2 * rx:2 * rx + 2].any()
    assert not w.rho[4][4 * ry:4 * ry + 4, 4 * rx:4 * rx + 4].any()
    return int(w.rho[2][ry, rx]) * 625 + int(r3 @ 5 ** np.arange(3, -1, -1))


def _flat_h4(f):
    """Flat root state (rho, cert, 4 rho, 4 cert, 16 tiles) -> index; -2
    if a certificate or tile is not zero (outside the support)."""
    if f[1] != 0 or f[6:].any():
        return -2
    return int(f[0]) * 625 + int(f[2:6] @ 5 ** np.arange(3, -1, -1))


def exact_h4(w, ry, rx):
    keep = cc.get_subtree(w, 2, ry, rx)
    neg = np.empty(17 * 625)
    for s in range(len(neg)):
        _set_h4(w, ry, rx, s)
        e, nv = cc.subtree_energy(w, 2, ry, rx)
        assert nv == 0
        neg[s] = -e
    cc.set_subtree(w, 2, ry, rx, keep)
    p = np.exp(neg - neg.max())
    return p / p.sum()


def _joint(idx):
    """(root rho, sum of children rho) as one index over 17 x 17."""
    r0 = idx // 625
    c = idx % 625
    sm = sum((c // 5 ** k) % 5 for k in range(4))
    return r0 * 17 + sm


def pick_h4_root(w):
    r2 = w.rho[2]
    for ry in range(r2.shape[0] - 1):
        for rx in range(r2.shape[1]):
            if w.Aof[2][w.cert[2][ry + 1, rx]] == 0 and 2 < r2[ry, rx] < 14:
                return ry, rx
    raise AssertionError("no forced-empty h = 4 root")


@pytest.mark.parametrize("withpred", [False, True])
def test_dc_exact_h4(withpred):
    w = world()
    ry, rx = pick_h4_root(w)
    p = exact_h4(w, ry, rx)
    pj = np.bincount(_joint(np.arange(len(p))), weights=p, minlength=289)
    pr = np.bincount(np.arange(len(p)) // 625, weights=p, minlength=17)
    pu = random_up(levels=(2, 3), scale=0.3, seed=9) if withpred else None
    n = 20000
    out, fresh = one_step(w, 2, ry, rx, p, lambda w_, s: _set_h4(w_, ry, rx, s), lambda w_: _get_h4(w_, ry, rx),
                          _flat_h4, 8, n, pu, 13)
    tv_j = _tv(np.bincount(_joint(out), minlength=289) / n, pj)
    tv_r = _tv(np.bincount(out // 625, minlength=17) / n, pr)
    fr = fresh[fresh >= 0]
    tv_q = _tv(np.bincount(_joint(fr), minlength=289) / max(len(fr), 1), pj)
    tol_j, tol_r = tv_tol(pj, n), tv_tol(pr, n)
    pv_j, pv_r, pv_q = chi_p(_joint(out), pj), chi_p(out // 625, pr), chi_p(_joint(fr), pj)
    print(f"h4 withpred={withpred}: root ({ry},{rx}) TV joint {tv_j:.4f} (tol {tol_j:.4f}, chi2 p={pv_j:.3g}) "
          f"TV root rho {tv_r:.4f} (tol {tol_r:.4f}, chi2 p={pv_r:.3g}); fresh particle: TV joint {tv_q:.3f} "
          f"chi2 p={pv_q:.3g} (alive {len(fr) / n:.2f})")
    assert tv_q > 3 * tol_j and pv_q < 1e-10
    assert tv_j < tol_j and tv_r < tol_r and pv_j > 1e-4 and pv_r > 1e-4
