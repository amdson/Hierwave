"""CondJax (castlegen.variants._cond_jax) against the numpy reference
_local.cond (soft, nv) and a brute-force lookahead written here (look), on
tiny() and the suite, periodic and open regions, random partial states."""
import itertools

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from castlegen.bench import problems as P  # noqa: E402
from castlegen.bench.problem import Region  # noqa: E402
from castlegen.variants._cond_jax import CondJax  # noqa: E402
from castlegen.variants._local import cond  # noqa: E402
from tests.bench.test_harness import tiny  # noqa: E402

NSITE = 192
_JIT = {}


def jitted(cj, chan):
    key = (id(cj), chan)
    if key not in _JIT:
        _JIT[key] = (cj, jax.jit(lambda s, Y, X: cj(s, chan, Y, X)))
    return _JIT[key][1]


def region_for(p, W, periodic):
    return Region(W) if periodic else Region(W, periodic=False, x0=W >> 1 if p.T else 4)


def partial_state(p, region, rng, q):
    x = p.designed_values(int(rng.integers(100)), region)
    for c in p.sampled():
        v = rng.integers(0, c.D, p.shape(c.name, region))
        x[c.name] = np.where(rng.random(v.shape) < q, -1, v)
    return x


def sites(p, region, chan, rng, n=NSITE, oob=True):
    """Random sites, about 10 % out of range when oob; returns Y, X, in-range mask."""
    h, w = p.shape(chan, region)
    Y = rng.integers(0, h, n)
    X = rng.integers(0, w, n)
    if oob:
        bad = rng.random(n) < 0.1
        Y = np.where(bad & (rng.random(n) < 0.5), rng.integers(-3, h + 3, n), Y)
        X = np.where(bad, rng.integers(-5, w + 5, n), X)
    m = (Y >= 0) & (Y < h) & (X >= 0) & (X < w)
    return Y, X, m


def to_jax(x):
    return {k: jnp.asarray(v, jnp.int32) for k, v in x.items()}


def check_soft(soft, nv, rs, rv):
    soft = np.asarray(soft, np.float64)
    scale = np.maximum(1.0, np.abs(rs).max(axis=1, keepdims=True))
    assert (np.abs(soft - rs) <= 1e-4 * scale).all(), np.abs(soft - rs).max()
    np.testing.assert_array_equal(np.asarray(nv), rv)


CASES = [("tiny", 8), ("tiny_hard", 8), ("toy_support", 32), ("toy_modes", 32), ("ground", 64), ("dimers", 32),
         ("columns", 64)]


def tiny_hard():
    """tiny() with rows / columns that no completion can satisfy, so the pair
    lookahead (both roles, by_pos parent, same level) and the count one fire."""
    p = tiny()
    fs = {f.name: f for f in p.factors}
    fs["vert"].table[2, :] = np.inf           # a = 2: no b works
    fs["vert"].table[:, 1] = np.inf           # b = 1: no a works
    fs["parent"].table[1, 1, :] = np.inf      # child position 1, t = 1: no parent works
    fs["parent"].table[3, :, 0] = np.inf      # parent lo = 0 against child position 3
    fs["cnt"].table[1, :2] = np.inf           # lo parent 1 needs both children on
    return p


def get(name):
    if name == "tiny":
        return tiny()
    if name == "tiny_hard":
        return tiny_hard()
    return P.get(name)


# ------------------------------------------------------------------ soft / nv
@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("name,W", CASES)
def test_soft_nv_match_local(name, W, periodic):
    p = get(name)
    region = region_for(p, W, periodic)
    cj = CondJax(p, W, periodic)
    rng = np.random.default_rng(hash((name, periodic)) % 2 ** 32)
    for q in (0.3, 0.7, 0.0, 1.0):
        x = partial_state(p, region, rng, q)
        xj = to_jax(x)
        for c in p.sampled():
            f = jitted(cj, c.name)
            for _ in range(2):
                Y, X, m = sites(p, region, c.name, rng)
                soft, nv, look = f(xj, jnp.asarray(Y, jnp.int32), jnp.asarray(X, jnp.int32))
                assert soft.shape == (NSITE, c.D) and nv.shape == look.shape == (NSITE, c.D)
                rs, rv = cond(p, x, region, c.name, Y[m], X[m])
                check_soft(np.asarray(soft)[m], np.asarray(nv)[m], rs, rv)


# ------------------------------------------------------------------ brute-force lookahead
def cell(p, x, region, cv, Y, X, pad):
    """('none',) | ('known', view value) | ('unknown', list of view values)."""
    c = p.chan[cv[0]]
    arr = x[c.name]
    h, w = arr.shape
    if region.periodic:
        X = X % w
    elif not 0 <= X < w:
        return ("none",)
    if not 0 <= Y < h:
        return ("known", pad) if pad >= 0 else ("none",)
    r = arr[Y, X]
    view = c.views[cv[1]]
    if r < 0:
        return ("unknown", [int(view[t]) for t in range(c.D)])
    return ("known", int(view[r]))


def within(f, la, Ya, Xa, lb, Yb, Xb):
    if f.within is None:
        return True
    L = f.within
    return ((Ya << la) >> L == (Yb << lb) >> L) and ((Xa << la) >> L == (Xb << lb) >> L)


def brute(p, x, region, chan, y, xx):
    """(soft, nv, look) at one site by explicit instance enumeration."""
    c = p.chan[chan]
    D = c.D
    soft, nv, look = np.zeros(D), np.zeros(D, int), np.zeros(D, int)
    inst = []      # (members, entry(t, values)), members: list of cell() results; 'child' for count rules
    for f in p.factors:
        la = p.chan[f.a[0]].level
        if f.kind == "unary":
            if f.a[0] == chan:
                inst.append(([], lambda t, v, f=f: f.table[c.views[f.a[1]][t]], None))
            continue
        lb = p.chan[f.b[0]].level
        dy, dx = f.off
        if f.kind == "pair":
            if f.a[0] == chan:
                Yb, Xb = (y + dy, xx + dx) if lb == la else ((y >> 1) + dy, (xx >> 1) + dx)
                cb = cell(p, x, region, f.b, Yb, Xb, f.pad_b)
                if cb[0] != "none" and within(f, la, y, xx, lb, Yb, Xb):
                    k = 2 * (y & 1) + (xx & 1)
                    if f.by_pos:
                        e = lambda t, v, f=f, k=k: f.table[k, c.views[f.a[1]][t], v[0]]
                    else:
                        e = lambda t, v, f=f: f.table[c.views[f.a[1]][t], v[0]]
                    inst.append(([cb], e, None))
            if f.b[0] == chan:
                if lb == la:
                    Ya, Xa = y - dy, xx - dx
                    ca = cell(p, x, region, f.a, Ya, Xa, f.pad_a)
                    if ca[0] != "none" and within(f, la, Ya, Xa, lb, y, xx):
                        inst.append(([ca], lambda t, v, f=f: f.table[v[0], c.views[f.b[1]][t]], None))
                else:
                    Yp, Xp = y - dy, xx - dx
                    if 0 <= Yp < (p.H >> lb):
                        for i, j in itertools.product((0, 1), (0, 1)):
                            Ya, Xa = 2 * Yp + i, 2 * Xp + j
                            ca = cell(p, x, region, f.a, Ya, Xa, -1)
                            if ca[0] != "none" and within(f, la, Ya, Xa, lb, y, xx):
                                k = 2 * i + j
                                if f.by_pos:
                                    e = lambda t, v, f=f, k=k: f.table[k, v[0], c.views[f.b[1]][t]]
                                else:
                                    e = lambda t, v, f=f: f.table[v[0], c.views[f.b[1]][t]]
                                inst.append(([ca], e, None))
        else:
            if f.a[0] == chan:
                k = 2 * (y & 1) + (xx & 1)
                if k in f.subset:
                    Yq, Xq = y >> 1, xx >> 1
                    cp = cell(p, x, region, f.b, Yq, Xq, -1)
                    sib = [cell(p, x, region, f.a, 2 * Yq + (kk >> 1), 2 * Xq + (kk & 1), -1)
                           for kk in f.subset if kk != k]
                    if cp[0] != "none" and all(s[0] != "none" for s in sib):
                        inst.append(([cp] + sib, ("child", f), "child"))
            if f.b[0] == chan:
                ch = [cell(p, x, region, f.a, 2 * y + (kk >> 1), 2 * xx + (kk & 1), -1) for kk in f.subset]
                if all(s[0] != "none" for s in ch):
                    inst.append((ch, ("parent", f), "parent"))
    for members, e, role in inst:
        unknown = [m for m in members if m[0] == "unknown"]
        if role is None:
            if not unknown:
                vals = [m[1] for m in members]
                for t in range(D):
                    E = e(t, vals)
                    if np.isinf(E):
                        nv[t] += 1
                    else:
                        soft[t] += E
            else:      # pair with the other cell unknown: enumerate its values
                for t in range(D):
                    look[t] += not any(np.isfinite(e(t, [v])) for v in unknown[0][1])
            continue
        f = e[1]
        if role == "child":
            cp, sib = members[0], members[1:]
            if cp[0] == "unknown":
                continue
            vb = cp[1]
            own = lambda t: c.views[f.a[1]][t]
        else:
            vb, sib = None, members
            own = lambda t: 0
        ks = sum(m[1] for m in sib if m[0] == "known")
        sums = {0}
        for m in sib:
            if m[0] == "unknown":
                sums = {s + v for s in sums for v in m[1]}
        closed = len(sums) == 1 and not any(m[0] == "unknown" for m in sib)
        for t in range(D):
            b = vb if role == "child" else c.views[f.b[1]][t]
            ent = [f.table[b, ks + own(t) + s] for s in sums if ks + own(t) + s < f.table.shape[1]]
            if closed:
                E = ent[0] if ent else f.table[b, -1]
                if np.isinf(E):
                    nv[t] += 1
                else:
                    soft[t] += E
            else:
                look[t] += not any(np.isfinite(ent))
    return soft, nv, look


LOOK_CASES = [("tiny", 8), ("tiny_hard", 8), ("toy_support", 16), ("columns", 32), ("ground", 32)]


@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("name,W", LOOK_CASES)
def test_look_matches_brute_force(name, W, periodic):
    p = get(name)
    region = region_for(p, W, periodic)
    cj = CondJax(p, W, periodic)
    rng = np.random.default_rng(7 + periodic)
    nchk = 0
    for q in (0.3, 0.7, 1.0):
        x = partial_state(p, region, rng, q)
        xj = to_jax(x)
        for c in p.sampled():
            Y, X, m = sites(p, region, c.name, rng, n=NSITE)
            soft, nv, look = jitted(cj, c.name)(xj, jnp.asarray(Y, jnp.int32), jnp.asarray(X, jnp.int32))
            soft, nv, look = np.asarray(soft), np.asarray(nv), np.asarray(look)
            for i in np.flatnonzero(m)[:48]:
                bs, bv, bl = brute(p, x, region, c.name, int(Y[i]), int(X[i]))
                np.testing.assert_array_equal(look[i], bl, err_msg=f"{c.name} {Y[i]} {X[i]}")
                np.testing.assert_array_equal(nv[i], bv)
                assert np.allclose(soft[i], bs, rtol=1e-4, atol=1e-4 * max(1, np.abs(bs).max()))
                nchk += int(bl.sum() > 0)
    if name in ("tiny_hard", "columns", "ground"):          # tiny and toy_support: no row is infeasible for every completion
        assert nchk > 0, "lookahead never fired: the test is vacuous"


# ------------------------------------------------------------------ jit + vmap
def test_jit_vmap_matches_per_world():
    p = P.get("ground")
    W = 32
    region = Region(W)
    cj = CondJax(p, W, True)
    rng = np.random.default_rng(3)
    xs = [partial_state(p, region, rng, q) for q in (0.3, 0.5, 0.7, 0.0)]
    batch = {k: jnp.stack([jnp.asarray(x[k], jnp.int32) for x in xs]) for k in xs[0]}
    for chan in ("t", "rho2", "cert3"):
        Y, X, _ = sites(p, region, chan, rng, n=64, oob=False)
        Yj, Xj = jnp.asarray(Y, jnp.int32), jnp.asarray(X, jnp.int32)
        f = jax.jit(jax.vmap(lambda s, Y, X: cj(s, chan, Y, X), in_axes=(0, None, None)))
        bs, bv, bl = f(batch, Yj, Xj)
        for w, x in enumerate(xs):
            s, v, l = cj(to_jax(x), chan, Yj, Xj)
            np.testing.assert_allclose(np.asarray(bs[w]), np.asarray(s), rtol=1e-6, atol=1e-5)
            np.testing.assert_array_equal(np.asarray(bv[w]), np.asarray(v))
            np.testing.assert_array_equal(np.asarray(bl[w]), np.asarray(l))
