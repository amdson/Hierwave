"""Chain training for certland (notes/certland_chain_spec.md): the predictor's
numba bias against its JAX twin, the subtree energy against a brute
full-world designed energy written here, forced passes, validity of moves, and
p* invariance of the subtree moves on an enumerable h = 2 root."""
import functools

import numpy as np
import pytest

from castlegen.channels import certland as cl, certland_fit as cf

cp = pytest.importorskip("castlegen.channels.certland_pred")
cc = pytest.importorskip("castlegen.channels.certland_chain")

H, W = 32, 64


# ------------------------------------------------------------------ brute designed energy
def brute_energy(world):
    """(soft, violations) of the whole world from certland's definitions:
    count prior, smoothness (each horizontal pair once, x periodic), top
    noise; support (B above > A below, nothing below the map) and row honour
    (sum A < A_P, sum B > B_P per child row) counted per violated inequality."""
    soft, nv = 0.0, 0
    for i, h in enumerate(cl.HS):
        r = world.rho[i]
        A, B = world.Aof[i][world.cert[i]], world.Bof[i][world.cert[i]]
        soft += world.lam * np.abs(r - np.roll(r, -1, 1)).sum() / h
        if i == 0:
            soft += world.beta * ((r - h * h * world.fill) ** 2).sum() / (h * h)
        else:
            R, C = r.shape
            d = r.reshape(R // 2, 2, C // 2, 2).sum((1, 3)) - world.rho[i - 1]
            soft += world.kappa * (d ** 2).sum() / (h * h)
            pc = world.cert[i - 1]
            pA = np.repeat(world.Aof[i - 1][pc], 2, 0)
            pB = np.repeat(world.Bof[i - 1][pc], 2, 0)
            nv += int((A.reshape(R, C // 2, 2).sum(2) < pA).sum())
            nv += int((B.reshape(R, C // 2, 2).sum(2) > pB).sum())
        nv += int((B[:-1] > A[1:]).sum())
    return float(soft), nv


def _make_world(seed=0, noise=1):
    w = cl.World(H, W)
    cf.set_noise(w, noise)
    cl.generate(w, sweeps=3, seed=seed)
    return w


@functools.lru_cache(None)
def _world0():
    return _make_world()


def world():
    """A fresh copy of a valid generated world (designed only)."""
    w0 = _world0()
    w = cl.World(H, W)
    w.fill[:] = w0.fill
    for i in range(cl.NL):
        w.rho[i][:] = w0.rho[i]
        if i < cl.NL - 1:
            w.cert[i][:] = w0.cert[i]
    return w


def _dom(w, i, ch):
    return cl.HS[i] ** 2 + 1 if ch == cl.RHO else len(w.Aof[i])


def random_preds(levels=(0, 1, 2, 3), scale=0.5, seed=0, nh=16, proposal=False):
    rng = np.random.default_rng(seed)
    preds = cp.QPreds(nh) if proposal else cp.Preds(nh)
    for i in levels:
        for ch in (0, 1):
            p = cp.init_params(i, ch, nh, seed + 10 * i + ch, preds.nf)
            p = {k: (rng.normal(0, scale, np.shape(v)) if isinstance(v, np.ndarray) else v) for k, v in p.items()}
            preds.set(i, ch, p)
    return preds


def numba_bias(w, preds, i, ch, y, x):
    la = cp.level_args(w, i)
    th, nh, on = preds.args(i, ch)
    out = np.zeros(_dom(w, i, ch))
    cp.cell_bias(la[0], ch, y, x, *la[1:], th, nh, on, out)
    return out


def test_world_valid():
    s, nv = brute_energy(world())
    assert nv == 0 and np.isfinite(s)
    assert cl.overhangs(world().rho[cl.NL - 1]) == 0


# ------------------------------------------------------------------ predictor
def test_cell_bias_matches_jax():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    w = world()
    preds = random_preds()
    rng = np.random.default_rng(1)
    for i in range(cl.NL - 1):
        rows, cols = w.rho[i].shape
        for ch in (0, 1):
            p = preds.params(i, ch)
            cells = [(0, 0), (rows - 1, cols - 1)] + [(int(rng.integers(rows)), int(rng.integers(cols))) for _ in range(6)]
            feats = np.stack([cp.features_np(w, i, ch, y, x) for y, x in cells])
            assert feats.shape == (len(cells), cp.NF) and np.all(np.isfinite(feats))
            jb = np.asarray(cp.jax_bias({k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v) for k, v in p.items()},
                                        jnp.asarray(feats)))
            nb = np.stack([numba_bias(w, preds, i, ch, y, x) for y, x in cells])
            assert jb.shape == nb.shape == (len(cells), _dom(w, i, ch))
            assert np.abs(nb).max() > 1e-6, (i, ch)                  # the random predictor is not trivially zero
            assert np.allclose(nb, jb, atol=1e-9), (i, ch, np.abs(nb - jb).max())
            # pack / unpack round trip
            q = cp.unpack(cp.pack(p), i, ch, preds.nh)
            for k, v in p.items():
                if isinstance(v, np.ndarray):
                    assert np.allclose(np.asarray(q[k]), v)


def test_zero_predictor_zero_bias():
    w = world()
    none = cp.Preds(16)
    zero = cp.Preds(16)
    for i in range(cl.NL - 1):
        for ch in (0, 1):
            p = cp.init_params(i, ch, 16, 0)
            zero.set(i, ch, {k: (np.zeros_like(v) if isinstance(v, np.ndarray) else v) for k, v in p.items()})
            for y, x in [(0, 0), (w.rho[i].shape[0] - 1, 1)]:
                assert np.all(numba_bias(w, none, i, ch, y, x) == 0)
                assert np.all(numba_bias(w, zero, i, ch, y, x) == 0)


def test_generate_pred_none_valid():
    w = world()
    bad = cp.generate_pred(w, cp.Preds(16), sweeps=2, seed=3)
    assert sum(bad) == 0
    assert cl.overhangs(w.rho[cl.NL - 1]) == 0
    assert brute_energy(w)[1] == 0


# ------------------------------------------------------------------ chain: energies and passes
def _roots(w, i, n, rng):
    rows, cols = w.rho[i].shape
    out = {(0, 0), (rows - 1, cols - 1)}
    while len(out) < min(n, rows * cols):
        out.add((int(rng.integers(rows)), int(rng.integers(cols))))
    return sorted(out)


def _same(a, b):
    return all(np.array_equal(r1, r2) and np.array_equal(c1, c2) for (r1, c1), (r2, c2) in zip(a, b))


def test_subtree_slices_shape():
    for i in range(cl.NL - 1):
        sl = cc.subtree_slices(i, 1, 1)
        assert [s[0] for s in sl] == list(range(i, cl.NL))
        for l, y0, y1, x0, x1 in sl:
            n = 2 ** (l - i)
            assert (y0, y1, x0, x1) == (n, 2 * n, n, 2 * n)


def test_subtree_energy_matches_brute():
    w = world()
    rng = np.random.default_rng(2)
    for i in range(cl.NL - 1):
        for ry, rx in _roots(w, i, 3, rng):
            s0, v0 = cc.subtree_energy(w, i, ry, rx)
            b0, n0 = brute_energy(w)
            keep = cc.get_subtree(w, i, ry, rx)
            for trial in range(3):
                st = cc.get_subtree(w, i, ry, rx)
                new = []
                for l, (r, c) in zip(range(i, cl.NL), st):
                    h = cl.HS[l]
                    if l == cl.NL - 1:
                        t = rng.integers(0, 2, r.shape) if trial else r.copy()
                        if not trial:                        # a small valid-ish change: flip one tile
                            t.flat[int(rng.integers(t.size))] ^= 1
                        new.append((t, t))
                    else:
                        new.append((rng.integers(0, h * h + 1, r.shape), rng.integers(0, len(w.Aof[l]), c.shape)))
                cc.set_subtree(w, i, ry, rx, new)
                s1, v1 = cc.subtree_energy(w, i, ry, rx)
                b1, n1 = brute_energy(w)
                assert np.isclose(s1 - s0, b1 - b0, atol=1e-8), (i, ry, rx, trial, s1 - s0, b1 - b0)
                assert v1 - v0 == n1 - n0, (i, ry, rx, trial, v1 - v0, n1 - n0)
                cc.set_subtree(w, i, ry, rx, keep)
            assert brute_energy(w) == (b0, n0)


@pytest.mark.parametrize("withpred", [False, True])
def test_forced_pass_finite(withpred):
    w = world()
    preds = random_preds() if withpred else cp.Preds(16)
    rng = np.random.default_rng(3)
    for i in range(cl.NL - 1):
        for ry, rx in _roots(w, i, 2 if i < 2 else 4, rng):
            st = cc.get_subtree(w, i, ry, rx)
            logq, dead = cc.pass_subtree(w, preds, i, ry, rx, rng, forced=True)
            assert not dead and np.isfinite(logq) and logq <= 1e-9, (i, ry, rx, logq, dead)
            assert _same(cc.get_subtree(w, i, ry, rx), st)      # a forced pass leaves the state as found


def _logsm(e, v):
    ok = np.isfinite(e)
    lg = np.where(ok, -np.where(ok, e, 0.0), -np.inf)
    mx = lg.max()
    return lg[v] - mx - np.log(np.exp(lg - mx).sum())


@pytest.mark.parametrize("proposal", [False, True])
def test_record_pass_matches_forced_pass(proposal):
    """The proposal's recorded contexts (record_pass: features with the
    pass's known flags, bias-free energies) reproduce the pass: the forced
    log q with a predictor minus the one without equals the sum over the
    recorded cells of log softmax(-(offs + b(feats)))[v] - log softmax(-offs)[v]."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    w = world()
    preds = random_preds(proposal=proposal)
    rng = np.random.default_rng(7)
    for i in range(cl.NL - 1):
        for ry, rx in _roots(w, i, 2, rng):
            st = cc.get_subtree(w, i, ry, rx)
            lq1, d1 = cc.pass_subtree(w, preds, i, ry, rx, rng, forced=True)
            lq0, d0 = cc.pass_subtree(w, cp.Preds(16), i, ry, rx, rng, forced=True)
            assert not d0 and not d1
            rec = cc.record_pass(w, i, ry, rx, st, proposal=proposal)
            assert _same(cc.get_subtree(w, i, ry, rx), st)
            tot = 0.0
            for (l, ch), (f, o, v) in rec.items():
                assert f.shape == (len(v), preds.nf) and o.shape == (len(v), _dom(w, l, ch))
                p = preds.params(l, ch)
                b = np.asarray(cp.jax_bias({k: jnp.asarray(x) for k, x in p.items()}, jnp.asarray(f)))
                for k in range(len(v)):
                    assert np.isfinite(o[k, v[k]])
                    tot += _logsm(o[k] + b[k], v[k]) - _logsm(o[k], v[k])
            assert np.isclose(lq1 - lq0, tot, atol=1e-8), (i, ry, rx, lq1 - lq0, tot)


def test_feature_layout():
    """wide / widek: every slot of the 4 x 4 same-level window and of the
    parent's 3 x 3 encodes the cell it names (compared with that cell's own
    features: its rho from its cert-channel context and vice versa)."""
    if not cp.WIDE:
        pytest.skip("layout v0")
    w = world()
    rng = np.random.default_rng(8)
    for i in range(1, cl.NL - 1):
        rows, cols = w.rho[i].shape
        prow, pcol = w.rho[i - 1].shape
        for _ in range(6):
            y, x = int(rng.integers(rows)), int(rng.integers(cols))
            for ch in (0, 1):
                f = cp.features_np(w, i, ch, y, x)
                ya, xa = y - (y & 1) - 1, x - (x & 1) - 1
                for d in range(16):
                    yy, xx = ya + d // 4, (xa + d % 4) % cols
                    slot = f[cp.OFF_WIN + d * cp.NCELL:cp.OFF_WIN + (d + 1) * cp.NCELL]
                    if yy < 0:
                        assert slot[0] == 1 and not slot[1:].any()
                        continue
                    if yy >= rows:
                        assert slot[1] == 1 and slot[0] == 0
                        continue
                    me = cp.OFF_WIN + ((yy & 1) + 1) * 4 * cp.NCELL + ((xx & 1) + 1) * cp.NCELL
                    g_c = cp.features_np(w, i, 0, yy, xx)[me:me + cp.NCELL]    # its cert
                    g_r = cp.features_np(w, i, 1, yy, xx)[me:me + cp.NCELL]    # its rho
                    own = yy == y and xx == x
                    if not (own and ch == 0):
                        assert np.array_equal(slot[2:2 + cp.NR], g_r[2:2 + cp.NR])
                    if not (own and ch == 1):
                        assert np.array_equal(slot[2 + cp.NR:], g_c[2 + cp.NR:])
                py, px = y // 2, x // 2
                for d in range(9):
                    yy, xx = py + d // 3 - 1, (px + d % 3 - 1) % pcol
                    slot = f[cp.OFF_PN + d * cp.NCELL:cp.OFF_PN + (d + 1) * cp.NCELL]
                    if 0 <= yy < prow:
                        me = cp.OFF_WIN + ((yy & 1) + 1) * 4 * cp.NCELL + ((xx & 1) + 1) * cp.NCELL
                        g_r = cp.features_np(w, i - 1, 1, yy, xx)[me:me + cp.NCELL]
                        assert np.array_equal(slot[2:2 + cp.NR], g_r[2:2 + cp.NR])
                if cp.KF:
                    assert np.all(f[cp.OFF_KN:cp.OFF_KN + cp.NSLOT] == 1)


def _brute_boundary(w, i, ry, rx, l, y, x):
    """boundary_features written out in numpy."""
    out = np.zeros(cp.NBND)
    s = 1 << (l - i)
    fy0, fx0, fs = (y, x, 1) if l == i else (y - (y & 1), x - (x & 1), 2)
    for k in range(1, cp.NDEP + 1):
        lp = l + k
        if lp > 4:
            break
        m, hq = 1 << k, 16 >> lp
        rows, cols = w.rho[lp].shape
        Y0, X0, S = ry * s * m, rx * s * m, s * m
        if lp == 4:
            R = A = B = w.rho[4].astype(float)
        else:
            R = w.rho[lp].astype(float)
            A, B = w.Aof[lp][w.cert[lp]].astype(float), w.Bof[lp][w.cert[lp]].astype(float)
        for side in range(4):
            b = (side * cp.NDEP + k - 1) * cp.NBS
            if side < 2:
                yy = Y0 - 1 if side == 0 else Y0 + S
                if yy < 0:
                    continue
                if yy >= rows:
                    out[b + 1:b + cp.NBS] = 1
                    continue
                cells = [(yy, (fx0 * m + t) % cols) for t in range(fs * m)]
            else:
                xx = (X0 - 1) % cols if side == 2 else (X0 + S) % cols
                cells = [(fy0 * m + t, xx) for t in range(fs * m)]
            r = np.array([R[c] for c in cells]) / hq ** 2
            a = np.array([A[c] for c in cells]) / hq
            bb = np.array([B[c] for c in cells]) / hq
            out[b:b + cp.NBS] = [1, r.mean(), a.mean(), bb.mean(), a.min(), bb.max()]
    return out


def test_boundary_features_brute():
    w = world()
    rng = np.random.default_rng(9)
    rhos, certs, Aofs, Bofs = cc._tup(w)
    for i in range(cl.NL - 1):
        for ry, rx in _roots(w, i, 3, rng):
            for l, y0, y1, x0, x1 in cc.subtree_slices(i, ry, rx)[:-1]:
                for _ in range(3):
                    y, x = int(rng.integers(y0, y1)), int(rng.integers(x0, x1))
                    out = np.zeros(cp.NBND)
                    cp.boundary_features(i, ry, rx, l, y, x, rhos, certs, Aofs, Bofs, out, 0)
                    assert np.allclose(out, _brute_boundary(w, i, ry, rx, l, y, x)), (i, ry, rx, l, y, x)


def test_move_level_valid():
    w = world()
    preds = random_preds(scale=0.3)
    rng = np.random.default_rng(4)
    for i in (3, 2, 1, 0):
        out = cc.move_level(w, preds, i, 2, rng)
        assert out["n"] == w.rho[i].size
        assert 0 <= out["changed"] <= 1 and 0 <= out["dead"] <= 1 and 1 <= out["ess"] <= 3 + 1e-9
        assert brute_energy(w)[1] == 0, i
        assert cl.overhangs(w.rho[cl.NL - 1]) == 0
    cc.tile_sweep(w, rng)
    assert brute_energy(w)[1] == 0


# ------------------------------------------------------------------ exactness on one h = 2 root
def _set_h2(w, ry, rx, r, c, bits):
    w.rho[3][ry, rx] = r
    w.cert[3][ry, rx] = c
    t = w.rho[4]
    for k in range(4):
        t[2 * ry + k // 2, 2 * rx + k % 2] = (bits >> k) & 1


def _get_h2(w, ry, rx):
    t = w.rho[4][2 * ry:2 * ry + 2, 2 * rx:2 * rx + 2].ravel()
    return (int(w.rho[3][ry, rx]) * 6 + int(w.cert[3][ry, rx])) * 16 + int(t @ (1 << np.arange(4)))


def exact_h2(w, ry, rx):
    """-E over the 480 states of the h = 2 root's subtree (-inf when invalid),
    from the brute energy; the world is restored."""
    keep = _get_h2(w, ry, rx)
    neg = np.full(480, -np.inf)
    for s in range(480):
        _set_h2(w, ry, rx, s // 96, (s // 16) % 6, s % 16)
        e, nv = brute_energy(w)
        if nv == 0:
            neg[s] = -e
    _set_h2(w, ry, rx, keep // 96, (keep // 16) % 6, keep % 16)
    p = np.exp(neg - neg.max())
    return p / p.sum()


def pick_h2_root(w, ncand=12, seed=0):
    """The candidate h = 2 root (cells with 0 < rho < 4) whose p* has the most entropy."""
    rng = np.random.default_rng(seed)
    ys, xs = np.nonzero((w.rho[3] > 0) & (w.rho[3] < 4))
    best = None
    for j in rng.permutation(len(ys))[:ncand]:
        p = exact_h2(w, int(ys[j]), int(xs[j]))
        ent = -(p[p > 0] * np.log(p[p > 0])).sum()
        if best is None or ent > best[0]:
            best = (ent, int(ys[j]), int(xs[j]), p)
    return best


def exact_q_h2(w, preds, ry, rx, p):
    """The proposal's exact law on the valid states by forced passes."""
    keep = _get_h2(w, ry, rx)
    q = np.zeros(480)
    rng = np.random.default_rng(0)
    for s in np.nonzero(p > 0)[0]:
        _set_h2(w, ry, rx, s // 96, (s // 16) % 6, s % 16)
        lq, dead = cc.pass_subtree(w, preds, 3, ry, rx, rng, forced=True)
        q[s] = 0.0 if dead else np.exp(lq)
    _set_h2(w, ry, rx, keep // 96, (keep // 16) % 6, keep % 16)
    return q


def tau_int(x, c=5.0):
    """Integrated autocorrelation time (Sokal's window)."""
    x = np.asarray(x, float) - np.mean(x)
    if x.var() == 0:
        return float(len(x))                                  # never moved: no information
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n] / (x.var() * n)
    tau = 1.0
    for m in range(1, n):
        tau += 2 * ac[m]
        if m >= c * tau:
            break
    return max(tau, 1.0)


def tv_tolerance(p, states, z=3.0):
    """z times the expected TV of an iid histogram of N_eff draws from p
    (normal approximation E|p_hat - p| = sqrt(2 p (1 - p) / (pi N))), N_eff =
    N / tau with tau the largest integrated autocorrelation time of the
    indicators of the 6 most probable states and the log p* trace."""
    states = np.asarray(states)
    top = np.argsort(p)[::-1][:6]
    lp = np.log(np.maximum(p[states], 1e-300))
    tau = max([tau_int(states == s) for s in top] + [tau_int(lp)])
    neff = len(states) / tau
    return z * 0.5 * np.sqrt(2 * p * (1 - p) / (np.pi * neff)).sum(), tau, neff


def run_h2_chain(w, preds, ry, rx, n, move, rng, **kw):
    out = np.empty(n, np.int64)
    for t in range(n):
        move(w, preds, 3, ry, rx, rng=rng, **kw)
        out[t] = _get_h2(w, ry, rx)
    return out


def check_invariance(move, n, label, proposal=False, **kw):
    w = world()
    ent, ry, rx, p = pick_h2_root(w)
    assert ent > 0.5, ent                                     # a non-trivial root
    preds = random_preds(levels=(3,), scale=0.2, seed=7, proposal=proposal)   # TV(q, p*) ~ 0.75: a wrong proposal
    q = exact_q_h2(w, preds, ry, rx, p)
    assert q.sum() <= 1 + 1e-9
    tv_q = 0.5 * np.abs(q / q.sum() - p).sum()
    rng = np.random.default_rng(11)
    # start from an exact draw of p*
    s0 = int(rng.choice(480, p=p))
    _set_h2(w, ry, rx, s0 // 96, (s0 // 16) % 6, s0 % 16)
    states = run_h2_chain(w, preds, ry, rx, n, move, rng, **kw)
    hist = np.bincount(states, minlength=480) / n
    tv = 0.5 * np.abs(hist - p).sum()
    tol, tau, neff = tv_tolerance(p, states)
    print(f"{label}: root ({ry},{rx}) H(p*)={ent:.2f} TV(q,p*)={tv_q:.3f} (sum q {q.sum():.3f}) "
          f"TV(chain,p*)={tv:.4f} tol={tol:.4f} tau={tau:.1f} N_eff={neff:.0f}")
    assert tv_q > tol, "the proposal is too close to p* for the test to have power"
    assert tv < tol, (label, tv, tol)
    assert brute_energy(w)[1] == 0


@pytest.mark.parametrize("K", [1, 4])
def test_subtree_move_invariant(K):
    check_invariance(lambda w, preds, i, ry, rx, rng: cc.subtree_move(w, preds, i, ry, rx, K, rng),
                     40000, f"subtree_move K={K}")


def test_subtree_move_invariant_proposal_preds():
    """The same with a proposal-only predictor (cp.QPreds: boundary context)."""
    check_invariance(lambda w, preds, i, ry, rx, rng: cc.subtree_move(w, preds, i, ry, rx, 2, rng),
                     40000, "subtree_move K=2 QPreds", proposal=True)


def test_annealed_move_invariant():
    ca = pytest.importorskip("castlegen.channels.certland_anneal")
    check_invariance(lambda w, preds, i, ry, rx, rng: ca.annealed_move(w, preds, i, ry, rx, 2, rng, M=2, L0=3.0),
                     12000, "annealed_move K=2 M=2")
