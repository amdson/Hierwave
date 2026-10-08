"""JAX fit of the conv potential (notes/convpot_test.md, "The fit").

The potential (the definition shared with the kernel, core.py / kernel.py):
for a grid z, an embedding E : (D, k) and the 3 x 3 offsets d indexed 0..8
row-major over (dy, dx) in (-1, 0, 1)^2 (index 4 the centre),

    E(z) = sum_p  a . e(z_p)
                + sum_{d != 4} e(z_p)^T A_d e(z_{p+d})
                + v . softplus( sum_d W_d^T e(z_{p+d}) + b )      (absent when m = 0)

the sum over the grid's cells p; off-grid neighbours take E[pad] when
pad >= 0, else the zero vector.  Offset d and its opposite are indices i
and 8 - i.

API
    energy_window(params, E, windows, pad)        (N,) energies of (N, R, C) windows
    init_params(D, k, m, seed, scale, tie)         params (A as (4, k, k) when tie)
    expand(params)                                 full params (A : (9, k, k), centre zero)
    fit(windows, targets, E, pad, m, ...)          hand-written Adam (optax is not installed)
    fit_bilinear_ls(windows, targets, E, pad, l2)  closed-form ridge, m = 0, fixed E
    materialise_pairs(params, E, pad)              (u : (D,), g : (9, D, D))
    to_factor(params, E, view, pad, name)          Factor.convpot(...)

params are dicts of arrays {"a" (k,), "A" (9 or 4, k, k), "W" (9, k, m),
"b" (m,), "v" (m,)} plus "c" (a scalar offset) when the fit has no reference
window.  With tie=True the stored A holds the offsets 5..8 = (0, 1), (1, -1),
(1, 0), (1, 1) and A_{8-i} = A_i^T; expand() turns it into the (9, k, k)
form the kernel reads.

The regression: loss = mean_n (E(z_n) - E(ref) - t_n)^2 + l2 * sum of the
squared free params (the stored ones: tied A counted once, E not penalised),
the reference window all `ref` (default: pad).  Everything here runs in
float64 under a scoped jax.enable_x64 (the rest of the repo's jax is 32 bit).
"""
import time

import numpy as np

HALF = (5, 6, 7, 8)                       # (0, 1), (1, -1), (1, 0), (1, 1); their opposites are 3, 2, 1, 0
OFFS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]


def _jax():
    import jax
    import jax.numpy as jnp
    return jax, jnp


# ------------------------------------------------------------------ energy
def expand(params):
    """Full-form params: A (9, k, k) with the centre zero.  numpy in, numpy out."""
    A = params["A"]
    xp = np if isinstance(A, np.ndarray) else _jax()[1]
    k = A.shape[1]
    if A.shape[0] == 4:
        full = [None] * 9
        full[4] = xp.zeros((k, k), A.dtype)
        for j, i in enumerate(HALF):
            full[i] = A[j]
            full[8 - i] = A[j].T
        A = xp.stack(full)
    else:
        A = A * (xp.arange(9) != 4)[:, None, None]
    out = dict(params)
    out["A"] = A
    return out


def _energy_batch(p, E, windows, pad):
    """The definition on (N, R, C) windows, p in full form: (N,) energies.
    Since the domain is small everything goes through per-value tables
    (differentiable in E as well): with Ex = E plus a row D for off-window
    cells (E[pad], or zero when pad < 0),
        U  = (E a)[z_p]
        B  = sum_d G_d[z_p, z_{p+d}],     G_d = E A_d Ex^T      (9, D, D + 1)
        h  = sum_d K_d[z_{p+d}] + b,      K_d = Ex W_d          (9, D + 1, m)."""
    jax, jnp = _jax()
    N, R, C = windows.shape
    D, k = E.shape
    epad = E[pad] if pad >= 0 else jnp.zeros((k,), E.dtype)
    Ex = jnp.concatenate([E, epad[None]], 0)
    Z = jnp.full((N, R + 2, C + 2), D, windows.dtype).at[:, 1:-1, 1:-1].set(windows)
    nb = jnp.stack([Z[:, 1 + dy:1 + dy + R, 1 + dx:1 + dx + C] for dy, dx in OFFS], 3)   # (N, R, C, 9)
    A = p["A"].at[4].set(0.0)
    G = jnp.einsum("vk,dkl,wl->dvw", E, A, Ex)
    d9 = jnp.arange(9)
    e = (E @ p["a"])[windows] + G[d9, windows[..., None], nb].sum(3)              # (N, R, C)
    m = p["v"].shape[0]
    if m > 0:
        K = jnp.einsum("vk,dkm->dvm", Ex, p["W"])
        rows = jnp.take(K.reshape(9 * (D + 1), m), (nb + (D + 1) * d9).reshape(-1), axis=0)
        pre = rows.reshape(N, R, C, 9, m).sum(3) + p["b"]                         # (N, R, C, m)
        e = e + jax.nn.softplus(pre) @ p["v"]
    return e.sum((1, 2))


def energy_window(params, E, windows, pad=-1):
    """(N,) energies of the (N, R, C) int windows (a single (R, C) window
    gives a scalar) under params (tied or full A), off-window at pad."""
    jax, jnp = _jax()
    with jax.enable_x64(True):
        p = {kk: jnp.asarray(v, jnp.float64) for kk, v in params.items() if kk != "c"}
        p = expand(p)
        Ej = jnp.asarray(E, jnp.float64)
        w = jnp.asarray(windows, jnp.int32)
        if w.ndim == 2:
            return float(_energy_batch(p, Ej, w[None], int(pad))[0])
        return np.asarray(jax.jit(_energy_batch, static_argnums=3)(p, Ej, w, int(pad)))


# ------------------------------------------------------------------ params
def init_params(D, k, m, seed=0, scale=0.01, tie=True):
    """Small Gaussian params (numpy float64).  D is unused by the shapes
    (kept for the signature of the note); a, A, v ~ N(0, scale^2), b = 0 and
    W ~ N(0, 1 / (9 k)), which makes the pre-activations O(1) for embedding
    rows of norm sqrt(k) (fit rescales W to the actual mean row norm)."""
    rng = np.random.default_rng(seed)
    nA = 4 if tie else 9
    A = scale * rng.standard_normal((nA, k, k))
    if not tie:
        A[4] = 0.0
    return dict(a=scale * rng.standard_normal(k), A=A,
                W=rng.standard_normal((9, k, m)) / np.sqrt(9 * k),
                b=np.zeros(m), v=scale * rng.standard_normal(m))


# -------------------------------------------------------------------- fit
def _adam(grad_fn, theta, steps, lr, lr_end=None, b1=0.9, b2=0.999, eps=1e-8):
    """Hand-written Adam: returns a jitted step (theta, mu, nu, t, key) -> ...
    and the initial state.  The step size decays from lr to lr_end (default
    lr: constant) on a cosine over `steps`."""
    jax, jnp = _jax()
    tm = jax.tree_util.tree_map

    @jax.jit
    def step(theta, mu, nu, t, key):
        loss, g = grad_fn(theta, key)
        mu = tm(lambda m, gg: b1 * m + (1 - b1) * gg, mu, g)
        nu = tm(lambda n, gg: b2 * n + (1 - b2) * gg * gg, nu, g)
        t = t + 1
        c1, c2 = 1 - b1 ** t, 1 - b2 ** t
        le = lr if lr_end is None else lr_end
        eta = le + 0.5 * (lr - le) * (1 + jnp.cos(jnp.pi * jnp.minimum(t / steps, 1.0)))
        theta = tm(lambda th, m, n: th - eta * (m / c1) / (jnp.sqrt(n / c2) + eps), theta, mu, nu)
        return theta, mu, nu, t, loss

    zeros = tm(jnp.zeros_like, theta)
    return step, (theta, zeros, tm(jnp.zeros_like, theta), jnp.asarray(0.0))


def fit(windows, targets, E, pad=-1, m=0, steps=2000, lr=1e-2, l2=1e-6, seed=0, tie=True,
        learn_E=False, batch=None, ref=None, scale=0.01, init=None, lr_end=None, log_every=0,
        log=print):
    """Adam on mean (E(z) - E(ref) - t)^2 + l2 |params|^2.

    windows (N, R, C) int, targets (N,), E (D, k) (the init when learn_E).
    ref: the reference value (default pad); with ref and pad both < 0 there
    is no reference window and a free scalar offset "c" is fitted instead.
    batch: minibatch size (None = full batch).  init: params to start from.
    lr_end: cosine decay of the step size from lr to lr_end (None: constant).
    Returns dict(params (full form, numpy), E, curve (per-step loss), resid_rms,
    target_rms, rel (= resid / target), seconds, steps_per_s)."""
    jax, jnp = _jax()
    ref = pad if ref is None else ref
    D, k = np.asarray(E).shape
    with jax.enable_x64(True):
        W_ = jnp.asarray(windows, jnp.int32)
        T_ = jnp.asarray(targets, jnp.float64)
        N, R, C = W_.shape
        refw = jnp.full((1, R, C), max(ref, 0), jnp.int32)
        if init is None:                          # W rescaled so the head's pre-activations are O(1) for this E
            p0 = init_params(D, k, m, seed, scale, tie)
            p0["W"] = p0["W"] * np.sqrt(k) / max(float(np.linalg.norm(np.asarray(E), axis=1).mean()), 1e-12)
        else:
            p0 = dict(init)
        theta = {kk: jnp.asarray(v, jnp.float64) for kk, v in p0.items() if kk != "c"}
        if ref < 0:
            theta["c"] = jnp.asarray(float(p0.get("c", 0.0)))
        if learn_E:
            theta["E"] = jnp.asarray(E, jnp.float64)
        E_fixed = jnp.asarray(E, jnp.float64)

        def pred(th, w):
            Eu = th["E"] if learn_E else E_fixed
            p = expand({kk: th[kk] for kk in ("a", "A", "W", "b", "v")})
            e = _energy_batch(p, Eu, w, pad)
            if ref >= 0:
                e = e - _energy_batch(p, Eu, refw, pad)[0]
            else:
                e = e + th["c"]
            return e

        def loss_fn(th, key):
            if batch is None or batch >= N:
                w, t = W_, T_
            else:
                idx = jax.random.choice(key, N, (batch,), replace=False)
                w, t = W_[idx], T_[idx]
            r = pred(th, w) - t
            reg = sum(jnp.sum(th[kk] ** 2) for kk in ("a", "A", "W", "b", "v"))
            return jnp.mean(r ** 2) + l2 * reg

        step, (th, mu, nu, t) = _adam(jax.value_and_grad(loss_fn), theta, steps, lr, lr_end)
        key = jax.random.PRNGKey(seed)
        curve = np.empty(steps)
        th, mu, nu, t, l0 = step(th, mu, nu, t, key)               # compile
        curve[0] = float(l0)
        t0 = time.perf_counter()
        for s in range(1, steps):
            key, sub = jax.random.split(key)
            th, mu, nu, t, l = step(th, mu, nu, t, sub)
            curve[s] = float(l)
            if log_every and s % log_every == 0:
                log(f"  adam step {s}: loss {curve[s]:.6g}")
        sec = time.perf_counter() - t0
        res = np.asarray(jax.jit(pred)(th, W_)) - np.asarray(T_)
        full = expand({kk: th[kk] for kk in ("a", "A", "W", "b", "v")})
        params = {kk: np.asarray(v) for kk, v in full.items()}
        if ref < 0:
            params["c"] = float(th["c"])
        Eo = np.asarray(th["E"]) if learn_E else np.asarray(E, float)
    tr = float(np.sqrt(np.mean(np.asarray(targets, float) ** 2)))
    rr = float(np.sqrt(np.mean(res ** 2)))
    return dict(params=params, E=Eo, curve=curve, resid_rms=rr, target_rms=tr,
                rel=rr / max(tr, 1e-300), seconds=sec, steps_per_s=(steps - 1) / max(sec, 1e-12))


# ----------------------------------------------------------- closed form
def bilinear_features(windows, E, pad=-1, tie=True):
    """(N, F) features of the m = 0 potential, linear in (a, A): F = k + nA k^2
    (nA = 4 tied, 8 untied, the A blocks in the order HALF resp. offsets
    0..8 without 4).  Tied: the block of offset i is F_i + F_{8-i}^T."""
    W = np.asarray(windows)
    E = np.asarray(E, float)
    N, R, C = W.shape
    k = E.shape[1]
    emb = E[W]                                                       # (N, R, C, k)
    epad = E[pad] if pad >= 0 else np.zeros(k)
    P = np.broadcast_to(epad, (N, R + 2, C + 2, k)).copy()
    P[:, 1:-1, 1:-1] = emb
    Fd = {}
    for i, (dy, dx) in enumerate(OFFS):
        if i == 4:
            continue
        nb = P[:, 1 + dy:1 + dy + R, 1 + dx:1 + dx + C]
        Fd[i] = np.einsum("nrck,nrcl->nkl", emb, nb)
    blocks = [emb.sum((1, 2))]
    if tie:
        for i in HALF:
            blocks.append((Fd[i] + Fd[8 - i].transpose(0, 2, 1)).reshape(N, -1))
    else:
        for i in range(9):
            if i != 4:
                blocks.append(Fd[i].reshape(N, -1))
    return np.concatenate(blocks, 1)


def fit_bilinear_ls(windows, targets, E, pad=-1, l2=1e-6, tie=True, ref=None):
    """The closed-form minimiser of the fit's loss for m = 0, fixed E:
    (X^T X / N + l2 I) theta = X^T y / N, X the features of z minus those of
    the reference window (or, with no reference, an unpenalised constant
    column).  Returns the same dict as fit (params in full form)."""
    ref = pad if ref is None else ref
    E = np.asarray(E, float)
    W = np.asarray(windows)
    N, R, C = W.shape
    k = E.shape[1]
    y = np.asarray(targets, float)
    t0 = time.perf_counter()
    X = bilinear_features(W, E, pad, tie)
    F = X.shape[1]
    pen = np.full(F, l2)
    if ref >= 0:
        X = X - bilinear_features(np.full((1, R, C), ref), E, pad, tie)
    else:
        X = np.concatenate([X, np.ones((N, 1))], 1)
        pen = np.append(pen, 0.0)
    G = X.T @ X / N + np.diag(pen)
    th = np.linalg.solve(G + 1e-12 * np.eye(len(pen)), X.T @ y / N)
    res = X @ th - y
    a = th[:k]
    nA = 4 if tie else 8
    Ab = th[k:k + nA * k * k].reshape(nA, k, k)
    A = np.zeros((9, k, k))
    if tie:
        for j, i in enumerate(HALF):
            A[i], A[8 - i] = Ab[j], Ab[j].T
    else:
        A[[i for i in range(9) if i != 4]] = Ab
    params = dict(a=a, A=A, W=np.zeros((9, k, 0)), b=np.zeros(0), v=np.zeros(0))
    if ref < 0:
        params["c"] = float(th[-1])
    tr = float(np.sqrt(np.mean(y ** 2)))
    rr = float(np.sqrt(np.mean(res ** 2)))
    loss = float(np.mean(res ** 2) + l2 * np.sum(th[:F] ** 2))
    return dict(params=params, E=E, loss=loss, resid_rms=rr, target_rms=tr,
                rel=rr / max(tr, 1e-300), seconds=time.perf_counter() - t0)


# ------------------------------------------------------------ materialise
def materialise_pairs(params, E, pad):
    """The implied unary and pair tables on an all-pad background (pad >= 0):
        u[v]       = E(v at c) - E(all pad)
        g[i][v, w] = E(v at c, w at c + d_i) - u[v] - u[w] - E(all pad),  g[4] = 0
    on a 5 x 5 window (c its centre), which holds every term a one- or
    two-cell change touches, so the window's own off-grid handling equals an
    infinite pad plane.  Exact for m = 0 (then E = const + sum u + sum over
    unordered adjacent pairs of g), the pair projection for m > 0.
    g[8 - i] = g[i]^T: as tables of a pairwise model use the four offsets
    HALF (each g[i] already holds both ordered bilinear terms)."""
    assert pad >= 0, "materialise_pairs needs a reference (pad) value"
    E = np.asarray(E, float)
    D = E.shape[0]
    c = 2
    base = np.full((5, 5), pad, np.int32)
    one = np.repeat(base[None], D, 0)
    one[:, c, c] = np.arange(D)
    e0 = energy_window(params, E, base[None], pad)[0]
    u = energy_window(params, E, one, pad) - e0
    g = np.zeros((9, D, D))
    vv, ww = np.meshgrid(np.arange(D), np.arange(D), indexing="ij")
    for i, (dy, dx) in enumerate(OFFS):
        if i == 4:
            continue
        two = np.repeat(base[None], D * D, 0)
        two[:, c, c] = vv.ravel()
        two[:, c + dy, c + dx] = ww.ravel()
        e = energy_window(params, E, two, pad).reshape(D, D)
        g[i] = e - u[:, None] - u[None, :] - e0
    return u, g


# ----------------------------------------------------------------- factor
def to_factor(params, E, view, pad=-1, name="convpot"):
    """Factor.convpot((chan, view), E, a_vec, A, W, b, v, pad, name) from fitted
    params (full form; a fitted "c" is a constant and is dropped)."""
    from .core import Factor
    p = expand({kk: np.asarray(v, float) for kk, v in params.items() if kk != "c"})
    E = np.asarray(E, float)
    if not hasattr(Factor, "convpot"):
        # TODO: Factor.convpot(a=(chan, view), E, a_vec, A, W, b, v, pad=-1, name)
        # (notes/convpot_test.md) is not in core.py yet; return its arrays.
        return dict(a=view, E=E, a_vec=p["a"], A=p["A"], W=p["W"], b=p["b"], v=p["v"], pad=pad, name=name)
    return Factor.convpot(view, E, p["a"], p["A"], p["W"], p["b"], p["v"], pad=pad, name=name)
