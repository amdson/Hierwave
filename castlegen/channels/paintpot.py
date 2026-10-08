"""Paint potentials: the induced free energy of a coarse level as a linear
local energy over its painted channel at fine resolution
(notes/paintpot_test.md).  Generic: nothing here knows a channel set.

    F_theta(paint) = sum_t u(paint_t) + sum_{d in OFF} sum_t g_d(paint_t, paint_{t+d})

paint : (H, W) ints in 0..V-1, OFF a list of (dy, dx) offsets (each
unordered neighbour pair once), off-grid partners contribute nothing.
theta = (u (V,), g (len(OFF), V, V)) flattened as [u, g_0, g_1, ...], row
= the cell, column = its partner at t + d.

features     count features (V + len(OFF) V^2,) of a map or a batch of maps
energy       features . theta
fit          closed-form ridge (per-window mean loss): theta, diagnostics
unpack       theta -> (u, g)
materialise  the implied coarse unary and pair tables (exact for a linear
             potential): u_c(v) = F(paint_single(v)) - F(paint_single(ref)),
             g_c[d](v, v') = F(paint_pair(v, v', d)) - u_c(v) - u_c(v') - F(ref)
canonical    gauge fix for reading theta: pair tables double-centred, their
             row / column means folded into u (same energy up to boundary
             terms that cancel in window differences)."""
import numpy as np

OFF8 = [(0, 1), (1, 0), (1, 1), (1, -1)]
OFF2 = OFF8 + [(0, 2), (2, 0), (1, 2), (2, 1), (2, 2), (1, -2), (2, -1), (2, -2)]


def nfeat(V, offsets):
    return V + len(offsets) * V * V


def _shift_pair(p, dy, dx):
    """(a, b): the cells t and their partners t + (dy, dx), both on the grid
    (p is (N, H, W))."""
    H, W = p.shape[1:]
    ys, ye = 0, H - dy                       # dy >= 0 always
    xs, xe = max(0, -dx), W - max(0, dx)
    a = p[:, ys:ye, xs:xe]
    b = p[:, ys + dy:ye + dy, xs + dx:xe + dx]
    return a, b


def features(paint, V, offsets=OFF8):
    """Count features of one (H, W) map -> (F,), or of a batch (N, H, W) -> (N, F)."""
    p = np.asarray(paint, np.int64)
    single = p.ndim == 2
    if single:
        p = p[None]
    N = p.shape[0]
    rows = np.arange(N)[:, None, None]
    out = [np.bincount((rows * V + p).ravel(), minlength=N * V).reshape(N, V)]
    for dy, dx in offsets:
        assert dy > 0 or (dy == 0 and dx > 0), "offsets: one of each unordered pair, (dy > 0) or (0, dx > 0)"
        a, b = _shift_pair(p, dy, dx)
        idx = (np.broadcast_to(rows, a.shape) * V + a) * V + b
        out.append(np.bincount(idx.ravel(), minlength=N * V * V).reshape(N, V * V))
    X = np.concatenate(out, 1).astype(float)
    return X[0] if single else X


def unpack(theta, V, offsets=OFF8):
    theta = np.asarray(theta, float)
    return theta[:V].copy(), theta[V:].reshape(len(offsets), V, V).copy()


def pack(u, g):
    return np.concatenate([np.asarray(u, float).ravel(), np.asarray(g, float).ravel()])


def energy(theta, paint, V, offsets=OFF8):
    return features(paint, V, offsets) @ np.asarray(theta, float)


def fit(X, y, l2=1e-6, idx=None):
    """Ridge on the per-window mean loss: theta = (X'X / N + l2 I)^-1 X'y / N.
    idx: rows to fit on (default all).  Returns (theta, diag) with diag the
    in-sample residual RMS, target RMS and their ratio."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    if idx is not None:
        X, y = X[idx], y[idx]
    N, F = X.shape
    th = np.linalg.solve(X.T @ X / N + l2 * np.eye(F), X.T @ y / N)
    r = X @ th - y
    rr, tr = float(np.sqrt(np.mean(r ** 2))), float(np.sqrt(np.mean(y ** 2)))
    return th, dict(resid_rms=rr, target_rms=tr, rel=rr / tr)


def rel_on(theta, X, y):
    r = np.asarray(X) @ theta - y
    return float(np.sqrt(np.mean(r ** 2)) / np.sqrt(np.mean(np.asarray(y) ** 2)))


def materialise(theta, paint_single, paint_pair, D, dirs, V, offsets=OFF8, ref=0):
    """(u_c (D,), g_c (len(dirs), D, D)) of the coarse values 0..D-1.
    paint_single(v) -> painted region with v at the canvas centre on the
    reference background; paint_pair(v, vp, d) -> the same with vp at
    centre + d.  All regions must have the same shape and hold both
    footprints and their spill.  Exact for a linear potential."""
    th = np.asarray(theta, float)
    e0 = energy(th, paint_single(ref), V, offsets)
    u = np.array([energy(th, paint_single(v), V, offsets) - e0 for v in range(D)])
    g = np.zeros((len(dirs), D, D))
    for k, d in enumerate(dirs):
        P = np.stack([paint_pair(v, vp, d) for v in range(D) for vp in range(D)])
        e = (features(P, V, offsets) @ th).reshape(D, D)
        g[k] = e - u[:, None] - u[None, :] - e0
    return u, g


def canonical(theta, V, offsets=OFF8):
    """(u, g) in the gauge g_d double-centred: g_d(v, w) = r_v + c_w - m + gdc
    moves r_v + c_v - m into u(v) (equivalent on interior cells; the
    difference lives on the region boundary, which cancels in the window
    differences of the fit and of materialise).  u is then centred."""
    u, g = unpack(theta, V, offsets)
    for k in range(len(offsets)):
        r, c, m = g[k].mean(1), g[k].mean(0), g[k].mean()
        u = u + r + c - m
        g[k] = g[k] - r[:, None] - c[None, :] + m
    return u - u.mean(), g
