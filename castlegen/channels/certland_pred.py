"""Per-cell bias predictors for certland (notes/certland_chain_spec.md,
section certland_pred).

Per (level i <= 3, channel) a small MLP over the cell's context gives a
full-domain bias b(ctx) (length D = h*h+1 for rho, len(Aof[i]) for cert);
the predictor conditional is softmax_t(-E_designed(t) - b_t).

Features (fixed length NF, the same for every level and channel):

  ordinal of rho (8):   u = rho / h^2, 5 triangular bumps on u (centres
                        0, .25, .5, .75, 1, half-width .25), 1{rho = 0},
                        1{rho = h^2}
  ordinal of cert (15): a = A / h and b = B / h each as u + 5 bumps (12),
                        b - a, 1{A = h}, 1{B = 0}

  [0:24]    parent:   1{no parent}, then the parent's rho (8) and cert (15)
                      at its own side 2h (all zero when there is no parent)
  [24:26]   quadrant: y & 1, x & 1 (the cell's place among its siblings;
                      zero when there is no parent)
  Layout "wide" (the default; env CERTLAND_PRED_FEATS, read at import):
  [26:426]  the same-level window covering the cell's parent block and one
            ring around it: with (y0, x0) = (y & ~1, x & ~1) the top-left of
            the cell's 2 x 2 block (at the top level the same pairing), rows
            y0-1 .. y0+2 and columns x0-1 .. x0+2, row-major, 16 slots of 25
            each: 1{row above the map}, 1{row below the map}, rho (8), cert
            (15).  Above the map the value blocks are zero; below the map
            the cell counts as full (rho = h^2, A = B = h).  x is periodic.
            At the cell itself (slot (y-y0+1)*4 + x-x0+1) the cell's own
            channel is left zero, the other channel is filled in.
  [426:651] the parent's 3 x 3 neighbourhood at the parent level (rows
            py-1..py+1, columns px-1..px+1, row-major), 9 slots of 25 in the
            parent's side 2h, the same encoding; all zero without a parent.
  Layout "widek": "wide" plus [651:667] a known flag per same-level slot
            (cell_features_k): 1 = a real value, 0 = a placeholder of the
            subtree pass (its value features are then zeroed; at the cell's
            own slot the flag is that of its OTHER channel).  The parent
            level is always known.  cell_features (generation, Gibbs
            recording) sets every flag to 1.
  Layout "v0" (the old features, for comparisons):
  [26:251]  the 3 x 3 same-level window, row-major, 9 slots of 25.
  Under a non-default layout the kernels of this module are compiled
  without numba's on-disk cache (the cache does not see the flag).

MLP (NF -> nh -> nh -> D), tanh hidden units:
    z1 = tanh(f W1 + b1), z2 = tanh(z1 W2 + b2), bias = z2 W3 + b3.
Packed flat float64 as [W1 (NF, nh) | b1 | W2 (nh, nh) | b2 | W3 (nh, D) | b3].
init_params zeros W3 and b3, so a fresh predictor is exactly the designed
conditional.  The first layer skips zero features (most of the vector)."""
import os

import numpy as np
from numba import njit as _njit

from . import certland as cl

LAYOUT = os.environ.get("CERTLAND_PRED_FEATS", "wide")
assert LAYOUT in ("wide", "v0", "widek"), LAYOUT
WIDE = LAYOUT in ("wide", "widek")
KF = LAYOUT == "widek"       # known flags
_CACHE = LAYOUT == "wide"


def njit(*a, **kw):
    """numba.njit with cache only under the default layout."""
    if "cache" in kw:
        kw["cache"] = kw["cache"] and _CACHE
    return _njit(*a, **kw)


NB = 5                      # bumps per ordinal
NR = 3 + NB                 # rho block
NC = 2 * (1 + NB) + 3       # cert block
NPAR = 1 + NR + NC          # parent block (with the no-parent flag)
NQ = 2                      # quadrant
NCELL = 2 + NR + NC         # one window cell (two off-grid flags)
OFF_WIN = NPAR + NQ             # 26: the same-level window
NSLOT = 16 if WIDE else 9       # same-level slots: 4 x 4 (wide) / 3 x 3 (v0)
OFF_PN = OFF_WIN + NSLOT * NCELL    # 426 (wide): the parent's 3 x 3
OFF_KN = OFF_PN + 9 * NCELL         # 651 (wide): known flags
NF = OFF_KN + NSLOT if KF else (OFF_KN if WIDE else OFF_PN)    # 667 / 651 / 251
_KN1 = np.ones(NSLOT, np.int64)
NNZ = 160.0 if WIDE else 60.0     # typical nonzero features (init scale of the first layer)
# proposal-only context (stage 2): the subtree's fixed boundary at finer levels
NDEP = 3                        # finer levels l + 1 .. l + 3
NBS = 6                         # per (side, depth): in-map flag, mean rho, mean A, mean B, min A, max B
NBND = 4 * NDEP * NBS           # sides top, bottom, left, right: 72
NFQ = NF + NBND


def domain_size(i, ch):
    h = cl.HS[i]
    return h * h + 1 if ch == cl.RHO else (h + 1) * (h + 2) // 2


# ------------------------------------------------------------------ features
@njit(cache=True, inline="always")
def _ord(u, out, o):
    """u and NB triangular bumps on [0, 1] into out[o:o+1+NB]."""
    out[o] = u
    for k in range(NB):
        d = abs(u - k / (NB - 1)) * (NB - 1)
        out[o + 1 + k] = 1.0 - d if d < 1.0 else 0.0


@njit(cache=True, inline="always")
def _rho_feats(r, R, out, o):
    _ord(r / R, out, o)
    out[o + 1 + NB] = 1.0 if r == 0 else 0.0
    out[o + 2 + NB] = 1.0 if r == R else 0.0


@njit(cache=True, inline="always")
def _cert_feats(A, B, h, out, o):
    a, b = A / h, B / h
    _ord(a, out, o)
    _ord(b, out, o + 1 + NB)
    out[o + 2 + 2 * NB] = b - a
    out[o + 3 + 2 * NB] = 1.0 if A == h else 0.0
    out[o + 4 + 2 * NB] = 1.0 if B == 0 else 0.0


@njit(cache=True, inline="always")
def _enc_cell(h, yy, xx, rho, cert, Aof, Bof, skip_rho, skip_cert, out, o):
    """One cell of side h at (yy, xx) (x periodic) into out[o:o+NCELL]:
    off-grid flags, rho, cert; above the map zero values, below it full."""
    rows, cols = rho.shape
    if yy < 0:
        out[o] = 1.0
        return
    if yy >= rows:
        out[o + 1] = 1.0
        _rho_feats(h * h, h * h, out, o + 2)
        _cert_feats(h, h, h, out, o + 2 + NR)
        return
    xx = xx % cols
    if not skip_rho:
        _rho_feats(rho[yy, xx], h * h, out, o + 2)
    if not skip_cert:
        c = cert[yy, xx]
        _cert_feats(Aof[c], Bof[c], h, out, o + 2 + NR)


@njit(cache=True)
def cell_features(h, ch, y, x, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, has_parent, out):
    """The context of channel ch at (y, x) of a level of side h into out[:NF]
    (every slot known: the Gibbs context of the shipped sampler)."""
    cell_features_k(h, ch, y, x, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, has_parent, _KN1, out)


@njit(cache=True)
def cell_features_k(h, ch, y, x, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, has_parent, kn, out):
    """cell_features with per-slot known flags kn (NSLOT,) over the
    same-level window slots (row-major).  Under layout widek a slot with
    kn = 0 (a placeholder in the subtree pass) has its value features
    zeroed and its flag 0; other layouts ignore kn."""
    for j in range(NF):
        out[j] = 0.0
    if has_parent:
        py, px = y // 2, x // 2
        hp = 2 * h
        _rho_feats(prho[py, px], hp * hp, out, 1)
        c = pcert[py, px]
        _cert_feats(pAof[c], pBof[c], hp, out, 1 + NR)
        out[NPAR] = y & 1
        out[NPAR + 1] = x & 1
    else:
        out[0] = 1.0
    if WIDE:
        ya, xa = y - (y & 1) - 1, x - (x & 1) - 1
        for d in range(16):
            if KF and kn[d] == 0:
                continue
            yy, xx = ya + d // 4, xa + d % 4
            centre = yy == y and xx == x
            _enc_cell(h, yy, xx, rho, cert, Aof, Bof,
                      centre and ch == cl.RHO, centre and ch == cl.CERT, out, OFF_WIN + d * NCELL)
        if has_parent:
            py, px = y // 2, x // 2
            for d in range(9):
                _enc_cell(2 * h, py + d // 3 - 1, px + d % 3 - 1, prho, pcert, pAof, pBof, False, False, out,
                          OFF_PN + d * NCELL)
        if KF:
            for k in range(NSLOT):
                out[OFF_KN + k] = 1.0 if kn[k] != 0 else 0.0
    else:
        for d in range(9):
            centre = d == 4
            _enc_cell(h, y + d // 3 - 1, x + d % 3 - 1, rho, cert, Aof, Bof,
                      centre and ch == cl.RHO, centre and ch == cl.CERT, out, OFF_WIN + d * NCELL)


@njit(cache=True)
def _mlp(f, th, nh, D, out):
    """out[:D] = MLP(f) with the packed layout of pack."""
    _mlp_n(f, th, nh, D, NF, out)


@njit(cache=True)
def _mlp_n(f, th, nh, D, nf, out):
    """_mlp with nf input features."""
    z1 = np.empty(nh)
    z2 = np.empty(nh)
    oW1 = 0
    ob1 = oW1 + nf * nh
    oW2 = ob1 + nh
    ob2 = oW2 + nh * nh
    oW3 = ob2 + nh
    ob3 = oW3 + nh * D
    for j in range(nh):
        z1[j] = th[ob1 + j]
    for k in range(nf):
        v = f[k]
        if v != 0.0:
            r = oW1 + k * nh
            for j in range(nh):
                z1[j] += v * th[r + j]
    for j in range(nh):
        z1[j] = np.tanh(z1[j])
        z2[j] = th[ob2 + j]
    for k in range(nh):
        v = z1[k]
        r = oW2 + k * nh
        for j in range(nh):
            z2[j] += v * th[r + j]
    for j in range(nh):
        z2[j] = np.tanh(z2[j])
    for t in range(D):
        out[t] = th[ob3 + t]
    for k in range(nh):
        v = z2[k]
        r = oW3 + k * D
        for t in range(D):
            out[t] += v * th[r + t]


@njit(cache=True)
def cell_bias(h, ch, y, x, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, has_parent, th, nh, on, out):
    """out[:D] = b(ctx) for channel ch at (y, x); zeros when on == 0."""
    cell_bias_k(h, ch, y, x, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, has_parent, _KN1, th, nh, on, out)


@njit(cache=True)
def cell_bias_k(h, ch, y, x, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, has_parent, kn, th, nh, on, out):
    """cell_bias with known flags kn (see cell_features_k)."""
    D = h * h + 1 if ch == cl.RHO else Aof.shape[0]
    if on == 0:
        for t in range(D):
            out[t] = 0.0
        return
    f = np.empty(NF)
    cell_features_k(h, ch, y, x, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, has_parent, kn, f)
    _mlp(f, th, nh, D, out)


# ------------------------------------------------------------------ proposal-only context
@njit(cache=True)
def boundary_features(i, ry, rx, l, y, x, rhos, certs, Aofs, Bofs, out, o):
    """The fixed cells adjacent to the subtree of root (i, ry, rx), on all
    four sides, at the finer levels l + 1 .. l + NDEP, seen from cell (y, x)
    of level l: with F the cell's parent block clipped to the subtree box
    (the cell itself at the root level) and F' its footprint at level l',
    the row just above / just below the subtree box under F's columns and
    the column just left / right of the box beside F's rows.  Per (side,
    depth) NBS numbers into out[o:]: 1{in the map}, mean rho / h'^2, mean
    A / h', mean B / h', min A / h', max B / h' (tiles: A = B = rho = t).
    Below the map counts as full (values 1, flag 0), above it as empty
    (zeros); levels beyond the tiles are zeros.  Every cell read is outside
    the subtree: fixed during a pass."""
    for j in range(NBND):
        out[o + j] = 0.0
    s = 1 << (l - i)
    Y0, X0 = ry * s, rx * s
    if l == i:
        fy0, fx0, fs = y, x, 1
    else:
        fy0, fx0, fs = y - (y & 1), x - (x & 1), 2
    for k in range(1, NDEP + 1):
        lp = l + k
        if lp > 4:
            break
        m = 1 << k
        hq = 16 >> lp
        rho = rhos[lp]
        cert = certs[lp]
        Aof, Bof = Aofs[lp], Bofs[lp]
        rows, cols = rho.shape
        BY0, BX0, BS = Y0 * m, X0 * m, s * m
        cy0, cx0, cs = fy0 * m, fx0 * m, fs * m
        for side in range(4):
            b = o + (side * NDEP + k - 1) * NBS
            if side < 2:
                yy = BY0 - 1 if side == 0 else BY0 + BS
                if yy < 0:
                    continue
                if yy >= rows:
                    for j in range(1, NBS):
                        out[b + j] = 1.0
                    continue
            else:
                xx = (BX0 - 1) % cols if side == 2 else (BX0 + BS) % cols
            sr = 0.0
            sa = 0.0
            sb = 0.0
            mna = 1e9
            mxb = -1e9
            for t in range(cs):
                if side < 2:
                    cy, cx = yy, (cx0 + t) % cols
                else:
                    cy, cx = cy0 + t, xx
                r = rho[cy, cx]
                if lp == 4:
                    A = r
                    B = r
                else:
                    c = cert[cy, cx]
                    A = Aof[c]
                    B = Bof[c]
                sr += r / (hq * hq)
                sa += A / hq
                sb += B / hq
                mna = min(mna, A / hq)
                mxb = max(mxb, B / hq)
            out[b] = 1.0
            out[b + 1] = sr / cs
            out[b + 2] = sa / cs
            out[b + 3] = sb / cs
            out[b + 4] = mna
            out[b + 5] = mxb


@njit(cache=True)
def cell_bias_q(h, ch, y, x, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, has_parent, kn,
                i, ry, rx, l, rhos, certs, Aofs, Bofs, th, nh, out):
    """The proposal-only predictor's bias (input NFQ: cell_features_k then
    boundary_features), used inside the subtree pass only."""
    D = h * h + 1 if ch == cl.RHO else Aof.shape[0]
    f = np.empty(NFQ)
    cell_features_k(h, ch, y, x, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, has_parent, kn, f)
    boundary_features(i, ry, rx, l, y, x, rhos, certs, Aofs, Bofs, f, NF)
    _mlp_n(f, th, nh, D, NFQ, out)


# ------------------------------------------------------------------ params
KEYS = ("W1", "b1", "W2", "b2", "W3", "b3")


def init_params(i, ch, nh, seed, nf=None):
    """Random hidden layers (W ~ N(0, 1/fan_in), with fan_in of the first
    layer the typical number of nonzero features NNZ, ~60 (v0) / ~160
    (wide)),
    zero output layer."""
    rng = np.random.default_rng(seed)
    D = domain_size(i, ch)
    nf = NF if nf is None else nf
    return dict(W1=rng.normal(0, 1 / np.sqrt(NNZ), (nf, nh)), b1=np.zeros(nh),
                W2=rng.normal(0, 1 / np.sqrt(nh), (nh, nh)), b2=np.zeros(nh),
                W3=np.zeros((nh, D)), b3=np.zeros(D))


def pack(p):
    return np.concatenate([np.asarray(p[k], np.float64).ravel() for k in KEYS])


def unpack(flat, i, ch, nh, nf=None):
    D = domain_size(i, ch)
    nf = NF if nf is None else nf
    shapes = dict(W1=(nf, nh), b1=(nh,), W2=(nh, nh), b2=(nh,), W3=(nh, D), b3=(D,))
    out, o = {}, 0
    for k in KEYS:
        n = int(np.prod(shapes[k]))
        out[k] = np.array(flat[o:o + n]).reshape(shapes[k])
        o += n
    assert o == len(flat), (o, len(flat))
    return out


def jax_bias(p, feats):
    """(N, NF) -> (N, D), the same function as cell_bias (JAX, x64)."""
    import jax
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy
    z = jnp.tanh(feats @ p["W1"] + p["b1"])
    z = jnp.tanh(z @ p["W2"] + p["b2"])
    return z @ p["W3"] + p["b3"]


# ------------------------------------------------------------------ world glue
def level_args(world, i):
    par = i - 1 if i > 0 else i
    return (cl.HS[i], world.rho[i], world.cert[i], world.Aof[i], world.Bof[i],
            world.rho[par], world.cert[par], world.Aof[par], world.Bof[par], int(i > 0))


def features_np(world, i, ch, y, x):
    h, *rest = level_args(world, i)
    out = np.zeros(NF)
    cell_features(h, ch, y, x, *rest, out)
    return out


class Preds:
    """th[i][ch]: packed float64 params or None (no bias), for i <= 3.
    nf: input width, NF (the Gibbs predictor: the shipped sampler and, by
    default, the subtree pass) or NFQ (a proposal-only predictor: the
    subtree pass only, with the boundary context; see QPreds)."""

    def __init__(self, nh=64, nf=None):
        self.nh = nh
        self.nf = NF if nf is None else nf
        self.th = [[None, None] for _ in range(cl.NL - 1)]

    @property
    def proposal(self):
        return self.nf == NFQ and NFQ != NF

    def args(self, i, ch):
        th = self.th[i][ch]
        return (np.zeros(1), self.nh, 0) if th is None else (th, self.nh, 1)

    def set(self, i, ch, params):
        flat = pack({k: np.asarray(params[k]) for k in KEYS})
        assert len(flat) == self.nf * self.nh + self.nh + self.nh * self.nh + self.nh + \
            (self.nh + 1) * domain_size(i, ch)
        self.th[i][ch] = np.ascontiguousarray(flat)

    def params(self, i, ch):
        th = self.th[i][ch]
        return None if th is None else unpack(th, i, ch, self.nh, self.nf)


def QPreds(nh=64):
    """A proposal-only predictor set (input NFQ = NF + NBND)."""
    return Preds(nh, NFQ)


# ------------------------------------------------------------------ the shipped sampler
@njit(cache=True)
def sweep_pred(i, h, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, fill, kappa, lam, beta, has_parent,
               th_r, on_r, th_c, on_c, nh, n, seed):
    """n raster sweeps (rho then cert) of single-site Gibbs on designed soft
    (+inf on violation) + bias.  Returns no-candidate site visits."""
    np.random.seed(seed)
    rows, cols = rho.shape
    Dr = h * h + 1
    Dc = Aof.shape[0]
    Dm = max(Dr, Dc)
    soft = np.empty(Dm)
    viol = np.empty(Dm, np.int64)
    e = np.empty(Dm)
    bias = np.empty(Dm)
    dum = np.zeros((2, cl.KPHI))
    th0 = np.zeros(1)
    bad = 0
    for _ in range(n):
        for ch in range(2):
            D = Dr if ch == cl.RHO else Dc
            g = rho if ch == cl.RHO else cert
            th = th_r if ch == cl.RHO else th_c
            on = on_r if ch == cl.RHO else on_c
            for y in range(rows):
                for x in range(cols):
                    cl.site_terms(i, h, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, fill, kappa, lam, beta,
                                  dum, dum, dum, dum, th0, 0, 0, y, x, ch, soft[:D], viol[:D])
                    cell_bias(h, ch, y, x, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, has_parent,
                              th, nh, on, bias)
                    for t in range(D):
                        e[t] = soft[t] + bias[t] if viol[t] == 0 else np.inf
                    pick = cl._gumbel(e, D)
                    if pick < 0:
                        bad += 1
                    else:
                        g[y, x] = pick
    return bad


def generate_pred(world, preds, sweeps=20, seed=0, upto=cl.NL - 1):
    """cl.generate's shape with the predictors' biases on levels h >= 2;
    designed terms only (learned = 0); tiles designed only.  Returns
    no-candidate counts per level."""
    rng = np.random.default_rng(seed)
    bad = []
    cl.init_top(world)
    for i in range(upto + 1):
        if i > 0:
            cl.refine(world, i, rng)
        s = int(rng.integers(1 << 30))
        if i == cl.NL - 1:
            a = list(world.args(i))
            a[-3], a[-2], a[-1] = np.zeros(1), 0, 0
            bad.append(int(cl.sweep_level(*a, sweeps, s)))
        else:
            h, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, hp = level_args(world, i)
            th_r, nh, on_r = preds.args(i, cl.RHO)
            th_c, _, on_c = preds.args(i, cl.CERT)
            bad.append(int(sweep_pred(i, h, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, world.fill,
                                      world.kappa, world.lam, world.beta, hp, th_r, on_r, th_c, on_c, nh,
                                      sweeps, s)))
    return bad
