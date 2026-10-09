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
  [26:251]  the 3 x 3 same-level window, row-major (dy, dx in -1..1), 25
            each: 1{row above the map}, 1{row below the map}, rho (8),
            cert (15).  Above the map the value blocks are zero; below the
            map the cell counts as full (rho = h^2, A = B = h).  x is
            periodic.  At the centre the cell's own channel is left zero,
            the other channel is filled in.

MLP (NF -> nh -> nh -> D), tanh hidden units:
    z1 = tanh(f W1 + b1), z2 = tanh(z1 W2 + b2), bias = z2 W3 + b3.
Packed flat float64 as [W1 (NF, nh) | b1 | W2 (nh, nh) | b2 | W3 (nh, D) | b3].
init_params zeros W3 and b3, so a fresh predictor is exactly the designed
conditional.  The first layer skips zero features (most of the vector)."""
import numpy as np
from numba import njit

from . import certland as cl

NB = 5                      # bumps per ordinal
NR = 3 + NB                 # rho block
NC = 2 * (1 + NB) + 3       # cert block
NPAR = 1 + NR + NC          # parent block (with the no-parent flag)
NQ = 2                      # quadrant
NCELL = 2 + NR + NC         # one window cell (two off-grid flags)
OFF_WIN = NPAR + NQ
NF = OFF_WIN + 9 * NCELL    # 251


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


@njit(cache=True)
def cell_features(h, ch, y, x, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, has_parent, out):
    """The context of channel ch at (y, x) of a level of side h into out[:NF]."""
    rows, cols = rho.shape
    for j in range(NF):
        out[j] = 0.0
    R = h * h
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
    for d in range(9):
        dy, dx = d // 3 - 1, d % 3 - 1
        o = OFF_WIN + d * NCELL
        yy = y + dy
        if yy < 0:
            out[o] = 1.0
            continue
        if yy >= rows:
            out[o + 1] = 1.0
            _rho_feats(R, R, out, o + 2)
            _cert_feats(h, h, h, out, o + 2 + NR)
            continue
        xx = (x + dx) % cols
        centre = d == 4
        if not (centre and ch == cl.RHO):
            _rho_feats(rho[yy, xx], R, out, o + 2)
        if not (centre and ch == cl.CERT):
            c = cert[yy, xx]
            _cert_feats(Aof[c], Bof[c], h, out, o + 2 + NR)


@njit(cache=True)
def _mlp(f, th, nh, D, out):
    """out[:D] = MLP(f) with the packed layout of pack."""
    z1 = np.empty(nh)
    z2 = np.empty(nh)
    oW1 = 0
    ob1 = oW1 + NF * nh
    oW2 = ob1 + nh
    ob2 = oW2 + nh * nh
    oW3 = ob2 + nh
    ob3 = oW3 + nh * D
    for j in range(nh):
        z1[j] = th[ob1 + j]
    for k in range(NF):
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
    D = h * h + 1 if ch == cl.RHO else Aof.shape[0]
    if on == 0:
        for t in range(D):
            out[t] = 0.0
        return
    f = np.empty(NF)
    cell_features(h, ch, y, x, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, has_parent, f)
    _mlp(f, th, nh, D, out)


# ------------------------------------------------------------------ params
KEYS = ("W1", "b1", "W2", "b2", "W3", "b3")


def init_params(i, ch, nh, seed):
    """Random hidden layers (W ~ N(0, 1/fan_in), with fan_in of the first
    layer the typical ~60 nonzero features), zero output layer."""
    rng = np.random.default_rng(seed)
    D = domain_size(i, ch)
    return dict(W1=rng.normal(0, 1 / np.sqrt(60.0), (NF, nh)), b1=np.zeros(nh),
                W2=rng.normal(0, 1 / np.sqrt(nh), (nh, nh)), b2=np.zeros(nh),
                W3=np.zeros((nh, D)), b3=np.zeros(D))


def pack(p):
    return np.concatenate([np.asarray(p[k], np.float64).ravel() for k in KEYS])


def unpack(flat, i, ch, nh):
    D = domain_size(i, ch)
    shapes = dict(W1=(NF, nh), b1=(nh,), W2=(nh, nh), b2=(nh,), W3=(nh, D), b3=(D,))
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
    """th[i][ch]: packed float64 params or None (no bias), for i <= 3."""

    def __init__(self, nh=64):
        self.nh = nh
        self.th = [[None, None] for _ in range(cl.NL - 1)]

    def args(self, i, ch):
        th = self.th[i][ch]
        return (np.zeros(1), self.nh, 0) if th is None else (th, self.nh, 1)

    def set(self, i, ch, params):
        flat = pack({k: np.asarray(params[k]) for k in KEYS})
        assert len(flat) == NF * self.nh + self.nh + self.nh * self.nh + self.nh + (self.nh + 1) * domain_size(i, ch)
        self.th[i][ch] = np.ascontiguousarray(flat)

    def params(self, i, ch):
        th = self.th[i][ch]
        return None if th is None else unpack(th, i, ch, self.nh)


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
