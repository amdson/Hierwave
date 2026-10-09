"""Certified terrain: density channels coupled to a column certificate by
learned potentials only (notes/certland_test.md).

Levels h = 16, 8, 4, 2, 1 (level index i = 0..4), each a grid of H/h x W/h
blocks, x periodic.  At every level h >= 2 two sampled channels:
  rho   0..h^2      solid tiles in the block
  cert  (A, B)      0 <= A <= B <= h: A promised full columns, B allowed
                    occupied columns (bounds, not counts)
At h = 1 the tile is one channel t in {0, 1} and its certificate is computed
exactly from it: A = B = t.

Designed terms (the only hand-written ones):
  rho   count prior    kappa (sum of the four children's rho - parent rho)^2 / h^2
        smoothness     lam |rho_p - rho_q| / h, horizontal neighbours
        noise (top)    beta (rho - h^2 f_p)^2 / h^2, f the block's fill target
  cert  support        B(p) <= A(p below); below the map is full     (hard)
        row honour     per row of the parent's 2 x 2 children:
                       sum A >= A_parent, sum B <= B_parent          (hard)
There is no hand-written rule between rho and cert above the tiles.

Learned (h >= 2): a joint energy over the level's two channels and their
parent, a convpot over a summed embedding
  phi_p = Er[rho_p] + Ec[cert_p] + Epr[rho_P] + Epc[cert_P]      (P the parent)
  Phi = sum_p [ a.phi_p + phi_p' S phi_p + sum_{d in D+} phi_p' A_d phi_{p+d}
               + v . softplus(b + sum_{d in 3x3} W_d' phi_{p+d}) ]
D+ = (0,1), (1,-1), (1,0), (1,1); off-grid rows give phi = 0 and heads
centred off the grid do not exist.  The embeddings are fixed ordinal
features (no per-value parameters); a, S, A, W, b, v are learned.

Kernel: single-site Gibbs, raster order, every channel by the same code.
Hard terms are counted as violations; the annealed energy (AIS) is
beta * (soft + L * violations)."""
import numpy as np
from numba import njit

HS = (16, 8, 4, 2, 1)
NL = len(HS)
RHO, CERT = 0, 1
D_PLUS = np.array([[0, 1], [1, -1], [1, 0], [1, 1]], np.int64)


# ------------------------------------------------------------------ domains
def cert_domain(h):
    """(A, B) arrays over the certificate values of a level of side h."""
    if h == 1:
        return np.array([0, 1], np.int64), np.array([0, 1], np.int64)
    ab = [(a, b) for b in range(h + 1) for a in range(b + 1)]
    return np.array([a for a, _ in ab], np.int64), np.array([b for _, b in ab], np.int64)


def emb_rho(h):
    """(h^2 + 1, 5) ordinal features of x = rho / h^2."""
    x = np.arange(h * h + 1) / (h * h)
    return np.stack([x, x ** 2, x ** 3, (x == 0) * 1.0, (x == 1) * 1.0], 1)


def emb_cert(h):
    """(Dc, 7) features of a = A / h, b = B / h."""
    A, B = cert_domain(h)
    a, b = A / h, B / h
    return np.stack([a, b, a * a, b * b, a * b, (a == 1) * 1.0, (b == 0) * 1.0], 1)


KR, KC = 5, 7
KPHI = 2 * (KR + KC)


def level_embeddings(i):
    """(Er, Ec, Epr, Epc) placed in their blocks of the KPHI-dim space;
    the parent blocks are zero rows of width 1 at the top."""
    h = HS[i]
    Er = np.zeros((h * h + 1, KPHI))
    Er[:, :KR] = emb_rho(h)
    Ec = np.zeros((len(cert_domain(h)[0]), KPHI))
    Ec[:, KR:KR + KC] = emb_cert(h)
    if i == 0:
        Epr, Epc = np.zeros((1, KPHI)), np.zeros((1, KPHI))
    else:
        hp = HS[i - 1]
        Epr = np.zeros((hp * hp + 1, KPHI))
        Epr[:, KR + KC:2 * KR + KC] = emb_rho(hp)
        Epc = np.zeros((len(cert_domain(hp)[0]), KPHI))
        Epc[:, 2 * KR + KC:] = emb_cert(hp)
    return Er, Ec, Epr, Epc


# ------------------------------------------------------------------ params
def zero_params(m=0):
    k = KPHI
    return dict(a=np.zeros(k), S=np.zeros((k, k)), A=np.zeros((4, k, k)),
                W=np.zeros((9, k, max(m, 1))), b=np.zeros(max(m, 1)), v=np.zeros(max(m, 1)), m=m)


def pack_params(p):
    """Flat float64 [a | S | A | W | b | v] and m (m = 0: no head)."""
    m = p["m"]
    W = p["W"].reshape(9, KPHI, -1)[:, :, :max(m, 1)]
    S = 0.5 * (p["S"] + p["S"].T)
    return np.concatenate([p["a"].ravel(), S.ravel(), p["A"].ravel(), W.ravel(),
                           np.ravel(p["b"])[:max(m, 1)], np.ravel(p["v"])[:max(m, 1)]]).astype(np.float64), m


@njit(cache=True)
def _unpack(th, m):
    k = KPHI
    mm = max(m, 1)
    o = 0
    a = th[o:o + k]; o += k
    S = th[o:o + k * k].reshape((k, k)); o += k * k
    A = th[o:o + 4 * k * k].reshape((4, k, k)); o += 4 * k * k
    W = th[o:o + 9 * k * mm].reshape((9, k, mm)); o += 9 * k * mm
    b = th[o:o + mm]; o += mm
    v = th[o:o + mm]
    return a, S, A, W, b, v


# ------------------------------------------------------------------ the state
class World:
    """All levels' grids and the designed constants.  rho[i], cert[i] int64
    (H/h, W/h); at the tile level cert[4] is rho[4] (the same array)."""

    def __init__(self, H=128, W=256, kappa=1.0, lam=0.5, beta=4.0, L=30.0):
        assert H % 16 == 0 and W % 16 == 0
        self.H, self.W = H, W
        self.kappa, self.lam, self.beta, self.L = kappa, lam, beta, L
        self.rho, self.cert = [], []
        self.Aof, self.Bof = [], []
        for i, h in enumerate(HS):
            r = np.zeros((H // h, W // h), np.int64)
            self.rho.append(r)
            self.cert.append(r if h == 1 else np.zeros_like(r))
            A, B = cert_domain(h)
            self.Aof.append(A)
            self.Bof.append(B)
        self.fill = np.zeros((H // 16, W // 16))           # top-level fill target in [0, 1]
        self.emb = [level_embeddings(i) for i in range(NL - 1)]
        self.params = [None] * (NL - 1)                    # packed (theta, m) per level h >= 2, None = Phi = 0

    def args(self, i):
        """The tuple the numba routines of level i read."""
        h = HS[i]
        par = i - 1 if i > 0 else i
        Er, Ec, Epr, Epc = self.emb[i] if i < NL - 1 else (np.zeros((2, KPHI)),) * 4
        if i < NL - 1 and self.params[i] is not None:
            th, m = self.params[i]
            learned = 1
        else:
            th, m, learned = np.zeros(1), 0, 0
        return (i, h, self.rho[i], self.cert[i], self.Aof[i], self.Bof[i], self.rho[par], self.cert[par],
                self.Aof[par], self.Bof[par], self.fill, self.kappa, self.lam, self.beta,
                Er, Ec, Epr, Epc, th, m, learned)


# ------------------------------------------------------------------ energies
@njit(cache=True, inline="always")
def _phi(i, y, x, rho, cert, prho, pcert, Er, Ec, Epr, Epc, out):
    """phi at (y, x) (y on the grid; x already wrapped)."""
    for j in range(KPHI):
        out[j] = Er[rho[y, x], j] + Ec[cert[y, x], j]
    if i > 0:
        py, px = y // 2, x // 2
        for j in range(KPHI):
            out[j] += Epr[prho[py, px], j] + Epc[pcert[py, px], j]


@njit(cache=True)
def _learned_site(i, y, x, ch, rho, cert, prho, pcert, Er, Ec, Epr, Epc, th, m, D, out):
    """out[t] += Phi(z with channel ch at (y, x) = t) - const, t < D."""
    rows, cols = rho.shape
    k = KPHI
    a, S, A, W, b, v = _unpack(th, m)
    E = Er if ch == RHO else Ec
    cur = rho[y, x] if ch == RHO else cert[y, x]
    r = np.empty(k)
    _phi(i, y, x, rho, cert, prho, pcert, Er, Ec, Epr, Epc, r)
    for j in range(k):
        r[j] -= E[cur, j]
    # linear coefficient gv on u_t: a + 2 S r + sum_d A_d phi_{p+d} + A_d' phi_{p-d}
    gv = a.copy()
    for j in range(k):
        s = 0.0
        for l in range(k):
            s += S[j, l] * r[l]
        gv[j] += 2.0 * s
    ph = np.empty(k)
    for di in range(4):
        dy, dx = D_PLUS[di, 0], D_PLUS[di, 1]
        Ad = A[di]
        ny, nx = y + dy, (x + dx) % cols
        if 0 <= ny < rows:
            _phi(i, ny, nx, rho, cert, prho, pcert, Er, Ec, Epr, Epc, ph)
            for j in range(k):
                s = 0.0
                for l in range(k):
                    s += Ad[j, l] * ph[l]
                gv[j] += s
        ny, nx = y - dy, (x - dx) % cols
        if 0 <= ny < rows:
            _phi(i, ny, nx, rho, cert, prho, pcert, Er, Ec, Epr, Epc, ph)
            for l in range(k):
                s = 0.0
                for j in range(k):
                    s += ph[j] * Ad[j, l]
                gv[l] += s
    for t in range(D):
        s = 0.0
        for j in range(k):
            uj = E[t, j]
            if uj != 0.0:
                s += uj * gv[j]
                q = 0.0
                for l in range(k):
                    q += S[j, l] * E[t, l]
                s += uj * q
        out[t] += s
    if m == 0:
        return
    # heads of every q with p in its 3 x 3 window
    hr = np.empty(m)
    for dq in range(9):
        qy, qx = y - (dq // 3 - 1), (x - (dq % 3 - 1)) % cols
        if qy < 0 or qy >= rows:
            continue
        for j in range(m):
            hr[j] = b[j]
        for d2 in range(9):
            ny, nx = qy + d2 // 3 - 1, (qx + d2 % 3 - 1) % cols
            if ny < 0 or ny >= rows:
                continue
            if d2 == dq:
                for l in range(k):
                    ph[l] = r[l]
            else:
                _phi(i, ny, nx, rho, cert, prho, pcert, Er, Ec, Epr, Epc, ph)
            Wd = W[d2]
            for l in range(k):
                if ph[l] != 0.0:
                    for j in range(m):
                        hr[j] += Wd[l, j] * ph[l]
        Wd = W[dq]
        for t in range(D):
            s = 0.0
            for j in range(m):
                hj = hr[j]
                for l in range(k):
                    hj += Wd[l, j] * E[t, l]
                s += v[j] * (hj + np.log1p(np.exp(-hj)) if hj > 0 else np.log1p(np.exp(hj)))
            out[t] += s


@njit(cache=True)
def _designed_rho(i, h, y, x, t, rho, prho, fill, kappa, lam, beta):
    """Soft designed energy of rho_p = t (count prior of p's parent block,
    smoothness with both horizontal neighbours, top noise)."""
    rows, cols = rho.shape
    e = 0.0
    if i > 0:
        py, px = y // 2, x // 2
        s = t
        for yy in range(2 * py, 2 * py + 2):
            for xx in range(2 * px, 2 * px + 2):
                if yy != y or xx != x:
                    s += rho[yy, xx]
        d = s - prho[py, px]
        e += kappa * d * d / (h * h)
    e += lam * (abs(t - rho[y, (x - 1) % cols]) + abs(t - rho[y, (x + 1) % cols])) / h
    if i == 0:
        d = t - h * h * fill[y, x]
        e += beta * d * d / (h * h)
    return e


@njit(cache=True)
def _viol_cert(i, h, y, x, A, B, cert, Aof, Bof, pcert, pAof, pBof):
    """Hard violations of cert_p = (A, B) (support with the cells above and
    below, the row honour of p's row in its parent block)."""
    rows, cols = cert.shape
    n = 0
    if y > 0 and Bof[cert[y - 1, x]] > A:
        n += 1
    if y + 1 < rows and B > Aof[cert[y + 1, x]]:
        n += 1
    if i > 0:
        py, px = y // 2, x // 2
        xs = x ^ 1
        if A + Aof[cert[y, xs]] < pAof[pcert[py, px]]:
            n += 1
        if B + Bof[cert[y, xs]] > pBof[pcert[py, px]]:
            n += 1
    return n


@njit(cache=True)
def site_terms(i, h, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, fill, kappa, lam, beta,
               Er, Ec, Epr, Epc, th, m, learned, y, x, ch, soft, viol):
    """soft[t], viol[t] for every value t of channel ch at (y, x).  At the
    tile level ch is ignored: t sets rho and the certificate together."""
    tied = h == 1
    D = soft.shape[0]
    for t in range(D):
        soft[t] = 0.0
        viol[t] = 0
    if ch == RHO or tied:
        for t in range(D):
            soft[t] += _designed_rho(i, h, y, x, t, rho, prho, fill, kappa, lam, beta)
    if ch == CERT or tied:
        for t in range(D):
            viol[t] += _viol_cert(i, h, y, x, Aof[t], Bof[t], cert, Aof, Bof, pcert, pAof, pBof)
    if learned and not tied:
        _learned_site(i, y, x, ch, rho, cert, prho, pcert, Er, Ec, Epr, Epc, th, m, D, soft)


@njit(cache=True)
def _gumbel(e, n):
    best, arg = -np.inf, -1
    for j in range(n):
        if e[j] < np.inf:
            g = -e[j] - np.log(-np.log(np.random.random()))
            if g > best:
                best, arg = g, j
    return arg


@njit(cache=True)
def sweep_level(i, h, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, fill, kappa, lam, beta,
                Er, Ec, Epr, Epc, th, m, learned, n, seed):
    """n sweeps (rho then cert, raster order; one tied sweep at the tiles).
    Returns the number of site visits with no finite candidate."""
    np.random.seed(seed)
    rows, cols = rho.shape
    tied = h == 1
    Dr = h * h + 1
    Dc = Aof.shape[0]
    soft = np.empty(max(Dr, Dc))
    viol = np.empty(max(Dr, Dc), np.int64)
    e = np.empty(max(Dr, Dc))
    bad = 0
    for _ in range(n):
        for ch in range(1 if tied else 2):
            D = Dr if ch == RHO else Dc
            g = rho if ch == RHO else cert
            for y in range(rows):
                for x in range(cols):
                    site_terms(i, h, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, fill, kappa, lam, beta,
                               Er, Ec, Epr, Epc, th, m, learned, y, x, ch, soft[:D], viol[:D])
                    for t in range(D):
                        e[t] = soft[t] if viol[t] == 0 else np.inf
                    pick = _gumbel(e, D)
                    if pick < 0:
                        bad += 1
                    else:
                        g[y, x] = pick
    return bad


# ------------------------------------------------------------------ window energy
@njit(cache=True)
def _learned_total_window(i, rho, cert, prho, pcert, Er, Ec, Epr, Epc, th, m, y0, y1, x0, x1):
    """Phi terms touching any cell of the window rows [y0, y1), columns
    x0..x1-1 (mod cols), each once."""
    rows, cols = rho.shape
    k = KPHI
    a, S, A, W, b, v = _unpack(th, m)
    ph = np.empty(k)
    pn = np.empty(k)
    tot = 0.0
    wc = x1 - x0

    def inwin(yy, xx):
        return y0 <= yy < y1 and ((xx - x0) % cols) < wc

    # scan the window dilated by 2
    for yy in range(y0 - 2, y1 + 2):
        if yy < 0 or yy >= rows:
            continue
        for xo in range(-2, wc + 2):
            xx = (x0 + xo) % cols
            inside = inwin(yy, xx)
            _phi(i, yy, xx, rho, cert, prho, pcert, Er, Ec, Epr, Epc, ph)
            if inside:
                for j in range(k):
                    tot += a[j] * ph[j]
                    s = 0.0
                    for l in range(k):
                        s += S[j, l] * ph[l]
                    tot += ph[j] * s
            for di in range(4):
                ny, nx = yy + D_PLUS[di, 0], (xx + D_PLUS[di, 1]) % cols
                if ny < 0 or ny >= rows:
                    continue
                if not (inside or inwin(ny, nx)):
                    continue
                _phi(i, ny, nx, rho, cert, prho, pcert, Er, Ec, Epr, Epc, pn)
                for j in range(k):
                    s = 0.0
                    for l in range(k):
                        s += A[di, j, l] * pn[l]
                    tot += ph[j] * s
            if m > 0:
                touch = False
                for d2 in range(9):
                    if inwin(yy + d2 // 3 - 1, (xx + d2 % 3 - 1) % cols):
                        touch = True
                if touch:
                    hq = b[:m].copy()
                    for d2 in range(9):
                        ny, nx = yy + d2 // 3 - 1, (xx + d2 % 3 - 1) % cols
                        if ny < 0 or ny >= rows:
                            continue
                        _phi(i, ny, nx, rho, cert, prho, pcert, Er, Ec, Epr, Epc, pn)
                        for l in range(k):
                            if pn[l] != 0.0:
                                for j in range(m):
                                    hq[j] += W[d2, l, j] * pn[l]
                    for j in range(m):
                        hj = hq[j]
                        tot += v[j] * (hj + np.log1p(np.exp(-hj)) if hj > 0 else np.log1p(np.exp(hj)))
    return tot


@njit(cache=True)
def window_terms(i, h, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, fill, kappa, lam, beta,
                 Er, Ec, Epr, Epc, th, m, learned, by, bx):
    """(soft, violations) of every term touching the 2 x 2 children of parent
    block (by, bx) at level i, each once.  Needs at least 6 columns at level
    i (the window and its reach must not wrap onto themselves)."""
    rows, cols = rho.shape
    tied = h == 1
    y0, x0 = 2 * by, 2 * bx
    soft = 0.0
    nv = 0
    # count prior of the block
    s = rho[y0, x0] + rho[y0, x0 + 1] + rho[y0 + 1, x0] + rho[y0 + 1, x0 + 1]
    d = s - prho[by, bx]
    soft += kappa * d * d / (h * h)
    for y in range(y0, y0 + 2):
        # smoothness: the pair inside and the two to the halo
        soft += lam * (abs(rho[y, x0] - rho[y, x0 + 1]) + abs(rho[y, x0] - rho[y, (x0 - 1) % cols])
                       + abs(rho[y, x0 + 1] - rho[y, (x0 + 2) % cols])) / h
        # row honour
        pa, pb = pAof[pcert[by, bx]], pBof[pcert[by, bx]]
        if Aof[cert[y, x0]] + Aof[cert[y, x0 + 1]] < pa:
            nv += 1
        if Bof[cert[y, x0]] + Bof[cert[y, x0 + 1]] > pb:
            nv += 1
    for x in range(x0, x0 + 2):
        # support: above the block, inside, below
        if y0 > 0 and Bof[cert[y0 - 1, x]] > Aof[cert[y0, x]]:
            nv += 1
        if Bof[cert[y0, x]] > Aof[cert[y0 + 1, x]]:
            nv += 1
        if y0 + 2 < rows and Bof[cert[y0 + 1, x]] > Aof[cert[y0 + 2, x]]:
            nv += 1
    if learned and not tied:
        soft += _learned_total_window(i, rho, cert, prho, pcert, Er, Ec, Epr, Epc, th, m, y0, y0 + 2, x0, x0 + 2)
    return soft, nv


# ------------------------------------------------------------------ AIS / exact on one child block
@njit(cache=True)
def block_logz_exact_tiles(i, h, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, fill, kappa, lam, beta,
                           Er, Ec, Epr, Epc, th, m, learned, by, bx, Lcap):
    """log Z of the 2 x 2 tiles of block (by, bx), by enumeration (tiles:
    16 states), a violation costing Lcap (inf: forbidden), the same
    convention as block_logz_ais; the block is restored."""
    y0, x0 = 2 * by, 2 * bx
    save = rho[y0:y0 + 2, x0:x0 + 2].copy()
    es = np.empty(16)
    for c in range(16):
        rho[y0, x0] = c & 1
        rho[y0, x0 + 1] = (c >> 1) & 1
        rho[y0 + 1, x0] = (c >> 2) & 1
        rho[y0 + 1, x0 + 1] = (c >> 3) & 1
        s, nv = window_terms(i, h, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, fill, kappa, lam, beta,
                             Er, Ec, Epr, Epc, th, m, learned, by, bx)
        es[c] = s if nv == 0 else s + Lcap * nv
    rho[y0:y0 + 2, x0:x0 + 2] = save
    mn = es.min()
    if mn == np.inf:
        return -np.inf
    return -mn + np.log(np.exp(-(es - mn)).sum())


@njit(cache=True)
def block_logz_ais(i, h, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, fill, kappa, lam, beta,
                   Er, Ec, Epr, Epc, th, m, learned, by, bx, K, M, Lcap, seed):
    """log Z of the 2 x 2 block's rho and cert (8 variables) by AIS from the
    uniform distribution, E_beta = beta (soft + L viol), linear schedule, K
    steps, M chains.  Returns (log Z estimate, log weights).  The block is
    restored."""
    np.random.seed(seed)
    y0, x0 = 2 * by, 2 * bx
    srho = rho[y0:y0 + 2, x0:x0 + 2].copy()
    scert = cert[y0:y0 + 2, x0:x0 + 2].copy()
    Dr = h * h + 1
    Dc = Aof.shape[0]
    logz0 = 4.0 * (np.log(Dr) + np.log(Dc))
    soft = np.empty(max(Dr, Dc))
    viol = np.empty(max(Dr, Dc), np.int64)
    e = np.empty(max(Dr, Dc))
    logw = np.zeros(M)
    for c in range(M):
        for yy in range(y0, y0 + 2):
            for xx in range(x0, x0 + 2):
                rho[yy, xx] = np.random.randint(Dr)
                cert[yy, xx] = np.random.randint(Dc)
        lw = 0.0
        prev = 0.0
        for kk in range(1, K + 1):
            bt = kk / K
            s, nv = window_terms(i, h, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, fill, kappa, lam, beta,
                                 Er, Ec, Epr, Epc, th, m, learned, by, bx)
            lw -= (bt - prev) * (s + Lcap * nv)
            prev = bt
            for ch in range(2):
                D = Dr if ch == RHO else Dc
                g = rho if ch == RHO else cert
                for yy in range(y0, y0 + 2):
                    for xx in range(x0, x0 + 2):
                        site_terms(i, h, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, fill, kappa, lam, beta,
                                   Er, Ec, Epr, Epc, th, m, learned, yy, xx, ch, soft[:D], viol[:D])
                        for t in range(D):
                            e[t] = bt * (soft[t] + Lcap * viol[t])
                        g[yy, xx] = _gumbel(e, D)
        logw[c] = lw
    rho[y0:y0 + 2, x0:x0 + 2] = srho
    cert[y0:y0 + 2, x0:x0 + 2] = scert
    mx = logw.max()
    return logz0 + mx + np.log(np.exp(logw - mx).mean()), logw


# ------------------------------------------------------------------ noise, init, generation
def value_noise_2d(H, W, seed, octaves=((32, 1.0), (16, 0.5), (8, 0.25))):
    """(H, W) summed cosine-interpolated value noise, periodic in x."""
    rng = np.random.default_rng(seed)
    out = np.zeros((H, W))
    yy, xx = np.mgrid[:H, :W]
    for sp, amp in octaves:
        ny, nx = H // sp + 2, W // sp
        knots = rng.uniform(-1, 1, (ny, nx))
        iy, fy = yy // sp, (yy % sp) / sp
        ix, fx = xx // sp, (xx % sp) / sp
        wy, wx = (1 - np.cos(np.pi * fy)) / 2, (1 - np.cos(np.pi * fx)) / 2
        k00 = knots[iy, ix % nx]
        k01 = knots[iy, (ix + 1) % nx]
        k10 = knots[iy + 1, ix % nx]
        k11 = knots[iy + 1, (ix + 1) % nx]
        out += amp * ((1 - wy) * ((1 - wx) * k00 + wx * k01) + wy * ((1 - wx) * k10 + wx * k11))
    return out


def density_field(H, W, seed, grad=48.0, amp=1.0):
    """The tile density d(y, x) = (y - H/2) / grad + amp * noise; solid where
    d > 0.  Its 2D noise makes overhangs and floating pieces."""
    yy = np.arange(H)[:, None]
    return (yy - H / 2) / grad + amp * value_noise_2d(H, W, seed)


def set_fill(world, d, sharp=8.0):
    """Top-level fill targets: block means of sigmoid(sharp d)."""
    f = 1 / (1 + np.exp(-sharp * d))
    H, W = d.shape
    world.fill[:] = f.reshape(H // 16, 16, W // 16, 16).mean((1, 3))


def _fillpos(n, h, left):
    """(left child, right child) of n columns of a 2h-wide block, packed to
    one side."""
    a, b = min(n, h), max(0, n - h)
    return (a, b) if left else (b, a)


def refine(world, i, rng):
    """Level i (>= 1) as a consistent refinement of level i - 1: certificates
    packed to one side per parent column (the side drawn per column so that
    vertically stacked parents agree), top child row A = pack(A_P),
    B = pack(B_P), bottom row A = B = pack(B_P); rho split evenly."""
    h = HS[i]
    P = i - 1
    pr, pc = world.rho[P].shape
    side = rng.random(pc) < 0.5
    Aof, Bof = world.Aof[i], world.Bof[i]
    index = {(int(a), int(b)): j for j, (a, b) in enumerate(zip(Aof, Bof))}
    for py in range(pr):
        for px in range(pc):
            cA, cB = world.Aof[P][world.cert[P][py, px]], world.Bof[P][world.cert[P][py, px]]
            top = list(zip(_fillpos(cA, h, side[px]), _fillpos(cB, h, side[px])))
            bot = [(b, b) for b in _fillpos(cB, h, side[px])]
            for r, row in enumerate((top, bot)):
                for c, (a, b) in enumerate(row):
                    y, x = 2 * py + r, 2 * px + c
                    if h == 1:
                        world.rho[i][y, x] = a                   # tiles: the top row takes A, the bottom B
                    else:
                        world.cert[i][y, x] = index[(a, b)]
            if h > 1:
                q, rem = divmod(int(world.rho[P][py, px]), 4)
                vals = np.full(4, q)
                vals[rng.permutation(4)[:rem]] += 1
                world.rho[i][2 * py:2 * py + 2, 2 * px:2 * px + 2] = np.minimum(vals.reshape(2, 2), h * h)


def init_top(world):
    world.cert[0][:] = 0                                     # (A, B) = (0, 0): valid everywhere
    world.rho[0][:] = np.round(world.fill * 256).astype(np.int64)


def generate(world, sweeps=20, seed=0, upto=NL - 1):
    """Top-down: the top from its init, every lower level from a refinement,
    `sweeps` sweeps each.  Returns no-candidate counts per level."""
    rng = np.random.default_rng(seed)
    bad = []
    init_top(world)
    for i in range(upto + 1):
        if i > 0:
            refine(world, i, rng)
        bad.append(int(sweep_level(*world.args(i), sweeps, int(rng.integers(1 << 30)))))
    return bad


# ------------------------------------------------------------------ metrics
def overhangs(tiles):
    """Solid tiles directly above air."""
    return int((tiles[:-1].astype(bool) & ~tiles[1:].astype(bool)).sum())


def block_exact(tiles, h):
    """(rho, A, B) of the tiles in each h x h block: solid count, full
    columns, occupied columns."""
    H, W = tiles.shape
    t = tiles.reshape(H // h, h, W // h, h)
    col = t.sum(1)                                           # (H/h, W/h, h): solid per column
    return t.sum((1, 3)), (col == h).sum(2), (col > 0).sum(2)


def metrics(world):
    tiles = world.rho[NL - 1]
    out = dict(overhangs=overhangs(tiles))
    for i, h in enumerate(HS[:-1]):
        r, A, B = block_exact(tiles, h)
        pr = world.rho[i]
        pA, pB = world.Aof[i][world.cert[i]], world.Bof[i][world.cert[i]]
        out[f"h{h}"] = dict(
            rho_err=float(np.abs(r - pr).mean() / (h * h)),          # tiles realised vs this level's density
            A_gap=float((A - pA).mean() / h), B_gap=float((pB - B).mean() / h),
            A_kept=float((A >= pA).mean()), B_kept=float((B <= pB).mean()),
            cert_vs_rho=float(np.corrcoef(pA + pB, pr)[0, 1]) if pr.std() > 0 and (pA + pB).std() > 0 else np.nan)
    out["fill_err"] = float(np.abs(block_exact(tiles, 16)[0] / 256 - world.fill).mean())
    return out


def render(world, px=2):
    t = world.rho[NL - 1]
    col = np.where(t[..., None] == 1, np.array([138, 98, 66], np.uint8), np.array([169, 212, 240], np.uint8))
    return np.repeat(np.repeat(col, px, 0), px, 1)
