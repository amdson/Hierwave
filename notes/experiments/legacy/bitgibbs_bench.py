"""Benchmarks of the bit-sliced sampler (castlegen/bitgibbs.py), single core.

1. toy: random level tables, S tiles, F factors: frequencies of 2e6 draws
   against the exact distribution (max |z| over tiles) and ns per draw.
2. pairs: the demo tile set's pair MRF (E_pair - logz) on a 128^2 torus from
   a random start, SW sweeps: the C sampler (energies quantized, unit ln 2 or
   ln 2 / 2, each also Metropolis-Hastings-corrected to the float energies) against a numba float heat-bath; ns per site and the final exact
   E_pair - logz per cell (both chains target nearly the same p).
3. field: block_field.py's single-block cases (castle exemplar, parent term,
   maze-style port patterns), the C field sampler (tile | fields bit-sliced,
   then (g, d) | tiles exactly, validity hard) against blockfield.sweep (the
   numba joint closed form): success (blockconn's exact score 0) and ms per
   sweep, same schedule (lam 0 -> LAMAX over SW, HOLD more).
PARTS (default "toy,pairs,field") picks parts."""
import os, time
import numpy as np
from numba import njit
from castlegen import bitgibbs as BG
from castlegen.legacy import blockconn as BC, blockfield as BF, tileset

PARTS = os.environ.get("PARTS", "toy,pairs,field").split(",")
LN2 = float(np.log(2))
ts = tileset.load("demo")
t = ts.np_tables
Eh, Ev, logz = (np.ascontiguousarray(t[k], np.float64) for k in ("Eh", "Ev", "logz"))
S = ts.n_sig

# ------------------------------------------------------------------ 1. toy
if "toy" in PARTS:
    rng = np.random.default_rng(0)
    print("toy: S, F -> max |z| (2e6 draws), ns per draw")
    for S_ in (16, 64, 128, 256):
        for F in (4, 8):
            L = rng.integers(0, 4, (F, S_))
            L[rng.random(L.shape) < 0.05] = -1
            stk = BG.stacks(L, 4)
            legal = BG.masks(np.ones(S_, bool))
            cw = BG.weights(LN2)
            p = BG.exact(L, np.ones(S_, bool), cw)
            n = 2_000_000
            BG.sample_many(stk, legal, cw, 1, 10)
            t0 = time.perf_counter()
            x = BG.sample_many(stk, legal, cw, 7, n)
            dt = time.perf_counter() - t0
            f = np.bincount(x[x >= 0], minlength=S_) / n
            z = np.abs(f - p)[p > 0] / np.sqrt(p * (1 - p) / n)[p > 0]
            print(f"  S {S_:3d} F {F}: max |z| {z.max():.2f}  {dt / n * 1e9:5.0f} ns")


# ------------------------------------------------------------------ 2. pairs
@njit(cache=True)
def nb_sweep(T, Eh, Ev, logz, sweeps):
    """Float heat-bath (Gumbel-max), checkerboard, torus."""
    H, W = T.shape
    S = logz.shape[0]
    for s in range(sweeps):
        for c in range(2):
            for y in range(H):
                for x in range((y + c) & 1, W, 2):
                    n0, n1 = T[(y - 1) % H, x], T[y, (x + 1) % W]
                    n2, n3 = T[(y + 1) % H, x], T[y, (x - 1) % W]
                    best, bt = -np.inf, 0
                    for tt in range(S):
                        v = -logz[tt] + Ev[n0, tt] + Eh[tt, n1] + Ev[tt, n2] + Eh[n3, tt]
                        g = -v - np.log(-np.log(np.random.random()))
                        if g > best:
                            best, bt = g, tt
                    T[y, x] = bt


def e_cell(T):
    return float((Eh[T, np.roll(T, -1, 1)].sum() + Ev[T, np.roll(T, -1, 0)].sum() - logz[T].sum()) / T.size)


if "pairs" in PARTS:
    N, SW = 128, int(os.environ.get("PSW", 50))
    T0 = np.random.default_rng(0).integers(0, S, (N, N)).astype(np.int32)
    print(f"pairs: demo tile set, {N}^2 torus, {SW} sweeps from random")
    T = T0.astype(np.int64)
    nb_sweep(T.copy(), Eh, Ev, logz, 1)
    t0 = time.perf_counter()
    nb_sweep(T, Eh, Ev, logz, SW)
    dt = time.perf_counter() - t0
    print(f"  numba float      {dt / SW / N ** 2 * 1e9:6.0f} ns/site  E/cell {e_cell(T):.3f}")
    for unit in (LN2, LN2 / 2):
        FB = 4
        pair = BG.pair_tables(Eh, Ev, unit, FB)
        unary = BG.stacks(BG.quantize(-logz, unit, FB), FB)
        legal = BG.masks(np.ones(S, bool))
        cw = BG.weights(unit)
        Ex = np.ascontiguousarray(np.stack([Ev, Eh.T, Ev.T, Eh]))
        lost = (BG.quantize(Ex, unit, FB) < 0) & np.isfinite(Ex)
        for mh in (False, True):
            BG.sweep_pairs(T0.copy(), pair, unary, legal, cw, 3, 1, (Ex, -logz) if mh else None, unit)   # warm-up
            T = T0.copy()
            t0 = time.perf_counter()
            a = BG.sweep_pairs(T, pair, unary, legal, cw, 3, SW, (Ex, -logz) if mh else None, unit)
            dt = time.perf_counter() - t0
            print(f"  C unit {unit:.3f}{' +MH' if mh else '    '} {dt / SW / N ** 2 * 1e9:6.0f} ns/site  E/cell {e_cell(T):.3f}"
                  + (f"  accept {a:.3f}" if mh else f"  (finite pair energies cut to forbidden: {lost.mean():.3f})"))

# ------------------------------------------------------------------ 3. field
if "field" in PARTS:
    K, NPAR, SEEDS = 16, int(os.environ.get("NPAR", 9)), int(os.environ.get("SEEDS", 3))
    SW, HOLD, LAMAX = int(os.environ.get("SW", 80)), int(os.environ.get("HOLD", 40)), float(os.environ.get("LAMAX", 8))
    G, D, MU, EPS, DELTA = 8, 96, 1.0, 0.1, 0.05
    UNIT = float(os.environ.get("UNIT", LN2))
    E = np.load("cache/castle_ex.npy").astype(np.int64)
    m = E.shape[0]
    tabs = BC.tables(ts)
    node, Dh, Dv, sock = tabs
    allowed = np.zeros(S, bool)
    allowed[np.unique(E)] = True
    PC = BC.parent_cost(E, S, 2)
    GRP = np.ones(BC.MLAB + 1, np.int64)
    yy, xx = np.mgrid[:K, :K]
    lam_of = lambda s: LAMAX * min(1.0, (s + 1) / SW)
    rng0 = np.random.default_rng(0)
    cases = []
    while len(cases) < NPAR:                                                # block_field.py's cases
        c = int(rng0.integers(m * m))
        Y, X = (c // m - K // 2 + yy) % m, (c % m - K // 2 + xx) % m
        Wt = np.ascontiguousarray(E[Y, X])
        sides = np.flatnonzero((BC.pattern(Wt, tabs).reshape(4, 2) > 0).any(1))
        if len(sides) < 2:
            continue
        down = rng0.choice(sides, int(rng0.integers(1, 3)), replace=False)
        lab = np.zeros(8, np.int64)
        for d in sides:
            lab[2 * d:2 * d + 2] = 1 if d in down else BC.OPEN
        cases.append(dict(lab=lab, W=Wt, pc=np.ascontiguousarray(PC[Y * m + X])))

    FB = 4
    pair = BG.pair_tables(Eh, Ev, UNIT, FB)
    J, openout, all_, seambad, nodem = BG.field_tables(ts, tabs)
    cw = BG.weights(UNIT)

    def site_masks(lab):
        """exits (K K, NW): nodes opening onto a class port; legal: allowed."""
        ex_ = np.zeros((K, K, S), bool)
        for d in range(4):
            for y in range(K):
                for x in range(K):
                    q = BF._port(y, x, d, K)
                    if q >= 0 and 0 < lab[q] < BC.OPEN:
                        ex_[y, x] |= node & sock[d]
        return BG.masks(ex_.reshape(K * K, S)), BG.masks(np.broadcast_to(allowed, (K * K, S)).copy())

    def gcost(lam):
        mu, eps = lam * MU, lam * MU * EPS
        g = np.arange(G + 2)[:, None]
        Eg = np.where(node[None], mu * g, eps * g)
        return BG.stacks(BG.quantize(Eg, UNIT, FB), FB)

    def run_c(cs, seed):
        T = cs["W"].astype(np.int32).copy()
        Gf, Df = (a.astype(np.int32) for a in BF.init_fields(cs["W"].copy(), cs["lab"], tabs, G, D))
        unary = BG.stacks(BG.quantize((-logz[None, None] + cs["pc"]).reshape(K * K, S), UNIT, FB), FB)
        exits, legal = site_masks(cs["lab"])
        lab9 = np.concatenate([cs["lab"], [-1]]).reshape(1, 1, 9)
        t0 = time.perf_counter()
        for s in range(SW + HOLD):
            lam = lam_of(s)
            BG.sweep_field(T, Gf, Df, K, pair, unary, gcost(lam), J, legal, exits, openout, all_, seambad, lab9,
                           int(round(lam / UNIT)), 0, nodem, ts.WALL, G, D, lam * MU, lam * MU * EPS, DELTA, cw,
                           seed * 1000 + s, 1)
        dt = time.perf_counter() - t0
        return T.astype(np.int64), dt, BF.field_energy_ok(T.astype(np.int64), Gf.astype(np.int64),
                                                          Df.astype(np.int64), cs["lab"], tabs, G)

    def run_nb(cs, seed):
        rng = np.random.default_rng(seed)
        T = cs["W"].copy()
        Gf, Df = BF.init_fields(T, cs["lab"], tabs, G, D)
        t0 = time.perf_counter()
        for s in range(SW + HOLD):
            BF.sweep(T, Gf, Df, int(rng.integers(2 ** 31)), Eh, Ev, logz, allowed, cs["pc"], cs["lab"], *tabs,
                     lam_of(s), MU, 1000.0, EPS, DELTA, G, D, ts.WALL)
        return T, time.perf_counter() - t0, None

    print(f"field: {NPAR} cases x {SEEDS} seeds, {SW}+{HOLD} sweeps, lam -> {LAMAX}, unit {UNIT:.3f}")
    run_c(cases[0], 0)
    for name, fn in (("C bit-sliced", run_c), ("numba closed form", run_nb)):
        ok, tt, chk = [], [], []
        for ci, cs in enumerate(cases):
            for sd in range(SEEDS):
                T, dt, extra = fn(cs, 100 * ci + sd)
                ok.append(BC.parts(T, cs["lab"], GRP, *tabs, 0).sum() == 0)
                tt.append(dt / (SW + HOLD))
                if extra is not None:
                    chk.append(extra)
        line = f"  {name:18s} success {np.mean(ok):.3f}  {np.mean(tt) * 1e3:7.3f} ms/sweep ({np.mean(tt) / K ** 2 * 1e9:.0f} ns/site)"
        if chk:
            line += f"  fields: invalid {sum(c[0] for c in chk)}, nodes g > 0 {sum(c[1] for c in chk)}"
        print(line, flush=True)
