"""The Channel-sampler interface of notes/dsl_interface.md for a table
channel on the numba kernel (kernel.sweep_cap over core.Model.compile(home,
hard_first=True)).  Generic: nothing here knows a channel set.

Sampler(model, home, K, hb, seed)
  init()            dormancy (dsl_updates C1): A_p = {t : E_par,p(t) < inf}, E_par the
                    hard rows that read other channels (parents, painted channels) or
                    are unaries; |A_p| = 1 sets z_p and marks p dormant; an hb x hb
                    block of dormant or fixed sites is inactive.  Returns the dormant
                    fraction.  Call again after the parents change.
  candidates(y, x)  C_p: the admissible values (every hard row), or with K the kernel's
                    capped set {z_p} + (K - 1) admissible values != z_p uniformly
                    without replacement.  Gibbs over C_p leaves exp(-E) invariant since
                    P(C | z) is the same for every admissible z in C.
  sweep(n, T)       the kernel with the cap and the active / dormant / fixed skips
  energy()          (finite total, inf count) of the rows homed on `home` (Model.energy)
  site_energies(y, x)   (D,) energies, inf where a hard row forbids
  sweep_tempered(beta, n), unary_logZ(), energy_rest()   AIS's kernel (induce._Anneal):
                    E_beta = E_un + beta E_rest, E_un the unaries of home at full
                    strength, E_rest every other row with inf -> L
  ais_log_z(...)    induce.ais_log_z on this sampler's kernel: free cells are those
                    neither fixed nor dormant
  relax(n, soft_model)  n sweeps under soft_model's packing of home (parents
                    softened by the caller), dormant sites kept

Seeding: each call seeds the numba RNG from self.rng (or from `seed` when
given, as Model.sweep does), so a Sampler is reproducible whatever ran
before it.

Certificates (K = None only): dormancy keeps a site active when its one
admitted value has mass > 0 (its d still has to be drawn); a dormant site
holds a mass-0 value with d = INF, valid with no witness, so active
neighbours see a dead end (dsl_updates C1).

ChannelSampler is the protocol every channel's sampler meets (Sampler here,
coord.CoordSampler for exemplar coordinates); generate() is the schedule of
dsl_interface.md over it, with no per-model cases."""
import numpy as np

from . import kernel
from .core import PAIR, UNARY
from .induce import betas_of, logmeanexp


class ChannelSampler:
    """The sampler a channel owns (dsl_interface.md, Channel).  Required:
      name                    the home channel
      init() -> float         dormancy where the parents admit one value; the dormant fraction
      sweep(n, T, seed)       n sweeps of the channel's own kernel; returns the no-candidate count
      energy()                (finite total, inf count) of the terms the kernel samples
      candidates(y, x)        the set the kernel draws over at (y, x)
    Optional (NotImplementedError where the kernel cannot): relax(n, ..., seed),
    sweep_tempered(beta, n, ...), unary_logZ(), energy_rest()."""
    name = ""

    def init(self):
        return 0.0

    def sweep(self, n=1, T=1.0, seed=None):
        raise NotImplementedError

    def energy(self):
        raise NotImplementedError

    def candidates(self, y, x):
        raise NotImplementedError

    def relax(self, n, *a, seed=None, **kw):
        raise NotImplementedError(f"{type(self).__name__}: no relaxation")

    def sweep_tempered(self, beta, n=1, *a, **kw):
        raise NotImplementedError(f"{type(self).__name__}: no tempered kernel")

    def unary_logZ(self, *a, **kw):
        raise NotImplementedError(f"{type(self).__name__}: no unary log Z")

    def energy_rest(self, *a, **kw):
        raise NotImplementedError(f"{type(self).__name__}: no E_rest")


def generate(levels, S, rng=None, painters=None, relax=None):
    """The forward chain of dsl_interface.md: for each level (coarsest first),
    painters[l]() (paints level l's inputs from the levels above: designed
    channels, refinements; None to skip), then for each sampler init(),
    S[l] sweeps, relax[l] relaxation sweeps.  S[l] is an int (one seed per
    call from rng, or the sampler's own rng when rng is None) or a list of
    (n, seed) chunks (replays a script that reseeds between reports).
    Returns the dormant fraction of every sampler, per level."""
    out = []
    for l, level in enumerate(levels):
        if painters is not None and painters[l] is not None:
            painters[l]()
        fr = []
        for c in level:
            fr.append(c.init())
            chunks = S[l] if not np.isscalar(S[l]) else [(int(S[l]), None if rng is None else int(rng.integers(1 << 30)))]
            for n, sd in chunks:
                c.sweep(n, seed=sd)
            if relax is not None and relax[l]:
                c.relax(relax[l], seed=None if rng is None else int(rng.integers(1 << 30)))
        out.append(fr)
    return out


class Sampler(ChannelSampler):
    def __init__(self, model, home, K=None, hb=1, seed=0, soft_model=None):
        self.model, self.home, self.hb = model, home, int(hb)
        self.name, self.soft_model = home, soft_model
        self.K = int(K) if K else 0
        self.hc = model.chan(home)
        self.D = self.hc.D
        self.rng = np.random.default_rng(seed)
        self.P = self._pack(model)
        rows, cols = self.hc.grid.shape
        self.dormant = np.zeros((rows, cols), bool)
        self.active = np.ones((-(-rows // self.hb), -(-cols // self.hb)), bool)
        self._soft = {}
        self._temper = {}

    def _pack(self, model):
        P = model.compile(self.home, hard_first=True)
        assert np.shares_memory(P.grids[P.home], self.hc.grid), "home grid must be contiguous int32"
        assert not (self.K and P.cert[4]), "candidate cap: not for certificate channels"
        return P

    def refresh(self):
        """Recompile after the model's tables or channel arrays were replaced."""
        self.P = self._pack(self.model)
        self._temper = {}

    def _seed(self, seed=None):
        kernel.seed(int(self.rng.integers(1 << 30)) if seed is None else int(seed))

    def _run(self, P, tabs, n, T=1.0):
        bad = 0
        fixed = np.ascontiguousarray(self.hc.fixed)
        for _ in range(n):
            bad = kernel.sweep_cap(P.home, P.grids, P.hs, P.views, P.fac, tabs, fixed, self.dormant, self.active,
                                   self.hb, P.colours, P.ncol, P.cert, P.joins, P.delta, float(T), P.convs, P.nhard,
                                   self.K, self.D)
        return bad

    # ------------------------------------------------------------ C1
    def parent_rows(self):
        """Packed hard rows that dormancy reads: unaries, and pairs whose b is not home."""
        f = self.P.fac[:self.P.nhard]
        return np.flatnonzero((f[:, 0] == UNARY) | ((f[:, 0] == PAIR) & (f[:, 1] != self.P.home))).astype(np.int64)

    def admissible(self):
        """(rows, cols, D) bool: A_p under the parent rows."""
        P = self.P
        return kernel.admissible(P.home, P.grids, P.hs, P.views, P.fac, P.tabs, P.convs, self.parent_rows(), self.D)

    def init(self):
        g, fixed, hb = self.hc.grid, self.hc.fixed, self.hb
        P = self.P
        ok = self.admissible()
        one, val = ok.sum(axis=2) == 1, ok.argmax(axis=2)
        if P.cert[4]:
            one &= P.views[P.cert[0]][val] == 0                  # a value with mass still draws its d
        self.dormant[:] = one & ~fixed
        g[self.dormant] = val[self.dormant]
        if P.cert[4]:
            P.grids[P.cert[2]][self.dormant] = P.cert[3] + 1      # d = INF: valid with no witness
        off = self.dormant | fixed
        rows, cols = g.shape
        for by in range(self.active.shape[0]):
            for bx in range(self.active.shape[1]):
                self.active[by, bx] = not off[by * hb:(by + 1) * hb, bx * hb:(bx + 1) * hb].all()
        return float(self.dormant.mean())

    # ------------------------------------------------------- sampling
    def candidates(self, y, x):
        P = self.P
        if self.K:
            self._seed()
        cand, e = kernel.site_candidates(y, x, P.home, P.grids, P.hs, P.views, P.fac, P.tabs, P.convs, P.nhard,
                                         self.K, self.D)
        return cand if self.K else cand[np.isfinite(e)]

    def sweep(self, n=1, T=1.0, seed=None):
        """n sweeps; returns the last sweep's count of sites with no finite candidate."""
        self._seed(seed)
        return self._run(self.P, self.P.tabs, n, T)

    def relax(self, n, soft_model=None, seed=None):
        """n sweeps under soft_model's packing of home (same channel objects;
        None: the one given to the constructor)."""
        soft_model = self.soft_model if soft_model is None else soft_model
        assert soft_model is not None, "relax: no soft model"
        P = self._soft.get(id(soft_model))
        if P is None:
            P = self._soft[id(soft_model)] = self._pack(soft_model)
        self._seed(seed)
        return self._run(P, P.tabs, n)

    def energy(self):
        P = self.P
        return kernel.total_energy(P.home, P.grids, P.hs, P.views, P.fac, P.tabs, P.cert, P.joins, P.delta, P.convs)

    def site_energies(self, y, x):
        P = self.P
        return kernel.site_candidates(y, x, P.home, P.grids, P.hs, P.views, P.fac, P.tabs, P.convs, P.nhard, 0,
                                      self.D)[1]

    # ------------------------------------------------------------ AIS
    def _tempering(self, L):
        """(full tabs with inf -> L on non-unary rows, anneal mask per tab, u (D,))."""
        if L not in self._temper:
            P = self.P
            assert P.cert[4] == 0, "AIS: certificates are not annealed"
            ann = np.zeros(len(P.tabs), bool)
            u = np.zeros(self.D)
            for f in range(P.fac.shape[0]):
                kind = P.fac[f, 0]
                assert kind in (0, 1, 2), f"row {f}: kind {kind} is not annealed"
                if kind == UNARY:
                    u += P.tabs[P.fac[f, 7]][P.views[P.fac[f, 2]], 0]
                else:
                    ann[P.fac[f, 7]] = True
            full = tuple(np.where(np.isinf(t), L, t) if a else t for t, a in zip(P.tabs, ann))
            self._temper[L] = (full, ann, u)
        return self._temper[L]

    def tempered_tabs(self, beta, L=30.0):
        full, ann, _ = self._tempering(L)
        return tuple(t * beta if a else t for t, a in zip(full, ann))

    def sweep_tempered(self, beta, n=1, L=30.0, seed=None):
        """n sweeps of the kernel for p_beta ~ exp(-E_un - beta E_rest)."""
        self._seed(seed)
        return self._run(self.P, self.tempered_tabs(beta, L), n)

    def energy_rest(self, L=30.0):
        full, ann, _ = self._tempering(L)
        P = self.P
        rest = tuple(t if a else np.zeros_like(t) for t, a in zip(full, ann))
        e, nv = kernel.total_energy(P.home, P.grids, P.hs, P.views, P.fac, rest, P.cert, P.joins, P.delta, P.convs)
        assert nv == 0
        return e

    def free(self):
        return ~(self.hc.fixed | self.dormant)

    def unary_logZ(self, L=30.0):
        """log Z_0 = sum over free sites of log sum_t exp(-u(t)), minus u at the
        fixed and dormant sites (induce._Anneal's convention)."""
        u = self._tempering(L)[2]
        fin = np.isfinite(u)
        assert fin.any(), "every value has an infinite unary"
        umin = u[fin].min()
        free = self.free()
        return float(free.sum() * (np.log(np.exp(-(u - umin)).sum()) - umin) - u[self.hc.grid[~free]].sum())

    def ais_log_z(self, K=32, M=16, seed=0, L=30.0, schedule="linear"):
        """(log Z, log w (M,)): induce.ais_log_z with this sampler's kernel
        (cap, dormancy); the same draws as induce.ais_log_z when K is None
        and nothing is dormant.  Free cells are overwritten."""
        _, _, u = self._tempering(L)
        logZ0 = self.unary_logZ(L)
        betas = betas_of(K, schedule)
        tabs = [self.tempered_tabs(b, L) for b in betas[:-1]]
        rng = np.random.default_rng(seed)
        g = self.hc.grid
        fy, fx = np.nonzero(self.free())
        p = np.exp(-(u - u[np.isfinite(u)].min()))
        cdf = np.cumsum(p / p.sum())
        logw = np.zeros(M)
        for m in range(M):
            g[fy, fx] = np.minimum(np.searchsorted(cdf, rng.random(len(fy)) * cdf[-1], side="right"),
                                   len(cdf) - 1).astype(np.int32)
            kernel.seed(int(rng.integers(1 << 30)))
            lw = 0.0
            for k in range(K):
                self._run(self.P, tabs[k], 1)
                lw -= (betas[k + 1] - betas[k]) * self.energy_rest(L)
            logw[m] = lw
        return logZ0 + logmeanexp(logw), logw
