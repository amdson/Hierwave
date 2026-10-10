"""Training the certland predictors from a persistent subtree-move chain on
the designed distribution p* (notes/certland_chain_spec.md, section
certland_train; the plan doc's "One update").

State: one or more valid training worlds (forward generation by the shipped
sampler, then burn-in moves), predictors theta (certland_pred.Preds), a
replay archive.  One update:

1. per root level i in cfg.levels (top-down): certland_chain.move_level with
   cfg.K[i] proposals (cfg.anneal[i] = (M, L0) -> certland_anneal.annealed_move);
2. every state j of every move with wbar_j > 0 is recorded through the
   collect callback: for every non-tile cell of the subtree and both
   channels, (features in z_j, designed offsets with +inf on hard-excluded
   values in z_j, the value in z_j, wbar_j) -- the Gibbs conditional the
   shipped sampler uses, read from the states;
3. one exact tile sweep;
4. cfg.steps Adam steps per (level, channel) on the weighted cross-entropy
   over this update's examples plus an archive sample, then Preds.set; the
   update's examples join the archive (never stale: they are p* states
   whatever theta was; burn-in is not recorded);
5. logs per level, and every cfg.eval_every updates a held-out forward
   generation (certland_pred.generate_pred) with metrics and a picture.

Sign convention: an energy; the predictor conditional is
softmax(-(offs + b_theta(feats)))."""
import functools
import os
import time
from dataclasses import dataclass, field

import numpy as np
from numba import njit

from . import certland as cl
from . import certland_pred as cp

IMAGES = "/Users/amdson/dev/Hierwave/images"


def _jax():
    import jax
    jax.config.update("jax_enable_x64", True)
    return jax, jax.numpy


def _chain():
    from . import certland_chain as cc
    return cc


# ------------------------------------------------------------------ config
@dataclass
class Cfg:
    levels: tuple = (0, 1, 2, 3)          # root levels moved, top-down (i = 0..3 for h = 16..2)
    K: dict = field(default_factory=lambda: {0: 8, 1: 4, 2: 1, 3: 1})
    anneal: dict = field(default_factory=dict)   # {i: (M, L0)} -> annealed_move at that root level
    proposal: bool = False                # a proposal-only predictor preds.q (cp.QPreds, boundary context)
                                          # drives the subtree pass, trained on the pass's own examples
                                          # (keys (i + QOFF, ch)); preds (Gibbs examples only) ships
    steps: int = 50                       # Adam steps per (level, channel) per update
    lr: float = 1e-3
    l2: float = 1e-6
    batch: int = 1024                     # minibatch per Adam step (fixed shape: one compile per (i, ch))
    archive_sample: int = 8192            # archive examples joined to the update's pool
    archive_cap: int = 50_000             # examples per (i, ch) ...
    archive_floats: int = 10_000_000      # ... and at most this many offset floats per (i, ch)
    nh: int = 64
    burn: int = 5                         # burn-in rounds (moves at every level + tile sweep), not recorded
    gen_sweeps: int = 20                  # forward generation sweeps (init and eval)
    eval_every: int = 10
    eval_seeds: tuple = (1001, 1002)
    eval_HW: tuple = None                 # eval world size; None: the training world's
    tag: str = "run"
    ckpt: str = None                      # .npz path for predictor params; None: no checkpoint
    ckpt_every: int = 10
    seed: int = 0


# ------------------------------------------------------------------ example stores
class Examples:
    """Growable per-(i, ch) example lists (one update's fresh examples)."""

    def __init__(self):
        self.d = {}

    def add(self, i, ch, feats, offs, val, w, upd):
        self.d.setdefault((i, ch), []).append((feats, offs, val, w, upd))

    def get(self, i, ch):
        parts = self.d.get((i, ch))
        if not parts:
            return None
        return dict(feats=np.concatenate([p[0] for p in parts]), offs=np.concatenate([p[1] for p in parts]),
                    val=np.concatenate([p[2] for p in parts]), w=np.concatenate([p[3] for p in parts]),
                    upd=np.concatenate([p[4] for p in parts]))

    def keys(self):
        return sorted(k for k, v in self.d.items() if v)

    def count(self, i, ch):
        return sum(len(p[2]) for p in self.d.get((i, ch), []))


class Archive:
    """Replay archive per (i, ch): feats (N, NF) and offs (N, D) as float32
    (offs +inf on excluded values), val (N,), w (N,), upd (N,).  A ring
    buffer of capacity min(cap, floats // D) per (i, ch): the oldest
    examples go first.  Burn-in is never recorded; drop_before removes
    examples from earlier updates if wanted.  Also holds the optimiser state
    (`opt`, per (i, ch)) so that update() keeps the spec's signature."""

    def __init__(self, cap=50_000, floats=10_000_000):
        self.cap, self.floats = cap, floats
        self.buf = {}     # (i, ch) -> dict of arrays
        self.n = {}       # filled
        self.pos = {}     # next write
        self.opt = {}     # (i, ch) -> adam state

    def _alloc(self, i, ch, D, nf=None):
        c = int(max(1, min(self.cap, self.floats // D)))
        self.buf[(i, ch)] = dict(feats=np.zeros((c, cp.NF if nf is None else nf), np.float32), offs=np.zeros((c, D), np.float32),
                                 val=np.zeros(c, np.int64), w=np.zeros(c), upd=np.zeros(c, np.int64))
        self.n[(i, ch)] = 0
        self.pos[(i, ch)] = 0

    def capacity(self, i, ch):
        return len(self.buf[(i, ch)]["val"]) if (i, ch) in self.buf else 0

    def size(self, i, ch):
        return self.n.get((i, ch), 0)

    def add(self, i, ch, feats, offs, val, w, upd):
        val = np.asarray(val)
        N = len(val)
        if N == 0:
            return
        if (i, ch) not in self.buf:
            self._alloc(i, ch, offs.shape[1], feats.shape[1])
        b = self.buf[(i, ch)]
        c = len(b["val"])
        upd = np.broadcast_to(np.asarray(upd, np.int64), (N,))
        w = np.broadcast_to(np.asarray(w, np.float64), (N,))
        if N > c:                                   # keep the last c
            feats, offs, val, w, upd = feats[-c:], offs[-c:], val[-c:], w[-c:], upd[-c:]
            N = c
        idx = (self.pos[(i, ch)] + np.arange(N)) % c
        b["feats"][idx] = feats
        b["offs"][idx] = offs
        b["val"][idx] = val
        b["w"][idx] = w
        b["upd"][idx] = upd
        self.pos[(i, ch)] = int((self.pos[(i, ch)] + N) % c)
        self.n[(i, ch)] = min(c, self.n[(i, ch)] + N)

    def sample(self, i, ch, n, rng):
        """Up to n examples drawn uniformly without replacement (dict of
        arrays; feats and offs float64)."""
        m = self.size(i, ch)
        if m == 0:
            return None
        b = self.buf[(i, ch)]
        idx = rng.choice(m, min(n, m), replace=False) if n < m else np.arange(m)
        return dict(feats=b["feats"][idx].astype(np.float64), offs=b["offs"][idx].astype(np.float64),
                    val=b["val"][idx].copy(), w=b["w"][idx].copy(), upd=b["upd"][idx].copy())

    def drop_before(self, upd):
        """Remove examples recorded before update `upd` (compacts)."""
        for key, b in self.buf.items():
            m = self.n[key]
            keep = np.flatnonzero(b["upd"][:m] >= upd) if m < len(b["val"]) else np.flatnonzero(b["upd"] >= upd)
            for k in b:
                b[k][:len(keep)] = b[k][keep]
            self.n[key] = len(keep)
            self.pos[key] = len(keep) % len(b["val"])


# ------------------------------------------------------------------ recording kernel
_cell_features = cp.cell_features
_site_terms = cl.site_terms


@njit
def _record_block(i, h, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, has_parent, fill, kappa, lam, beta,
                  Er, Ec, Epr, Epc, y0, y1, x0, x1, ch, feats, offs, vals):
    """Features, designed offsets (+inf where a hard row excludes the value
    against the current cells) and values of channel ch over rows [y0, y1),
    columns [x0, x1) of level i, in raster order."""
    D = offs.shape[1]
    soft = np.empty(D)
    viol = np.empty(D, np.int64)
    th0 = np.zeros(1)
    g = rho if ch == 0 else cert
    k = 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            _cell_features(h, ch, y, x, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, has_parent, feats[k])
            _site_terms(i, h, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, fill, kappa, lam, beta,
                        Er, Ec, Epr, Epc, th0, 0, 0, y, x, ch, soft, viol)
            for t in range(D):
                offs[k, t] = soft[t] if viol[t] == 0 else np.inf
            vals[k] = g[y, x]
            k += 1


def domain(i, ch):
    h = cl.HS[i]
    return h * h + 1 if ch == cl.RHO else len(cl.cert_domain(h)[0])


def record_cells(world, l, ch, y0, y1, x0, x1):
    """(feats (n, NF), offs (n, D), vals (n,)) of channel ch over a block of
    level l (< 4) of the world as it stands."""
    a = world.args(l)
    i, h, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, fill, kappa, lam, beta, Er, Ec, Epr, Epc = a[:18]
    hp = int(l > 0)
    n = (y1 - y0) * (x1 - x0)
    feats = np.empty((n, cp.NF))
    offs = np.empty((n, domain(l, ch)))
    vals = np.empty(n, np.int64)
    _record_block(i, h, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, hp, fill, kappa, lam, beta,
                  Er, Ec, Epr, Epc, y0, y1, x0, x1, ch, feats, offs, vals)
    return feats, offs, vals


QOFF = 4      # store key (i + QOFF, ch): the proposal-only predictor's examples


class _Collect:
    def __init__(self, store, upd, proposal=False):
        self.store, self.upd, self.proposal = store, upd, proposal
        self.time = 0.0
        self.calls = 0

    def __call__(self, world, i, ry, rx, states, wbar):
        t0 = time.time()
        cc = _chain()
        save = cc.get_subtree(world, i, ry, rx)
        sl = [s for s in cc.subtree_slices(i, ry, rx) if s[0] < cl.NL - 1]
        acc = {}
        for j, st in enumerate(states):
            wj = float(wbar[j])
            if not wj > 0:
                continue
            cc.set_subtree(world, i, ry, rx, st)
            for (l, y0, y1, x0, x1) in sl:
                for ch in (cl.RHO, cl.CERT):
                    f, o, v = record_cells(world, l, ch, y0, y1, x0, x1)
                    if not np.isfinite(o[np.arange(len(v)), v]).all():
                        cc.set_subtree(world, i, ry, rx, save)
                        raise AssertionError(f"collect: state {j} of root ({i}, {ry}, {rx}) violates a hard row "
                                             f"at level {l} channel {ch}")
                    acc.setdefault((l, ch), []).append((f, o, v, np.full(len(v), wj)))
            if self.proposal:
                # the proposal-only predictor's examples (known flags + boundary), stored apart
                for (l, ch), (f, o, v) in cc.record_pass(world, i, ry, rx, st, proposal=True).items():
                    acc.setdefault((l + QOFF, ch), []).append((f, o, v, np.full(len(v), wj)))
            elif cp.KF:
                # the proposal's own context (known flags, placeholders): the same value and weight
                for (l, ch), (f, o, v) in cc.record_pass(world, i, ry, rx, st).items():
                    acc.setdefault((l, ch), []).append((f, o, v, np.full(len(v), wj)))
        cc.set_subtree(world, i, ry, rx, save)
        for (l, ch), parts in acc.items():
            f = np.concatenate([p[0] for p in parts])
            self.store.add(l, ch, f.astype(np.float32), np.concatenate([p[1] for p in parts]).astype(np.float32),
                           np.concatenate([p[2] for p in parts]), np.concatenate([p[3] for p in parts]),
                           np.full(len(f), self.upd, np.int64))
        self.calls += 1
        self.time += time.time() - t0


def collector(archive, upd, proposal=False):
    """collect(world, i, ry, rx, states, wbar) for move_level: records every
    state with wbar_j > 0 into `archive` (an Archive or Examples) and leaves
    the world as found.  .time accumulates the recording time.  Under the
    known-flag layout (cp.KF) each state gives two examples per cell and
    channel: the Gibbs one (all slots known, for the shipped sampler) and
    the proposal's own (certland_chain.record_pass: the pass's known mask,
    placeholders and bias-free energies), the same value and weight.
    With proposal=True the second example is instead the proposal-only
    predictor's (record_pass(proposal=True), width cp.NFQ), stored under
    (l + QOFF, ch)."""
    return _Collect(archive, upd, proposal)


# ------------------------------------------------------------------ loss and Adam
def _ce_sums(p, feats, offs, val, w):
    """(sum w * nll, sum w); offs +inf masks a value out; rows with w = 0
    contribute nothing (padding)."""
    jax, jnp = _jax()
    b = cp.jax_bias(p, feats)
    ok = jnp.isfinite(offs)
    logits = jnp.where(ok, -(jnp.where(ok, offs, 0.0) + b), -jnp.inf)
    lse = jax.scipy.special.logsumexp(logits, axis=1)
    lv = jnp.take_along_axis(logits, val[:, None], 1)[:, 0]
    nll = jnp.where(w > 0, lse - lv, 0.0)
    return jnp.sum(w * nll), jnp.sum(w)


def weighted_ce(p, feats, offs, val, w):
    """-sum w log softmax(-(offs + b_theta(feats)))[val] / sum w (JAX)."""
    _, jnp = _jax()
    s, ws = _ce_sums(p, feats, offs, val, w)
    return s / jnp.maximum(ws, 1e-300)


def ce_numpy(offs, val, w, b=None):
    """The same loss in numpy (b None: theta = 0)."""
    o = np.asarray(offs, np.float64)
    ok = np.isfinite(o)
    lg = np.where(ok, -(np.where(ok, o, 0.0) + (0.0 if b is None else b)), -np.inf)
    mx = lg.max(1, keepdims=True)
    lse = mx[:, 0] + np.log(np.exp(lg - mx).sum(1))
    nll = lse - lg[np.arange(len(val)), val]
    w = np.asarray(w, np.float64)
    return float((w * np.where(w > 0, nll, 0.0)).sum() / max(w.sum(), 1e-300))


@functools.lru_cache(maxsize=None)
def _jitted():
    jax, jnp = _jax()

    def obj(p, feats, offs, val, w, l2):
        s, ws = _ce_sums(p, feats, offs, val, w)
        ce = s / jnp.maximum(ws, 1e-300)
        reg = sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(p))
        return ce + l2 * reg, ce

    vg = jax.value_and_grad(obj, has_aux=True)

    @jax.jit
    def step(p, m, v, t, feats, offs, val, w, lr, l2):
        (_, ce), g = vg(p, feats, offs, val, w, l2)
        t = t + 1.0
        m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
        v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * b * b, v, g)
        p = jax.tree_util.tree_map(
            lambda x, a, b: x - lr * (a / (1 - 0.9 ** t)) / (jnp.sqrt(b / (1 - 0.999 ** t)) + 1e-8), p, m, v)
        return p, m, v, t, ce

    sums = jax.jit(_ce_sums)
    return step, sums


def _split(p):
    """(array leaves, static entries) of a param dict."""
    arr, static = {}, {}
    for k, v in p.items():
        if v is None or isinstance(v, (bool, int, float, str, np.integer)):
            static[k] = v
        else:
            arr[k] = np.asarray(v, np.float64)
    return arr, static


def _pad(batch, idx, n):
    """Rows idx of batch padded to n rows with w = 0, offs = 0."""
    k = len(idx)
    out = {}
    for key, dt in (("feats", np.float64), ("offs", np.float64), ("val", np.int64), ("w", np.float64)):
        a = batch[key]
        z = np.zeros((n,) + a.shape[1:], dt)
        z[:k] = a[idx]
        out[key] = z
    return out


def fit_steps(p, batch, steps, lr, adam_state=None, batch_size=None, rng=None, l2=0.0):
    """`steps` Adam steps on weighted_ce over `batch` (dict feats, offs,
    val, w), minibatches of batch_size rows drawn without replacement each
    step (None: the full batch every step; rows padded to a fixed shape).
    Returns (params dict as numpy, adam_state, mean CE over the steps)."""
    jax, jnp = _jax()
    step, _ = _jitted()
    rng = np.random.default_rng() if rng is None else rng
    arr, static = _split(p)
    N = len(batch["val"])
    bs = N if batch_size is None else batch_size
    if adam_state is None:
        adam_state = dict(m={k: np.zeros_like(v) for k, v in arr.items()},
                          v={k: np.zeros_like(v) for k, v in arr.items()}, t=0.0)
    P = {k: jnp.asarray(v) for k, v in arr.items()}
    M = {k: jnp.asarray(v) for k, v in adam_state["m"].items()}
    V = {k: jnp.asarray(v) for k, v in adam_state["v"].items()}
    T = jnp.asarray(float(adam_state["t"]))
    tot = 0.0
    for _ in range(steps):
        idx = rng.choice(N, bs, replace=False) if bs < N else np.arange(N)
        mb = _pad(batch, idx, bs)
        P, M, V, T, ce = step({**P}, M, V, T, jnp.asarray(mb["feats"]), jnp.asarray(mb["offs"]),
                              jnp.asarray(mb["val"]), jnp.asarray(mb["w"]), lr, l2)
        tot += float(ce)
    out = {k: np.asarray(v) for k, v in P.items()}
    out.update(static)
    st = dict(m={k: np.asarray(v) for k, v in M.items()}, v={k: np.asarray(v) for k, v in V.items()}, t=float(T))
    return out, st, tot / max(steps, 1)


def eval_ce(p, batch, chunk=4096):
    """weighted_ce over a batch of any size, in fixed-size chunks (p None:
    theta = 0, numpy)."""
    if p is None:
        return ce_numpy(batch["offs"], batch["val"], batch["w"])
    _, sums = _jitted()
    _, jnp = _jax()
    arr, static = _split(p)
    P = {k: jnp.asarray(v) for k, v in arr.items()}
    N = len(batch["val"])
    s = ws = 0.0
    for a in range(0, N, chunk):
        idx = np.arange(a, min(N, a + chunk))
        mb = _pad(batch, idx, chunk)
        u, v = sums(P, jnp.asarray(mb["feats"]), jnp.asarray(mb["offs"]), jnp.asarray(mb["val"]),
                    jnp.asarray(mb["w"]))
        s += float(u)
        ws += float(v)
    return s / max(ws, 1e-300)


# ------------------------------------------------------------------ chain init
def set_noise(world, seed):
    cl.set_fill(world, cl.density_field(world.H, world.W, seed))


@njit
def _count_viol_level(i, h, rho, cert, Aof, Bof, prho, pcert, pAof, pBof):
    """Hard violations of the current certificates at level i (each counted
    from each end; zero iff valid)."""
    rows, cols = cert.shape
    n = 0
    for y in range(rows):
        for x in range(cols):
            c = cert[y, x]
            n += cl._viol_cert(i, h, y, x, Aof[c], Bof[c], cert, Aof, Bof, pcert, pAof, pBof)
    return n


def count_violations(world):
    """Per level, the number of hard-row violations in the world's state."""
    out = []
    for i in range(cl.NL):
        par = i - 1 if i > 0 else i
        out.append(int(_count_viol_level(i, cl.HS[i], world.rho[i], world.cert[i], world.Aof[i], world.Bof[i],
                                         world.rho[par], world.cert[par], world.Aof[par], world.Bof[par])))
    return out


def _move_fn(cfg, i):
    if i in cfg.anneal:
        from . import certland_anneal as ca
        M, L0 = cfg.anneal[i]
        return functools.partial(ca.annealed_move, M=M, L0=L0)
    return None


def init_chain(world, preds, noise_seed, burn, rng, cfg=None, tries=8):
    """Noise field from noise_seed, one forward generation by the shipped
    sampler (retrying generation seeds until the state is valid), then
    `burn` rounds of moves at cfg.levels (+ a tile sweep), not recorded.
    Returns dict(bad=generation stuck counts, viol, time)."""
    cc = _chain()
    cfg = Cfg() if cfg is None else cfg
    t0 = time.time()
    world.params = [None] * (cl.NL - 1)
    set_noise(world, noise_seed)
    for k in range(tries):
        bad = cp.generate_pred(world, preds, sweeps=cfg.gen_sweeps, seed=int(rng.integers(1 << 30)))
        viol = count_violations(world)
        if sum(viol) == 0:
            break
    else:
        raise RuntimeError(f"init_chain: no valid forward generation in {tries} tries (viol {viol})")
    for _ in range(burn):
        for i in cfg.levels:
            cc.move_level(world, preds, i, cfg.K[i], rng, collect=None, move=_move_fn(cfg, i))
        cc.tile_sweep(world, rng)
    return dict(bad=[int(b) for b in bad], viol=viol, time=time.time() - t0)


def make_chains(cfg, preds, H, W, noise_seeds, rng, log=print, **world_kw):
    """Training worlds of size H x W, one per noise seed, initialised."""
    worlds = []
    for s in noise_seeds:
        w = cl.World(H, W, **world_kw)
        info = init_chain(w, preds, s, cfg.burn, rng, cfg)
        log(f"init chain seed {s}: gen stuck {info['bad']} burn {cfg.burn} in {info['time']:.1f}s")
        worlds.append(w)
    return worlds


# ------------------------------------------------------------------ one update
def _params_or_init(preds, i, ch, seed):
    p = preds.params(i, ch)
    return cp.init_params(i, ch, preds.nh, seed, preds.nf) if p is None else p


def _mover(preds, cfg):
    """The predictor set that drives the subtree pass."""
    if cfg.proposal:
        if getattr(preds, "q", None) is None:
            preds.q = cp.QPreds(preds.nh)
        return preds.q
    return preds


def update(world, preds, archive, cfg, upd, rng):
    """The plan's "One update" over one world or a list of worlds.  Returns
    a log dict: per root level ess / changed / dead / n and move time; per
    (i, ch) example counts and losses (loss_pre with the incoming theta on
    the pool, loss after the steps, loss0 at theta = 0, fit = mean minibatch
    CE); timings per phase."""
    cc = _chain()
    worlds = world if isinstance(world, (list, tuple)) else [world]
    fresh = Examples()
    col = collector(fresh, upd, cfg.proposal)
    mp = _mover(preds, cfg)
    log = dict(upd=upd, levels={}, fit={}, time={})
    tm = time.time()
    for i in cfg.levels:
        t0 = time.time()
        rec0 = col.time
        stats = []
        for w in worlds:
            stats.append(cc.move_level(w, mp, i, cfg.K[i], rng, collect=col, move=_move_fn(cfg, i)))
        n = sum(s["n"] for s in stats)
        log["levels"][i] = dict(
            ess=float(sum(s["ess"] * s["n"] for s in stats) / max(n, 1)),
            changed=float(np.mean([s["changed"] for s in stats])),
            dead=float(np.mean([s["dead"] for s in stats])), n=n, K=cfg.K[i],
            time=time.time() - t0, record=col.time - rec0)
    log["time"]["moves"] = time.time() - tm - col.time
    log["time"]["record"] = col.time
    t0 = time.time()
    for w in worlds:
        cc.tile_sweep(w, rng)
    log["time"]["tiles"] = time.time() - t0
    t0 = time.time()
    for (i, ch) in fresh.keys():
        new = fresh.get(i, ch)
        old = archive.sample(i, ch, cfg.archive_sample, rng)
        pool = new if old is None else {k: np.concatenate([new[k].astype(old[k].dtype), old[k]]) for k in new}
        pool["feats"] = pool["feats"].astype(np.float64)
        pool["offs"] = pool["offs"].astype(np.float64)
        tgt, li = (mp, i - QOFF) if i >= QOFF else (preds, i)
        p0 = _params_or_init(tgt, li, ch, cfg.seed + 101 * i + ch)
        pre = eval_ce(p0, pool)
        p, st, fit = fit_steps(p0, pool, cfg.steps, cfg.lr, archive.opt.get((i, ch)), cfg.batch, rng, cfg.l2)
        archive.opt[(i, ch)] = st
        tgt.set(li, ch, p)
        log["fit"][(i, ch)] = dict(n_new=len(new["val"]), n_arch=0 if old is None else len(old["val"]),
                                   w_new=float(new["w"].sum()), loss_pre=pre, loss=eval_ce(p, pool),
                                   loss0=eval_ce(None, pool), fit=fit)
        archive.add(i, ch, new["feats"], new["offs"], new["val"], new["w"], new["upd"])
    log["time"]["fit"] = time.time() - t0
    return log


# ------------------------------------------------------------------ eval, checkpoint, loop
def evaluate(world, preds, cfg, upd, save=True):
    """Held-out forward generations on cfg.eval_seeds: overhangs, stuck
    sites, fill error, solid fraction, mean cert slack per level; picture of
    the first seed to images/certland_chain_<tag>_u<upd>.png."""
    H, W = cfg.eval_HW or (world.H, world.W)
    ew = cl.World(H, W, kappa=world.kappa, lam=world.lam, beta=world.beta, L=world.L)
    out = []
    for k, s in enumerate(cfg.eval_seeds):
        set_noise(ew, s)
        bad = cp.generate_pred(ew, preds, sweeps=cfg.gen_sweeps, seed=int(s))
        mk = cl.metrics(ew)
        out.append(dict(seed=s, stuck=int(sum(bad)), overhangs=mk["overhangs"], fill_err=mk["fill_err"],
                        solid=float(ew.rho[cl.NL - 1].mean()),
                        A_gap={h: mk[f"h{h}"]["A_gap"] for h in cl.HS[:-1]},
                        B_gap={h: mk[f"h{h}"]["B_gap"] for h in cl.HS[:-1]}))
        if save and k == 0:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            os.makedirs(IMAGES, exist_ok=True)
            plt.imsave(os.path.join(IMAGES, f"certland_chain_{cfg.tag}_u{upd}.png"), cl.render(ew, px=3))
    return out


def save_ckpt(path, preds, upd):
    """Predictor params (and those of a proposal-only set preds.q, if any)."""
    d = dict(nh=np.int64(preds.nh), upd=np.int64(upd))
    q = getattr(preds, "q", None)
    for i in range(cl.NL - 1):
        for ch in (0, 1):
            if preds.th[i][ch] is not None:
                d[f"th_{i}_{ch}"] = np.asarray(preds.th[i][ch])
            if q is not None and q.th[i][ch] is not None:
                d[f"thq_{i}_{ch}"] = np.asarray(q.th[i][ch])
    np.savez(path, **d)


def load_ckpt(path, preds=None):
    z = np.load(path)
    preds = cp.Preds(nh=int(z["nh"])) if preds is None else preds
    for i in range(cl.NL - 1):
        for ch in (0, 1):
            if f"th_{i}_{ch}" in z:
                preds.set(i, ch, cp.unpack(z[f"th_{i}_{ch}"], i, ch, preds.nh))
            if f"thq_{i}_{ch}" in z:
                if getattr(preds, "q", None) is None:
                    preds.q = cp.QPreds(preds.nh)
                preds.q.set(i, ch, cp.unpack(z[f"thq_{i}_{ch}"], i, ch, preds.nh, cp.NFQ))
    return preds, int(z["upd"])


def format_log(lg):
    s = [f"u{lg['upd']}"]
    for i, d in lg["levels"].items():
        s.append(f"h{cl.HS[i]}: ess {d['ess']:.2f} chg {d['changed']:.3f} dead {d['dead']:.2f} ({d['time']:.1f}s)")
    for (i, ch), d in sorted(lg["fit"].items()):
        s.append(f"[{'q' if i >= QOFF else ''}{cl.HS[i % QOFF]}{'rc'[ch]} n{d['n_new']}+{d['n_arch']} {d['loss_pre']:.3f}->{d['loss']:.3f} "
                 f"(0: {d['loss0']:.3f})]")
    t = lg["time"]
    s.append(f"t moves {t['moves']:.1f} rec {t['record']:.1f} tiles {t['tiles']:.2f} fit {t['fit']:.1f}")
    return " | ".join(s)


def train(world, preds, cfg, n_updates, log=print, eval_every=None, eval_seeds=None, archive=None, start=0):
    """n_updates updates on an initialised chain (world or list of worlds;
    see init_chain / make_chains).  Every eval_every updates (default
    cfg.eval_every) a held-out evaluation; every cfg.ckpt_every a
    checkpoint to cfg.ckpt.  Returns the history: a list of update logs
    (with 'eval' where evaluated).  Pass `archive` (and `start`) to resume
    with the replay archive and optimiser state of an earlier call."""
    rng = np.random.default_rng(cfg.seed)
    eval_every = cfg.eval_every if eval_every is None else eval_every
    if eval_seeds is not None:
        cfg.eval_seeds = tuple(eval_seeds)
    archive = Archive(cfg.archive_cap, cfg.archive_floats) if archive is None else archive
    w0 = world[0] if isinstance(world, (list, tuple)) else world
    hist = []
    for u in range(start, start + n_updates):
        lg = update(world, preds, archive, cfg, u, rng)
        log(format_log(lg))
        if eval_every and (u + 1) % eval_every == 0:
            t0 = time.time()
            ev = evaluate(w0, preds, cfg, u + 1)
            lg["eval"] = ev
            lg["time"]["eval"] = time.time() - t0
            log("  eval: " + "; ".join(f"seed {e['seed']} overhangs {e['overhangs']} stuck {e['stuck']} "
                                        f"fill_err {e['fill_err']:.3f} solid {e['solid']:.3f}" for e in ev)
                + f" ({lg['time']['eval']:.1f}s)")
        if cfg.ckpt and (u + 1) % cfg.ckpt_every == 0:
            save_ckpt(cfg.ckpt, preds, u + 1)
        hist.append(lg)
    if cfg.ckpt:
        save_ckpt(cfg.ckpt, preds, start + n_updates)
    return hist
