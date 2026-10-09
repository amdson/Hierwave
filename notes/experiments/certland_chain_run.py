"""Certland chain training (notes/certland_chain_spec.md; the plan doc's
tests 1-6), one subcommand per test, each gating the next:

  ref         reference chains at theta = 0 (designed-only proposals) from
              different forward initialisations on one noise field: per-level
              certificate marginals, corr(A + B, rho), tile fill, overhangs,
              certificate slack (A_gap / B_gap) over time; R-hat across chains
  ess         per root level ESS / changed / dead for K = 1, 4, 8 at theta = 0,
              and annealed proposals (certland_anneal) at h = 8
  capacity    fit the h = 2 heads alone to the EXACT h = 2 conditional
              (designed site terms + the 2 x 2 tiles' exact free energy)
  h2          train h = 2 by the chain; cross-entropy gap to the exact oracle
              on held-out contexts
  curriculum  roots h = 2, then + h = 4, then + h = 8; ESS / changed / dead per
              level per update
  e2e         the shipped sampler (generate_pred, S = 1 and 20) against the
              chain on held-out noise; then the full 128 x 256 world

Defaults: world 32 x 128 tiles, roots at h = 8, 4, 2 (h = 16 held fixed).
Every subcommand prints a compact table and writes
images/certland_chain_<cmd>_log.txt; figures go to images/.

Usage: NUMBA_NUM_THREADS=1 python notes/experiments/certland_chain_run.py <cmd> [--n N] [--seed S] ...
       (see --help of each subcommand)"""
import argparse
import functools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from castlegen.channels import certland as cl                       # noqa: E402
from castlegen.channels import certland_pred as cp                  # noqa: E402

IMG = os.environ.get("CERTLAND_IMG", "/Users/amdson/dev/Hierwave/images")
CKPT = os.path.join(IMG, "certland_chain_preds.npz")
LOGF = None


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    if LOGF is not None:
        LOGF.write(s + "\n")
        LOGF.flush()


def cc_():
    from castlegen.channels import certland_chain as cc
    return cc


def ct_():
    from castlegen.channels import certland_train as ct
    return ct


def ca_():
    from castlegen.channels import certland_anneal as ca
    return ca


def ilist(s):
    return tuple(int(v) for v in s.split(",") if v != "")


def kdict(s, levels):
    """'4' -> {i: 4}; '1:8,2:4,3:1' -> per level."""
    if ":" not in s:
        return {i: int(s) for i in range(cl.NL - 1)}
    d = {i: 1 for i in range(cl.NL - 1)}
    for part in s.split(","):
        a, b = part.split(":")
        d[int(a)] = int(b)
    return d


# ------------------------------------------------------------------ world helpers
def new_world(H, W, noise):
    w = cl.World(H, W)
    cl.set_fill(w, cl.density_field(H, W, noise))
    return w


def copy_world(w):
    v = cl.World(w.H, w.W, kappa=w.kappa, lam=w.lam, beta=w.beta, L=w.L)
    v.fill[:] = w.fill
    for i in range(cl.NL):
        v.rho[i][:] = w.rho[i]
        if i < cl.NL - 1:
            v.cert[i][:] = w.cert[i]
    return v


def gen_below(world, preds, sweeps, seed, start=1):
    """Levels start..4 from the world's level start - 1 as in generate_pred
    (refine + sweeps), the levels above kept.  Returns stuck counts."""
    rng = np.random.default_rng(seed)
    bad = []
    for i in range(start, cl.NL):
        cl.refine(world, i, rng)
        s = int(rng.integers(1 << 30))
        if i == cl.NL - 1:
            a = list(world.args(i))
            a[-3], a[-2], a[-1] = np.zeros(1), 0, 0
            bad.append(int(cl.sweep_level(*a, sweeps, s)))
        else:
            h, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, hp = cp.level_args(world, i)
            th_r, nh, on_r = preds.args(i, cl.RHO)
            th_c, _, on_c = preds.args(i, cl.CERT)
            bad.append(int(cp.sweep_pred(i, h, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, world.fill,
                                         world.kappa, world.lam, world.beta, hp, th_r, on_r, th_c, on_c, nh,
                                         sweeps, s)))
    return bad


def nviol(world):
    return int(sum(ct_().count_violations(world)))


def stats(world):
    """Flat dict of the per-level statistics the chains are compared on."""
    with np.errstate(invalid="ignore", divide="ignore"):
        mk = cl.metrics(world)
    out = {}
    for i, h in enumerate(cl.HS[:-1]):
        A = world.Aof[i][world.cert[i]]
        B = world.Bof[i][world.cert[i]]
        r = world.rho[i]
        out[f"A/h@{h}"] = float(A.mean() / h)
        out[f"B/h@{h}"] = float(B.mean() / h)
        out[f"rho@{h}"] = float(r.mean() / (h * h))
        s = A + B
        out[f"corr@{h}"] = float(np.corrcoef(s.ravel(), r.ravel())[0, 1]) if s.std() > 0 and r.std() > 0 else np.nan
        out[f"Agap@{h}"] = mk[f"h{h}"]["A_gap"]
        out[f"Bgap@{h}"] = mk[f"h{h}"]["B_gap"]
        out[f"rhoerr@{h}"] = mk[f"h{h}"]["rho_err"]
    out["fill"] = float(world.rho[cl.NL - 1].mean())
    out["fill_err"] = mk["fill_err"]
    out["overhangs"] = mk["overhangs"]
    out["viol"] = nviol(world)
    return out


def round_moves(world, preds, levels, K, rng, anneal=None, collect=None):
    """One round: move_level at each root level top-down, then a tile sweep.
    Returns {i: move_level stats (+ time)}."""
    cc = cc_()
    out = {}
    for i in levels:
        t0 = time.time()
        mv = None
        if anneal and i in anneal:
            M, L0 = anneal[i]
            mv = functools.partial(ca_().annealed_move, M=M, L0=L0)
        s = cc.move_level(world, preds, i, K[i], rng, collect=collect, move=mv)
        s = dict(s)
        s["time"] = time.time() - t0
        out[i] = s
    cc.tile_sweep(world, rng)
    return out


def rhat(x):
    """Gelman-Rubin R-hat of (chains, n) samples."""
    x = np.asarray(x, float)
    m, n = x.shape
    if n < 2:
        return np.nan
    W = x.var(1, ddof=1).mean()
    B = n * x.mean(1).var(ddof=1)
    if W <= 0:
        return np.nan if B <= 0 else np.inf
    return float(np.sqrt(((n - 1) / n * W + B / n) / W))


def save_preds(path, preds):
    d = dict(nh=np.int64(preds.nh), upd=np.int64(0))
    for i in range(cl.NL - 1):
        for ch in (0, 1):
            if preds.th[i][ch] is not None:
                d[f"th_{i}_{ch}"] = np.asarray(preds.th[i][ch])
    np.savez(path, **d)


def load_preds(path):
    if path and os.path.exists(path):
        return ct_().load_ckpt(path)[0]
    log(f"(no checkpoint at {path}: theta = 0)")
    return cp.Preds()


def plt_():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ------------------------------------------------------------------ 1. ref
def cmd_ref(a):
    preds = cp.Preds(a.nh)
    K = kdict(a.K, a.levels)
    rng = np.random.default_rng(a.seed)
    base = new_world(a.H, a.W, a.noise)
    bad0 = cp.generate_pred(base, preds, sweeps=a.gen_sweeps, seed=a.seed)
    chains = []
    # different forward initialisations below the held top (or of everything when h = 16 is a root):
    # alternate generate_pred / cl.generate-shaped sweeps and vary the sweep count
    sweeps = [a.gen_sweeps, 1, 5, a.gen_sweeps, 2]
    for c in range(a.chains):
        w = copy_world(base)
        if 0 in a.levels:
            if c % 2:
                bad = cl.generate(w, sweeps=sweeps[c % 5], seed=a.seed + 1000 * (c + 1))
            else:
                bad = cp.generate_pred(w, preds, sweeps=sweeps[c % 5], seed=a.seed + 1000 * (c + 1))
        else:
            bad = gen_below(w, preds, sweeps[c % 5], a.seed + 1000 * (c + 1))
        assert nviol(w) == 0, f"chain {c}: invalid init"
        chains.append(w)
        log(f"chain {c}: init sweeps {sweeps[c % 5]} stuck {bad}")
    log(f"ref: {a.H}x{a.W} noise {a.noise} roots {[cl.HS[i] for i in a.levels]} K {[K[i] for i in a.levels]} "
        f"chains {a.chains} N {a.n} (top stuck {bad0})")
    keys = list(stats(chains[0]).keys())
    tr = np.zeros((a.chains, a.n + 1, len(keys)))
    mvs = []
    for c, w in enumerate(chains):
        tr[c, 0] = [stats(w)[k] for k in keys]
    t0 = time.time()
    for t in range(1, a.n + 1):
        for c, w in enumerate(chains):
            mv = round_moves(w, preds, a.levels, K, rng)
            mvs.append(mv)
            tr[c, t] = [stats(w)[k] for k in keys]
        if t % max(1, a.n // 10) == 0 or t == a.n:
            j = keys.index
            log(f"  t={t:4d} ({time.time() - t0:.0f}s) " + "  ".join(
                f"c{c}: B/h@8 {tr[c, t, j('B/h@8')]:.3f} A/h@2 {tr[c, t, j('A/h@2')]:.3f} "
                f"fill {tr[c, t, j('fill')]:.3f} ovh {tr[c, t, j('overhangs')]:.0f}" for c in range(a.chains)))
    half = tr[:, a.n // 2 + 1:, :] if a.n >= 2 else tr[:, -1:, :]
    log("\nstat            " + " ".join(f"{'chain' + str(c):>8s}" for c in range(a.chains))
        + "   init-mean   between-sd  within-sd   R-hat   (second half of the run)")
    for k, key in enumerate(keys):
        cm = half[:, :, k].mean(1)
        wsd = np.sqrt(half[:, :, k].var(1, ddof=1).mean()) if half.shape[1] > 1 else np.nan
        log(f"{key:16s}" + " ".join(f"{v:8.3f}" for v in cm)
            + f"   {tr[:, 0, k].mean():9.3f}   {cm.std(ddof=1) if a.chains > 1 else 0:9.4f}  {wsd:9.4f}"
            f"   {rhat(half[:, :, k]):6.2f}")
    ovh = tr[:, :, keys.index("overhangs")]
    vio = tr[:, :, keys.index("viol")]
    log(f"\noverhangs max over all chains and times: {ovh.max():.0f}; violations max: {vio.max():.0f}")
    for i in a.levels:
        e = np.mean([m[i]["ess"] for m in mvs])
        ch = np.mean([m[i]["changed"] for m in mvs])
        d = np.mean([m[i]["dead"] for m in mvs])
        tt = np.mean([m[i]["time"] for m in mvs])
        log(f"moves h={cl.HS[i]:2d}: K {K[i]} mean ESS {e:.2f} changed {ch:.3f} dead {d:.3f}  {tt:.2f}s per level pass")
    # traces
    plt = plt_()
    hs = cl.HS[:-1]
    rows = [("A/h", "A/h"), ("B/h", "B/h"), ("corr", "corr(A+B, rho)"), ("Agap", "A_gap (tiles' full - A)/h"),
            ("Bgap", "B_gap (B - tiles' occupied)/h")]
    extra = ["fill", "fill_err", "overhangs", "viol", "rhoerr@2"]
    fig, ax = plt.subplots(len(rows), 5, figsize=(17, 2.4 * len(rows)), squeeze=False)
    x = np.arange(a.n + 1)
    for r, (pre, name) in enumerate(rows):
        for c, h in enumerate(hs):
            k = keys.index(f"{pre}@{h}")
            for ci in range(a.chains):
                ax[r, c].plot(x, tr[ci, :, k], lw=1)
            ax[r, c].set_title(f"{name}, h={h}" + ("" if any(cl.HS[i] == h for i in a.levels) else " (held)"),
                               fontsize=8)
            ax[r, c].tick_params(labelsize=7)
        k = keys.index(extra[r])
        for ci in range(a.chains):
            ax[r, 4].plot(x, tr[ci, :, k], lw=1, label=f"chain {ci}")
        ax[r, 4].set_title(extra[r], fontsize=8)
        ax[r, 4].tick_params(labelsize=7)
    ax[0, 4].legend(fontsize=7)
    for c in range(5):
        ax[-1, c].set_xlabel("update (round of moves)", fontsize=8)
    fig.suptitle(f"certland reference chains, theta = 0, {a.H}x{a.W}, roots h={[cl.HS[i] for i in a.levels]}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "certland_chain_ref.png"), dpi=100)
    plt.close(fig)
    log("wrote", os.path.join(IMG, "certland_chain_ref.png"))
    np.savez(os.path.join(IMG, "certland_chain_ref_traces.npz"), tr=tr, keys=np.array(keys))


# ------------------------------------------------------------------ 2. ess
def cmd_ess(a):
    cc = cc_()
    preds = cp.Preds(a.nh)
    rng = np.random.default_rng(a.seed)
    w = new_world(a.H, a.W, a.noise)
    cp.generate_pred(w, preds, sweeps=a.gen_sweeps, seed=a.seed)
    K0 = {i: 1 for i in range(4)}
    for _ in range(a.burn):
        round_moves(w, preds, a.levels, K0, rng)
    assert nviol(w) == 0
    log(f"ess: {a.H}x{a.W} noise {a.noise} theta = 0, burn {a.burn} rounds of K=1, {a.n} level passes per cell")
    log(f"{'root':>5s} {'move':>14s} {'K':>3s} {'ESS':>6s} {'ESS-1/K':>8s} {'changed':>8s} {'dead':>6s} "
        f"{'s/pass':>7s} {'ms/root':>8s}")
    rows = []
    for i in a.levels:
        for K in a.Ks:
            for kind, anneal in [("plain", None)] + ([(f"anneal M{M} L{a.L0:g}", (M, a.L0)) for M in a.M]
                                                     if i == 1 else []):
                if anneal and K not in a.Ka:
                    continue
                v = copy_world(w)
                r = np.random.default_rng(a.seed + 7 * i + K)
                ss = []
                for _ in range(a.n):
                    t0 = time.time()
                    mv = None if anneal is None else functools.partial(ca_().annealed_move, M=anneal[0],
                                                                      L0=anneal[1])
                    s = dict(cc.move_level(v, preds, i, K, r, move=mv))
                    s["time"] = time.time() - t0
                    ss.append(s)
                assert nviol(v) == 0 and cl.overhangs(v.rho[cl.NL - 1]) == 0
                e = np.mean([s["ess"] for s in ss])
                row = dict(h=cl.HS[i], kind=kind, K=K, ess=e, essn=(e - 1) / K,
                           changed=np.mean([s["changed"] for s in ss]), dead=np.mean([s["dead"] for s in ss]),
                           t=np.mean([s["time"] for s in ss]), n=ss[0]["n"])
                rows.append(row)
                log(f"{'h=' + str(row['h']):>5s} {kind:>14s} {K:3d} {e:6.2f} {row['essn']:8.3f} "
                    f"{row['changed']:8.3f} {row['dead']:6.3f} {row['t']:7.2f} {1000 * row['t'] / row['n']:8.1f}")
    return rows


# ------------------------------------------------------------------ exact h = 2 conditional
def exact_h2_examples(world, cells=None, rng=None, nmax=None):
    """For level 3 (h = 2) cells, both channels: features, designed offsets
    (+inf excluded), and the exact conditional pi over the channel's domain:
    softmax(-(designed site terms - log Z of the cell's 2 x 2 tiles given the
    value)), log Z by cl.block_logz_exact_tiles (Lcap = inf).  Returns
    {ch: dict(feats, offs, pi)}."""
    i = 3
    a3 = world.args(i)
    a4 = world.args(cl.NL - 1)
    rows, cols = world.rho[i].shape
    if cells is None:
        cells = [(y, x) for y in range(rows) for x in range(cols)]
        if nmax is not None and len(cells) > nmax:
            sel = rng.choice(len(cells), nmax, replace=False)
            cells = [cells[j] for j in sel]
    out = {}
    for ch in (cl.RHO, cl.CERT):
        D = 5 if ch == cl.RHO else len(world.Aof[i])
        g = world.rho[i] if ch == cl.RHO else world.cert[i]
        F, O, P = [], [], []
        soft = np.zeros(D)
        viol = np.zeros(D, np.int64)
        for y, x in cells:
            cl.site_terms(*a3, y, x, ch, soft, viol)
            offs = np.where(viol == 0, soft, np.inf)
            cur = g[y, x]
            lz = np.empty(D)
            for t in range(D):
                g[y, x] = t
                lz[t] = cl.block_logz_exact_tiles(*a4, y, x, np.inf)
            g[y, x] = cur
            e = np.where(np.isfinite(offs) & np.isfinite(lz), offs - lz, np.inf)
            assert np.isfinite(e[cur]), "the current value must be admissible"
            lp = -(e - e[np.isfinite(e)].min())
            p = np.where(np.isfinite(e), np.exp(lp), 0.0)
            F.append(cp.features_np(world, i, ch, y, x))
            O.append(offs)
            P.append(p / p.sum())
        out[ch] = dict(feats=np.array(F), offs=np.array(O), pi=np.array(P))
    return out


def merge(ds):
    return {k: np.concatenate([d[k] for d in ds]) for k in ds[0]}


def soft_batch(d):
    """Soft targets as weighted hard examples: one row per (context, value
    with pi > 0), weight pi.  weighted_ce on it is the mean soft-target CE
    over contexts (sum_t pi_t = 1 per context)."""
    n, D = d["pi"].shape
    r, t = np.nonzero(d["pi"] > 0)
    return dict(feats=d["feats"][r], offs=d["offs"][r], val=t.astype(np.int64), w=d["pi"][r, t])


def soft_ce(d, b=None):
    """Mean over contexts of -sum_t pi_t log softmax(-(offs + b))_t."""
    o = d["offs"]
    ok = np.isfinite(o)
    lg = np.where(ok, -(np.where(ok, o, 0.0) + (0.0 if b is None else b)), -np.inf)
    mx = lg.max(1, keepdims=True)
    lsm = lg - (mx + np.log(np.exp(lg - mx).sum(1, keepdims=True)))
    pi = d["pi"]
    return float(-(pi * np.where(pi > 0, lsm, 0.0)).sum(1).mean())


def entropy(d):
    pi = d["pi"]
    return float(-(np.where(pi > 0, pi * np.log(np.where(pi > 0, pi, 1.0)), 0.0)).sum(1).mean())


def bias_np(p, feats):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    return np.asarray(cp.jax_bias({k: jnp.asarray(v) for k, v in p.items()}, jnp.asarray(feats)))


def oracle_worlds(a, seeds, preds, moves, rng, levels=(1, 2, 3)):
    """Forward-generated worlds (theta = preds) on the noise seeds, plus
    `moves` rounds of chain moves (K = 1) so that contexts are chain states."""
    out = []
    for s in seeds:
        w = new_world(a.H, a.W, s)
        cp.generate_pred(w, preds, sweeps=a.gen_sweeps, seed=s)
        for _ in range(moves):
            round_moves(w, preds, levels, {i: 1 for i in range(4)}, rng)
        assert nviol(w) == 0
        out.append(w)
    return out


# ------------------------------------------------------------------ 3. capacity
def cmd_capacity(a):
    ct = ct_()
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    tr_w = oracle_worlds(a, a.train_seeds, cp.Preds(a.nh), a.moves, rng)
    te_w = oracle_worlds(a, a.test_seeds, cp.Preds(a.nh), a.moves, rng)
    tr = [exact_h2_examples(w, rng=rng, nmax=a.cells) for w in tr_w]
    te = [exact_h2_examples(w, rng=rng, nmax=a.cells) for w in te_w]
    log(f"capacity: {a.H}x{a.W}, contexts from noise {a.train_seeds} (train) / {a.test_seeds} (held out), "
        f"chain moves {a.moves}, nh {a.nh}, steps {a.steps}, lr {a.lr} ({time.time() - t0:.0f}s to build)")
    log(f"{'ch':>4s} {'n tr':>6s} {'n te':>6s} {'H(pi) te':>9s} {'CE0 te':>8s} {'CE fit tr':>10s} {'CE fit te':>10s} "
        f"{'KL0 te':>7s} {'KL fit te':>9s}")
    res = {}
    for ch in (cl.RHO, cl.CERT):
        dtr = merge([d[ch] for d in tr])
        dte = merge([d[ch] for d in te])
        p = cp.init_params(3, ch, a.nh, a.seed + ch)
        batch = soft_batch(dtr)
        st = None
        curve = []
        for k in range(0, a.steps, a.chunk):
            p, st, fit = ct.fit_steps(p, batch, min(a.chunk, a.steps - k), a.lr, st, batch_size=min(a.bs, len(batch["val"])),
                                      rng=rng)
            curve.append((k + a.chunk, fit, soft_ce(dte, bias_np(p, dte["feats"]))))
        ce_tr = soft_ce(dtr, bias_np(p, dtr["feats"]))
        ce_te = soft_ce(dte, bias_np(p, dte["feats"]))
        Hte, ce0 = entropy(dte), soft_ce(dte)
        res[ch] = dict(Hte=Hte, ce0=ce0, ce_tr=ce_tr, ce_te=ce_te, curve=curve, Htr=entropy(dtr), ce0tr=soft_ce(dtr))
        log(f"{'rc'[ch]:>4s} {len(dtr['pi']):6d} {len(dte['pi']):6d} {Hte:9.4f} {ce0:8.4f} {ce_tr:10.4f} {ce_te:10.4f} "
            f"{ce0 - Hte:7.4f} {ce_te - Hte:9.4f}")
        log("     curve (step, minibatch CE, held-out soft CE): "
            + " ".join(f"({s},{f:.3f},{c:.3f})" for s, f, c in curve))
        log(f"     train: H(pi) {res[ch]['Htr']:.4f} CE0 {res[ch]['ce0tr']:.4f}")
    plt = plt_()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for ch in (0, 1):
        c = np.array(res[ch]["curve"])
        ax[ch].plot(c[:, 0], c[:, 2], label="held-out soft CE (fit)")
        ax[ch].plot(c[:, 0], c[:, 1], label="minibatch CE (train)", alpha=0.6)
        ax[ch].axhline(res[ch]["Hte"], color="k", ls="--", label="H(pi) held out")
        ax[ch].axhline(res[ch]["ce0"], color="r", ls=":", label="theta = 0")
        ax[ch].set_title(f"h=2 {'rho' if ch == 0 else 'cert'}: fit to the exact conditional", fontsize=9)
        ax[ch].set_xlabel("Adam steps")
        ax[ch].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "certland_chain_capacity.png"), dpi=100)
    plt.close(fig)
    log("wrote", os.path.join(IMG, "certland_chain_capacity.png"))
    return res


# ------------------------------------------------------------------ 4. h2
def oracle_report(a, preds, tag, rng):
    """CE of preds at h = 2 against the exact oracle on held-out contexts
    (forward generation + moves with theta = 0 on the held-out noise)."""
    te_w = oracle_worlds(a, a.test_seeds, cp.Preds(a.nh), a.moves, rng)
    te = [exact_h2_examples(w, rng=rng, nmax=a.cells) for w in te_w]
    log(f"{tag}: h=2 vs the exact oracle on held-out contexts (noise {a.test_seeds})")
    log(f"{'ch':>4s} {'n':>6s} {'H(pi)':>8s} {'CE0':>8s} {'CE trained':>11s} {'gap0':>7s} {'gap':>7s}")
    out = {}
    for ch in (0, 1):
        d = merge([x[ch] for x in te])
        p = preds.params(3, ch)
        ce = soft_ce(d, None if p is None else bias_np(p, d["feats"]))
        Hh, ce0 = entropy(d), soft_ce(d)
        out[ch] = (Hh, ce0, ce)
        log(f"{'rc'[ch]:>4s} {len(d['pi']):6d} {Hh:8.4f} {ce0:8.4f} {ce:11.4f} {ce0 - Hh:7.4f} {ce - Hh:7.4f}")
    return out


def make_cfg(a, levels, K, tag):
    ct = ct_()
    return ct.Cfg(levels=tuple(levels), K=dict(K), anneal=dict(a.anneal), steps=a.steps, lr=a.lr, nh=a.nh,
                  burn=a.burn, gen_sweeps=a.gen_sweeps, eval_every=a.eval_every,
                  eval_seeds=tuple(a.test_seeds), tag=tag, ckpt=None, seed=a.seed)


def train_logged(a, worlds, preds, cfg, n, archive, start, rows):
    ct = ct_()

    def lg(s):
        log("  " + s)
    hist = ct.train(worlds, preds, cfg, n, log=lg, eval_every=a.eval_every, archive=archive, start=start)
    for h in hist:
        for i, d in h["levels"].items():
            rows.append(dict(upd=h["upd"], h=cl.HS[i], ess=d["ess"], changed=d["changed"], dead=d["dead"],
                             K=d["K"], t=d["time"]))
    return hist


def cmd_h2(a):
    ct = ct_()
    rng = np.random.default_rng(a.seed)
    preds = cp.Preds(a.nh)
    K = kdict(a.K, (3,))
    cfg = make_cfg(a, (3,), K, "h2")
    log(f"h2: {a.H}x{a.W}, chains on noise {a.train_seeds}, K {K[3]}, {a.n} updates, steps {a.steps}, nh {a.nh}")
    worlds = ct.make_chains(cfg, preds, a.H, a.W, a.train_seeds, rng, log=log)
    rows = []
    t0 = time.time()
    hist = train_logged(a, worlds, preds, cfg, a.n, ct.Archive(cfg.archive_cap, cfg.archive_floats), 0, rows)
    log(f"training {time.time() - t0:.0f}s")
    path = os.path.join(IMG, "certland_chain_h2_preds.npz")
    save_preds(path, preds)
    log("saved", path)
    res = oracle_report(a, preds, "h2", rng)
    for ch in (0, 1):
        f = [h["fit"][(3, ch)] for h in hist if (3, ch) in h["fit"]]
        if f:
            log(f"chain pool CE {'rc'[ch]}: first update {f[0]['loss']:.4f} (theta=0 {f[0]['loss0']:.4f}), "
                f"last {f[-1]['loss']:.4f} (theta=0 {f[-1]['loss0']:.4f})")
    return res


# ------------------------------------------------------------------ 5. curriculum
def cmd_curriculum(a):
    ct = ct_()
    rng = np.random.default_rng(a.seed)
    preds = load_preds(a.init) if a.init else cp.Preds(a.nh)
    K = kdict(a.K, (1, 2, 3))
    stages = [tuple(int(v) for v in s.split(",")) for s in a.stages.split(";")]
    cfg = make_cfg(a, stages[0], K, "curriculum")
    log(f"curriculum: {a.H}x{a.W}, noise {a.train_seeds}, stages {[[cl.HS[i] for i in s] for s in stages]}, "
        f"K {K}, anneal {a.anneal}, {a.n} updates per stage")
    worlds = ct.make_chains(cfg, preds, a.H, a.W, a.train_seeds, rng, log=log)
    archive = ct.Archive(cfg.archive_cap, cfg.archive_floats)
    rows = []
    start = 0
    for s in stages:
        cfg = make_cfg(a, s, K, "curriculum")
        log(f"--- stage roots h={[cl.HS[i] for i in s]}")
        train_logged(a, worlds, preds, cfg, a.n, archive, start, rows)
        start += a.n
        save_preds(CKPT, preds)
    log("saved", CKPT)
    log(f"\n{'upd':>4s} " + " ".join(f"{'h' + str(cl.HS[i]) + ' ess/chg/dead':>22s}" for i in (1, 2, 3)))
    for u in range(start):
        cells = []
        for i in (1, 2, 3):
            r = [x for x in rows if x["upd"] == u and x["h"] == cl.HS[i]]
            cells.append(f"{r[0]['ess']:6.2f}/{r[0]['changed']:.3f}/{r[0]['dead']:.2f}" if r else "-")
        log(f"{u:4d} " + " ".join(f"{c:>22s}" for c in cells))
    plt = plt_()
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.2))
    for i in (1, 2, 3):
        r = [x for x in rows if x["h"] == cl.HS[i]]
        if not r:
            continue
        u = [x["upd"] for x in r]
        for k, key in enumerate(("ess", "changed", "dead")):
            ax[k].plot(u, [x[key] for x in r], ".-", label=f"h={cl.HS[i]}")
    for k, key in enumerate(("ESS of K+1 weights", "fraction of subtree cells changed", "dead / proposals")):
        ax[k].set_title(key, fontsize=9)
        ax[k].set_xlabel("update")
        for s in range(1, len(stages)):
            ax[k].axvline(s * a.n - 0.5, color="gray", lw=0.5)
        ax[k].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "certland_chain_curriculum.png"), dpi=100)
    plt.close(fig)
    log("wrote", os.path.join(IMG, "certland_chain_curriculum.png"))
    oracle_report(a, preds, "curriculum", rng)


# ------------------------------------------------------------------ 6. e2e
SUM_KEYS = ["fill", "fill_err", "overhangs"] + [f"{p}@{h}" for h in (8, 4, 2) for p in ("A/h", "B/h", "corr",
                                                                                     "Agap", "Bgap", "rhoerr")]


def cmd_e2e(a):
    preds = load_preds(a.ckpt)
    on = [(i, ch) for i in range(4) for ch in (0, 1) if preds.th[i][ch] is not None]
    K = kdict(a.K, a.levels)
    rng = np.random.default_rng(a.seed)
    log(f"e2e: predictors on {[(cl.HS[i], 'rc'[ch]) for i, ch in on]} from {a.ckpt}; held-out noise {a.test_seeds}")
    res = {"S=1": [], "S=20": [], "chain": []}
    ims = {"S=1": [], "S=20": [], "chain": []}
    for s in a.test_seeds:
        for S in (1, 20):
            w = new_world(a.H, a.W, s)
            bad = cp.generate_pred(w, preds, sweeps=S, seed=s)
            st = stats(w)
            st["stuck"] = sum(bad)
            res[f"S={S}"].append(st)
            ims[f"S={S}"].append(cl.render(w, px=3))
        # the chain: from the S = 20 sample, rounds of moves with the trained proposals (p* is
        # theta-independent), statistics over the second half
        w = new_world(a.H, a.W, s)
        cp.generate_pred(w, preds, sweeps=20, seed=s)
        tr = []
        for t in range(a.n):
            round_moves(w, preds, a.levels, K, rng)
            if t >= a.n // 2:
                tr.append(stats(w))
        st = {k: float(np.mean([x[k] for x in tr])) for k in tr[0]}
        st["stuck"] = 0
        res["chain"].append(st)
        ims["chain"].append(cl.render(w, px=3))
    log(f"{'stat':12s} {'S=1':>9s} {'S=20':>9s} {'chain':>9s}   (mean over held-out seeds; chain: 2nd half of {a.n} rounds)")
    for k in SUM_KEYS + ["viol", "stuck"]:
        log(f"{k:12s} " + " ".join(f"{np.nanmean([r[k] for r in res[n]]):9.3f}" for n in ("S=1", "S=20", "chain")))
    sheet([(n, ims[n]) for n in ("S=1", "S=20", "chain")], os.path.join(IMG, "certland_chain_e2e.png"))
    # the full world with h = 16
    log(f"\nfull {a.FH}x{a.FW} world (h = 16 included), held-out noise {a.test_seeds}")
    fims = {"S=1": [], "S=20": [], f"S=20 + {a.full_rounds} rounds h16..2": []}
    for s in a.test_seeds:
        for S in (1, 20):
            w = new_world(a.FH, a.FW, s)
            t0 = time.time()
            bad = cp.generate_pred(w, preds, sweeps=S, seed=s)
            st = stats(w)
            log(f"  seed {s} S={S:2d}: stuck {sum(bad)} overhangs {st['overhangs']} viol {st['viol']} "
                f"fill_err {st['fill_err']:.3f} fill {st['fill']:.3f} A/h@2 {st['A/h@2']:.3f} B/h@2 {st['B/h@2']:.3f} "
                f"({time.time() - t0:.1f}s)")
            fims[f"S={S}"].append(cl.render(w))
        if a.full_rounds:
            t0 = time.time()
            for _ in range(a.full_rounds):
                round_moves(w, preds, (0, 1, 2, 3), K, rng)
            st = stats(w)
            log(f"  seed {s} + {a.full_rounds} rounds h16..2: overhangs {st['overhangs']} viol {st['viol']} "
                f"fill_err {st['fill_err']:.3f} fill {st['fill']:.3f} ({time.time() - t0:.0f}s)")
            fims[f"S=20 + {a.full_rounds} rounds h16..2"].append(cl.render(w))
    sheet([(n, v) for n, v in fims.items() if v], os.path.join(IMG, "certland_chain_e2e_full.png"))
    log("wrote", os.path.join(IMG, "certland_chain_e2e.png"), os.path.join(IMG, "certland_chain_e2e_full.png"))


def sheet(rows, path):
    plt = plt_()
    nr, nc = len(rows), max(len(r[1]) for r in rows)
    H, W = rows[0][1][0].shape[:2]
    fig, ax = plt.subplots(nr, nc, figsize=(4.5 * nc, (4.5 * H / W + 0.4) * nr), squeeze=False)
    for r, (name, ims) in enumerate(rows):
        for c in range(nc):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if c < len(ims):
                ax[r, c].imshow(ims[c], interpolation="nearest")
            if c == 0:
                ax[r, c].set_ylabel(name, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    global LOGF
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, n, K="1:4,2:4,3:1"):
        p.add_argument("--n", type=int, default=n, help="updates / rounds / passes")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--H", type=int, default=32)
        p.add_argument("--W", type=int, default=128)
        p.add_argument("--noise", type=int, default=1, help="noise field seed")
        p.add_argument("--levels", type=ilist, default=(1, 2, 3), help="root levels (0..3 = h 16..2)")
        p.add_argument("--K", default=K, help="proposals: one int or 'i:K,...'")
        p.add_argument("--nh", type=int, default=64)
        p.add_argument("--gen-sweeps", dest="gen_sweeps", type=int, default=20)
        p.add_argument("--train-seeds", dest="train_seeds", type=ilist, default=(1, 2))
        p.add_argument("--test-seeds", dest="test_seeds", type=ilist, default=(101, 102))
        p.add_argument("--moves", type=int, default=2, help="chain rounds after the forward init (oracle contexts)")
        p.add_argument("--cells", type=int, default=None, help="h = 2 cells per world for the oracle (None: all)")
        p.add_argument("--steps", type=int, default=50)
        p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--burn", type=int, default=3)
        p.add_argument("--eval-every", dest="eval_every", type=int, default=0)
        p.add_argument("--anneal", default="", help="'i:M:L0,...' annealed moves at those root levels")
        return p

    p = common(sub.add_parser("ref"), 40)
    p.add_argument("--chains", type=int, default=3)
    p = common(sub.add_parser("ess"), 3)
    p.add_argument("--Ks", type=ilist, default=(1, 4, 8))
    p.add_argument("--Ka", type=ilist, default=(1, 4), help="K for the annealed rows")
    p.add_argument("--M", type=ilist, default=(4, 8))
    p.add_argument("--L0", type=float, default=3.0)
    p = common(sub.add_parser("capacity"), 0)
    p.set_defaults(steps=3000, train_seeds=(1, 2, 3, 4), lr=3e-3)
    p.add_argument("--bs", type=int, default=2048)
    p.add_argument("--chunk", type=int, default=250)
    common(sub.add_parser("h2"), 20)
    p = common(sub.add_parser("curriculum"), 20)
    p.add_argument("--stages", default="3;2,3;1,2,3")
    p.add_argument("--init", default=None, help="start from a checkpoint (e.g. the h2 one)")
    p = common(sub.add_parser("e2e"), 20)
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--FH", type=int, default=128)
    p.add_argument("--FW", type=int, default=256)
    p.add_argument("--full-rounds", dest="full_rounds", type=int, default=1)
    a = ap.parse_args()
    a.anneal = {int(x.split(":")[0]): (int(x.split(":")[1]), float(x.split(":")[2]))
                for x in a.anneal.split(",") if x}
    os.makedirs(IMG, exist_ok=True)
    LOGF = open(os.path.join(IMG, f"certland_chain_{a.cmd}_log.txt"), "w")
    log("argv:", " ".join(sys.argv[1:]))
    t0 = time.time()
    dict(ref=cmd_ref, ess=cmd_ess, capacity=cmd_capacity, h2=cmd_h2, curriculum=cmd_curriculum, e2e=cmd_e2e)[a.cmd](a)
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
