"""Training the certland potentials by the bootstrap of notes/dsl_updates.md
C2: level by level from the tiles up, contexts from forward runs of the
current model, at each sampled site the collapsed conditional over a capped
candidate set,

    pi_p(t) ~ exp(-E_designed(t) + log Z_children(t)),

log Z of the site's 2 x 2 child block (both child channels) under the child
level's designed terms and installed potential, by enumeration at the tiles
and AIS above; then the level's potential Phi fitted by cross-entropy of pi
against softmax(-(E_designed + Delta Phi)) over the candidates (JAX, Adam).
Phi's definition is certland's; local_energy below must agree with
certland._learned_site up to a per-site constant (tested)."""
import time

import numpy as np

from . import certland as cl


# ------------------------------------------------------------------ phi windows
def phi_grid(world, i):
    """(rows, cols, KPHI): phi at every cell of level i."""
    Er, Ec, Epr, Epc = world.emb[i]
    rho, cert = world.rho[i], world.cert[i]
    g = Er[rho] + Ec[cert]
    if i > 0:
        pr, pc = world.rho[i - 1], world.cert[i - 1]
        g += np.repeat(np.repeat(Epr[pr] + Epc[pc], 2, 0), 2, 1)
    return g


def window(pg, y, x):
    """(5, 5, KPHI) phi around (y, x), x wrapped, rows off the grid zero;
    (5, 5) on-grid mask."""
    rows, cols, k = pg.shape
    out = np.zeros((5, 5, k))
    mask = np.zeros((5, 5))
    for dy in range(5):
        yy = y + dy - 2
        if 0 <= yy < rows:
            out[dy] = pg[yy, (x + np.arange(5) - 2) % cols]
            mask[dy] = 1
    return out, mask


# ------------------------------------------------------------------ JAX model
def _jax():
    import jax
    jax.config.update("jax_enable_x64", True)
    return jax, jax.numpy


def local_energy(p, win, mask, m):
    """Phi terms touching the centre of a 5 x 5 phi window (the definition
    in certland), batched over leading axes of win (..., 5, 5, k)."""
    jax, jnp = _jax()
    S = 0.5 * (p["S"] + p["S"].T)
    c = win[..., 2, 2, :]
    e = c @ p["a"] + jnp.einsum("...j,jl,...l->...", c, S, c)
    for di, (dy, dx) in enumerate(cl.D_PLUS):
        fwd = win[..., 2 + dy, 2 + dx, :]
        bwd = win[..., 2 - dy, 2 - dx, :]
        e = e + jnp.einsum("...j,jl,...l->...", c, p["A"][di], fwd) + jnp.einsum("...j,jl,...l->...", bwd, p["A"][di], c)
    if m > 0:
        for dq in range(9):
            qy, qx = 2 - (dq // 3 - 1), 2 - (dq % 3 - 1)
            pre = p["b"]
            for d2 in range(9):
                pre = pre + win[..., qy + d2 // 3 - 1, qx + d2 % 3 - 1, :] @ p["W"][d2]
            e = e + mask[..., qy, qx] * (jax.nn.softplus(pre) @ p["v"])
    return e


def fit(data, m=0, steps=1500, lr=3e-2, l2=1e-3, init=None, seed=0):
    """Adam on mean cross-entropy of the targets against the model's
    candidate softmax.  data: dict of stacked arrays win (N, C, 5, 5, k),
    mask (N, C, 5, 5), ed (N, C) designed energies, pi (N, C), ok (N, C).
    Returns (params dict, final loss, loss with Phi = 0, target entropy)."""
    jax, jnp = _jax()
    k = cl.KPHI
    rng = np.random.default_rng(seed)
    if init is None:
        p = dict(a=np.zeros(k), S=np.zeros((k, k)), A=np.zeros((4, k, k)))
        if m > 0:
            p.update(W=rng.normal(0, 0.05, (9, k, m)), b=np.zeros(m), v=rng.normal(0, 0.05, m))
    else:
        p = {key: np.asarray(init[key]) for key in (("a", "S", "A", "W", "b", "v") if m > 0 else ("a", "S", "A"))}
    win, mask = jnp.asarray(data["win"]), jnp.asarray(data["mask"])
    ed, pi, ok = jnp.asarray(data["ed"]), jnp.asarray(data["pi"]), jnp.asarray(data["ok"])

    def loss(p):
        e = local_energy(p, win, mask, m)
        logits = jnp.where(ok, -(ed + e), -jnp.inf)
        logq = jax.nn.log_softmax(logits, axis=-1)
        ce = -jnp.sum(jnp.where(ok, pi * logq, 0.0), -1).mean()
        reg = sum(jnp.sum(v ** 2) for v in p.values())
        return ce + l2 * reg

    def ce0():
        logits = jnp.where(ok, -ed, -jnp.inf)
        logq = jax.nn.log_softmax(logits, axis=-1)
        return float(-jnp.sum(jnp.where(ok, pi * logq, 0.0), -1).mean())

    H = float(-jnp.sum(jnp.where(ok & (pi > 0), pi * jnp.log(jnp.where(pi > 0, pi, 1.0)), 0.0), -1).mean())
    vg = jax.jit(jax.value_and_grad(loss))
    mu = {key: np.zeros_like(v) for key, v in p.items()}
    nu = {key: np.zeros_like(v) for key, v in p.items()}
    p = {key: jnp.asarray(v) for key, v in p.items()}
    for t in range(1, steps + 1):
        val, g = vg(p)
        a_t = lr * (0.05 ** (t / steps))
        for key in p:
            mu[key] = 0.9 * mu[key] + 0.1 * g[key]
            nu[key] = 0.999 * nu[key] + 0.001 * g[key] ** 2
            p[key] = p[key] - a_t * (mu[key] / (1 - 0.9 ** t)) / (jnp.sqrt(nu[key] / (1 - 0.999 ** t)) + 1e-8)
    final = float(loss(p)) - l2 * float(sum(jnp.sum(v ** 2) for v in p.values()))
    out = {key: np.asarray(v) for key, v in p.items()}
    out["m"] = m
    if m == 0:
        out.update(W=np.zeros((9, k, 1)), b=np.zeros(1), v=np.zeros(1))
    return out, final, ce0(), H


# ------------------------------------------------------------------ targets
FREE = True   # train on candidates from the whole domain with the site's own hard rows left out of
              # target and model alike (they add the same inf to both); False: the sampler's admissible
              # set, which never contrasts values the hard rows keep apart at one site, so the learned
              # free energy is unidentified there and the parent level's AIS queries it anyway


def candidates(world, i, y, x, ch, rng, C=8):
    """The current value, then (rho) 3 values within h of it and the rest
    uniform, or (cert) uniform values (admissible ones only when not FREE);
    distinct, at most C."""
    h = cl.HS[i]
    if ch == cl.RHO:
        D = h * h + 1
        cur = int(world.rho[i][y, x])
        pool = [cur]
        for _ in range(3):
            pool.append(int(np.clip(cur + rng.integers(-h, h + 1), 0, D - 1)))
        pool += list(rng.integers(0, D, C))
    else:
        a = world.args(i)
        D = len(world.Aof[i])
        soft, viol = np.zeros(D), np.zeros(D, np.int64)
        cl.site_terms(*a[:-1], 0, y, x, ch, soft, viol)
        adm = np.arange(D) if FREE else np.flatnonzero(viol == 0)
        cur = int(world.cert[i][y, x])
        pool = [cur] + list(rng.permutation(adm))
    out = []
    for v in pool:
        if v not in out:
            out.append(v)
        if len(out) == C:
            break
    return out


def designed(world, i, y, x, ch):
    """(D,) designed energies of channel ch at (y, x): the soft part when
    FREE, else inf where a hard row forbids."""
    a = world.args(i)
    h = cl.HS[i]
    D = h * h + 1 if ch == cl.RHO else len(world.Aof[i])
    soft, viol = np.zeros(D), np.zeros(D, np.int64)
    cl.site_terms(*a[:-1], 0, y, x, ch, soft, viol)
    return soft if FREE else np.where(viol == 0, soft, np.inf)


def child_logz(world, i, y, x, K, M, seed):
    """log Z of the children of (y, x) of level i (level i + 1's 2 x 2 block)."""
    a = world.args(i + 1)
    if cl.HS[i + 1] == 1:
        return float(cl.block_logz_exact_tiles(*a, y, x, world.L if FREE else np.inf))
    lz, _ = cl.block_logz_ais(*a, y, x, K, M, world.L, seed)
    return float(lz)


def site_record(world, i, y, x, ch, rng, K=64, M=8, C=8):
    """One context: candidates, designed energies, targets, phi windows."""
    cand = candidates(world, i, y, x, ch, rng, C)
    ed_all = designed(world, i, y, x, ch)
    g = world.rho[i] if ch == cl.RHO else world.cert[i]
    E = world.emb[i][0] if ch == cl.RHO else world.emb[i][1]
    cur = int(g[y, x])
    pg = phi_grid(world, i)
    w0, mask = window(pg, y, x)
    ed, lz, wins = [], [], []
    for t in cand:
        g[y, x] = t
        lz.append(child_logz(world, i, y, x, K, M, int(rng.integers(1 << 30))))
        g[y, x] = cur
        ed.append(ed_all[t])
        w = w0.copy()
        w[2, 2] = w0[2, 2] - E[cur] + E[t]
        wins.append(w)
    ed, lz = np.array(ed), np.array(lz)
    logit = -ed + lz
    ok = np.isfinite(logit)
    pi = np.zeros(len(cand))
    pi[ok] = np.exp(logit[ok] - logit[ok].max())
    pi /= pi.sum()
    return dict(win=np.array(wins), mask=np.repeat(mask[None], len(cand), 0), ed=np.where(np.isfinite(ed), ed, 0.0),
                pi=pi, ok=ok & np.isfinite(ed))


def stack(recs, C=8):
    n = len(recs)
    out = dict(win=np.zeros((n, C, 5, 5, cl.KPHI)), mask=np.zeros((n, C, 5, 5)), ed=np.zeros((n, C)),
               pi=np.zeros((n, C)), ok=np.zeros((n, C), bool))
    for j, r in enumerate(recs):
        c = len(r["pi"])
        for key in out:
            out[key][j, :c] = r[key]
    return out


def contexts(world, i, n_sites, rng, K, M, seed):
    """Forward-run the world (all levels) and record n_sites sites per
    channel at level i."""
    cl.generate(world, seed=seed)
    rows, cols = world.rho[i].shape
    recs = []
    for ch in (cl.RHO, cl.CERT):
        for _ in range(n_sites):
            y, x = int(rng.integers(rows)), int(rng.integers(cols))
            recs.append(site_record(world, i, y, x, ch, rng, K, M))
    return recs


def train(world, noise_seeds, iters=3, n_sites=100, m=0, K=64, M=8, log=print, levels=(3, 2, 1, 0)):
    """Install learned potentials level by level from h = 2 up.  Each
    iteration draws a new noise field, generates, records contexts, refits
    on everything recorded at that level.  Returns per-level diagnostics."""
    rng = np.random.default_rng(0)
    diag = {}
    for i in levels:
        recs, p = [], None
        for it in range(iters):
            set_noise(world, noise_seeds[(it + 7 * i) % len(noise_seeds)])
            t0 = time.time()
            recs += contexts(world, i, n_sites, rng, K, M, seed=int(rng.integers(1 << 30)))
            t1 = time.time()
            p, ce, ce0, H = fit(stack(recs), m=m, init=p)
            world.params[i] = cl.pack_params(p)
            mk = cl.metrics(world)
            log(f"h={cl.HS[i]} it={it} contexts={len(recs)} targets {t1 - t0:.0f}s fit {time.time() - t1:.0f}s "
                f"CE {ce:.3f} (Phi=0 {ce0:.3f}, entropy {H:.3f}) | forward before refit: overhangs {mk['overhangs']} "
                f"fill_err {mk['fill_err']:.3f}")
            diag[cl.HS[i]] = dict(ce=ce, ce0=ce0, H=H, n=len(recs))
    return diag


def set_noise(world, seed):
    cl.set_fill(world, cl.density_field(world.H, world.W, seed))
