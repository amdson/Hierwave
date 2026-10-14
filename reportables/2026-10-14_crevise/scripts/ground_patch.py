"""The three-level ground patch of reportables/2026-10-14_ground_mcmc/mcmc.py (T = 2, H = 8) on a periodic
width of 16 (the coloured-pass layout needs every level's width to be a multiple of 4, so the 8-wide patch
cannot be laid out; the fill pattern [[4, 8], [16, 16]] is tiled twice, which keeps every local boundary of
the 8 x 8 patch: the seam between copies is the same row-3 / row-2 step as the 8-periodic wrap).

python ground_patch.py mcmc <seconds>      annealed-importance reference samples (CPU, numpy; mcmc.py's sampler)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from castlegen.bench import energy as en
from castlegen.bench.problem import INF, Chan, Problem, Region, count, pair
from castlegen.bench.problems.ground import BETA, KAPPA, LAM, cert_domain, cert_index

T, H, W = 2, 8, 16
OUT = Path(__file__).resolve().parent.parent / "runs"
FILL = np.array([[4, 8], [16, 16]])
SURFACE = np.array([3, 3, 3, 3, 2, 2, 2, 2] * 2)


def build():
    chans = [Chan("fill", T, 17, designed=lambda s, Y, X, p: FILL[np.asarray(Y), np.asarray(X) % 2])]
    for l in range(T, 0, -1):
        h = 1 << l
        A, B = cert_domain(h)
        chans.append(Chan(f"rho{l}", l, h * h + 1))
        chans.append(Chan(f"cert{l}", l, A.size, views={"A": A, "B": B}))
    chans.append(Chan("t", 0, 2, views={"A": [0, 1], "B": [0, 1]}))
    rho = lambda l: "t" if l == 0 else f"rho{l}"
    cert = lambda l: "t" if l == 0 else f"cert{l}"
    n = (1 << T) ** 2
    r = np.arange(n + 1)[:, None]
    f = np.arange(17)[None, :] / 16
    fs = [pair((rho(T), "id"), ("fill", "id"), (0, 0), BETA * (r - n * f) ** 2 / n, name="noise")]
    for l in range(T, -1, -1):
        h = 1 << l
        v = np.arange(h * h + 1)
        fs.append(pair((rho(l), "id"), (rho(l), "id"), (0, 1), LAM * np.abs(v[:, None] - v[None, :]) / h,
                       name=f"smooth{l}"))
        Bv, Av = np.arange(h + 1)[:, None], np.arange(h + 1)[None, :]
        fs.append(pair((cert(l), "B"), (cert(l), "A"), (1, 0), np.where(Bv > Av, INF, 0.0), pad_b=h,
                       name=f"support{l}"))
        if l == 0:
            continue
        hc = h >> 1
        s = np.arange(4 * hc * hc + 1)[None, :]
        fs.append(count((rho(l - 1), "id"), (rho(l), "id"), KAPPA * (s - v[:, None]) ** 2 / (hc * hc),
                        name=f"prior{l}"))
        P = np.arange(h + 1)[:, None]
        s = np.arange(2 * hc + 1)[None, :]
        for row, sub in (("top", (0, 1)), ("bot", (2, 3))):
            fs.append(count((cert(l - 1), "A"), (cert(l), "A"), np.where(s < P, INF, 0.0), subset=sub,
                            name=f"honourA{l}{row}"))
            fs.append(count((cert(l - 1), "B"), (cert(l), "B"), np.where(s > P, INF, 0.0), subset=sub,
                            name=f"honourB{l}{row}"))
    return Problem("ground_patch16", T=T, H=H, chans=chans, factors=fs, W_train=W, W_eval=W)


def lift(t):
    out = {"t": np.asarray(t, np.int64)}
    for l in range(1, T + 1):
        h = 1 << l
        b = out["t"].reshape(H // h, h, W // h, h)
        col = b.sum(1)
        out[f"rho{l}"] = b.sum((1, 3))
        out[f"cert{l}"] = cert_index(h)[(col == h).sum(2), (col > 0).sum(2)]
    return out


# ------------------------------------------------------------------ MCMC reference (mcmc.py's sampler)
def classes(l):
    h, w = H >> l, W >> l
    for cy in range(4):
        for cx in range(4):
            Y, X = np.meshgrid(np.arange(cy, h, 4), np.arange(cx, w, 4), indexing="ij")
            if Y.size:
                yield Y.ravel(), X.ravel()


def sweep(p, x, region, lam, rng, tau=1.0):
    from castlegen.variants._local import cond
    for l in range(T, -1, -1):
        for Y, X in classes(l):
            for c in p.level_chans(l):
                if c.designed is not None:
                    continue
                soft, nv = cond(p, x, region, c.name, Y, X)
                if np.isinf(lam):
                    logits = np.where(nv == 0, -soft / tau, -np.inf)
                    bad = ~np.isfinite(logits).any(1)
                    logits[bad, x[c.name][Y[bad], X[bad]]] = 0.0
                else:
                    logits = -soft / tau - lam * nv
                g = rng.gumbel(size=logits.shape)
                x[c.name][Y, X] = np.argmax(logits + g, 1)


def mcmc(budget):
    p = build()
    region = Region(W)
    rng = np.random.default_rng(0)
    des = p.designed_values(0, region)
    lam0, lam1, steps, burn = 0.5, 60.0, 150, 20
    lams = lam0 * (lam1 / lam0) ** (np.arange(steps + 1) / steps)
    H_of = lambda x: sum(en.energy(p, x, region).viol.values())
    samples, logw, S = [], [], []
    t0, run = time.time(), 0
    while time.time() - t0 < budget:
        x = {**des}
        for c in p.sampled():
            x[c.name] = rng.integers(0, c.D, p.shape(c.name, region))
        for _ in range(burn):
            sweep(p, x, region, lam0, rng)
        lw = 0.0
        for k in range(1, steps + 1):
            lw -= (lams[k] - lams[k - 1]) * H_of(x)
            sweep(p, x, region, lams[k], rng)
        h_end = H_of(x)
        if h_end > 0:
            lw = -np.inf
        else:
            for _ in range(5):
                sweep(p, x, region, np.inf, rng)
        samples.append(x["t"].copy())
        logw.append(lw)
        S.append(en.energy(p, x, region).soft)
        run += 1
        print(f"run {run} t {time.time() - t0:.0f} logw {lw:.2f} dirt {x['t'].mean():.3f}", flush=True)
    logw = np.array(logw)
    fin = np.isfinite(logw)
    w = np.exp(logw - logw[fin].max()) if fin.any() else np.zeros_like(logw)
    w = w / w.sum()
    ess = 1.0 / np.sum(w ** 2)
    S = np.array(S)
    samples = np.array(samples)
    pd = (w[:, None, None] * samples).sum(0)
    out = {"runs": run, "valid": int(fin.sum()), "ess": float(ess), "dirt_weighted": float(pd.mean()),
           "dirt_unweighted": float(samples[fin].mean()), "soft_weighted": float(np.sum(w * S)),
           "p_dirt": pd.tolist(), "p_dirt_fold8": (0.5 * (pd[:, :8] + pd[:, 8:])).tolist(),
           "distinct": len({s.tobytes() for s in samples[fin]})}
    np.save(OUT / "ground_patch_mcmc_samples.npy", samples)
    np.save(OUT / "ground_patch_mcmc_logw.npy", logw)
    json.dump(out, open(OUT / "ground_patch_mcmc.json", "w"))
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("p_dirt")}))


if __name__ == "__main__":
    if sys.argv[1] == "mcmc":
        mcmc(float(sys.argv[2]))
