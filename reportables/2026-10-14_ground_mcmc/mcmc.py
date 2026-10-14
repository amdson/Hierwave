"""Brute-force MCMC on a small ground patch, to see what the target looks like.

Ground's factors (castlegen/bench/problems/ground.py) with T = 2: levels 0-2,
an 8 x 8 periodic patch (64 tiles), rho/cert at levels 1-2, a fixed designed
fill at the top level.  Sampler: parallel tempering over the hard-constraint
strength.  Replica k runs single-site heat-bath Gibbs (exact local
conditionals from castlegen/variants/_local.cond, colour classes in parallel)
on exp(-S - lam_k H); the last replica has lam = inf (the exact target, hard
constraints as hard).  Neighbouring replicas swap with the Metropolis rule; the
hard replica accepts only valid states.

python reportables/2026-10-14_ground_mcmc/mcmc.py [seconds]"""
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from castlegen.bench import energy as en
from castlegen.bench.problem import INF, Chan, Problem, Region, count, pair
from castlegen.bench.problems.ground import BETA, KAPPA, LAM, cert_domain, cert_index
from castlegen.variants._local import cond

T, H, W = 2, 8, 8
OUT = Path(__file__).parent
# designed fill per top block (level-2 cells, 2 x 2): the line sits at row 3 in the left
# column of blocks and row 2 in the right, so top blocks are 4/16 and 8/16 dirt
FILL = np.array([[4, 8], [16, 16]])
SURFACE = np.array([3, 3, 3, 3, 2, 2, 2, 2])          # the designed line, for drawing


def build():
    chans = [Chan("fill", T, 17, designed=lambda s, Y, X, p: FILL[Y, X % 2])]
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
    return Problem("ground_small", T=T, H=H, chans=chans, factors=fs, W_train=W, W_eval=W)


def lift(t):
    out = {"t": t.astype(np.int64)}
    for l in range(1, T + 1):
        h = 1 << l
        b = t.reshape(H // h, h, W // h, h)
        col = b.sum(1)
        out[f"rho{l}"] = b.sum((1, 3))
        out[f"cert{l}"] = cert_index(h)[(col == h).sum(2), (col > 0).sum(2)]
    return out


def classes(p, l):
    h, w = H >> l, W >> l
    for cy in range(4):
        for cx in range(4):
            Y, X = np.meshgrid(np.arange(cy, h, 4), np.arange(cx, w, 4), indexing="ij")
            if Y.size:
                yield Y.ravel(), X.ravel()


def sweep(p, x, region, lam, rng):
    for l in range(T, -1, -1):
        for Y, X in classes(p, l):
            for c in p.level_chans(l):
                if c.designed is not None:
                    continue
                soft, nv = cond(p, x, region, c.name, Y, X)
                if np.isinf(lam):
                    logits = np.where(nv == 0, -soft, -np.inf)
                    # a site with no valid value keeps its current one (cannot happen from a valid state)
                    bad = ~np.isfinite(logits).any(1)
                    logits[bad, x[c.name][Y[bad], X[bad]]] = 0.0
                else:
                    logits = -soft - lam * nv
                g = rng.gumbel(size=logits.shape)
                x[c.name][Y, X] = np.argmax(logits + g, 1)


def main():
    """Annealed importance sampling: each run starts from uniform random values,
    burns in at lam0, then raises lam geometrically to LAM1 and finally to inf
    (hard), with single-site Gibbs sweeps at every step.  Weight: log w +=
    -(lam_k - lam_{k-1}) H(x) before each move (S cancels); the last step to
    inf keeps only valid states.  Exact in expectation up to the burn-in at
    lam0 (approximate start)."""
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 300
    p = build()
    region = Region(W)
    rng = np.random.default_rng(0)
    des = p.designed_values(0, region)
    lam0, lam1, steps, burn = 0.5, 60.0, 150, 20
    lams = lam0 * (lam1 / lam0) ** (np.arange(steps + 1) / steps)

    def H_of(x):
        return sum(en.energy(p, x, region).viol.values())

    samples, logw, trace = [], [], []
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
            lw = -np.inf                                      # cannot reach the hard target
        else:
            for _ in range(5):
                sweep(p, x, region, np.inf, rng)
        samples.append(x["t"].copy())
        logw.append(lw)
        trace.append({"run": run, "t": time.time() - t0, "logw": lw, "viol_end": int(h_end),
                      "dirt": float(x["t"].mean())})
        run += 1
    logw = np.array(logw)
    w = np.exp(logw - logw[np.isfinite(logw)].max()) if np.isfinite(logw).any() else np.zeros_like(logw)
    ess = w.sum() ** 2 / max((w ** 2).sum(), 1e-300)
    print(f"{run} annealed runs in {time.time() - t0:.0f} s; valid at the end: {np.isfinite(logw).mean():.2f}; "
          f"ESS {ess:.1f}")
    np.save(OUT / "samples.npy", np.array(samples))
    np.save(OUT / "logw.npy", logw)
    json.dump({"lam0": lam0, "lam1": lam1, "steps": steps, "runs": run, "ess": ess, "trace": trace},
              open(OUT / "trace.json", "w"))
    report(np.array(samples), w)


def report(S, w):
    valid = w > 0
    keep, wk = S[valid], w[valid] / w[valid].sum()
    order = np.argsort(-wk)
    pal = np.array([[169, 212, 240], [138, 98, 66]], np.uint8)
    # 24 valid samples (unweighted, in run order), titled with their normalised weight
    idx = np.arange(len(keep))[:24]
    fig, ax = plt.subplots(3, 8, figsize=(12, 5.0))
    for a in ax.ravel():
        a.axis("off")
    for a, i in zip(ax.ravel(), idx):
        a.imshow(pal[keep[i]], interpolation="nearest")
        a.plot(np.arange(W + 1) - 0.5, np.r_[SURFACE, SURFACE[-1]] - 0.5, drawstyle="steps-post", color="k", lw=0.8)
        a.set_title(f"w {wk[i]:.2f}", fontsize=7)
    fig.suptitle("ground, 8 × 8 patch: annealed samples of the exact target (black: designed line; w = importance weight)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "samples.png", dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.4))
    ax[0].imshow((wk[:, None, None] * keep).sum(0), cmap="copper_r", vmin=0, vmax=1)
    ax[0].plot(np.arange(W + 1) - 0.5, np.r_[SURFACE, SURFACE[-1]] - 0.5, drawstyle="steps-post", color="c", lw=1)
    ax[0].set(title="P(dirt) per cell (weighted)")
    ax[1].hist(np.log(wk[wk > 0]), bins=30)
    ax[1].set(title="log normalised weights", xlabel="log w")
    fig.tight_layout()
    fig.savefig(OUT / "traces.png", dpi=110)
    plt.close(fig)
    distinct = len({k.tobytes() for k in keep})
    print(f"valid samples {len(keep)}, distinct {distinct}, weighted dirt {(wk[:, None, None] * keep).sum(0).mean():.3f}, "
          f"unweighted dirt {keep.mean():.3f}")


if __name__ == "__main__":
    main()
