"""Pictures for section 7: certificate-free MCMC from far starts, snapshots per sweep.
Rows: start (probe / random / air / dirt) for one fill seed; columns: sweeps.  Red: the designed
ground line (first dirt row of the target).  Plus a strip of final worlds for 4 fill seeds, and
full-ground chain samples (cgraded, valid at lam = 20) for comparison."""
from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from castlegen.bench.problems import ground as G
from castlegen.kernels import colour_gibbs as cg
from castlegen.variants._cpass_core import Layout

sys.path.insert(0, "reportables/2026-10-14_cmix")
sys.path.insert(0, "reportables/2026-10-14_cgraded")
from mix import starts, stripped, W                                 # noqa: E402
from allair import init_state                                       # noqa: E402

D = "reportables/2026-10-14_cmix"
SNAPS = [0, 10, 50, 100, 250, 1000]
STARTS = ["probe", "random", "air", "dirt"]


def img(t, seed, scale=2):
    rgb = G.render({"t": t}, scale=scale).copy()
    sr = G.surface_row(seed, np.arange(t.shape[1]), t.shape[1])
    for x, y in enumerate(sr):
        if 0 <= y < t.shape[0]:
            rgb[y * scale:(y + 1) * scale, x * scale:(x + 1) * scale] = [220, 40, 40]
    return rgb


def main():
    p, st0, fam, fseed = starts(8)
    ps = stripped(p)
    lay = Layout(ps, W, True)
    base = {k: np.asarray(v) for k, v in st0.items() if k in ps.chan}
    seeds = [0, 1, 2, 3]
    idx = np.array([np.flatnonzero((fam == 0) & (fseed == s))[0] for s in seeds])
    b = {k: v[idx] for k, v in base.items()}
    rng = np.random.default_rng(0)
    snaps = {}
    for kind in STARTS:
        st = init_state(kind, ps, b, rng)
        lam = 20.0 if kind == "random" else None
        key = jax.random.PRNGKey(1)
        for s in range(SNAPS[-1] + 1):
            if s:
                key, sub = jax.random.split(key)
                st = cg.sweep_all(lay, st, 1.0, lam, sub)
            if s in SNAPS:
                snaps[(kind, s)] = np.asarray(st["t"])
        print(kind, "done", flush=True)
    np.savez_compressed(f"{D}/allair_snaps.npz", **{f"{k}_{s}": v for (k, s), v in snaps.items()})

    # figure 1: evolution for fill seed 0
    fig, axs = plt.subplots(len(STARTS), len(SNAPS), figsize=(3.0 * len(SNAPS), 1.75 * len(STARTS)))
    for i, kind in enumerate(STARTS):
        for j, s in enumerate(SNAPS):
            ax = axs[i, j]
            ax.imshow(img(snaps[(kind, s)][0], seeds[0]))
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"sweep {s}", fontsize=10)
            if j == 0:
                ax.set_ylabel(kind, fontsize=10)
    fig.suptitle("certificate-free MCMC on ground from far starts (fill seed 0; red = designed ground line)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{D}/allair_evolution.png", dpi=100)
    fig.savefig("images/cmix_allair_evolution.png", dpi=100)

    # figure 2: final worlds (sweep 1000) for 4 fills, from the random start, vs full-ground chain samples
    z = np.load("reportables/2026-10-14_cgraded/final_binary_lam20.npz")
    fig, axs = plt.subplots(2, 4, figsize=(12, 3.6))
    for j, s in enumerate(seeds):
        axs[0, j].imshow(img(snaps[("random", 1000)][j], s))
        axs[1, j].imshow(img(z["t"][s], s))
        axs[0, j].set_title(f"fill seed {s}", fontsize=10)
        for ax in axs[:, j]:
            ax.set_xticks([]); ax.set_yticks([])
    axs[0, 0].set_ylabel("cert-free\nMCMC, 1000", fontsize=9)
    axs[1, 0].set_ylabel("full ground\nchain (cgraded)", fontsize=9)
    fig.suptitle("final worlds: certificate-free MCMC alone vs the full-ground chain (red = designed ground line)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{D}/allair_finals.png", dpi=100)
    fig.savefig("images/cmix_allair_finals.png", dpi=100)


if __name__ == "__main__":
    main()
