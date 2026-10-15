"""Mixing metrics and figure for cmix traces.  Usage: analyse.py arm1,arm2,... [figure name]"""
from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from castlegen.bench.problems import ground as G

D = "reportables/2026-10-14_cmix"
W = 128
FAM = ["smooth", "ragged", "up", "down"]
FCOL = ["#2a78d6", "#1baf7a", "#eb6834", "#e87ba4"]
OBS = ["offset", "rough", "S", "c1re", "c1im", "c2re", "c2im"]


def observables(tr):
    h = tr["h"].astype(np.float64)                     # (T, B, W)
    line = np.stack([G.surface_row(int(s), np.arange(W), W) for s in tr["fseed"]])   # (B, W)
    dev = h - line[None]
    f = np.fft.rfft(dev, axis=-1) / W
    o = {"offset": dev.mean(-1), "rough": np.abs(np.diff(h, axis=-1, append=h[..., :1])).mean(-1),
         "S": tr["S"], "c1re": f[..., 1].real, "c1im": f[..., 1].imag, "c2re": f[..., 2].real, "c2im": f[..., 2].imag}
    return o, dev


def rhat(x):
    """x: list of 1-D arrays (one per chain).  Gelman-Rubin (non-split) R-hat; inf if a chain is empty."""
    if any(len(a) < 2 for a in x):
        return np.inf
    m = np.array([a.mean() for a in x])
    v = np.array([a.var(ddof=1) for a in x])
    n = min(len(a) for a in x)
    Wv = v.mean()
    Bv = n * m.var(ddof=1)
    if Wv <= 1e-12:
        return np.inf if Bv > 1e-12 else 1.0
    return float(np.sqrt(((n - 1) / n * Wv + Bv / n) / Wv))


def tau_int(x):
    """Integrated autocorrelation time (in records) with Sokal's window c = 5."""
    x = x - x.mean()
    n = len(x)
    if n < 10 or x.var() < 1e-12:
        return np.nan
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac /= ac[0]
    t = 1.0
    for k in range(1, n):
        t += 2 * ac[k]
        if k >= 5 * t:
            break
    return t


def analyse(arm):
    tr = dict(np.load(f"{D}/trace_{arm}.npz"))
    o, dev = observables(tr)
    sw = tr["sweep"]
    valid = tr["N"] == 0                               # (T, B)
    fam, fs = tr["fam"], tr["fseed"]
    T = len(sw)
    res = {"arm": arm, "secs": float(tr["secs"]), "sweeps": int(sw[-1]),
           "valid_frac": float(valid[1:].mean()), "mean_N": float(tr["N"][1:].mean()),
           "mean_M": float(tr["M"][1:].mean())}
    # distance travelled from the start (rows^2 per column), valid records only
    msd = ((dev - dev[0:1]) ** 2).mean(-1)            # (T, B)
    res["msd_end"] = float(np.median(msd[-1]))
    # family agreement over time: window [t/2, t]
    rh_t = []
    for ti in range(1, T):
        lo = ti // 2
        per_seed = []
        for s in np.unique(fs):
            chains = [np.flatnonzero(fs == s)[fam[fs == s] == k][0] for k in range(4)]
            worst = 1.0
            for name in OBS:
                x = [o[name][lo:ti + 1, b][valid[lo:ti + 1, b]] for b in chains]
                worst = max(worst, rhat(x))
            per_seed.append(worst)
        rh_t.append(float(np.median(per_seed)))
    rh_t = np.array(rh_t)
    res["rhat_end"] = float(rh_t[-1])
    lo = (T - 1) // 2
    byobs = {}
    for name in OBS:
        vals = []
        for s in np.unique(fs):
            chains = [np.flatnonzero(fs == s)[fam[fs == s] == k][0] for k in range(4)]
            vals.append(rhat([o[name][lo:, b][valid[lo:, b]] for b in chains]))
        byobs[name] = round(float(np.median(vals)), 3)
    res["rhat_end_by_obs"] = byobs
    res["mean_by_family_late"] = {name: [round(float(np.nanmean(np.where(valid[lo:], o[name][lo:], np.nan)[:, fam == k])), 3)
                                         for k in range(4)] for name in OBS}
    ok = np.flatnonzero(rh_t < 1.1)
    res["agree_sweep"] = int(sw[1:][ok[0]]) if ok.size else None
    # autocorrelation (second half, valid-only chains), in sweeps
    half = slice(T // 2, T)
    taus = {}
    for name in ("offset", "c1re", "S"):
        tt = [tau_int(o[name][half, b]) for b in range(fam.size) if valid[half, b].all()]
        tt = [t for t in tt if np.isfinite(t)]
        taus[name] = float(np.median(tt)) * (sw[1] - sw[0]) if tt else None
    res["tau_sweeps"] = taus
    # shift recovery: mean offset per family at the end vs the smooth family
    rel = {}
    for k in (2, 3):
        m = np.abs(o["offset"][:, fam == k].mean(1))
        rel[FAM[k]] = {thr: (int(sw[np.argmax(m < v)]) if (m < v).any() else None)
                       for thr, v in (("half", 3.0), ("tenth", 0.6))}
    res["shift_relax_sweeps"] = rel
    res["offset_end_by_family"] = {FAM[k]: float(o["offset"][-1][fam == k].mean()) for k in range(4)}
    res["offset_start_by_family"] = {FAM[k]: float(o["offset"][0][fam == k].mean()) for k in range(4)}
    for k in ("swap_acc", "round_trips"):
        if k in tr:
            res[k] = np.asarray(tr[k]).tolist()
    return res, sw, o, rh_t, fam


def main(arms, figname):
    out = []
    fig, axs = plt.subplots(2, len(arms), figsize=(3.2 * len(arms), 6.4), squeeze=False, sharey="row")
    for j, arm in enumerate(arms):
        res, sw, o, rh_t, fam = analyse(arm)
        out.append(res)
        print(json.dumps(res))
        ax = axs[0, j]
        for k in range(4):
            ax.plot(sw, o["offset"][:, fam == k].mean(1), color=FCOL[k], lw=2, label=FAM[k])
        ax.axhline(0, color="#888", lw=0.6)
        ax.set_title(arm, fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
        ax = axs[1, j]
        ax.plot(sw[1:], np.minimum(rh_t, 10), color="#2a78d6", lw=2)
        ax.axhline(1.1, color="#888", lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_xlabel("sweep")
        ax.grid(alpha=0.25, lw=0.5)
    axs[0, 0].set_ylabel("mean surface offset (rows, + = lower)")
    axs[1, 0].set_ylabel("family R-hat, median over fills (cap 10)")
    axs[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("ground: do chains from 4 valid start families reach the same terrain distribution?", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{D}/{figname}.png", dpi=110)
    fig.savefig(f"images/cmix_{figname}.png", dpi=110)
    json.dump(out, open(f"{D}/metrics_{figname}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1].split(","), sys.argv[2] if len(sys.argv) > 2 else "pilot")
