"""The Potts test (notes/potts_test.md): train the six learned coarse tables
(mid_h, mid_v, mid_u, top_h, top_v, top_u) of the top-down forward model by
self-play against the designed bidirectional energy p*, with four schemes:

  S0   oracle PCD: gap = forward stats - Oracle.moments (per-site frequencies)
  S1   per-level RB, one step: from each forward sample, the Rao-Blackwellised
       expected count change under one-site moves of p*; top sites by the
       designed-only conditional site_probs(m_star, "top", below=True), mid
       sites by the collapsed conditional Oracle.mid_probs (tiles integrated out)
  S2   as S1 for the mid tables; for the top tables first relax (top, mid) K
       rounds under E_lam + theta_mid (no tiles), then the RB change; the top
       count change accumulated over the relaxation is included
  S3   end-to-end CD-K: K oracle sweeps (top moves, collapsed mid moves, 2
       tile sweeps) applied to the forward state; gap = -(N(relaxed) - N(fwd)
       + RB change at the relaxed state), so K = 0 is S1 and K -> inf is S0
  S2rb, S3rb  the same relaxations but only the RB change at the relaxed
       state (no accumulated count change), as a diagnostic
  S1f  S1 with the plain (frozen at kappa = 8) one-site mid conditional

The oracle target is symmetrised over the symmetries of p* (colour rotation
and reflection with the induced palette map, lattice transpose / mirror):
at the spec's dials p* is in its ordered phase and a finite oracle chain
sits in one colour sector.

gap[name] = -rb_gap[name] / n_sites(level) for the RB schemes.  The target
kernel is always p*'s (designed factors only, m_star); the counts are read on
m_full (designed + learned) sharing the same Channel objects.

    NUMBA_NUM_THREADS=1 PYTHONPATH=. python notes/experiments/potts_train.py
"""
import os, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from castlegen.channels.potts import Potts, Forward, Oracle, dist
from castlegen.channels.core import Model
from castlegen.channels.train import site_probs, rb_gap, step, counts

NT = int(os.environ.get("NT", 6))
JJ, LAM = float(os.environ.get("J", 1.0)), float(os.environ.get("LAM", 3.0))
ONLY8 = {"S1f", "S2rb", "S3rb"}                                       # diagnostics run at kappa = 8 only
KAPPAS = [float(k) for k in os.environ.get("KAPPAS", "8,1").split(",")]
SCHEMES = os.environ.get("SCHEMES", "S0,S1,S2,S3,S1f,S2rb,S3rb").split(",")
RUNS, STEPS, K = int(os.environ.get("RUNS", 4)), int(os.environ.get("STEPS", 150)), int(os.environ.get("K", 3))
BURN, SWEEPS = int(os.environ.get("BURN", 50)), int(os.environ.get("SWEEPS", 200))
S_T, S_M, S_F = int(os.environ.get("S_T", 30)), int(os.environ.get("S_M", 30)), int(os.environ.get("S_F", 20))
ETA = float(os.environ.get("ETA", 2.0))
EVAL, EVERY, FINAL = int(os.environ.get("EVAL", 16)), int(os.environ.get("EVERY", 10)), int(os.environ.get("FINAL", 64))
SEED = int(os.environ.get("SEED", 0))
OUT = os.environ.get("OUT", "images/potts_results.json")
TAG = os.environ.get("TAG", "")
FIT = ["mid_h", "mid_v", "mid_u", "top_h", "top_v", "top_u"]
TOPK = ["top_h", "top_v", "top_u"]
MON = ["tile_d", "mismatch", "pal_viol", "edge_same_mid", "edge_same_top"]
os.makedirs("images", exist_ok=True)
print(f"nty=ntx={NT}  RUNS {RUNS}  STEPS {STEPS}  K {K}  BURN {BURN}  SWEEPS {SWEEPS}  S_T/S_M/S_F {S_T}/{S_M}/{S_F}  "
      f"ETA {ETA}  EVAL {EVAL} every {EVERY}  FINAL {FINAL}  schemes {SCHEMES}  kappas {KAPPAS}")


def symmetrise(P, st):
    """Average the fitted stats over the symmetry group of p*: colour maps
    c -> s c + r (s = +-1) with the induced palette map, and the lattice
    transpose / mirror (h <-> v, a pair table <-> its transpose).  The
    monitors are invariant and kept."""
    q = P.q
    out = {k: np.zeros_like(np.asarray(st[k], float)) for k in FIT}
    n = 0
    for sgn in (1, -1):
        for r in range(q):
            sig = (sgn * np.arange(q) + r) % q                         # colour c -> sig[c]
            pi = np.array([np.flatnonzero((P.PAL[:, sig[P.PAL[p]]]).all(1))[0] for p in range(P.P)])
            for k in FIT:
                a = np.asarray(st[k], float)
                m = sig if k.startswith("mid") else pi
                b = np.zeros_like(a)
                if a.ndim == 1:
                    b[m] = a
                else:
                    b[np.ix_(m, m)] = a
                out[k] += b
            n += 1
    out = {k: v / n for k, v in out.items()}
    for lv in ("mid", "top"):
        h, v = out[lv + "_h"], out[lv + "_v"]
        s = (h + h.T + v + v.T) / 4
        out[lv + "_h"], out[lv + "_v"] = s, s.copy()
    res = dict(st)
    res.update(out)
    return res


def mean_stats(sts):
    return {k: np.mean([s[k] for s in sts], axis=0) for k in sts[0]}


def l1(a, b, keys):
    return {k: float(np.abs(np.asarray(a[k]) - np.asarray(b[k])).sum()) for k in keys}


def tolist(d):
    return {k: (np.asarray(v).tolist() if not isinstance(v, (float, int)) else v) for k, v in d.items()}


class Trainer:
    """One forward model whose channels ARE the oracle's channels, so the
    oracle's moves, the designed-only model m_star (= oracle.model) and the
    forward's full model m_full (= fw.model, rebuilt in run) all act on the
    same grids."""

    def __init__(self, P, theta, seed):
        self.P = P
        self.orc = Oracle(P, seed=seed + 1000)
        self.fw = Forward(P, theta, seed=seed)
        self.fw.top, self.fw.mid, self.fw.tile = self.orc.top, self.orc.mid, self.orc.tile
        self.m_star = self.orc.model
        self.names = FIT

    def run(self, theta):
        self.fw.theta = theta
        return self.fw.run(S_T, S_M, S_F, fresh=True)

    def mid_probs_all(self):
        P = self.P
        pr = np.empty((P.nmy, P.nmx, P.q))
        for i in range(P.nmy):
            for j in range(P.nmx):
                pr[i, j] = self.orc.mid_probs(i, j)
        return pr

    def rb(self, home, probs):
        return rb_gap(self.fw.model, home, self.names, probs)

    def gap(self, scheme, st, target):
        """Per-site gap (forward - target) for one forward sample in the shared grids."""
        P = self.P
        if scheme == "S0":
            return {k: st[k] - target[k] for k in FIT}
        nt, nm = P.nty * P.ntx, P.nmy * P.nmx
        if scheme in ("S1", "S2", "S1f", "S2rb"):
            pm = site_probs(self.m_star, "mid", below=True) if scheme == "S1f" else self.mid_probs_all()
            gm = self.rb("mid", pm)
            if scheme in ("S2", "S2rb"):           # relax (top, mid) under E_lam + theta_mid, no tiles
                n0 = counts(self.fw.model, TOPK)
                for _ in range(K):
                    self.fw.model.sweep("mid", 1, seed=int(self.orc.rng.integers(1 << 30)), below=False)
                    self.m_star.sweep("top", 1, seed=int(self.orc.rng.integers(1 << 30)), below=True)
            gt = self.rb("top", site_probs(self.m_star, "top", below=True))
            if scheme == "S2":
                n1 = counts(self.fw.model, TOPK)
                gt = {k: gt[k] + n1[k] - n0[k] if k in n1 else gt[k] for k in gt}
        elif scheme in ("S3", "S3rb"):
            n0 = counts(self.fw.model, FIT)
            self.orc.sweep(K, tile_sweeps=2)
            gm = self.rb("mid", self.mid_probs_all())
            gt = self.rb("top", site_probs(self.m_star, "top", below=True))
            if scheme == "S3":
                n1 = counts(self.fw.model, FIT)
                gm = {k: gm[k] + n1[k] - n0[k] for k in gm}
                gt = {k: gt[k] + n1[k] - n0[k] for k in gt}
        else:
            raise ValueError(scheme)
        return {k: -(gm[k] / nm if k.startswith("mid") else gt[k] / nt) for k in FIT}


def self_check(P):
    tr = Trainer(P, P.theta0(), seed=99)
    tr.run(P.theta0())
    rng = np.random.default_rng(5)
    tr.orc.mid.grid[:] = rng.integers(P.q, size=tr.orc.mid.grid.shape)
    tr.orc.top.grid[:] = rng.integers(P.P, size=tr.orc.top.grid.shape)
    pt = site_probs(tr.m_star, "top", below=True)
    mg = tr.orc.mid.grid
    for i in range(P.nty):
        for j in range(P.ntx):
            blk = mg[i * P.BT:(i + 1) * P.BT, j * P.BT:(j + 1) * P.BT].ravel()
            nout = np.array([(~P.PAL[p, blk]).sum() for p in range(P.P)])
            ref = np.exp(-P.lam * nout); ref /= ref.sum()
            assert np.allclose(pt[i, j], ref), (i, j, pt[i, j], ref)
    # rb_gap on the top unary: sum over sites of (probs - onehot(current))
    g = tr.rb("top", pt)
    ref = pt.sum((0, 1)) - np.bincount(tr.orc.top.grid.ravel(), minlength=P.P)
    assert np.allclose(g["top_u"], ref)
    assert np.allclose(g["mid_h"], 0)
    # the shared grids: m_star, m_full and the oracle see the same arrays
    assert tr.m_star.chan("mid").grid is tr.fw.model.chan("mid").grid is tr.orc.mid.grid
    print("self-check ok: site_probs(m_star, top) = softmax(-lam * #mid outside palette); rb top_u = sum(probs - onehot)")


def forward_eval(P, theta, n, seed):
    fw = Forward(P, theta, seed=seed)                     # fixed eval seed: common random numbers across evaluations
    sts = [fw.run(S_T, S_M, S_F, fresh=True) for _ in range(n)]
    return mean_stats(sts), fw


def save_render(P, chans, path, px=6):
    img = P.render(*chans)
    Image.fromarray(np.repeat(np.repeat(img, px, 0), px, 1)).save(path)


results = json.load(open(OUT)) if os.path.exists(OUT) and os.environ.get("APPEND", "1") == "1" else {}
CONFIG = dict(J=JJ, LAM=LAM, NT=NT, RUNS=RUNS, STEPS=STEPS, K=K, BURN=BURN, SWEEPS=SWEEPS, S_T=S_T, S_M=S_M, S_F=S_F,
                         ETA=ETA, EVAL=EVAL, EVERY=EVERY, FINAL=FINAL, SEED=SEED)

for kappa in KAPPAS:
    kk = f"{kappa:g}"
    P = Potts(NT, NT, J=JJ, kappa=kappa, lam=LAM)
    print(f"\n===== kappa {kk}: {P.nty}x{P.ntx} top, {P.nmy}x{P.nmx} mid, {P.H}x{P.W} tiles")
    self_check(P)
    t0 = time.time()
    orc = Oracle(P, seed=SEED + 7)
    raw = orc.moments(BURN, SWEEPS)
    target = symmetrise(P, raw)
    print("oracle raw mid_u", np.round(raw["mid_u"], 3), "top_u", np.round(raw["top_u"], 3),
          "-> symmetrised;  edge_same_top", round(raw["edge_same_top"], 3), " pal_viol", round(raw["pal_viol"], 4))
    t_orc = time.time() - t0
    print(f"oracle moments: {BURN}+{SWEEPS} sweeps {t_orc:.1f}s  ({t_orc / (BURN + SWEEPS) * 1000:.0f} ms/sweep)")
    save_render(P, (orc.top, orc.mid, orc.tile), f"images/potts{TAG}_k{kk}_oracle.png")
    st_u, fw_u = forward_eval(P, P.theta0(), FINAL, SEED + 555)
    save_render(P, (fw_u.top, fw_u.mid, fw_u.tile), f"images/potts{TAG}_k{kk}_untrained.png")
    R = dict(oracle=tolist(target), oracle_raw=tolist(raw), untrained=tolist(st_u), untrained_L1=l1(st_u, target, FIT), schemes={})
    print("untrained L1:", {k: round(v, 3) for k, v in R["untrained_L1"].items()})
    print(f"  edge_same_top oracle {target['edge_same_top']:.3f} untrained {st_u['edge_same_top']:.3f}")

    # magnitude check: S0 gap vs RB gaps for the untrained model, averaged over RUNS samples
    tr = Trainer(P, P.theta0(), seed=SEED + 3)
    for sc in ("S0", "S1", "S3"):
        if sc not in SCHEMES:
            continue
        gs = []
        for _ in range(RUNS):
            st = tr.run(P.theta0())
            gs.append(tr.gap(sc, st, target))
        g = mean_stats(gs)
        print(f"  untrained gap {sc}: " + "  ".join(f"{k} |g|_1 {np.abs(g[k]).sum():.3f}" for k in FIT))

    for sc in SCHEMES:
        if sc in ONLY8 and kappa != 8:
            continue
        theta = P.theta0()
        tr = Trainer(P, theta, seed=SEED + 11)
        curve = [None]
        ev0, _ = forward_eval(P, theta, EVAL, SEED + 777)
        curve[0] = dict(step=0, **l1(ev0, target, FIT))
        t_train = 0.0
        for s in range(STEPS):
            t1 = time.time()
            gs = []
            for r in range(RUNS):
                st = tr.run(theta)
                gs.append(tr.gap(sc, st, target))
            theta = step(theta, mean_stats(gs), eta=ETA)
            t_train += time.time() - t1
            if (s + 1) % EVERY == 0:
                ev, _ = forward_eval(P, theta, EVAL, SEED + 777)
                c = dict(step=s + 1, **l1(ev, target, FIT))
                curve.append(c)
                if (s + 1) % (EVERY * 5) == 0 or s + 1 == STEPS:
                    print(f"  {sc} step {s + 1:4d}  " + "  ".join(f"{k} {c[k]:.3f}" for k in FIT)
                          + f"  est {ev['edge_same_top']:.3f}  ({t_train / (s + 1):.2f} s/step)")
        stf, fwf = forward_eval(P, theta, FINAL, SEED + 555)
        save_render(P, (fwf.top, fwf.mid, fwf.tile), f"images/potts{TAG}_k{kk}_{sc}.png")
        st2, _ = forward_eval(P, theta, FINAL, SEED + 556)            # a second, independent final evaluation
        R["schemes"][sc] = dict(final=tolist(stf), L1=l1(stf, target, FIT), theta=tolist(theta), curve=curve,
                                s_per_step=t_train / STEPS, eval_noise=l1(stf, st2, FIT))
        print(f"{sc} done: {t_train:.0f}s ({t_train / STEPS:.2f} s/step)  final L1 "
              + "  ".join(f"{k} {v:.3f}" for k, v in R["schemes"][sc]["L1"].items())
              + f"  edge_same_top {stf['edge_same_top']:.3f} (oracle {target['edge_same_top']:.3f})")
    R["config"] = CONFIG
    results[f"k{kk}_J{JJ:g}"] = R

    # ---- figures
    scs = list(R["schemes"])
    fig, axs = plt.subplots(1, len(FIT), figsize=(3.2 * len(FIT), 3), constrained_layout=True)
    for ax, k in zip(axs, FIT):
        for sc in scs:
            cv = R["schemes"][sc]["curve"]
            ax.plot([c["step"] for c in cv], [c[k] for c in cv], label=sc, lw=1.5)
        ax.axhline(R["untrained_L1"][k], color="0.6", ls=":", lw=1)
        ax.set_title(k); ax.set_xlabel("step"); ax.set_ylim(bottom=0)
    axs[0].set_ylabel("L1 to oracle moments"); axs[-1].legend(fontsize=8, frameon=False)
    fig.suptitle(f"kappa {kk}: forward vs oracle, fitted features (eval {EVAL} runs, ETA {ETA})")
    fig.savefig(f"images/potts{TAG}_k{kk}_curves.png", dpi=110); plt.close(fig)

    cc = np.arange(P.q)
    ref_mid = P.BM * P.J * dist(cc[:, None], cc[None, :], P.q).astype(float)
    ref_mid -= ref_mid.mean()
    fig, axs = plt.subplots(2, len(scs) + 1, figsize=(2.7 * (len(scs) + 1), 5.4), constrained_layout=True)
    vm = max(np.abs(np.array(R["schemes"][sc]["theta"]["top_h"])).max() for sc in scs) or 1
    vmm = max(np.abs(ref_mid).max(), max(np.abs(np.array(R["schemes"][sc]["theta"]["mid_h"])).max() for sc in scs))
    axs[0, 0].imshow(ref_mid, cmap="RdBu_r", vmin=-vmm, vmax=vmm); axs[0, 0].set_title("BM J d (ref)")
    axs[1, 0].axis("off")
    for c, sc in enumerate(scs, 1):
        th = R["schemes"][sc]["theta"]
        axs[0, c].imshow(np.array(th["mid_h"]), cmap="RdBu_r", vmin=-vmm, vmax=vmm); axs[0, c].set_title(f"{sc} mid_h")
        im = axs[1, c].imshow(np.array(th["top_h"]), cmap="RdBu_r", vmin=-vm, vmax=vm); axs[1, c].set_title(f"{sc} top_h")
        for r_ in (0, 1):
            for i in range(P.q):
                for j in range(P.q):
                    v = np.array(th["mid_h" if r_ == 0 else "top_h"])[i, j]
                    axs[r_, c].text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7)
    for i in range(P.q):
        for j in range(P.q):
            axs[0, 0].text(j, i, f"{ref_mid[i, j]:+.1f}", ha="center", va="center", fontsize=7)
    for ax in axs.ravel():
        ax.set_xticks(range(P.q)); ax.set_yticks(range(P.q))
    fig.suptitle(f"kappa {kk}: learned tables (zero-mean); rows = left cell, cols = right cell")
    fig.savefig(f"images/potts{TAG}_k{kk}_tables.png", dpi=110); plt.close(fig)
    json.dump(results, open(OUT, "w"))
    print("wrote", OUT)

# ---- markdown tables (pasted into notes/potts_test.md)
lines = []
for kk in [f"{k:g}" for k in KAPPAS]:
    R = results[f"k{kk}_J{JJ:g}"]
    scs = list(R["schemes"])
    lines.append(f"\n### kappa = {kk}, J = {JJ:g}\n")
    lines.append("L1 distance to the oracle moments on the fitted features (final, " f"{FINAL} forward runs):\n")
    lines.append("| | " + " | ".join(FIT) + " |")
    lines.append("|---" * (len(FIT) + 1) + "|")
    lines.append("| untrained | " + " | ".join(f"{R['untrained_L1'][k]:.3f}" for k in FIT) + " |")
    for sc in scs:
        lines.append(f"| {sc} | " + " | ".join(f"{R['schemes'][sc]['L1'][k]:.3f}" for k in FIT) + " |")
    for sc in scs:
        lines.append(f"| eval noise {sc} | " + " | ".join(f"{R['schemes'][sc]['eval_noise'][k]:.3f}" for k in FIT) + " |")
    lines.append("\nMonitors (tile_d = fraction of tile pairs at distance 0/1/2; edge_same_top held out):\n")
    lines.append("| | tile_d | mismatch | pal_viol | edge_same_mid | edge_same_top | s/step |")
    lines.append("|---|---|---|---|---|---|---|")
    rows = [("oracle", R["oracle"], None), ("untrained", R["untrained"], None)] + \
           [(sc, R["schemes"][sc]["final"], R["schemes"][sc]["s_per_step"]) for sc in scs]
    for nm, st, sps in rows:
        td = "/".join(f"{v:.3f}" for v in st["tile_d"])
        lines.append(f"| {nm} | {td} | {st['mismatch']:.4f} | {st['pal_viol']:.4f} | {st['edge_same_mid']:.3f} | "
                     f"{st['edge_same_top']:.3f} | {'' if sps is None else f'{sps:.2f}'} |")
    P = Potts(NT, NT, J=JJ, kappa=float(kk), lam=LAM)
    cc = np.arange(P.q)
    ref = P.BM * P.J * dist(cc[:, None], cc[None, :], P.q).astype(float)
    ref -= ref.mean()
    lines.append(f"\nmid_h vs BM J d(m, m') zero-meaned (d = 0 / 1 / 2 reference {ref[0, 0]:+.2f} / {ref[0, 1]:+.2f} / "
                 f"{ref[0, 2]:+.2f}); learned mean over entries by distance, and max |mid_h - ref|; top_h rows:\n")
    lines.append("| | mid_h d=0 | d=1 | d=2 | max abs err | top_h (rows p = 0..3) |")
    lines.append("|---|---|---|---|---|---|")
    D = dist(cc[:, None], cc[None, :], P.q)
    for sc in scs:
        th = np.array(R["schemes"][sc]["theta"]["mid_h"]); tt = np.array(R["schemes"][sc]["theta"]["top_h"])
        md = [th[D == d].mean() for d in range(3)]
        trow = "; ".join(" ".join(f"{v:+.2f}" for v in row) for row in tt)
        lines.append(f"| {sc} | {md[0]:+.2f} | {md[1]:+.2f} | {md[2]:+.2f} | {np.abs(th - ref).max():.2f} | {trow} |")
md = "\n".join(lines)
open(OUT.replace(".json", f"{TAG}_J{JJ:g}_k{'_'.join(f'{k:g}' for k in KAPPAS)}.md"), "w").write(md)
print(md)
