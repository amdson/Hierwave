"""Self-play training of the learned coarse tables (mid_h, mid_v, mid_u, top_h,
top_v, top_u) of a top-down forward model against a designed bidirectional
energy p*, model-agnostic.  The model is chosen by env MODEL=potts|circles
(notes/potts_test.md, notes/circles_test.md); everything model-specific is in
a small hooks object (PottsHooks / CirclesHooks below).  Supersedes
notes/experiments/potts_train.py (kept as the record of the first run).

Schemes (gap = forward - target, per site; the target kernel is always p*'s):

  S0   oracle PCD: gap = forward stats - Oracle.moments (symmetrised)
  S1   per-level RB, one step: the Rao-Blackwellised expected count change
       under one-site moves of p* from each forward sample; top sites by the
       hook's top conditional, mid sites by Oracle.mid_probs (tiles
       integrated out)
  S2   as S1 for the mid tables; for the top tables first relax (top, mid) K
       rounds without tiles (hooks.relax_two_level), then the RB change; the
       top count change accumulated over the relaxation is included
  S3   end-to-end CD-K: K Oracle sweeps (top moves, mid moves, 2 tile sweeps)
       from the forward state; gap = -(N(relaxed) - N(fwd) + RB change at the
       relaxed state), so K = 0 is S1 and K -> inf is S0
  S1f  S1 with the frozen one-site mid conditional (tiles fixed)
  S2rb, S3rb (DIAG=1)  the relaxations with only the RB change at the relaxed
       state (no accumulated count change), as a diagnostic

S1f / S2rb / S3rb run at the first kappa of KAPPAS only.  The counts are read
on the forward's full model (designed + learned) whose Channel objects are the
oracle's, so the oracle's moves, the designed-only model and the full model
act on one state.

    MODEL=potts J=0.1 KAPPAS=8 NUMBA_NUM_THREADS=1 PYTHONPATH=. python notes/experiments/selfplay_train.py
    MODEL=circles KAPPAS=4 NUMBA_NUM_THREADS=1 PYTHONPATH=. python notes/experiments/selfplay_train.py

Env: MODEL; dials J, LAM (potts), MU, LAM (circles); KAPPAS (or KAPPA); NT,
SCHEMES, DIAG, RUNS, STEPS, K, BURN, SWEEPS, S_T, S_M, S_F, ETA, EVAL, EVERY,
FINAL, SEED, OUT, TAG, APPEND.  Figures images/{MODEL}{TAG}_k{kappa}_*.png.
"""
import os, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from castlegen.channels.core import Model
from castlegen.channels.train import site_probs, rb_gap, step, counts

FIT = ["mid_h", "mid_v", "mid_u", "top_h", "top_v", "top_u"]
TOPK = ["top_h", "top_v", "top_u"]
SHOW = ["mid_h", "top_h"]                       # tables drawn / reported against the reference


# ------------------------------------------------------------------ helpers
def centre(a):
    """Double centering for a pair table (row and column terms removed),
    centering for a unary: the identifiable part of a learned table."""
    a = np.asarray(a, float)
    if a.ndim == 1:
        return a - a.mean()
    return a - a.mean(0, keepdims=True) - a.mean(1, keepdims=True) + a.mean()


def compare_tables(learned, reference):
    """{key: dict(max_err, slope, corr)} after centering both; slope = least
    squares learned ~ slope * reference (1 = exact, < 1 = leak / shrink)."""
    out = {}
    for k, r in (reference or {}).items():
        if k not in learned:
            continue
        a, b = centre(learned[k]).ravel(), centre(r).ravel()
        nb = float(b @ b)
        out[k] = dict(max_err=float(np.abs(a - b).max()),
                      slope=float(a @ b / nb) if nb > 0 else float("nan"),
                      corr=float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan"))
    return out


def all_sites(fn, ny, nx):
    first = np.asarray(fn(0, 0), float)
    pr = np.empty((ny, nx) + first.shape)
    for i in range(ny):
        for j in range(nx):
            pr[i, j] = first if i == j == 0 else fn(i, j)
    return pr


def seed_of(orc):
    return int(orc.rng.integers(1 << 30))


# ------------------------------------------------------------- potts hooks
class PottsHooks:
    name = "potts"
    held_out = "edge_same_top"
    default_kappas = "8,1"

    def __init__(self):
        from castlegen.channels import potts
        self.mod = potts
        self.dials = dict(J=float(os.environ.get("J", 1.0)), lam=float(os.environ.get("LAM", 3.0)))

    def dial_tag(self):
        return f"J{self.dials['J']:g}"

    def make(self, nty, ntx, kappa):
        return self.mod.Potts(nty, ntx, J=self.dials["J"], kappa=kappa, lam=self.dials["lam"])

    def forward(self, P, theta, seed):
        return self.mod.Forward(P, theta, seed=seed)

    def oracle(self, P, seed):
        return self.mod.Oracle(P, seed=seed)

    def share(self, fw, orc):
        fw.top, fw.mid, fw.tile = orc.top, orc.mid, orc.tile

    def m_star(self, orc):
        return orc.model

    def chans3(self, obj):
        return obj.top, obj.mid, obj.tile

    def top_probs(self, orc, m_star):
        return site_probs(m_star, "top", below=True)

    def mid_probs(self, orc):
        return all_sites(orc.mid_probs, orc.P.nmy, orc.P.nmx)

    def mid_probs_plain(self, orc, m_star):
        return site_probs(m_star, "mid", below=True)

    def relax_two_level(self, orc, m_full, m_star, K):
        """(top, mid) under E_lam + theta_mid, no tiles."""
        for _ in range(K):
            m_full.sweep("mid", 1, seed=seed_of(orc), below=False)
            m_star.sweep("top", 1, seed=seed_of(orc), below=True)

    def relax_full(self, orc, K):
        orc.sweep(K, tile_sweeps=2)

    def reference(self, P):
        cc = np.arange(P.q)
        ref = P.BM * P.J * self.mod.dist(cc[:, None], cc[None, :], P.q).astype(float)
        ref -= ref.mean()
        return dict(mid_h=ref, mid_v=ref.copy())                 # kappa -> inf limit; no top reference

    def raw_moments(self, orc, burn, sweeps):
        return orc.moments(burn, sweeps)

    def symmetrise(self, P, st):
        """Average the fitted stats over the symmetry group of p*: colour maps
        c -> s c + r (s = +-1) with the induced palette map, and the lattice
        transpose / mirror (h <-> v, a pair table <-> its transpose)."""
        q = P.q
        out = {k: np.zeros_like(np.asarray(st[k], float)) for k in FIT}
        n = 0
        for sgn in (1, -1):
            for r in range(q):
                sig = (sgn * np.arange(q) + r) % q
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

    def self_check(self, P, tr):
        """Top conditionals = softmax(-lam * #mid cells outside the palette)
        at a random state."""
        rng = np.random.default_rng(5)
        tr.orc.mid.grid[:] = rng.integers(P.q, size=tr.orc.mid.grid.shape)
        tr.orc.top.grid[:] = rng.integers(P.P, size=tr.orc.top.grid.shape)
        pt = self.top_probs(tr.orc, tr.m_star)
        mg = tr.orc.mid.grid
        for i in range(P.nty):
            for j in range(P.ntx):
                blk = mg[i * P.BT:(i + 1) * P.BT, j * P.BT:(j + 1) * P.BT].ravel()
                nout = np.array([(~P.PAL[p, blk]).sum() for p in range(P.P)])
                ref = np.exp(-P.lam * nout); ref /= ref.sum()
                assert np.allclose(pt[i, j], ref), (i, j, pt[i, j], ref)
        return "site_probs(m_star, top) = softmax(-lam * #mid outside palette)"

    def pattern(self, P, th):
        """Markdown cells: mid_h mean by cyclic distance, top_h rows."""
        cc = np.arange(P.q)
        D = self.mod.dist(cc[:, None], cc[None, :], P.q)
        mh, tt = np.array(th["mid_h"]), np.array(th["top_h"])
        md = " / ".join(f"{mh[D == d].mean():+.2f}" for d in range(D.max() + 1))
        trow = "; ".join(" ".join(f"{v:+.2f}" for v in row) for row in tt)
        return [("mid_h by d = 0/1/2", md), ("top_h (rows p)", trow)]


# ----------------------------------------------------------- circles hooks
class CirclesHooks:
    name = "circles"
    held_out = "edge_air_top"
    default_kappas = "4"

    def __init__(self):
        from castlegen.channels import circles
        self.mod = circles
        self.dials = dict(mu=float(os.environ.get("MU", 0.3)), lam=float(os.environ.get("LAM", 3.0)))

    def dial_tag(self):
        return f"mu{self.dials['mu']:g}"

    def make(self, nty, ntx, kappa):
        return self.mod.Circles(nty, ntx, kappa=kappa, mu=self.dials["mu"], lam=self.dials["lam"])

    def forward(self, P, theta, seed):
        return self.mod.Forward(P, theta, seed=seed)

    def oracle(self, P, seed):
        return self.mod.Oracle(P, seed=seed)

    def share(self, fw, orc):
        # the painted channels are part of both models and must be shared too;
        # Forward.run builds its model from fw.chans
        fw.chans = orc.chans
        fw.top, fw.mid, fw.tile, fw.slot, fw.dem = orc.chans

    def m_star(self, orc):
        if getattr(orc, "model", None) is not None:
            return orc.model
        return orc.C.model(orc.chans)

    def chans3(self, obj):
        return obj.top, obj.mid, obj.tile

    def top_probs(self, orc, m_star):
        # the painted slot is not a factor site_probs can see: use the oracle's p*(T | mid)
        return all_sites(orc.top_probs_mid, orc.C.nty, orc.C.ntx)

    def mid_probs(self, orc):
        return all_sites(orc.mid_probs, orc.C.nmy, orc.C.nmx)

    def mid_probs_plain(self, orc, m_star):
        return all_sites(orc.mid_probs_plain, orc.C.nmy, orc.C.nmx)

    def relax_two_level(self, orc, m_full, m_star, K):
        """(top, mid) without tiles: top by top_move_mid over all top cells in
        random order (it repaints slot), mid by one sweep of the full model
        (lam via the painted slot + learned mid tables; kappa is homed on
        tile and not read with below=False).  dem repainted at the end so
        the state stays consistent (the RB evaluations read mid only)."""
        P = orc.C
        for _ in range(K):
            for k in orc.rng.permutation(P.nty * P.ntx):
                orc.top_move_mid(k // P.ntx, k % P.ntx)
            m_full.sweep("mid", 1, seed=seed_of(orc), below=False)
        P.paint_dem(orc.mid, orc.dem)

    def relax_full(self, orc, K):
        orc.sweep(K, tile_sweeps=2)

    def reference(self, P):
        return P.reference()

    def raw_moments(self, orc, burn, sweeps):
        return orc.moments(burn, sweeps, symmetrise=False)

    def symmetrise(self, P, st):
        return P.symmetrise(st)                 # dihedral group average (Circles.symmetrise)

    def self_check(self, P, tr):
        pt = self.top_probs(tr.orc, tr.m_star)
        assert np.allclose(pt.sum(-1), 1)
        return "top_probs_mid normalised"

    def pattern(self, P, th):
        tt = centre(th["top_h"])
        trow = "; ".join(" ".join(f"{v:+.2f}" for v in row) for row in tt)
        return [("top_h double-centred (rows T)", trow)]


HOOKS = {"potts": PottsHooks, "circles": CirclesHooks}


# --------------------------------------------------------------------- env
MODEL = os.environ.get("MODEL", "potts")
H = HOOKS[MODEL]()
NT = int(os.environ.get("NT", 6))
KAPPAS = [float(k) for k in os.environ.get("KAPPAS", os.environ.get("KAPPA", H.default_kappas)).split(",")]
DIAG = os.environ.get("DIAG", "0") == "1"
ONLY_FIRST = {"S1f", "S2rb", "S3rb"}                                   # diagnostics: first kappa only
SCHEMES = os.environ.get("SCHEMES", "S0,S1,S2,S3,S1f" + (",S2rb,S3rb" if DIAG else "")).split(",")
RUNS, STEPS, K = int(os.environ.get("RUNS", 4)), int(os.environ.get("STEPS", 150)), int(os.environ.get("K", 3))
BURN, SWEEPS = int(os.environ.get("BURN", 50)), int(os.environ.get("SWEEPS", 200))
S_T, S_M, S_F = int(os.environ.get("S_T", 30)), int(os.environ.get("S_M", 30)), int(os.environ.get("S_F", 20))
ETA = float(os.environ.get("ETA", 0.5))                    # 2.0 oscillated on Potts (notes/potts_test.md)
# SCALE=fwd: per-entry preconditioning, gap / max(forward frequency, FLOOR / table size) (the diagonal
# Fisher of a frequency, so the step is ~ eta * log(p_fwd / p_target)); each entry's step clipped to CLIP nats
SCALE, FLOOR, CLIP = os.environ.get("SCALE", ""), float(os.environ.get("FLOOR", 0.1)), float(os.environ.get("CLIP", 1.0))
EVAL, EVERY, FINAL = int(os.environ.get("EVAL", 16)), int(os.environ.get("EVERY", 10)), int(os.environ.get("FINAL", 64))
SEED = int(os.environ.get("SEED", 0))
OUT = os.environ.get("OUT", f"images/{MODEL}_selfplay.json")
TAG = os.environ.get("TAG", "")
os.makedirs("images", exist_ok=True)
print(f"MODEL {MODEL} dials {H.dials}  nty=ntx={NT}  RUNS {RUNS}  STEPS {STEPS}  K {K}  BURN {BURN}  SWEEPS {SWEEPS}  "
      f"S_T/S_M/S_F {S_T}/{S_M}/{S_F}  ETA {ETA} SCALE {SCALE or 'none'} FLOOR {FLOOR} CLIP {CLIP}  EVAL {EVAL} every {EVERY}  FINAL {FINAL}  schemes {SCHEMES}  "
      f"kappas {KAPPAS}")


def mean_stats(sts):
    return {k: np.mean([s[k] for s in sts], axis=0) for k in sts[0]}


def l1(a, b, keys):
    return {k: float(np.abs(np.asarray(a[k]) - np.asarray(b[k])).sum()) for k in keys}


def tolist(d):
    return {k: (np.asarray(v).tolist() if not isinstance(v, (float, int)) else v) for k, v in d.items()}


class Trainer:
    """One forward model whose channels ARE the oracle's channels, so the
    oracle's moves, the designed-only model m_star and the forward's full
    model m_full (= fw.model, rebuilt in run) all act on the same grids."""

    def __init__(self, P, theta, seed):
        self.P = P
        self.orc = H.oracle(P, seed + 1000)
        self.fw = H.forward(P, theta, seed)
        H.share(self.fw, self.orc)
        self.m_star = H.m_star(self.orc)

    def run(self, theta):
        self.fw.theta = theta
        return self.fw.run(S_T, S_M, S_F, fresh=True)

    def rb(self, home, probs):
        return rb_gap(self.fw.model, home, FIT, probs)

    def gap(self, scheme, st, target):
        """Per-site gap (forward - target) for one forward sample in the shared grids."""
        P = self.P
        if scheme == "S0":
            return {k: st[k] - target[k] for k in FIT}
        nt, nm = P.nty * P.ntx, P.nmy * P.nmx
        if scheme in ("S1", "S2", "S1f", "S2rb"):
            pm = H.mid_probs_plain(self.orc, self.m_star) if scheme == "S1f" else H.mid_probs(self.orc)
            gm = self.rb("mid", pm)
            if scheme in ("S2", "S2rb"):
                n0 = counts(self.fw.model, TOPK)
                H.relax_two_level(self.orc, self.fw.model, self.m_star, K)
            gt = self.rb("top", H.top_probs(self.orc, self.m_star))
            if scheme == "S2":
                n1 = counts(self.fw.model, TOPK)
                gt = {k: gt[k] + n1[k] - n0[k] if k in n1 else gt[k] for k in gt}
        elif scheme in ("S3", "S3rb"):
            n0 = counts(self.fw.model, FIT)
            H.relax_full(self.orc, K)
            gm = self.rb("mid", H.mid_probs(self.orc))
            gt = self.rb("top", H.top_probs(self.orc, self.m_star))
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
    for n in ("top", "mid", "tile"):                     # the shared grids
        assert tr.m_star.chan(n).grid is tr.fw.model.chan(n).grid is getattr(tr.orc, n).grid, n
    msg = H.self_check(P, tr)
    pt = H.top_probs(tr.orc, tr.m_star)
    g = tr.rb("top", pt)                                 # rb on the top unary: sum over sites of (probs - onehot)
    ref = pt.sum((0, 1)) - np.bincount(tr.orc.top.grid.ravel(), minlength=pt.shape[-1])
    assert np.allclose(g["top_u"], ref)
    assert np.allclose(g["mid_h"], 0)
    print(f"self-check ok: {msg}; rb top_u = sum(probs - onehot); grids shared")


def forward_eval(P, theta, n, seed):
    fw = H.forward(P, theta, seed)                       # fixed eval seed: common random numbers across evaluations
    sts = [fw.run(S_T, S_M, S_F, fresh=True) for _ in range(n)]
    return mean_stats(sts), fw


def save_render(P, obj, path, px=6):
    img = P.render(*H.chans3(obj))
    Image.fromarray(np.repeat(np.repeat(img, px, 0), px, 1)).save(path)


def fmt(v):
    a = np.asarray(v, float)
    return f"{float(a):.4f}" if a.ndim == 0 else "/".join(f"{x:.3f}" for x in a.ravel())


results = json.load(open(OUT)) if os.path.exists(OUT) and os.environ.get("APPEND", "1") == "1" else {}
CONFIG = dict(MODEL=MODEL, **H.dials, NT=NT, RUNS=RUNS, STEPS=STEPS, K=K, BURN=BURN, SWEEPS=SWEEPS, S_T=S_T,
              S_M=S_M, S_F=S_F, ETA=ETA, SCALE=SCALE, FLOOR=FLOOR, CLIP=CLIP, EVAL=EVAL, EVERY=EVERY, FINAL=FINAL, SEED=SEED)


def rkey(kk):
    return f"k{kk}_{H.dial_tag()}"


for ik, kappa in enumerate(KAPPAS):
    kk = f"{kappa:g}"
    pre = f"images/{MODEL}{TAG}_k{kk}"
    P = H.make(NT, NT, kappa)
    print(f"\n===== {MODEL} kappa {kk}: {P.nty}x{P.ntx} top, {P.nmy}x{P.nmx} mid, {P.H}x{P.W} tiles")
    self_check(P)
    REF = H.reference(P)
    t0 = time.time()
    orc = H.oracle(P, SEED + 7)
    raw = H.raw_moments(orc, BURN, SWEEPS)
    target = H.symmetrise(P, raw)
    MON = [k for k in target if k not in FIT]
    print("oracle raw mid_u", np.round(raw["mid_u"], 3), "top_u", np.round(raw["top_u"], 3),
          f"-> symmetrised;  {H.held_out}", round(float(raw[H.held_out]), 3))
    t_orc = time.time() - t0
    print(f"oracle moments: {BURN}+{SWEEPS} sweeps {t_orc:.1f}s  ({t_orc / (BURN + SWEEPS) * 1000:.0f} ms/sweep)")
    save_render(P, orc, f"{pre}_oracle.png")
    st_u, fw_u = forward_eval(P, P.theta0(), FINAL, SEED + 555)
    save_render(P, fw_u, f"{pre}_untrained.png")
    R = dict(oracle=tolist(target), oracle_raw=tolist(raw), untrained=tolist(st_u), untrained_L1=l1(st_u, target, FIT),
             reference=tolist(REF) if REF else None, schemes={})
    print("untrained L1:", {k: round(v, 3) for k, v in R["untrained_L1"].items()})
    print(f"  {H.held_out} oracle {target[H.held_out]:.3f} untrained {st_u[H.held_out]:.3f}")

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
        if sc in ONLY_FIRST and ik > 0:
            continue
        theta = P.theta0()
        tr = Trainer(P, theta, seed=SEED + 11)
        ev0, _ = forward_eval(P, theta, EVAL, SEED + 777)
        curve = [dict(step=0, **l1(ev0, target, FIT))]
        t_train = 0.0
        for s in range(STEPS):
            t1 = time.time()
            gs, sts = [], []
            for r in range(RUNS):
                st = tr.run(theta)
                sts.append({k: st[k] for k in FIT})
                gs.append(tr.gap(sc, st, target))
            g = mean_stats(gs)
            if SCALE == "fwd":
                pf = mean_stats(sts)
                g = {k: np.clip(g[k] / np.maximum(pf[k], FLOOR / np.size(pf[k])), -CLIP / ETA, CLIP / ETA)
                     for k in FIT}
            theta = step(theta, g, eta=ETA)
            t_train += time.time() - t1
            if (s + 1) % EVERY == 0:
                ev, _ = forward_eval(P, theta, EVAL, SEED + 777)
                c = dict(step=s + 1, **l1(ev, target, FIT))
                curve.append(c)
                if (s + 1) % (EVERY * 5) == 0 or s + 1 == STEPS:
                    print(f"  {sc} step {s + 1:4d}  " + "  ".join(f"{k} {c[k]:.3f}" for k in FIT)
                          + f"  {H.held_out} {ev[H.held_out]:.3f}  ({t_train / (s + 1):.2f} s/step)")
        stf, fwf = forward_eval(P, theta, FINAL, SEED + 555)
        save_render(P, fwf, f"{pre}_{sc}.png")
        st2, _ = forward_eval(P, theta, FINAL, SEED + 556)            # a second, independent final evaluation
        R["schemes"][sc] = dict(final=tolist(stf), L1=l1(stf, target, FIT), theta=tolist(theta), curve=curve,
                                s_per_step=t_train / max(STEPS, 1), eval_noise=l1(stf, st2, FIT),
                                vs_reference=compare_tables(theta, REF))
        cmp_ = R["schemes"][sc]["vs_reference"]
        print(f"{sc} done: {t_train:.0f}s ({t_train / max(STEPS, 1):.2f} s/step)  final L1 "
              + "  ".join(f"{k} {v:.3f}" for k, v in R["schemes"][sc]["L1"].items())
              + f"  {H.held_out} {stf[H.held_out]:.3f} (oracle {target[H.held_out]:.3f})"
              + "".join(f"\n    vs ref {k}: max err {c['max_err']:.3f} slope {c['slope']:.2f} corr {c['corr']:.2f}"
                        for k, c in cmp_.items()))
    R["config"] = CONFIG
    results[rkey(kk)] = R

    # ---- figures
    scs = list(R["schemes"])
    fig, axs = plt.subplots(1, len(FIT), figsize=(3.2 * len(FIT), 3), constrained_layout=True)
    for ax, k in zip(axs, FIT):
        for sc in scs:
            cv = R["schemes"][sc]["curve"]
            ax.plot([c["step"] for c in cv], [c[k] for c in cv], label=sc, lw=1.5)
        ax.axhline(R["untrained_L1"][k], color="0.6", ls=":", lw=1)
        ax.set_title(k); ax.set_xlabel("step"); ax.set_ylim(bottom=0)
    axs[0].set_ylabel("L1 to oracle moments")
    if scs:
        axs[-1].legend(fontsize=8, frameon=False)
    fig.suptitle(f"{MODEL} kappa {kk}: forward vs oracle, fitted features (eval {EVAL} runs, ETA {ETA})")
    fig.savefig(f"{pre}_curves.png", dpi=110); plt.close(fig)

    # tables: one row per SHOW key, a reference column then one per scheme; all double-centred
    ncol = len(scs) + 1
    fig, axs = plt.subplots(len(SHOW), ncol, figsize=(2.9 * ncol, 2.8 * len(SHOW)), constrained_layout=True,
                            squeeze=False)
    for r_, key in enumerate(SHOW):
        tabs = [centre(REF[key]) if REF and key in REF else None] + \
               [centre(R["schemes"][sc]["theta"][key]) for sc in scs]
        vm = max([np.abs(t).max() for t in tabs if t is not None] + [1e-9])
        for c, (t, title) in enumerate(zip(tabs, ["reference"] + scs)):
            ax = axs[r_, c]
            if t is None:
                ax.axis("off"); ax.set_title(f"{key} ref: none", fontsize=9)
                continue
            ax.imshow(t, cmap="RdBu_r", vmin=-vm, vmax=vm)
            ax.set_title(f"{title} {key}", fontsize=9)
            n = t.shape[0]
            if n <= 6:
                for i in range(n):
                    for j in range(t.shape[1]):
                        ax.text(j, i, f"{t[i, j]:+.1f}", ha="center", va="center", fontsize=7)
                ax.set_xticks(range(t.shape[1])); ax.set_yticks(range(n))
            else:
                ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{MODEL} kappa {kk}: learned tables, double-centred; rows = left / upper cell")
    fig.savefig(f"{pre}_tables.png", dpi=110); plt.close(fig)
    json.dump(results, open(OUT, "w"))
    print("wrote", OUT)

# ---- markdown tables
lines = []
for kk in [f"{k:g}" for k in KAPPAS]:
    R = results[rkey(kk)]
    scs = list(R["schemes"])
    P = H.make(NT, NT, float(kk))
    lines.append(f"\n### {MODEL}: kappa = {kk}, {H.dial_tag()}\n")
    lines.append(f"L1 distance to the oracle moments on the fitted features (final, {FINAL} forward runs):\n")
    lines.append("| | " + " | ".join(FIT) + " |")
    lines.append("|---" * (len(FIT) + 1) + "|")
    lines.append("| untrained | " + " | ".join(f"{R['untrained_L1'][k]:.3f}" for k in FIT) + " |")
    for sc in scs:
        lines.append(f"| {sc} | " + " | ".join(f"{R['schemes'][sc]['L1'][k]:.3f}" for k in FIT) + " |")
    for sc in scs:
        lines.append(f"| eval noise {sc} | " + " | ".join(f"{R['schemes'][sc]['eval_noise'][k]:.3f}" for k in FIT) + " |")
    MON = [k for k in R["oracle"] if k not in FIT]
    lines.append(f"\nMonitors ({H.held_out} held out):\n")
    lines.append("| | " + " | ".join(MON) + " | s/step |")
    lines.append("|---" * (len(MON) + 2) + "|")
    rows = [("oracle", R["oracle"], None), ("untrained", R["untrained"], None)] + \
           [(sc, R["schemes"][sc]["final"], R["schemes"][sc]["s_per_step"]) for sc in scs]
    for nm, st, sps in rows:
        lines.append(f"| {nm} | " + " | ".join(fmt(st[k]) for k in MON) + f" | {'' if sps is None else f'{sps:.2f}'} |")
    ref = R.get("reference")
    rk = list(ref) if ref else []
    pat = {sc: H.pattern(P, R["schemes"][sc]["theta"]) for sc in scs}
    pcols = [c for c, _ in pat[scs[0]]] if scs else []
    lines.append("\nLearned tables vs reference (double-centred; max abs err / least-squares slope learned ~ ref):\n")
    lines.append("| | " + " | ".join(f"{k} err / slope" for k in rk) + "".join(f" | {c}" for c in pcols) + " |")
    lines.append("|---" * (len(rk) + len(pcols) + 1) + "|")
    for sc in scs:
        c = R["schemes"][sc]["vs_reference"]
        lines.append(f"| {sc} | " + " | ".join(f"{c[k]['max_err']:.2f} / {c[k]['slope']:.2f}" for k in rk)
                     + "".join(f" | {v}" for _, v in pat[sc]) + " |")
md = "\n".join(lines)
open(OUT.replace(".json", f"{TAG}_{H.dial_tag()}_k{'_'.join(f'{k:g}' for k in KAPPAS)}.md"), "w").write(md)
print(md)
