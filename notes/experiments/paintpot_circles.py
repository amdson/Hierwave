"""The paint-potential test on circles (notes/paintpot_test.md, stages 1-3).

    NUMBA_NUM_THREADS=1 PYTHONPATH=. python notes/experiments/paintpot_circles.py [stages]

stages: any of "1" (mid, held-out), "a" (top, saved convpot windows with
their mid-ring halo), "b" (top, new windows with a ring of halo top cells),
"2" (= a + b), "3" (end to end).  Window data cached in
images/paintpot_circles_windows.npz, results in images/paintpot_circles.json,
log appended to images/paintpot_circles_log.txt.

Conventions copied from notes/experiments/convpot_circles.py: held-out set
(seed 20, 40% of the 136 unordered present-present pairs, symmetric),
reference-gauge scoring of mid tables, oracle moments from
images/induce_circles.json, forward evals 64 runs of 30 / 30 / 20 sweeps at
seed 555, eval noise against 556."""
import json, os, sys, time
import numpy as np

from castlegen.channels.circles import Circles, Forward
from castlegen.channels.core import Factor
from castlegen.channels.induce import windows
from castlegen.channels.induce_circles import (TopWindows, reference_halos, paint_region_mid, mid_window_F,
                                               paint_single_mid, paint_pair_mid, paint_region_top,
                                               paint_single_top, paint_pair_top)
from castlegen.channels import paintpot as pp
from castlegen.channels import convfit as cf

STAGES = sys.argv[1] if len(sys.argv) > 1 else "123"
if "2" in STAGES:
    STAGES += "ab"
SEED = int(os.environ.get("SEED", 0))
KT, MT, NT = 256, 4, int(os.environ.get("NT", 2000))
MTB = int(os.environ.get("MTB", 8))          # stage 2b: two AIS per target (F(z|H) and F(r|H)), M doubled
L2 = 1e-6
NM_ALL, NM_HO, HO_FRAC = 6000, 4000, 0.4
FIT = ["mid_h", "mid_v", "mid_u", "top_h", "top_v", "top_u"]
DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]             # east, south, SE, SW
OUT, WIN = "images/paintpot_circles.json", "images/paintpot_circles_windows.npz"
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
WD = dict(np.load(WIN)) if os.path.exists(WIN) else {}
IND = json.load(open("images/induce_circles.json"))
CPR = json.load(open("images/convpot_circles.json"))
C = Circles(6, 6)
REF = C.reference()
RAW = C.reference(centred=False)
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def save():
    json.dump(R, open(OUT, "w"))
    np.savez(WIN, **WD)


def dc(a):
    a = np.asarray(a, float)
    return a - a.mean(0, keepdims=True) - a.mean(1, keepdims=True) + a.mean()


def arr(d):
    return {k: np.asarray(v) for k, v in d.items()}


def tl(d):
    return {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


# ------------------------------------------------------------ comparisons (as convpot_circles.py)
ENH = [(1, 0), (3, 2), (0, 1), (2, 3)]
DEP = [(1, 2), (3, 0)]
TR = np.array([0, 2, 1, 3])
ORC = arr(IND["oracle"])


def contrast(t):
    t = dc(t)
    return float(np.mean([t[a] for a in DEP]) - np.mean([t[a] for a in ENH]))


def olog(orc, k="top_h"):
    tu = np.asarray(orc["top_u"])
    return dc(-np.log(np.asarray(orc[k]) / np.outer(tu, tu)))


def cmp_top(th, th_v, orc=ORC):
    o = olog(orc)
    t = dc(th)
    return dict(contrast_h=contrast(th), contrast_v=contrast(np.asarray(th_v)[np.ix_(TR, TR)]),
                corr_o=float(np.corrcoef(t.ravel(), o.ravel())[0, 1]),
                slope_o=float(t.ravel() @ o.ravel() / (o.ravel() @ o.ravel())),
                corr_r=float(np.corrcoef(t.ravel(), REF["top_h"].ravel())[0, 1]),
                corr_o_v=float(np.corrcoef(dc(th_v).ravel(), olog(orc, "top_v").ravel())[0, 1]))


UPAIRS = [(v, w) for v in range(1, C.D) for w in range(v, C.D)]


def val(cy, cx):
    return 1 + (cy - 2) * 4 + (cx - 2)


HAND = dict(A=(val(3, 5), val(3, 2)), B=(val(2, 5), val(2, 2)), C=(val(3, 5), val(3, 3)), D=(val(3, 4), val(3, 2)))


def heldout_mask(seed):
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(UPAIRS), int(round(HO_FRAC * len(UPAIRS))), replace=False)
    M = np.zeros((C.D, C.D), bool)
    for i in sel:
        v, w = UPAIRS[i]
        M[v, w] = M[w, v] = True
    return M


def has_pair_batch(Z, M):
    """(N,) bool: grid n holds a held-out pair at any of the 8 offsets (M symmetric)."""
    out = np.zeros(len(Z), bool)
    for dy, dx in pp.OFF8:
        a, b = pp._shift_pair(Z, dy, dx)
        out |= M[a, b].reshape(len(Z), -1).any(1)
    return out


def score_mid(g_h, g_v, M):
    """Reference gauge (absent row / column zero): pp entries, seen vs held out,
    h and v pooled; hand entries of g_h vs RAW mid_h."""
    out = {}
    pp_ = np.zeros((C.D, C.D), bool)
    pp_[1:, 1:] = True
    g_h, g_v = np.asarray(g_h), np.asarray(g_v)
    for part, mask in (("seen", pp_ & ~M), ("held", pp_ & M)):
        a = np.concatenate([g_h[mask], g_v[mask]])
        b = np.concatenate([RAW["mid_h"][mask], RAW["mid_v"][mask]])
        out[part] = dict(max_err=float(np.abs(a - b).max()), slope=float(a @ b / (b @ b)),
                         corr=float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-12 else 0.0,
                         rms_pred=float(np.sqrt(np.mean(a ** 2))), rms_ref=float(np.sqrt(np.mean(b ** 2))),
                         n=int(mask.sum()))
    out["hand"] = {n: [float(g_h[p, q]), float(RAW["mid_h"][p, q]), bool(M[p, q])] for n, (p, q) in HAND.items()}
    return out


def mat_mid(theta, offs):
    return pp.materialise(theta, lambda v: paint_single_mid(C, v), lambda v, w, d: paint_pair_mid(C, v, w, d),
                          C.D, DIRS, 4, offs)


def mat_top(theta, V, offs):
    return pp.materialise(theta, lambda v: paint_single_top(C, v), lambda v, w, d: paint_pair_top(C, v, w, d),
                          4, DIRS, V, offs)


# ======================================================================= 1
def mid_windows(n, rng, M=None):
    """n accepted (5, 5) mid grids (3 x 3 window + one-block halo, presence
    0.5, uniform offset); with M, reject grids holding a held-out pair."""
    out, got, ndraw = [], 0, 0
    while got < n:
        Z = ((rng.random((20000, 5, 5)) < 0.5) * rng.integers(1, C.D, (20000, 5, 5))).astype(np.int32)
        ok = np.ones(len(Z), bool) if M is None else ~has_pair_batch(Z, M)
        idx = np.flatnonzero(ok)[:n - got]
        out.append(Z[idx])
        got += len(idx)
        ndraw += int(idx[-1] + 1) if got == n else len(Z)
    return np.concatenate(out), ndraw


def mid_xy(Z):
    r = np.zeros((3, 3), np.int32)
    Pz = np.stack([paint_region_mid(C, z[1:-1, 1:-1], z) for z in Z])
    Pr = np.stack([paint_region_mid(C, r, z) for z in Z])
    y = C.fz[Pz].sum((1, 2)) - C.fz[Pr].sum((1, 2))
    X = pp.features(Pz, 4, pp.OFF8) - pp.features(Pr, 4, pp.OFF8)
    return X, y


def fit_mid(X, y, M):
    out = {}
    for name, nf, offs in (("unary", 4, []), ("unary+OFF8", X.shape[1], pp.OFF8)):
        t0 = time.time()
        th, d = pp.fit(X[:, :nf], y, L2)
        u_c, g_c = mat_mid(th, offs)
        sc = score_mid(g_c[0], g_c[1], M)
        uc, gcan = pp.canonical(th, 4, offs)
        fz = C.fz - C.fz.mean()
        out[name] = dict(rel=d["rel"], resid=d["resid_rms"], target=d["target_rms"], t=time.time() - t0, score=sc,
                         u_fit=uc.tolist(), u_err=float(np.abs(uc - fz).max()),
                         g_paint_rms=float(np.sqrt(np.mean(gcan ** 2))) if len(offs) else 0.0,
                         g_paint_max=float(np.abs(gcan).max()) if len(offs) else 0.0,
                         mid_u=u_c.tolist(), mid_h=g_c[0].tolist(), mid_v=g_c[1].tolist(),
                         diag_rms=[float(np.sqrt(np.mean(g_c[k] ** 2))) for k in (2, 3)],
                         diag_max=float(np.abs(g_c[2:]).max()),
                         u_c_err=float(np.abs(u_c + C.pres_e - RAW["mid_u"]).max()))
    return out


if "1" in STAGES:
    say("## stage 1: mid level, held-out generalisation")
    M = heldout_mask(20)
    say(f"  held-out seed 20: {int(np.triu(M[1:, 1:]).sum())} of {len(UPAIRS)} unordered pp pairs; hand held out: "
        + ", ".join(n for n, (p, q) in HAND.items() if M[p, q]))
    rng = np.random.default_rng(SEED + 21)
    t = time.time()
    Zh, ndraw = mid_windows(NM_HO, rng, M)
    Za, _ = mid_windows(NM_ALL, rng)
    Xh, yh = mid_xy(Zh)
    Xa, ya = mid_xy(Za)
    t_w = time.time() - t
    # sanity: F through the hook equals the paint-energy target
    z0 = Zh[0]
    assert abs(mid_window_F(C, z0[1:-1, 1:-1], z0) - mid_window_F(C, np.zeros((3, 3), np.int32), z0) - yh[0]) < 1e-9
    say(f"  windows: {NM_HO} accepted of {ndraw} drawn ({NM_HO / ndraw:.4f}; the whole 5 x 5 window + halo is "
        f"checked), {NM_ALL} all-pairs; exact F + features {t_w:.1f}s; target RMS held-out fit "
        f"{np.sqrt(np.mean(yh ** 2)):.1f}, all {np.sqrt(np.mean(ya ** 2)):.1f}")
    S1 = dict(accept=NM_HO / ndraw, ndraw=ndraw, t_windows=t_w, ho=fit_mid(Xh, yh, M), all=fit_mid(Xa, ya, M),
              mask=M.tolist())
    for fs in ("ho", "all"):
        for name, v in S1[fs].items():
            s, h = v["score"]["seen"], v["score"]["held"]
            hd = v["score"]["hand"]
            say(f"  [{fs}] {name}: rel {v['rel']:.2e} (resid {v['resid']:.2e} / {v['target']:.1f}); seen max "
                f"{s['max_err']:.2e} slope {s['slope']:.4f} corr {s['corr']:.4f}; held max {h['max_err']:.2e} slope "
                f"{h['slope']:.4f} corr {h['corr']:.4f}; hand " + " ".join(
                    f"{n}{'*' if x[2] else ''} {x[0]:+.3f}/{x[1]:+.3f}" for n, x in hd.items())
                + f"; u vs fz max {v['u_err']:.2e}; paint pair (canonical) rms {v['g_paint_rms']:.2e} max "
                f"{v['g_paint_max']:.2e}; diag rms {v['diag_rms'][0]:.2e}/{v['diag_rms'][1]:.2e}; mid_u err "
                f"{v['u_c_err']:.2e}; {v['t']:.2f}s")
    say(f"  fz = {np.round(C.fz, 4).tolist()} (centred {np.round(C.fz - C.fz.mean(), 4).tolist()}); "
        f"fitted u (unary, ho) {np.round(S1['ho']['unary']['u_fit'], 4).tolist()}")
    R["S1"] = S1
    save()


# ======================================================================= 2a
def regen_halos3():
    """The 32 mid-ring halos of convpot stage 1 (top3): same rng stream;
    checked by recomputing F(r, halo) for a few halos."""
    thm = arr(CPR["mid_exact"]["theta"])
    rng = np.random.default_rng(SEED + 11)
    halos = reference_halos(C, thm, 32, wy=3, wx=3, seed=int(rng.integers(1 << 30)))
    TW = TopWindows(C, thm, halos, 3, 3)
    fr, _ = windows(TW, [(np.zeros((3, 3), np.int32), h) for h in range(32)], "ais", 4 * KT, MT,
                    int(rng.integers(1 << 30)))
    return thm, halos, TW, float(np.abs(fr - WD_cp["top3_fr"]).max())


def top_variants(Xs, y, tr, te, mats):
    """Xs: name -> (X, materialiser (theta) -> (u_c, g_c)).  Fit all / train,
    score test, materialise from the all-window fit."""
    out = {}
    for name, (X, mat) in Xs.items():
        t0 = time.time()
        th, d = pp.fit(X, y, L2)
        th_tr, d_tr = pp.fit(X, y, L2, idx=tr)
        u_c, g_c = mat(th)
        out[name] = dict(rel=d["rel"], rel_tr=d_tr["rel"], rel_te=pp.rel_on(th_tr, X[te], y[te]),
                         nfeat=X.shape[1], top_u=u_c.tolist(), top_h=g_c[0].tolist(), top_v=g_c[1].tolist(),
                         diag_se=g_c[2].tolist(), diag_sw=g_c[3].tolist(), cmp=cmp_top(g_c[0], g_c[1]),
                         diag_dc_rms=[float(np.sqrt(np.mean(dc(g_c[k]) ** 2))) for k in (2, 3)],
                         hv_dc_rms=[float(np.sqrt(np.mean(dc(g_c[k]) ** 2))) for k in (0, 1)], t=time.time() - t0,
                         theta=th.tolist())
    return out


def say_top(S):
    for name, v in S.items():
        c = v["cmp"]
        say(f"  {name} ({v['nfeat']} feat): rel all {v['rel']:.3f} train {v['rel_tr']:.3f} test {v['rel_te']:.3f}; "
            f"contrast h/v {c['contrast_h']:.2f}/{c['contrast_v']:.2f}; top_h corr/slope vs oracle "
            f"{c['corr_o']:.2f}/{c['slope_o']:.2f}; top_v corr {c['corr_o_v']:.2f}; corr vs ref {c['corr_r']:.2f}; "
            f"dc rms h/v {v['hv_dc_rms'][0]:.3f}/{v['hv_dc_rms'][1]:.3f}, diag SE/SW {v['diag_dc_rms'][0]:.3f}/"
            f"{v['diag_dc_rms'][1]:.3f}")


def bilinear_mat(E):
    def f(th):
        k = E.shape[1]
        A = np.zeros((9, k, k))
        Ab = th[k:].reshape(4, k, k)
        for j, i in enumerate(cf.HALF):
            A[i], A[8 - i] = Ab[j], Ab[j].T
        params = dict(a=th[:k], A=A, W=np.zeros((9, k, 0)), b=np.zeros(0), v=np.zeros(0))
        u, g = cf.materialise_pairs(params, E, 0)
        return u, np.stack([g[5], g[7], g[8], g[6]])          # east, south, SE, SW
    return f


if "a" in STAGES:
    say("## stage 2a: top level, saved convpot 3 x 3 windows (mid-ring halo)")
    WD_cp = dict(np.load("images/convpot_circles_windows.npz"))
    t = time.time()
    thm, halos, TW, frerr = regen_halos3()
    say(f"  regenerated the 32 halos; F(r, halo) reproduced to max |diff| {frerr:.2e} ({time.time() - t:.1f}s)")
    zs, hs, y = WD_cp["top3_zs"], WD_cp["top3_hs"], WD_cp["top3_y"]
    n = len(y)
    perm = np.random.default_rng(SEED + 12).permutation(n)       # the convpot split
    tr, te = perm[:int(0.8 * n)], perm[int(0.8 * n):]
    # painted region at mid resolution (8 x 8): window slot (0 / 1) inside; ring = 2 + 2 slot + present(halo mid)
    sl = np.stack([TW.slot(z) for z in zs])
    sl_r = TW.slot(np.zeros((3, 3), np.int32))
    ring = np.ones((8, 8), bool)
    ring[1:-1, 1:-1] = False
    H = np.stack([halos[h] for h in hs])
    pres = (H > 0).astype(np.int32)

    def paint6(s):
        p = s.copy()
        p[:, ring] = 2 + 2 * s[:, ring] + pres[:, ring]
        return p
    P6, P6r = paint6(sl), paint6(np.broadcast_to(sl_r, sl.shape).copy())

    def paint19(s):                       # ring = 2 + the halo mid value (offset included; ring slot dropped)
        p = s.copy()
        p[:, ring] = 2 + H[:, ring]
        return p
    P19, P19r = paint19(sl), paint19(np.broadcast_to(sl_r, sl.shape).copy())
    E4 = np.eye(4)
    Xb0 = cf.bilinear_features(zs, E4, 0) - cf.bilinear_features(np.zeros((1, 3, 3), np.int32), E4, 0)
    Xs = {"(b0) one-hot convpot, no halo features": (Xb0, bilinear_mat(E4))}
    for oname, offs in (("OFF8", pp.OFF8), ("OFF2", pp.OFF2)):
        Xs[f"paint slot V2 {oname} (halo not painted)"] = (
            pp.features(sl, 2, offs) - pp.features(sl_r, 2, offs), lambda th, o=offs: mat_top(th, 2, o))
        Xs[f"paint slot+halo presence V6 {oname}"] = (
            pp.features(P6, 6, offs) - pp.features(P6r, 6, offs), lambda th, o=offs: mat_top(th, 6, o))
    Xs["paint slot+halo value V19 OFF8"] = (pp.features(P19, 19) - pp.features(P19r, 19),
                                            lambda th: mat_top(th, 19, pp.OFF8))
    S2a = dict(target_rms=float(np.sqrt(np.mean(y ** 2))), frerr=frerr, variants=top_variants(Xs, y, tr, te, None))
    ais_floor = float(np.sqrt(np.mean(WD_cp["top3_se"] ** 2) + np.mean(WD_cp["top3_se_r"] ** 2)))
    S2a["ais_floor"] = ais_floor
    say(f"  target RMS {S2a['target_rms']:.3f} (convpot: 1.680, halo floor 0.54 of target); AIS-only floor "
        f"sqrt(mean se^2 + mean se_r^2) = {ais_floor:.3f} = {ais_floor / S2a['target_rms']:.2f} of target")
    say_top(S2a["variants"])
    R["S2a"] = S2a
    save()


# ======================================================================= 2b
if "b" in STAGES:
    say("## stage 2b: top level, new 3 x 3 windows with a ring of halo TOP cells (mid of all 5 x 5 blocks free)")
    thm = arr(CPR["mid_exact"]["theta"])
    TW5 = TopWindows(C, thm, None, 5, 5)                 # halo None: the 10 x 10 mid region, free boundary
    rng = np.random.default_rng(SEED + 31)
    if "b_y" not in WD:
        Z = rng.integers(4, size=(NT, 5, 5)).astype(np.int32)        # window = centre 3 x 3, ring = halo tops
        Zr = Z.copy()
        Zr[:, 1:-1, 1:-1] = 0
        t = time.time()
        F, se = windows(TW5, [(z, None) for z in Z], "ais", KT, MTB, int(rng.integers(1 << 30)))
        Fr, ser = windows(TW5, [(z, None) for z in Zr], "ais", KT, MTB, int(rng.integers(1 << 30)))
        t_w = time.time() - t
        WD.update(b_Z=Z, b_F=F, b_Fr=Fr, b_se=se, b_ser=ser, b_y=F - Fr)
        # AIS floor: 50 windows x 4 repeats of both F's
        t = time.time()
        Zf = Z[:50]
        Zfr = Zf.copy()
        Zfr[:, 1:-1, 1:-1] = 0
        Ff, _ = windows(TW5, [(z, None) for z in Zf for _ in range(4)], "ais", KT, MTB, int(rng.integers(1 << 30)))
        Ffr, _ = windows(TW5, [(z, None) for z in Zfr for _ in range(4)], "ais", KT, MTB, int(rng.integers(1 << 30)))
        WD["b_floor_y"] = (Ff - Ffr).reshape(50, 4)
        R["t_2b"] = dict(windows=t_w, ms=1000 * t_w / (2 * NT), floor=time.time() - t)
        save()
    Z, y = WD["b_Z"], WD["b_y"]
    n = len(y)
    perm = np.random.default_rng(SEED + 32).permutation(n)
    tr, te = perm[:int(0.8 * n)], perm[int(0.8 * n):]
    Zr = Z.copy()
    Zr[:, 1:-1, 1:-1] = 0
    floor = float(np.sqrt(WD["b_floor_y"].var(1, ddof=1).mean()))
    trms = float(np.sqrt(np.mean(y ** 2)))
    say(f"  {n} windows, AIS K {KT} M {MTB}: {R['t_2b']['windows']:.0f}s ({R['t_2b']['ms']:.0f} ms / AIS run), "
        f"mean se {WD['b_se'].mean():.3f} / {WD['b_ser'].mean():.3f}; target RMS {trms:.3f}; AIS floor (50 x 4) sd "
        f"{floor:.3f} = {floor / trms:.2f} of target")
    E4 = np.eye(4)
    zin = np.ascontiguousarray(Z[:, 1:-1, 1:-1])
    Xs = {"(b0) one-hot convpot, window only": (
        cf.bilinear_features(zin, E4, 0) - cf.bilinear_features(np.zeros((1, 3, 3), np.int32), E4, 0),
        bilinear_mat(E4)),
        "(b0+halo) one-hot convpot, halo tops as features": (
        cf.bilinear_features(Z, E4, 0) - cf.bilinear_features(Zr, E4, 0), bilinear_mat(E4))}
    Sl = np.stack([paint_region_top(C, z[1:-1, 1:-1], z) for z in Z])
    Slr = np.stack([paint_region_top(C, z[1:-1, 1:-1], z) for z in Zr])
    for oname, offs in (("OFF8", pp.OFF8), ("OFF2", pp.OFF2)):
        Xs[f"paint slot V2 {oname}"] = (pp.features(Sl, 2, offs) - pp.features(Slr, 2, offs),
                                        lambda th, o=offs: mat_top(th, 2, o))
    S2b = dict(target_rms=trms, floor_sd=floor, variants=top_variants(Xs, y, tr, te, None), t=R["t_2b"])
    say_top(S2b["variants"])
    R["S2b"] = S2b
    save()


# ======================================================================= 3
S_T, S_M, S_F, FINAL = 30, 30, 20, 64


def mean_stats(sts):
    return {k: np.mean([s[k] for s in sts], axis=0) for k in sts[0]}


def forward_eval(theta, seed, extra=()):
    Forward(C, theta, seed=1, extra=extra).run(2, 2, 2)
    fw = Forward(C, theta, seed=seed, extra=extra)
    t = time.perf_counter()
    sts = [fw.run(S_T, S_M, S_F, fresh=True) for _ in range(FINAL)]
    return mean_stats(sts), (time.perf_counter() - t) / FINAL


def l1(a, b, keys=FIT):
    return {k: float(np.abs(np.asarray(a[k]) - np.asarray(b[k])).sum()) for k in keys}


if "3" in STAGES:
    say("## stage 3: end to end")
    m = R["S1"]["all"]["unary+OFF8"]
    th_mid = dict(mid_h=np.array(m["mid_h"]), mid_v=np.array(m["mid_v"]), mid_u=np.array(m["mid_u"]))
    TPV = os.environ.get("TOPV", "paint slot V2 OFF2")
    tv = R["S2b"]["variants"][TPV]
    T = ("top", "pal")
    configs = {}
    th = dict(th_mid, top_h=np.array(tv["top_h"]), top_v=np.array(tv["top_v"]), top_u=np.array(tv["top_u"]))
    configs[f"paint: mid (unary+OFF8, all pairs) + top ({TPV}) h+v"] = (th, [])
    configs[f"paint: same + top diagonals"] = (th, [Factor.pair(T, T, (1, 1), np.array(tv["diag_se"]), name="top_se"),
                                                    Factor.pair(T, T, (1, -1), np.array(tv["diag_sw"]), name="top_sw")])
    S3 = {}
    for name, (thc, ex) in configs.items():
        st, sec = forward_eval(thc, SEED + 555, ex)
        st2, _ = forward_eval(thc, SEED + 556, ex)
        S3[name] = dict(final=tl(st), L1=l1(st, ORC), noise=l1(st, st2), sec_per_run=sec)
        say(f"  {name}: L1 " + " ".join(f"{k} {v:.3f}" for k, v in S3[name]["L1"].items())
            + "; noise " + " ".join(f"{v:.3f}" for v in S3[name]["noise"].values())
            + "; " + " ".join(f"{k} {st[k]:.4f}" for k in ("present", "viol", "conflict", "contact", "slot_viol",
                                                             "edge_air_top")) + f"; {1000 * sec:.1f} ms/run")
    R["S3"] = dict(rows=S3, top=TPV)
    save()

open("images/paintpot_circles_log.txt", "a").write("\n".join(LOG) + "\n")
