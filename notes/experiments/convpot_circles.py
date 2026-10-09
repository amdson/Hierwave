"""The conv-potential test on circles (notes/convpot_test.md, "The test on
circles", stages 1-4).

    NUMBA_NUM_THREADS=1 PYTHONPATH=. python notes/experiments/convpot_circles.py [stages]

stages: a subset of "123" (default all; stage 4 = the timings collected by
1-3).  Window data (AIS / exact targets) are cached in
images/convpot_circles_windows.npz, results in images/convpot_circles.json.
Env: KT, MT (top AIS budget, default 256 / 4), NT (top windows, 2000), SEED.

Conventions as notes/experiments/induce_circles.py: the oracle moments are
read from images/induce_circles.json (BURN 50 + SWEEPS 400, seed 7,
symmetrised: the same Circles(6, 6) and code); forward evals 64 runs of
S_T / S_M / S_F = 30 / 30 / 20 at seed 555, eval noise against seed 556.
Mid tables used inside the top windows: exact F on 2 x 2 windows, N = 4000
(induce stage B)."""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from castlegen.channels.circles import Circles, Forward
from castlegen.channels.induce import windows, fit_tables, materialise
from castlegen.channels.induce_circles import (MidWindows, TopWindows, reference_halos, embed_identity,
                                               embed_patch)
from castlegen.channels import convfit as cf

STAGES = sys.argv[1] if len(sys.argv) > 1 else "123"
SEED = int(os.environ.get("SEED", 0))
KT, MT = int(os.environ.get("KT", 256)), int(os.environ.get("MT", 4))
NT = int(os.environ.get("NT", 2000))
NHALO, RIDGE = 32, 1e-3
L2T, L2M = 1e-6, 1e-6                    # convfit ridge (per-window mean loss)
NM_ALL, NM_HO, HO_FRAC = 6000, 4000, 0.4
FIT = ["mid_h", "mid_v", "mid_u", "top_h", "top_v", "top_u"]
OUT, WIN = "images/convpot_circles.json", "images/convpot_circles_windows.npz"
os.makedirs("images", exist_ok=True)
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
WD = dict(np.load(WIN)) if os.path.exists(WIN) else {}
IND = json.load(open("images/induce_circles.json"))
C = Circles(6, 6)
REF = C.reference()
RAW = C.reference(centred=False)
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def dc(a):
    a = np.asarray(a, float)
    return a - a.mean(0, keepdims=True) - a.mean(1, keepdims=True) + a.mean()


def tl(d):
    return {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


def arr(d):
    return {k: np.asarray(v) for k, v in d.items()}


def save():
    json.dump(R, open(OUT, "w"))
    np.savez(WIN, **WD)


def ptl(p):
    return {k: (np.asarray(v).tolist() if not np.isscalar(v) else v) for k, v in p.items()}


# ------------------------------------------------------------ comparisons
ENH = [(1, 0), (3, 2), (0, 1), (2, 3)]
DEP = [(1, 2), (3, 0)]
TR = np.array([0, 2, 1, 3])


def contrast(t):
    t = dc(t)
    return float(np.mean([t[a] for a in DEP]) - np.mean([t[a] for a in ENH]))


def contrast_v(t):
    return contrast(np.asarray(t)[np.ix_(TR, TR)])


def olog(orc, k="top_h"):
    tu = np.asarray(orc["top_u"])
    return dc(-np.log(np.asarray(orc[k]) / np.outer(tu, tu)))


def cmp_top(th, th_v, orc):
    o = olog(orc)
    t = dc(th)
    return dict(contrast_h=contrast(th), contrast_v=contrast_v(th_v),
                corr_o=float(np.corrcoef(t.ravel(), o.ravel())[0, 1]),
                slope_o=float(t.ravel() @ o.ravel() / (o.ravel() @ o.ravel())),
                corr_r=float(np.corrcoef(t.ravel(), REF["top_h"].ravel())[0, 1]),
                corr_o_v=float(np.corrcoef(dc(th_v).ravel(), olog(orc, "top_v").ravel())[0, 1]))


def fmt_tab(t):
    return "; ".join(" ".join(f"{x:+.2f}" for x in r) for r in dc(t))


# ------------------------------------------------------------- evaluation
S_T, S_M, S_F, FINAL = 30, 30, 20, 64
ORC = arr(IND["oracle"])


def mean_stats(sts):
    return {k: np.mean([s[k] for s in sts], axis=0) for k in sts[0]}


def forward_eval(theta, seed, extra=()):
    Forward(C, theta, seed=1, extra=extra).run(2, 2, 2)           # warm-up (numba compile), own seed
    fw = Forward(C, theta, seed=seed, extra=extra)
    t = time.perf_counter()
    sts = [fw.run(S_T, S_M, S_F, fresh=True) for _ in range(FINAL)]
    return mean_stats(sts), fw, (time.perf_counter() - t) / FINAL


def l1(a, b, keys=FIT):
    return {k: float(np.abs(np.asarray(a[k]) - np.asarray(b[k])).sum()) for k in keys}


def save_render(obj, path, px=6):
    img = C.render(obj.top, obj.mid, obj.tile)
    Image.fromarray(np.repeat(np.repeat(img, px, 0), px, 1)).save(path)


def orth_rows(D, k, seed):
    """(D, k) orthonormal rows (D <= k)."""
    q = np.linalg.qr(np.random.default_rng(seed).standard_normal((k, D)))[0]
    return q.T.copy()


def head_init(ls_params, k, m, seed, scale=0.01):
    """Tied params from an m = 0 ls solution plus a fresh head."""
    p0 = cf.init_params(1, k, m, seed, scale, tie=True)
    p0["a"] = np.asarray(ls_params["a"]).copy()
    p0["A"] = np.asarray(ls_params["A"])[list(cf.HALF)].copy()
    return p0


def predict(params, E, w, pad):
    R_, C_ = w.shape[1:]
    return cf.energy_window(params, E, w, pad) - cf.energy_window(params, E, np.full((1, R_, C_), pad), pad)[0]


def ls_hv(w, y, E, pad, l2):
    """fit_bilinear_ls restricted to the h and v offsets (diagonal A = 0):
    the 8-offset bilinear model minus its diagonals (keeps the pad terms)."""
    k = E.shape[1]
    N, R_, C_ = w.shape
    X = cf.bilinear_features(w, E, pad) - cf.bilinear_features(np.full((1, R_, C_), pad), E, pad)
    keep = np.r_[np.arange(k), k + np.arange(k * k), k + 2 * k * k + np.arange(k * k)]   # unary, east, south
    Xk = X[:, keep]
    th = np.linalg.solve(Xk.T @ Xk / N + l2 * np.eye(len(keep)), Xk.T @ y / N)
    full = np.zeros(X.shape[1])
    full[keep] = th
    A = np.zeros((9, k, k))
    Ab = full[k:].reshape(4, k, k)
    for j, i in enumerate(cf.HALF):
        A[i], A[8 - i] = Ab[j], Ab[j].T
    res = X @ full - y
    tr = float(np.sqrt(np.mean(y ** 2)))
    params = dict(a=full[:k], A=A, W=np.zeros((9, k, 0)), b=np.zeros(0), v=np.zeros(0))
    return dict(params=params, E=E, rel=float(np.sqrt(np.mean(res ** 2))) / tr, resid_rms=float(np.sqrt(np.mean(res ** 2))),
                target_rms=tr)


def rel_on(params, E, w, y, pad):
    r = predict(params, E, w, pad) - y
    return float(np.sqrt(np.mean(r ** 2)) / np.sqrt(np.mean(y ** 2)))


# ======================================================================= 1
def mid_tables_exact():
    """induce stage B: exact F on 2 x 2 mid windows, N = 4000, tabular fit."""
    MW = MidWindows(C)
    FR = MW.window_free_energy_exact(np.zeros((2, 2), np.int32))
    rng = np.random.default_rng(SEED + 2)
    zs = [((rng.random((2, 2)) < 0.5) * rng.integers(1, C.D, (2, 2))).astype(np.int32) for _ in range(4000)]
    fe, _ = windows(MW, zs, "exact")
    u, gh, gv, res, rms = fit_tables(np.array(zs), fe - FR, C.D, RIDGE)
    th = materialise(C.theta0(), u, gh, gv, ("mid_u", "mid_h", "mid_v"))
    return {k: th[k] for k in ("mid_h", "mid_v", "mid_u")}, res, rms


def top_windows(thm, w, rng, n, halo=True):
    """Halos (32, forward mid under reference tops), F(r, halo) by AIS at 4 KT,
    n uniform w x w windows by AIS (KT, MT); target F(z, h) - F(r, h).
    halo False: no ring (free window boundary), one F(r)."""
    t0 = time.time()
    halos = reference_halos(C, thm, NHALO, wy=w, wx=w, seed=int(rng.integers(1 << 30))) if halo else None
    TW = TopWindows(C, thm, halos, w, w)
    zr = np.zeros((w, w), np.int32)
    hl = list(range(NHALO)) if halo else [None]
    fr, se_r = windows(TW, [(zr, h) for h in hl], "ais", 4 * KT, MT, int(rng.integers(1 << 30)))
    t_r = time.time() - t0
    zs = rng.integers(4, size=(n, w, w)).astype(np.int32)
    hs = rng.integers(NHALO, size=n) if halo else np.zeros(n, int)
    t1 = time.time()
    F, se = windows(TW, [(z, int(h) if halo else None) for z, h in zip(zs, hs)], "ais", KT, MT,
                    int(rng.integers(1 << 30)))
    t_w = time.time() - t1
    return dict(zs=zs, hs=hs, y=F - fr[hs], se=se, se_r=se_r, fr=fr), TW, dict(t_ref=t_r, t_win=t_w,
                                                                                ms=1000 * t_w / n)


if "1" in STAGES:
    say("## stage 1: top level")
    t = time.time()
    thm, res_m, rms_m = mid_tables_exact()
    say(f"  mid tables (exact F, 2 x 2, N 4000): residual {res_m:.4f} / {rms_m:.2f} ({time.time() - t:.1f}s)")
    R["mid_exact"] = dict(theta=tl(thm), res=res_m, rms=rms_m)
    rng = np.random.default_rng(SEED + 11)
    timing = {}
    for key, w, halo in (("top3", 3, True), ("top2", 2, True), ("top3n", 3, False)):
        if key + "_y" not in WD:
            d, TW, tm = top_windows(thm, w, rng, NT, halo)
            for kk, v in d.items():
                WD[f"{key}_{kk}"] = v
            timing[key] = tm
            say(f"  top {w}x{w}: {NT} windows AIS K {KT} M {MT}: {tm['t_win']:.0f}s ({tm['ms']:.1f} ms/window), "
                f"F(r, halo) {tm['t_ref']:.0f}s; mean se {d['se'].mean():.3f}, F(r) se {d['se_r'].mean():.3f}")
            if key == "top3":           # halo + AIS floor: 50 z x 8 halos, within-z spread of the target
                t1 = time.time()
                zf = rng.integers(4, size=(50, 3, 3)).astype(np.int32)
                hf = np.stack([rng.choice(NHALO, 8, replace=False) for _ in range(50)])
                items = [(z, int(h)) for z, hh in zip(zf, hf) for h in hh]
                Ff, sef = windows(TW, items, "ais", KT, MT, int(rng.integers(1 << 30)))
                yf = (Ff - d["fr"][hf.ravel()]).reshape(50, 8)
                WD["floor_y"], WD["floor_se"] = yf, sef
                timing["floor"] = time.time() - t1
                say(f"  floor: within-z sd {np.sqrt(yf.var(1, ddof=1).mean()):.3f} ({time.time() - t1:.0f}s)")
            if key == "top3n":              # AIS floor: 50 z x 4 repeats
                t1 = time.time()
                zf = rng.integers(4, size=(50, 3, 3)).astype(np.int32)
                Ff, _ = windows(TW, [(z, None) for z in zf for _ in range(4)], "ais", KT, MT,
                                int(rng.integers(1 << 30)))
                WD["floorn_y"] = Ff.reshape(50, 4)
                timing["floorn"] = time.time() - t1
                say(f"  no-halo floor: within-z sd {np.sqrt(WD['floorn_y'].var(1, ddof=1).mean()):.3f}")
            save()
    R.setdefault("timing", {}).update(timing)
    yf = WD["floor_y"]
    floor_sd = float(np.sqrt(yf.var(1, ddof=1).mean()))
    floor_se = float(WD["floor_se"].mean())
    floorn_sd = float(np.sqrt(WD["floorn_y"].var(1, ddof=1).mean()))
    S1 = dict(floor_sd=floor_sd, floor_se=floor_se, floorn_sd=floorn_sd, variants={})
    split_rng = np.random.default_rng(SEED + 12)
    E16 = orth_rows(4, 16, SEED + 13)
    curves = {}
    for key, w in (("top3", 3), ("top2", 2), ("top3n", 3)):
        sfx = key[3:]
        zs, y = WD[f"{key}_zs"], WD[f"{key}_y"]
        n = len(y)
        perm = split_rng.permutation(n)
        tr, te = perm[:int(0.8 * n)], perm[int(0.8 * n):]
        trms = float(np.sqrt(np.mean(y ** 2)))
        S1[f"target_rms{sfx}"] = trms
        fl = {"3": floor_sd, "3n": floorn_sd}.get(sfx)
        say(f"  -- {key} windows: target RMS {trms:.3f}, mean AIS se {WD[key + '_se'].mean():.3f}"
            + (f", floor sd {fl:.3f} ({fl / trms:.2f} of target)" if fl else ""))
        # (a) tabular h + v
        t0 = time.time()
        u, gh, gv, res_all, _ = fit_tables(zs, y, 4, RIDGE, ref=0)
        ua, gha, gva, res_tr, _ = fit_tables(zs[tr], y[tr], 4, RIDGE, ref=0)
        from castlegen.channels.induce import window_features
        X = window_features(zs[te], 4, 0)
        # fit_tables zero-means its outputs; refit the raw solution for prediction on te
        Xt = window_features(zs[tr], 4, 0)
        th_raw = np.linalg.solve(Xt.T @ Xt + RIDGE * np.eye(Xt.shape[1]), Xt.T @ y[tr])
        rel_te = float(np.sqrt(np.mean((X @ th_raw - y[te]) ** 2)) / trms)
        S1["variants"][f"a{sfx}"] = dict(rel=res_all / trms, rel_tr=res_tr / np.sqrt(np.mean(y[tr] ** 2)),
                                       rel_te=rel_te, top_h=gh.tolist(), top_v=gv.tolist(),
                                       cmp=cmp_top(gh, gv, ORC), t=time.time() - t0)
        # (b0) one-hot k = 4, (b) orthonormal k = 16 (m = 0, closed form), (c) m = 32 from (b)
        specs = [("b0", np.eye(4), 0, L2T), ("b0hv", np.eye(4), -1, L2T)] + ([("b", E16, 0, L2T), ("c", E16, 32, L2T), ("c8", E16, 8, L2T),
                                                ("creg", E16, 32, 1e-3)] if w == 3 else [])
        ls_cache = {}
        for name, E, m, l2 in specs:
            out = {}
            for part, idx in (("tr", tr), ("all", np.arange(n))):
                t0 = time.time()
                if m == 0:
                    r = cf.fit_bilinear_ls(zs[idx], y[idx], E, pad=0, l2=L2T)
                    ls_cache[(name, part)] = r
                elif m < 0:
                    r = ls_hv(zs[idx], y[idx], E, 0, l2)
                else:
                    init = head_init(ls_cache[("b", part)]["params"], E.shape[1], m, SEED + 14)
                    r = cf.fit(zs[idx], y[idx], E, pad=0, m=m, steps=4000, lr=3e-3, lr_end=3e-5, l2=l2,
                               init=init, seed=SEED + 15)
                    curves[f"top_{name}{sfx}_{part}"] = r["curve"]
                out[part] = (r, time.time() - t0)
            r_tr, r_all = out["tr"][0], out["all"][0]
            u_, g = cf.materialise_pairs(r_all["params"], E, 0)
            v = dict(rel=r_all["rel"], rel_tr=r_tr["rel"], rel_te=rel_on(r_tr["params"], E, zs[te], y[te], 0),
                     top_h=g[5].tolist(), top_v=g[7].tolist(), diag_se=g[8].tolist(), diag_sw=g[6].tolist(),
                     u=u_.tolist(), cmp=cmp_top(g[5], g[7], ORC), t=out["all"][1], m=m, k=E.shape[1],
                     params=ptl(r_all["params"]), E=E.tolist())
            S1["variants"][f"{name}{sfx}"] = v
        for name, v in S1["variants"].items():
            if name[-len(sfx):] != sfx or (sfx == "3" and name.endswith("3n")):
                continue
            c = v["cmp"]
            say(f"  {name}: rel all {v['rel']:.3f} train {v['rel_tr']:.3f} test {v['rel_te']:.3f}; contrast h/v "
                f"{c['contrast_h']:.2f}/{c['contrast_v']:.2f}; corr/slope vs oracle {c['corr_o']:.2f}/"
                f"{c['slope_o']:.2f}, v corr {c['corr_o_v']:.2f}; {v['t']:.1f}s")
    R["S1"] = S1
    R["curves_top"] = {k: v.tolist() for k, v in curves.items()}
    save()

# ======================================================================= 2
UPAIRS = [(v, w) for v in range(1, C.D) for w in range(v, C.D)]


def val(cy, cx):
    return 1 + (cy - 2) * 4 + (cx - 2)


HAND = dict(A=(val(3, 5), val(3, 2)), B=(val(2, 5), val(2, 2)), C=(val(3, 5), val(3, 3)), D=(val(3, 4), val(3, 2)))
OFF4 = [(0, 1), (1, -1), (1, 0), (1, 1)]


def heldout_mask(seed):
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(UPAIRS), int(round(HO_FRAC * len(UPAIRS))), replace=False)
    M = np.zeros((C.D, C.D), bool)
    for i in sel:
        v, w = UPAIRS[i]
        M[v, w] = M[w, v] = True
    return M


def has_pair(z, M):
    H, W = z.shape
    for y in range(H):
        for x in range(W):
            for dy, dx in OFF4:
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and M[z[y, x], z[yy, xx]]:
                    return True
    return False


def score_mid(g_h, g_v, M):
    """Reference gauge (absent row / column zero, as RAW and as the
    materialised tables): pp entries, seen vs held out, h and v pooled."""
    out = {}
    pp = np.zeros((C.D, C.D), bool)
    pp[1:, 1:] = True
    for part, mask in (("seen", pp & ~M), ("held", pp & M)):
        a = np.concatenate([np.asarray(g_h)[mask], np.asarray(g_v)[mask]])
        b = np.concatenate([RAW["mid_h"][mask], RAW["mid_v"][mask]])
        ad = np.concatenate([dc(g_h)[mask], dc(g_v)[mask]])
        bd = np.concatenate([REF["mid_h"][mask], REF["mid_v"][mask]])
        out[part] = dict(max_err=float(np.abs(a - b).max()), slope=float(a @ b / (b @ b)),
                         corr=float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-12 else 0.0,
                         rms_pred=float(np.sqrt(np.mean(a ** 2))), rms_ref=float(np.sqrt(np.mean(b ** 2))),
                         dc_max_err=float(np.abs(ad - bd).max()),
                         dc_corr=float(np.corrcoef(ad, bd)[0, 1]) if ad.std() > 1e-12 else 0.0, n=int(mask.sum()))
    out["hand"] = {n: [float(dc(g_h)[p, q]), float(REF["mid_h"][p, q]), bool(M[p, q])] for n, (p, q) in HAND.items()}
    return out


if "2" in STAGES:
    say("## stage 2: mid level, generalisation across values")
    seed = SEED + 20
    while sum(heldout_mask(seed)[p, q] for p, q in HAND.values()) < 2:
        seed += 1
    M = heldout_mask(seed)
    say(f"  held-out seed {seed}: {int(np.triu(M[1:, 1:]).sum())} of {len(UPAIRS)} unordered pp pairs; hand held out: "
        + ", ".join(n for n, (p, q) in HAND.items() if M[p, q]))
    MW3 = MidWindows(C, 3, 3)
    FR3 = MW3.window_free_energy_exact(np.zeros((3, 3), np.int32))
    rng = np.random.default_rng(SEED + 21)

    def draw():
        return ((rng.random((3, 3)) < 0.5) * rng.integers(1, C.D, (3, 3))).astype(np.int32)
    t = time.time()
    z_all = np.array([draw() for _ in range(NM_ALL)])
    z_ho, ndraw = [], 0
    while len(z_ho) < NM_HO:
        z = draw()
        ndraw += 1
        if not has_pair(z, M):
            z_ho.append(z)
    z_ho = np.array(z_ho)
    y_all = np.array([MW3.window_free_energy_exact(z) for z in z_all]) - FR3
    y_ho = np.array([MW3.window_free_energy_exact(z) for z in z_ho]) - FR3
    contains = np.array([has_pair(z, M) for z in z_all])
    t_F = time.time() - t
    say(f"  windows: {NM_ALL} all-pairs, {NM_HO} accepted of {ndraw} drawn ({NM_HO / ndraw:.2f}); exact F {t_F:.1f}s; "
        f"{contains.sum()} all-pairs windows contain a held-out pair")
    E_id = embed_identity(C.D, 32, SEED + 22)
    E_pa = embed_patch(C, "mid", 32, SEED + 23)
    variants = [("b0", np.eye(C.D), 0), ("b", E_id, 0), ("c", E_id, 32), ("b-patch", E_pa, 0), ("c-patch", E_pa, 32)]
    S2 = dict(seed=seed, mask=M.tolist(), accept=NM_HO / ndraw, t_F=t_F, target_rms_ho=float(np.sqrt(np.mean(y_ho ** 2))),
              target_rms_all=float(np.sqrt(np.mean(y_all ** 2))), variants={}, all={}, l2sweep={})
    curves = {}
    ls = {}
    for fitset, zz, yy in (("ho", z_ho, y_ho), ("all", z_all, y_all)):
        for name, E, m in variants:
            t0 = time.time()
            if m == 0:
                r = cf.fit_bilinear_ls(zz, yy, E, pad=0, l2=L2M)
                ls[(fitset, name)] = r
            else:
                base = "b" if name == "c" else "b-patch"
                init = head_init(ls[(fitset, base)]["params"], E.shape[1], m, SEED + 24)
                r = cf.fit(zz, yy, E, pad=0, m=m, steps=3000, lr=1e-3, lr_end=1e-5, l2=L2M, init=init,
                           seed=SEED + 25)
                curves[f"mid_{name}_{fitset}"] = r["curve"]
            dt = time.time() - t0
            u, g = cf.materialise_pairs(r["params"], E, 0)
            sc = score_mid(g[5], g[7], M)
            v = dict(rel=r["rel"], resid=r["resid_rms"], t=dt, score=sc, m=m, k=E.shape[1],
                     g_h=g[5].tolist(), g_v=g[7].tolist(), diag_rms=float(np.sqrt(np.mean(g[8] ** 2))),
                     diag_max=float(np.abs(g[8]).max()), u=u.tolist())
            if fitset == "ho":
                v["rel_out"] = rel_on(r["params"], E, z_all[contains], y_all[contains], 0)
                S2["variants"][name] = v
                s, h = sc["seen"], sc["held"]
                say(f"  [held-out fit] {name}: in-sample rel {r['rel']:.2e} (resid {r['resid_rms']:.3f}); windows with "
                    f"held-out pairs rel {v['rel_out']:.3f}; seen max {s['max_err']:.2f} slope {s['slope']:.3f} corr "
                    f"{s['corr']:.3f}; held max {h['max_err']:.2f} slope {h['slope']:.3f} corr {h['corr']:.3f}; {dt:.1f}s")
            else:
                v["params"], v["E"] = ptl(r["params"]), E.tolist()
                S2["all"][name] = v
                say(f"  [all-pairs fit] {name}: rel {r['rel']:.2e}; pp max err {max(sc['seen']['max_err'], sc['held']['max_err']):.3f}"
                    f"; diag (SE) rms {v['diag_rms']:.3f} max {v['diag_max']:.3f}; {dt:.1f}s")
    for name, E in (("b", E_id), ("b-patch", E_pa)):        # ridge sweep (m = 0)
        for l2 in (1e-4, 1e-2, 1.0):
            r = cf.fit_bilinear_ls(z_ho, y_ho, E, pad=0, l2=l2)
            _, g = cf.materialise_pairs(r["params"], E, 0)
            sc = score_mid(g[5], g[7], M)
            S2["l2sweep"][f"{name} l2 {l2:g}"] = dict(rel=r["rel"], score=sc)
            say(f"  ridge {name} l2 {l2:g}: rel {r['rel']:.2e}; seen corr {sc['seen']['corr']:.3f}; held max "
                f"{sc['held']['max_err']:.2f} slope {sc['held']['slope']:.3f} corr {sc['held']['corr']:.3f}")
    R["S2"] = S2
    R["curves_mid"] = {k: v.tolist() for k, v in curves.items()}
    save()

# ======================================================================= 3
if "3" in STAGES:
    say("## stage 3: end to end")
    S1, S2 = R["S1"], R["S2"]
    top_name = os.environ.get("TOPV") or min(("b03", "b3", "c3", "c83", "creg3"), key=lambda n: S1["variants"][n]["rel_te"])
    mid_name = os.environ.get("MIDV", "b0")
    tv, mv = S1["variants"][top_name], S2["all"][mid_name]
    say(f"  top variant {top_name} (test rel {tv['rel_te']:.3f}), mid variant {mid_name} (all-pairs fit)")
    f_top = cf.to_factor(arr(tv["params"]), np.array(tv["E"]), ("top", "pal"), 0, "top_conv")
    f_mid = cf.to_factor(arr(mv["params"]), np.array(mv["E"]), ("mid", "self"), 0, "mid_conv")
    thm = arr(R["mid_exact"]["theta"])
    a3 = S1["variants"]["a3"]
    th_tab = dict(thm, top_h=np.array(a3["top_h"]), top_v=np.array(a3["top_v"]),
                  top_u=np.zeros(4))
    # top_u of (a): refit (fit_tables returns it; recompute from the cached windows)
    u_a, _, _, _, _ = fit_tables(WD["top3_zs"], WD["top3_y"], 4, RIDGE, ref=0)
    th_tab["top_u"] = u_a
    configs = {"tables: exact-mid + top (a, 3x3)": (th_tab, []),
               f"mixed: exact-mid tables + top convpot ({top_name})": (thm, [f_top]),
               f"convpot: mid ({mid_name}) + top ({top_name})": ({}, [f_mid, f_top])}
    tn = S1["variants"]["b03n"]
    f_topn = cf.to_factor(arr(tn["params"]), np.array(tn["E"]), ("top", "pal"), 0, "top_conv")
    configs["mixednohalo: exact-mid tables + top convpot (b03n, no halo)"] = (thm, [f_topn])
    S3 = {}
    for name, (th, ex) in configs.items():
        st, fw, sec = forward_eval(th, SEED + 555, ex)
        st2, _, _ = forward_eval(th, SEED + 556, ex)
        tag = name.split(":")[0]
        save_render(fw, f"images/convpot_circles_fwd_{tag}.png")
        S3[name] = dict(final=tl(st), L1=l1(st, ORC), noise=l1(st, st2), sec_per_run=sec)
        say(f"  {name}: L1 " + " ".join(f"{k} {v:.3f}" for k, v in S3[name]["L1"].items())
            + f"; present {st['present']:.3f} edge_air_top {st['edge_air_top']:.3f}; {1000 * sec:.1f} ms/run")
    # kernel time per sweep at each level, tables vs convpot
    kt = {}
    for name, (th, ex) in (("tables", (th_tab, [])), ("convpot", ({}, [f_mid, f_top]))):
        fw = Forward(C, th, seed=1, extra=ex)
        fw.run(S_T, S_M, S_F)
        mdl = fw.model
        for lev in ("top", "mid"):
            mdl.sweep(lev, 2, seed=3)
            t = time.perf_counter()
            mdl.sweep(lev, 50, seed=4)
            kt[f"{name}_{lev}_ms_per_sweep"] = 1000 * (time.perf_counter() - t) / 50
    say("  kernel " + " ".join(f"{k} {v:.3f}" for k, v in kt.items()))
    R["S3"] = dict(rows=S3, kernel=kt, top=top_name, mid=mid_name)
    save()

open("images/convpot_circles_log.txt", "a").write("\n".join(LOG) + "\n")
