"""The induced-potential test on circles (notes/induce_test.md, stages A-F).

    NUMBA_NUM_THREADS=1 PYTHONPATH=. python notes/experiments/induce_circles.py [stages]

stages: a subset of "ABCDE" (default all); results of earlier stages are
read back from images/induce_circles.json.  Env: KA, MA (mid AIS budget
for B / E), KT, MT (top AIS budget), NB, NC (window counts), SEED.
Evaluation conventions as notes/experiments/selfplay_train.py (circles):
oracle moments BURN 50 + SWEEPS 400 (seed 7), symmetrised; forward evals 64
runs of S_T / S_M / S_F = 30 / 30 / 20 (seed 555; eval noise against an
independent seed-556 evaluation)."""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from castlegen.channels.circles import Circles, Forward, Oracle
from castlegen.channels.induce import ais_log_z, windows, fit_tables, materialise
from castlegen.channels.induce_circles import MidWindows, TopWindows, reference_halos

STAGES = sys.argv[1] if len(sys.argv) > 1 else "ABCDE"
SEED = int(os.environ.get("SEED", 0))
KA, MA = int(os.environ.get("KA", 64)), int(os.environ.get("MA", 16))
KT, MT = int(os.environ.get("KT", 128)), int(os.environ.get("MT", 16))
NB, NC = int(os.environ.get("NB", 4000)), int(os.environ.get("NC", 2000))
NHALO = 32
RIDGE = 1e-3
FIT = ["mid_h", "mid_v", "mid_u", "top_h", "top_v", "top_u"]
OUT = "images/induce_circles.json"
os.makedirs("images", exist_ok=True)
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
C = Circles(6, 6)
REF = C.reference()
RAW = C.reference(centred=False)
MW = MidWindows(C)
F_R = MW.window_free_energy_exact(np.zeros((2, 2), np.int32))
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


def mid_windows(n, rng, pres=0.5):
    return [((rng.random((2, 2)) < pres) * rng.integers(1, C.D, (2, 2))).astype(np.int32) for _ in range(n)]


# ---------------------------------------------------------------- compare
def val(cy, cx):
    return 1 + (cy - 2) * 4 + (cx - 2)


HAND = dict(A=(val(3, 5), val(3, 2)), B=(val(2, 5), val(2, 2)), C=(val(3, 5), val(3, 3)), D=(val(3, 4), val(3, 2)))


def cmp_mid(th):
    """mid tables vs the exact reference (as circles_test.md)."""
    out = {}
    pp_r = dc(RAW["mid_h"])[1:, 1:]
    attr, rep = RAW["mid_h"][1:, 1:] < -1e-9, RAW["mid_h"][1:, 1:] > 1e-9
    zero = ~attr & ~rep
    for k in ("mid_h", "mid_v"):
        m = dc(th[k])
        a, b = m[1:, 1:].ravel(), dc(RAW[k])[1:, 1:].ravel()
        out[k] = dict(max_err=float(np.abs(m - REF[k]).max()), slope=float(a @ b / (b @ b)),
                      corr=float(np.corrcoef(a, b)[0, 1]))
    m = dc(th["mid_h"])[1:, 1:]
    out["means"] = [float(m[attr].mean()), float(m[rep].mean()), float(m[zero].mean())]
    out["means_ref"] = [float(pp_r[attr].mean()), float(pp_r[rep].mean()), float(pp_r[zero].mean())]
    out["hand"] = {n: [float(dc(th["mid_h"])[p, q]), float(REF["mid_h"][p, q])] for n, (p, q) in HAND.items()}
    u = np.asarray(th["mid_u"])
    out["u_gap"] = float(u[1:].mean() - u[0])
    out["u_spread"] = float(u[1:].std())
    ue = u + C.pres_e
    out["u_max_err"] = float(np.abs((ue - ue.mean()) - REF["mid_u"]).max())
    return out


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


def cmp_top(th, orc):
    o = olog(orc)
    t = dc(th["top_h"])
    return dict(contrast_h=contrast(th["top_h"]), contrast_v=contrast_v(th["top_v"]),
                corr_o=float(np.corrcoef(t.ravel(), o.ravel())[0, 1]),
                slope_o=float(t.ravel() @ o.ravel() / (o.ravel() @ o.ravel())),
                corr_r=float(np.corrcoef(t.ravel(), REF["top_h"].ravel())[0, 1]),
                corr_o_v=float(np.corrcoef(dc(th["top_v"]).ravel(), olog(orc, "top_v").ravel())[0, 1]),
                top_h_dc=t.tolist())


def mid_row(name, c):
    h = c["hand"]
    return (f"| {name} | {c['mid_h']['max_err']:.2f} | {c['mid_h']['slope']:.2f} / {c['mid_h']['corr']:.3f} | "
            f"{c['mid_v']['slope']:.2f} / {c['mid_v']['corr']:.3f} | "
            f"{c['means'][0]:+.2f} / {c['means'][1]:+.2f} / {c['means'][2]:+.2f} | "
            + " | ".join(f"{h[n][0]:+.2f}" for n in "ABCD")
            + f" | {c['u_gap']:.2f} / {c['u_spread']:.2f} | {c['u_max_err']:.2f} |")


MID_HDR = ("| tables | mid_h max abs err | mid_h pp slope / corr | mid_v pp slope / corr | "
           f"mean attract / repel / zero (ref {{:+.2f}} / {{:+.2f}} / {{:+.2f}}) | A ({REF['mid_h'][HAND['A']]:+.2f}) | "
           f"B ({REF['mid_h'][HAND['B']]:+.2f}) | C ({REF['mid_h'][HAND['C']]:+.2f}) | D ({REF['mid_h'][HAND['D']]:+.2f}) | "
           f"mid_u present - absent / spread (F_obj {C.object_cost():.2f} / 0) | mid_u + pres max err |\n"
           "|---|---|---|---|---|---|---|---|---|---|---|")


# ------------------------------------------------------------- top stage
def top_fit(theta_mid, rng, tag, z_extra=None, halo=True):
    """Halos under reference tops with the given mid tables, F(r, halo) by AIS
    at 4x the budget, NC windows (uniform corners, or z_extra first), AIS,
    ridge fit of top_u, top_h, top_v."""
    t0 = time.time()
    halos = reference_halos(C, theta_mid, NHALO, seed=int(rng.integers(1 << 30))) if halo else None
    TW = TopWindows(C, theta_mid, halos)
    zr = np.zeros((2, 2), np.int32)
    if halo:
        fr, se_r = windows(TW, [(zr, h) for h in range(NHALO)], "ais", 4 * KT, MT, int(rng.integers(1 << 30)))
    else:
        fr, se_r = windows(TW, [(zr, None)], "ais", 4 * KT, MT, int(rng.integers(1 << 30)))
    zs = list(z_extra or [])
    zs += [rng.integers(4, size=(2, 2)).astype(np.int32) for _ in range(NC - len(zs))]
    hs = rng.integers(NHALO, size=len(zs)) if halo else np.zeros(len(zs), int)
    F, se = windows(TW, [(z, (int(h) if halo else None)) for z, h in zip(zs, hs)], "ais", KT, MT,
                    int(rng.integers(1 << 30)))
    y = F - fr[hs]
    u, gh, gv, res, rms = fit_tables(np.array(zs), y, C.P, RIDGE, ref=0)
    dt = time.time() - t0
    say(f"  top fit {tag}: {len(zs)} windows, halo {halo}, KT {KT} MT {MT}: residual RMS {res:.3f} / target RMS "
        f"{rms:.3f}; mean AIS se {se.mean():.3f}, F(r) se {se_r.mean():.3f}; {dt:.0f}s")
    return dict(top_u=u, top_h=gh, top_v=gv), dict(res=res, rms=rms, se=float(se.mean()), t=dt, n=len(zs))


# ------------------------------------------------------------- evaluation
S_T, S_M, S_F, FINAL = 30, 30, 20, 64


def mean_stats(sts):
    return {k: np.mean([s[k] for s in sts], axis=0) for k in sts[0]}


def forward_eval(theta, seed):
    fw = Forward(C, theta, seed=seed)
    sts = [fw.run(S_T, S_M, S_F, fresh=True) for _ in range(FINAL)]
    return mean_stats(sts), fw


def l1(a, b, keys=FIT):
    return {k: float(np.abs(np.asarray(a[k]) - np.asarray(b[k])).sum()) for k in keys}


def oracle_moments():
    if "oracle" not in R:
        t = time.time()
        orc = Oracle(C, seed=SEED + 7)
        st = orc.moments(50, 400, symmetrise=True)
        R["oracle"] = tl(st)
        save_render(orc, "images/induce_circles_oracle.png")
        say(f"oracle moments 50 + 400 sweeps: {time.time() - t:.1f}s")
    return arr(R["oracle"])


def save_render(obj, path, px=6):
    img = C.render(obj.top, obj.mid, obj.tile)
    Image.fromarray(np.repeat(np.repeat(img, px, 0), px, 1)).save(path)


# ======================================================================= A
if "A" in STAGES:
    say("## stage A: estimator check")
    rng = np.random.default_rng(SEED + 1)
    zs = mid_windows(200, rng)
    fe, _ = windows(MW, zs, "exact")
    tgt = fe - F_R
    rows = []
    for K, M in [(8, 4), (16, 4), (32, 4), (64, 4), (8, 16), (16, 16), (32, 16), (64, 16), (128, 16), (256, 16),
                 (256, 4)]:
        t = time.time()
        fa, se = windows(MW, zs, "ais", K, M, seed=SEED + 100 + K + M)
        dt = (time.time() - t) / len(zs)
        err = (fa - F_R) - tgt                                     # F(r) by AIS is exact (no E_rest)
        within = float((np.abs(err) <= 3 * se + 1e-12).mean())
        r = dict(K=K, M=M, mean=float(err.mean()), sd=float(err.std()), rms=float(np.sqrt((err ** 2).mean())),
                 se=float(se.mean()), within3=within, ms=dt * 1000, err=err.tolist())
        rows.append(r)
        say(f"  K {K:3d} M {M:2d}: mean err {r['mean']:+.3f} sd {r['sd']:.3f} rms {r['rms']:.3f} mean se {r['se']:.3f} "
            f"within 3 se {within:.2f}  {dt * 1000:.1f} ms/window")
    # L check: circles has no inf entry, so the anneal tables are the same for any L
    sub = zs[:10]
    f30, _ = windows(MW, sub, "ais", 32, 4, seed=5, L=30.0)
    f5, _ = windows(MW, sub, "ais", 32, 4, seed=5, L=5.0)
    say(f"  L = 30 vs L = 5 (10 windows, same seed): max |dF| {np.abs(f30 - f5).max():.2e}")
    nobj = np.array([(z > 0).sum() for z in zs])
    R["A"] = dict(rows=rows, L_check=float(np.abs(f30 - f5).max()), tgt=tgt.tolist(), nobj=nobj.tolist())
    save()

# ======================================================================= B
if "B" in STAGES:
    say("## stage B: mid tables")
    rng = np.random.default_rng(SEED + 2)
    zs = mid_windows(NB, rng)
    t = time.time()
    fe, _ = windows(MW, zs, "exact")
    u, gh, gv, res, rms = fit_tables(np.array(zs), fe - F_R, C.D, RIDGE)
    t_ex = time.time() - t
    th_ex = materialise(C.theta0(), u, gh, gv, ("mid_u", "mid_h", "mid_v"))
    c_ex = cmp_mid(th_ex)
    say(f"  exact: residual RMS {res:.4f} / target RMS {rms:.2f}  ({t_ex:.1f}s)  {c_ex['mid_h']}")
    t = time.time()
    fa, se = windows(MW, zs, "ais", KA, MA, seed=SEED + 3)
    ua, gha, gva, resa, rmsa = fit_tables(np.array(zs), fa - F_R, C.D, RIDGE)
    t_ais = time.time() - t
    th_ais = materialise(C.theta0(), ua, gha, gva, ("mid_u", "mid_h", "mid_v"))
    c_ais = cmp_mid(th_ais)
    say(f"  ais K {KA} M {MA}: residual RMS {resa:.4f} / target RMS {rmsa:.2f}  ({t_ais:.0f}s)  {c_ais['mid_h']}")
    R["B"] = dict(exact=dict(theta=tl(th_ex), cmp=c_ex, res=res, rms=rms, t=t_ex),
                  ais=dict(theta=tl(th_ais), cmp=c_ais, res=resa, rms=rmsa, t=t_ais, K=KA, M=MA,
                           se=float(se.mean()), err_rms=float(np.sqrt(np.mean((fa - fe) ** 2)))))
    save()

# ======================================================================= C
if "C" in STAGES:
    say("## stage C: top tables")
    orc = oracle_moments()
    rng = np.random.default_rng(SEED + 4)
    R["C"] = {}
    for src in ("exact", "ais"):
        thm = arr(R["B"][src]["theta"])
        for halo in ((True, False) if src == "exact" else (True,)):
            tt, info = top_fit(thm, rng, f"mid {src}", halo=halo)
            key = f"{src}" + ("" if halo else "_nohalo")
            R["C"][key] = dict(theta=tl(tt), info=info, cmp=cmp_top(tt, orc))
            c = R["C"][key]["cmp"]
            say(f"    contrast h/v {c['contrast_h']:.2f}/{c['contrast_v']:.2f}  corr/slope vs oracle "
                f"{c['corr_o']:.2f}/{c['slope_o']:.2f}  corr vs ref {c['corr_r']:.2f}")
    save()

# ======================================================================= D
if "D" in STAGES:
    say("## stage D: end to end")
    orc = oracle_moments()
    R["D"] = {}
    st_u, _ = forward_eval(C.theta0(), SEED + 555)
    R["D"]["untrained"] = dict(final=tl(st_u), L1=l1(st_u, orc))
    combos = {"exact-mid + top(exact-mid)": ("exact", "exact"), "exact-mid + top(exact-mid, no halo)":
              ("exact", "exact_nohalo"), "ais-mid + top(ais-mid)": ("ais", "ais")}
    for name, (m, t) in combos.items():
        th = dict(arr(R["B"][m]["theta"]))
        th.update(arr(R["C"][t]["theta"]))
        tt = time.time()
        st, fw = forward_eval(th, SEED + 555)
        st2, _ = forward_eval(th, SEED + 556)
        save_render(fw, f"images/induce_circles_fwd_{m}_{t}.png")
        R["D"][name] = dict(final=tl(st), L1=l1(st, orc), noise=l1(st, st2), theta=tl(th), t=time.time() - tt)
        say(f"  {name}: L1 " + " ".join(f"{k} {v:.3f}" for k, v in R["D"][name]["L1"].items())
            + f"  present {st['present']:.3f} edge_air_top {st['edge_air_top']:.3f} ({time.time() - tt:.0f}s)")
    save()

# ======================================================================= E
if "E" in STAGES:
    say("## stage E: iterate with windows from forward samples")
    orc = oracle_moments()
    rng = np.random.default_rng(SEED + 5)
    th0 = arr(R["D"]["ais-mid + top(ais-mid)"]["theta"])
    fw = Forward(C, th0, seed=SEED + 9)
    mids, tops = [], []
    while len(mids) < NB // 2:
        fw.run(S_T, S_M, S_F, fresh=True)
        mg, tg = fw.mid.grid, fw.top.grid
        for _ in range(64):
            i, j = rng.integers(C.nmy - 1), rng.integers(C.nmx - 1)
            mids.append(mg[i:i + 2, j:j + 2].copy())
        for _ in range(32):
            i, j = rng.integers(C.nty - 1), rng.integers(C.ntx - 1)
            tops.append(tg[i:i + 2, j:j + 2].copy())
    mids = mids[:NB // 2]
    tops = tops[:NC // 2]
    pres_fwd = float(np.mean([(z > 0).mean() for z in mids]))
    zs = mids + mid_windows(NB - len(mids), rng)
    t = time.time()
    fa, se = windows(MW, zs, "ais", KA, MA, seed=SEED + 6)
    ua, gha, gva, resa, rmsa = fit_tables(np.array(zs), fa - F_R, C.D, RIDGE)
    th_m = materialise(C.theta0(), ua, gha, gva, ("mid_u", "mid_h", "mid_v"))
    t_mid = time.time() - t
    say(f"  mid refit: {len(mids)} forward-cut (presence {pres_fwd:.3f}) + {NB - len(mids)} uniform windows; "
        f"residual {resa:.3f} / {rmsa:.2f} ({t_mid:.0f}s)")
    tt, info = top_fit(th_m, rng, "E", z_extra=tops)
    th1 = dict(th_m)
    th1.update(tt)

    def move(a, b, k):
        a, b = np.asarray(a[k]), np.asarray(b[k])
        if a.ndim == 2:
            a, b = dc(a), dc(b)
        else:
            a, b = a - a.mean(), b - b.mean()
        d = a - b
        return dict(max=float(np.abs(d).max()), rms=float(np.sqrt((d ** 2).mean())), scale=float(np.sqrt((b ** 2).mean())))
    mv = {k: move(th1, th0, k) for k in FIT}
    for k in FIT:
        say(f"    move {k}: max {mv[k]['max']:.3f} rms {mv[k]['rms']:.3f} (table rms {mv[k]['scale']:.3f})")
    st, fwe = forward_eval(th1, SEED + 555)
    st2, _ = forward_eval(th1, SEED + 556)
    save_render(fwe, "images/induce_circles_fwd_iter.png")
    R["E"] = dict(theta=tl(th1), move=mv, final=tl(st), L1=l1(st, orc), noise=l1(st, st2), cmp_mid=cmp_mid(th1),
                  cmp_top=cmp_top(th1, orc), info=info, t_mid=t_mid, n_fwd_mid=len(mids), n_fwd_top=len(tops),
                  pres_fwd=pres_fwd, n_mid=len(zs), res_mid=resa, rms_mid=rmsa)
    say("  L1 " + " ".join(f"{k} {v:.3f}" for k, v in R["E"]["L1"].items()))
    save()

open("images/induce_circles_log.txt", "a").write("\n".join(LOG) + "\n")
