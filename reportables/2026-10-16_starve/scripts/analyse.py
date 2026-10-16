import sys, numpy as np
from castlegen.bench import shares
from scipy.stats import spearmanr
f = np.load(sys.argv[1]); ref = {1: .011, 2: .045, 3: .151, 4: .52}
rng = np.random.default_rng(0)
print("level set n  starved%  rho(direct)  medDeficit | slack(B-A) mean/q90 | rho_dirt mean")
for l in range(5):
    a = f[f"L{l}"]; lq, dw, sw, kind, wid, slack, rho, ag = a.T
    own = kind == 0
    for w, nm in ((dw, "direct"), (sw, "subtree")):
        s, ic = shares._robust_line(w[own], lq[own], rng); d = (s * w + ic) - lq
        for k, sn in ((0, "own"), (1, "probe"), (2, "pop")):
            m = kind == k
            r = spearmanr(w[m], lq[m])[0] if m.sum() > 2 and np.ptp(w[m]) > 0 else float("nan")
            print(f"L{l} {nm:7s} {sn:5s} n={m.sum():6d} starved={100*(d[m]>5).mean():5.1f}% rho={r:6.3f} medDef={np.median(d[m]):6.2f} | slack {slack[m].mean():.3f}/{np.quantile(slack[m],.9):.3f} | dirt {rho[m].mean():.3f}")
        if nm == "direct" and l >= 3:
            for k, sn in ((1, "probe"), (2, "pop")):
                m = kind == k; st = d > 5
                for lab, mm in (("starved", m & st), ("ok", m & ~st)):
                    if mm.sum(): print(f"    {sn} {lab:7s} n={mm.sum():5d} slack mean {slack[mm].mean():.3f} med {np.median(slack[mm]):.3f}  dirt {rho[mm].mean():.3f}  weight {dw[mm].mean():.1f}")
    if l >= 3:
        bins = np.quantile(slack[kind >= 0], [0, .5, .8, .9, .95, .99, 1]) if False else np.array([0, .05, .1, .2, .4, .8, 1.6, 3.2, 1e9])
        print(f"  slack hist L{l} (fraction per bin {bins[1:-1]}):")
        for k, sn in ((0, "own"), (1, "probe"), (2, "pop")):
            h = np.histogram(slack[kind == k], bins)[0]; print("   ", sn, np.round(h / h.sum(), 3))
