"""Summarise diag1_*.json: per model and level, binary violated fraction and graded mass (raw / final),
mass per violated instance, per-factor breakdown, per-world distribution and the distance histogram."""
import json, glob, sys
import numpy as np
R = "reportables/2026-10-14_graded/runs/"
order = ["cfast300", "cr2_hard_pen_s0", "w6_C_rkl123x2", "w6_D_rkl123x2_cg40", "w4_W1_hard", "w4_W2b_hard_rkl"]
rows_lvl, rows_fac, rows_world, rows_hist = [], [], [], []
for lab in order:
    d = json.load(open(R + f"diag1_{lab}.json"))
    W = d["worlds"]; n = len(W); T1 = len(d["nhard"]); nh = np.array(d["nhard"])
    for kind in ("raw", "final"):
        V = np.array([w[kind]["viol"] for w in W]); M = np.array([w[kind]["mass"] for w in W])
        lm = np.mean([w[kind]["stats"]["line_match"] for w in W]); df = np.mean([w[kind]["stats"]["dirt_frac"] for w in W])
        for l in range(T1):
            v, m = V[:, l].sum(), M[:, l].sum()
            rows_lvl.append((lab, kind, l, v, v / (n * nh[l]), m, m / (n * nh[l]), (m / v if v else float("nan")),
                             int((V[:, l] > 0).sum()), n))
        fac = {}
        for w in W:
            for k, (vv, mm) in w[kind]["per_factor"].items():
                a = fac.setdefault(k, [0, 0]); a[0] += vv; a[1] += mm
        for k, (vv, mm) in sorted(fac.items()):
            rows_fac.append((lab, kind, k, vv, mm))
        hist = {}
        for w in W:
            for l in range(T1):
                for dd, c in w[kind]["hist"][str(l)].items():
                    hist[(l, int(dd))] = hist.get((l, int(dd)), 0) + c
        rows_hist.append((lab, kind, dict(sorted(hist.items()))))
        tot_v, tot_m = V.sum(1), M.sum(1)
        rows_world.append((lab, kind, int((tot_v == 0).sum()), n, tot_v.tolist(), tot_m.tolist(), lm, df))
out = []
out.append("### Per level (16 held-out eval worlds each, W = 128; budget 2e-4)\n")
out.append("| model | kind | L | violated | frac | mass D | D / instances | D per violated | worlds with a violation |")
out.append("|---|---|---|---|---|---|---|---|---|")
for lab, kind, l, v, fr, m, mfr, ratio, nw, n in rows_lvl:
    out.append(f"| {lab} | {kind} | {l} | {v} | {fr:.1e} | {m} | {mfr:.1e} | {ratio:.2f} | {nw}/{n} |")
out.append("\n### Per factor (summed over the 16 worlds; violated instances, graded mass)\n")
out.append("| model | kind | factor | violated | mass |"); out.append("|---|---|---|---|---|")
for r in rows_fac: out.append("| " + " | ".join(str(x) for x in r) + " |")
out.append("\n### Distance histogram (level, d) -> violated instances, summed over worlds\n")
for lab, kind, h in rows_hist:
    out.append(f"- {lab} {kind}: " + ", ".join(f"L{l} d={dd}: {c}" for (l, dd), c in h.items()))
out.append("\n### Per world: total violated instances / total mass (all levels)\n")
out.append("| model | kind | valid worlds | violated per world | mass per world | line_match | dirt |"); out.append("|---|---|---|---|---|---|---|")
for lab, kind, nv, n, tv, tm, lm, df in rows_world:
    out.append(f"| {lab} | {kind} | {nv}/{n} | {tv} | {tm} | {lm:.3f} | {df:.3f} |")
open(R + "diag1_summary.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
