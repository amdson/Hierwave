"""kblock: per-run summary of loop.jsonl at equal updates (population and own-draw line_match, dirt, valid,
per-level violation fractions, refresh / kb cost).  usage: loopsum.py <run_dir> ..."""
import sys, json, numpy as np
for d in sys.argv[1:]:
    it = [json.loads(l) for l in open(d + "/loop.jsonl")]
    it = [r for r in it if "iter" in r]
    print(d, "iters", len(it), "updates", it[-1]["update"], "rkl", it[-1].get("rkl_updates"))
    for lo, hi in ((0, 200), (200, 400), (400, 600), (600, 900), (900, 10**6)):
        w = [r for r in it if lo < r["update"] <= hi]
        if not w: continue
        lv = [r for r in w if "lv_pop" in r]
        f = lambda g: round(float(np.mean([g(r) for r in w])), 4)
        print(f"  upd {lo}-{hi} n={len(w)} pop lm {f(lambda r: r['pop']['line_match'])} dirt {f(lambda r: r['pop']['dirt'])} "
              f"valid {f(lambda r: r['pop']['valid'])} viol/cell {f(lambda r: r['pop']['viol_per_cell']):.2e} | own8 lm "
              f"{np.mean([r['own8']['line_match'] for r in lv]):.4f} dirt {np.mean([r['own8']['dirt'] for r in lv]):.4f} | "
              f"lv_pop {np.round(np.mean([r['lv_pop'] for r in lv], 0), 5).tolist()} lv_own {np.round(np.mean([r['lv_own'] for r in lv], 0), 5).tolist()} | "
              f"refine {1000*f(lambda r: r['time_refine']):.0f} ms kb {f(lambda r: r.get('time_kb_ms', 0)):.0f} ms chg_t {f(lambda r: r.get('kb_chg_t', 0)):.4f}")
