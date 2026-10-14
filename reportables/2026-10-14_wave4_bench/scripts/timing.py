"""Per-iteration time split and update counts of loop.jsonl runs. usage: timing.py <run_dir>..."""
import json, sys
import numpy as np
for d in sys.argv[1:]:
    rs = [json.loads(l) for l in open(f"{d}/loop.jsonl")]
    su = [r for r in rs if r.get("setup_done")][0]
    it = [r for r in rs if "time_ml" in r]
    if not it:
        print(d, "no iterations"); continue
    g = lambda k: np.mean([r.get(k, 0.0) for r in it])
    ctx, ref, diag, ml, rkl = g("time_ctx"), g("time_refine"), g("time_diag"), g("time_ml"), g("time_rkl")
    loop_t = it[-1]["t"] - su["t"]
    print(f"{d}: setup {su['t']:.1f} s | iters {len(it)} in {loop_t:.1f} s = {loop_t/len(it):.3f} s/iter | "
          f"ctx {ctx:.3f} (wait {g('time_ctx_wait'):.3f}) refine {ref:.3f} diag {diag:.3f} ML {ml:.3f} RKL {rkl:.3f} | "
          f"ML upd {it[-1]['update']} RKL upd {it[-1].get('rkl_updates')} | ML upd/s {it[-1]['update']/loop_t:.2f} "
          f"RKL upd/s {(it[-1].get('rkl_updates') or 0)/loop_t:.2f}")
