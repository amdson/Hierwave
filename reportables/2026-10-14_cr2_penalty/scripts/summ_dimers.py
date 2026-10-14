"""Summarise dk_loop jsonl runs: updates, h/v/d of raw own draws and refined states (first 3 / last 10 iterations),
violations per cell, lambda at the end.  usage: summ_dimers.py <run.jsonl> ..."""
import json, sys
import numpy as np
for f in sys.argv[1:]:
    rs = [json.loads(l) for l in open(f)]
    it = [r for r in rs if "iter" in r]
    if not it:
        print(f, "no iterations"); continue
    def m(key, rows): return np.mean([r[key] for r in rows], 0).round(3).tolist()
    first, last = it[:3], it[-10:]
    print(f, "| updates", it[-1]["update"], "iters", len(it), "t", round(it[-1]["t"]),
          "\n  raw   first", m("orient_raw", first), "last", m("orient_raw", last),
          "\n  refnd first", m("orient_refined", first), "last", m("orient_refined", last),
          "\n  viol raw last", float(np.mean([r["viol_raw"] for r in last])), "refined", float(np.mean([r["viol_refined"] for r in last])),
          "valid_frac_raw", float(np.mean([r["valid_frac_raw"] for r in last])),
          "\n  rkl_raw_viol last", (m("rkl_raw_viol", last) if "rkl_raw_viol" in last[-1] else None), "lam end", it[-1].get("lam"))
