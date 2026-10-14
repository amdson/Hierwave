"""Summarise arm runs / bench rows (wave5-pop)."""
import sys, json, numpy as np
def blocks(b):
    if not b: return None, None
    L = b["levels"]
    rho = [L[l]["rho_pool"] if isinstance(L, list) else L[str(l)].get("rho_pool") for l in range(len(L))] if isinstance(L, list) else None
    return L
for d in sys.argv[1:]:
    m = json.load(open(d + "/eval.json"))
    loop = [json.loads(l) for l in open(d + "/loop.jsonl")]
    it = [r for r in loop if "iter" in r]
    lv = [r for r in it if "lv_own" in r]
    last = lambda key: np.mean([r[key] for r in lv[-5:]], 0).round(5).tolist() if lv else None
    print(d)
    print(" setup", round(m["timing"]["setup"], 1), "updates", m["updates"], "rkl", m["rkl_updates"])
    print(" final", {k: round(v, 6) for k, v in m["viol"]["final"].items()})
    print(" raw  ", {k: round(v, 6) for k, v in m["viol"]["raw"].items()})
    print(" F", m["free_energy"], "line_match", m["stats"]["final"]["line_match"], "dirt", m["stats"]["final"]["dirt_frac"],
          "n_valid_raw", m["n_valid_raw"], "probes", m["probes"])
    print(" lv_own(last5)", last("lv_own"), "pop valid", it[-1]["pop"]["valid"], "pop lm", it[-1]["pop"].get("line_match"),
          "own8 lm", np.mean([r["own8"]["line_match"] for r in lv[-5:]]).round(3), "lam", it[-1].get("lam"))
    b = m["blocks"]
    print(" blocks", json.dumps(b["levels"], default=float)[:1500] if b else None)
