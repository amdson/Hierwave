"""Print the training trend of a sweep arm: own8 line_match / dirt / offset per level, lv_own, lambda, rec diagnostics."""
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
rows = [r for r in rows if "iter" in r and r.get("own8")]
step = max(1, len(rows) // int(sys.argv[2] if len(sys.argv) > 2 else 10))
for r in rows[::step] + rows[-1:]:
    o = r["own8"]; rc = r.get("rec") or {}
    print(r["iter"], round(r["wall"]), "lm %.3f dirt %.3f" % (o["line_match"], o["dirt"]),
          "off", [round(x, 2) for x in o.get("offset", [])], "lv", [f"{x:.1e}" for x in r.get("lv_own", [])],
          "lam", [round(x, 1) for x in r.get("lam", [])],
          "ess3 %.2f ess4 %.2f Rf3 %.2f f13 %.2f" % (rc.get("ess_l3_rho3", 0), rc.get("ess_l4_rho4", 0), rc.get("Rfoot_l3_rho3", 0), rc.get("foot1_l3_rho3", 0)) if rc else "")
