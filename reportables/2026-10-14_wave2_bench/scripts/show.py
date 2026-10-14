import json, sys
for arm in sys.argv[1:]:
    L = [json.loads(l) for l in open(f"runs/{arm}/loop.jsonl")]
    print("==", arm)
    for r in L:
        if "update" not in r: continue
        if "lv_own_raw" in r or r is L[-1]:
            print(r["update"], r.get("rkl_updates"), "dirt %.2f lm %.3f" % (r["own_raw"]["dirt"], r["own_raw"]["line_match"]),
                  "rawL", [round(x, 4) for x in r.get("lv_own_raw", {}).get("frac", [])],
                  "refL", [round(x, 4) for x in r.get("lv_own_refined", {}).get("frac", [])],
                  "vs %.2f ess %.1f" % (r["refined_viol_states_frac"], r["ml_ess"]),
                  "acc %.2f/%.2f/%.2f" % (r.get("acc_flip_L3") or 0, r.get("acc_flip_L4") or 0, r.get("acc_mtm_L2") or 0),
                  "lam", [round(x, 1) for x in r.get("lam", [])], "rklv", [round(x, 4) for x in r.get("rkl_raw_viol", [])], "t %.0f" % r["t"])
