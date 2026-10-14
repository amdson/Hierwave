import json, sys
rows = [json.loads(l) for l in open("results/bench.jsonl")]
for v in sys.argv[1:]:
    r = [x for x in rows if x["variant"] == v][-1]
    print("==", v, r["run"], "train", r["train_budget_s"], "seeds", r["seeds"])
    for k, s in enumerate(r["per_seed"]):
        inf = r["info"][k]
        st = s["stats"]["raw"]
        b = s.get("blocks") or {}
        print(" seed", k, "updates", inf.get("updates"), "rkl", inf.get("rkl_updates"), "lam", [round(x, 1) for x in inf.get("lam", [])],
              "train_s", s.get("train_s"))
        print("   raw", [f"{float(x):.1e}" for x in s["viol"]["raw"].values()], "final", [f"{float(x):.1e}" for x in s["viol"]["final"].values()])
        print("   F %.3f" % s["free_energy"], "line_match", st["line_match"], "dirt", st["dirt_frac"])
        if b:
            print("   blocks", json.dumps({kk: vv for kk, vv in b.items() if kk in ("rho", "rho_pooled", "starved", "per_level")})[:1500])
            print("   block keys", list(b.keys()))
    print(" verdict worst final", r["verdict"]["final_violations"], "F", r["verdict"]["free_energy"], "accepted", r["verdict"]["accepted"])
print("-- v2 block check (direct [subtree]): rho_pool / starved_probe per level")
for v in sys.argv[1:]:
    r = [x for x in rows if x["variant"] == v][-1]
    for k, s in enumerate(r["per_seed"]):
        lv = s["blocks"]["levels"]
        f = lambda x: "None" if x is None else f"{x:.2f}"
        print(v, "seed", k, "rho", " / ".join(f"{f(lv[l]['direct']['rho_pool'])}[{f(lv[l]['subtree']['rho_pool'])}]" for l in sorted(lv)),
              "| starved %", " / ".join(f"{100*lv[l]['direct']['starved_probe']:.1f}" for l in sorted(lv)))
