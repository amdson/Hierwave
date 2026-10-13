"""Print cpass vs cordinal bench rows from results/bench.jsonl."""
import json

rs = [json.loads(line) for line in open("results/bench.jsonl")]
for r in rs:
    if r["variant"].startswith(("cpass", "cordinal")) and r["problem"] in ("toy_support", "ground", "columns"):
        v = r["verdict"]
        fails = [k for k, x in v.items() if isinstance(x, dict) and "ok" in x and not x["ok"]]
        print(r["variant"], r["problem"], r["tag"], "worst", v["final_violations"]["worst_frac"],
              "raw", v["raw_violations_worst"], "F", round(v["free_energy"], 3),
              "upd", [i["updates"] for i in r["info"]],
              "lam", [[round(x, 1) for x in i["lam"]] for i in r["info"]], "fail", fails)
        for s in r["per_seed"]:
            print("    ", {k: round(x[0], 3) for k, x in s["stats"]["raw"].items()}, "F", round(s["free_energy"], 3),
                  "rawmax", max(s["viol"]["raw"].values()))
