import json, glob, os
R = "reportables/2026-10-16_longfit2/runs"
print("| arm | seed | upd | raw lm | raw dirt | raw viol L0-4 | raw valid | offset | rough | |c1| | R15 lm | R15 viol | R15 valid | rho L0-4 | starved_probe L0-4 | starved_own L0-4 |")
print("|"+"---|"*16)
for arm in ("big300", "big600", "small600"):
    for s in (0, 1):
        f = f"{R}/{arm}/q2_s{s}.json"
        if not os.path.exists(f): continue
        q = json.load(open(f)); r = q["pass_raw"]; a = q["arms"][0]["final"] if q.get("arms") else None
        u = json.load(open(f"{R}/{arm}/updates_s{s}.json"))["updates"]
        b = {}
        try: b = json.loads(open(f"{R}/{arm}/blocks_s{s}.txt").read().strip().splitlines()[-1])
        except Exception: pass
        c = r["chain_obs"]; fm = lambda x: "/".join(f"{v:.3f}" for v in x) if x else "-"
        print(f"| {arm} | {s} | {u} | {r['stats']['line_match']:.3f} | {r['stats']['dirt_frac']:.3f} | {','.join(map(str,r['viol']))} | {r['valid']} | {c['offset']:+.2f} | {c['rough_circ']:.2f} | {c['c1abs']:.2f} | "
              + (f"{a['stats']['line_match']:.3f} | {','.join(map(str,a['viol']))} | {a['valid']} | " if a else "- | - | - | ")
              + f"{fm(b.get('rho_pool'))} | {fm(b.get('starved_probe'))} | {fm(b.get('starved_own'))} |")
