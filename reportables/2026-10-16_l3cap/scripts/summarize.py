import json, os, sys
R = sys.argv[1]
def L(f):
    return json.load(open(f)) if os.path.exists(f) else None
rows = []
print("| arm | seed | updates | iters | train s | raw line_match | raw viol L0-L4 | hi_cross starved % (probe/own) | hi_cross med deficit (probe/own) | slack0 starved % / med deficit (probe) | rho(q,D) hi_cross probe/own | rho(q,D) slack0 probe | L3 starved % relaxed (probe/own) | L3 rho_pool relaxed | L4 starved % relaxed (probe/own) | L4 rho_pool relaxed |")
print("|" + "---|" * 16)
for arm in ("ctl", "w2", "c2", "wc2"):
    for s in (0, 1):
        d = f"{R}/{arm}"
        u = L(f"{d}/updates_s{s}.json"); q = L(f"{d}/q2_s{s}.json"); c = L(f"{d}/classes_s{s}.json"); dc = L(f"{d}/decomp_s{s}.json"); rc = L(f"{d}/relaxcheck_s{s}.json")
        if not (u and q and c and rc): continue
        c = c[arm]; dc = dc[arm]; rx = rc[f"{arm}/relaxed"]
        hp, ho, sp = c["probe/hi_cross"], c["own/hi_cross"], c["probe/slack0"]
        pr = q["pass_raw"]
        print(f"| {arm} | {s} | {u['updates']} | {u['iters']} | {u['train_wall']:.0f} | {pr['stats']['line_match']:.3f} | {pr['viol']} | {100*hp['starved']:.1f} / {100*ho['starved']:.1f} | {hp['med_deficit']:.2f} / {ho['med_deficit']:.2f} | {100*sp['starved']:.1f} / {sp['med_deficit']:.2f} | {dc['probe/hi_cross']['rho_direct']:.2f} / {dc['own/hi_cross']['rho_direct']:.2f} | {dc['probe/slack0']['rho_direct']:.2f} | {100*rx['starved_probe'][3]:.1f} / {100*rx['starved_own'][3]:.1f} | {rx['rho_pool'][3]:.2f} | {100*rx['starved_probe'][4]:.1f} / {100*rx['starved_own'][4]:.1f} | {rx['rho_pool'][4]:.2f} |")
