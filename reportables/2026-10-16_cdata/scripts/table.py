import json, os, glob
R = "reportables/2026-10-16_cdata/runs"
print("| arm | starved probe/own L2 | L3 | L4 | rho_pool L3 | L4 | line_match | raw viol L0..L4 | offset L0..L4 (raw) | updates | upd/s | ms/iter | post-stage viol (R30) | stage rough | stage offset | |c1| | stage lm | tiles chg |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for d in sorted(glob.glob(f"{R}/arm_*/eval.json")):
    a = d.split("arm_")[1].split("/")[0]; m = json.load(open(d)); L = m["blocks"]["levels"]
    s = lambda l: f"{L[str(l)]['direct']['starved_probe']:.3f}/{L[str(l)]['direct']['starved_own']:.3f}"
    t = m["timing"]; q = f"{R}/q2_{a}.json"
    st = ["", "", "", "", "", ""]
    if os.path.exists(q):
        x = json.load(open(q))
        for ar in x["arms"]:
            if ar.get("init", 0) == 0:
                f = ar["final"]; st = [",".join(map(str, f["viol"])), f"{f['chain_obs']['rough_circ']:.2f}", f"{f['chain_obs']['offset']:+.2f}", f"{f['chain_obs']['c1abs']:.2f}", f"{f['stats']['line_match']:.3f}", f"{100*f['tile_changed']:.1f} %"]
    print(f"| {a} | {s(2)} | {s(3)} | {s(4)} | {L['3']['direct']['rho_pool']:.2f} | {L['4']['direct']['rho_pool']:.2f} | {m['stats']['raw']['line_match'][0]:.3f} | {','.join(f'{m['viol']['raw'][str(l)]:.1e}' for l in range(5))} | {','.join(f'{o:+.2f}' for o in m['offset_raw_L0_L4'])} | {m['updates']} | {m['updates']/m['train_s']:.2f} | {1000*t['total']/t['iters']:.0f} | " + " | ".join(st) + " |")
