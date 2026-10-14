"""Per-seed violation counts and pooled per-level final rates (Clopper-Pearson 95 %) of a bench row."""
import json, sys
from scipy.stats import beta
row = [json.loads(l) for l in open(sys.argv[1]) if l.strip()][-1]
D = "/workspace/hierwave_git/results/runs/" + row["run"]
T = 4; tot = {l: [0, 0] for l in range(T + 1)}
for k, m in enumerate(row["per_seed"]):
    L = [json.loads(l) for l in open(f"{D}/train_s{k}.jsonl")]
    st = [x["t"] for x in L if x.get("setup_done")][0]; it = [x for x in L if "iter" in x]
    n = m["viol"]["final_n"]; f = m["viol"]["final"]
    cnt = {l: round(f[str(l)] * n[str(l)]) for l in range(T + 1)}
    for l in cnt: tot[l][0] += cnt[l]; tot[l][1] += n[str(l)]
    lv = m["blocks"]["levels"]
    ok = max(f[str(l)] for l in range(T + 1)) < 2e-4
    print(f"seed {k}: setup {st:.1f} s, updates {it[-1]['update']}/{it[-1].get('rkl_updates')}, counts {[cnt[l] for l in range(T+1)]} "
          f"of {[n[str(l)] for l in range(T+1)]}, worst {max(f.values()):.2g}, {'PASS' if ok else 'fail'} viol, "
          f"F {m['free_energy']:.4f}, lm {m['stats']['final']['line_match'][0]:.3f}, rough {m['stats']['final']['roughness'][0]:.2f}, "
          f"nvalid {m['n_valid_raw']}, probes {None if m['probes'] is None else m['probes']['flag']}, "
          f"rho {[round(lv[str(l)]['direct']['rho_pool'], 2) for l in range(T+1)]}, "
          f"starved {[round(100*lv[str(l)]['direct']['starved_probe'], 1) for l in range(T+1)]}")
for l in range(T + 1):
    x, n = tot[l]
    lo = beta.ppf(0.025, x, n - x + 1) if x else 0.0; hi = beta.ppf(0.975, x + 1, n - x)
    print(f"pooled L{l}: {x}/{n} = {x/n:.2e}  95% CI [{lo:.2e}, {hi:.2e}]  {'over' if lo > 2e-4 else ('under' if hi < 2e-4 else 'straddles')} budget 2e-4")
print(json.dumps({k: v for k, v in row["verdict"].items() if k in ("final_violations", "across_seeds", "probes", "within_budget", "free_energy", "accepted", "raw_vs_final", "finite_support")}, default=float))
