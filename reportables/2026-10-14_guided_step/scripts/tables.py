"""Markdown tables from the offline_eval json files (no JAX)."""
import json, sys
import numpy as np
TOL = {"dirt_frac": 0.03, "dirt_excess": 0.02, "roughness": 0.15, "line_match": 0.03, "floating": 0.5, "fill_err": 0.01}
BUDGET = [52, 13, 3, 0, 0]          # violated instances allowed in 16 worlds at 2e-4 (262144/65536/16384/4096/512)
models = [("pen_s0", "`ckpop_hard_pen` cr2 arm seed 0 (bench 3/4 seeds)"), ("rkl_w2b", "`ckpop_hard_rkl` wave4 W2b, 300 s"),
          ("rkl123x2_C", "`ckpop_rkl123x2` wave6 C"), ("hard_w1", "`ckpop_hard` wave4 W1 (no violation term)")]
rows = ["raw", "repair", "jitter1", "jitter2", "guided1", "guided2", "guided4", "guided8", "guided4+repair", "guided8+repair"]
def agree(m1, s1, m2, s2, tol): return abs(m1 - m2) <= 3 * np.hypot(s1, s2) + tol
for key, title in models:
    r = json.load(open(f"reportables/2026-10-14_guided_step/runs/{key}.json"))
    K = r["kinds"]; raw = K["raw"]["stats"]
    print(f"\n**{title}** (`runs/{key}.json`; lam = {np.round(r['lam'], 1).tolist()})\n")
    print("| state | violated L0/L1/L2/L3/L4 (16 worlds) | total | budget | graded mass L0..L4 | cells changed / world | valid worlds | stats vs raw (bench rule) |")
    print("|---|---|---|---|---|---|---|---|")
    for kd in rows:
        x = K[kd]; v = [int(a) for a in x["viol"]]; m = [int(a) for a in x["mass"]]
        ok = all(v[l] <= BUDGET[l] for l in range(5))
        bad = [s for s in TOL if s in raw and not agree(raw[s][0], raw[s][1], x["stats"][s][0], x["stats"][s][1], TOL[s])]
        st = x["stats"]
        d = f"lm {st['line_match'][0]:.3f} dirt {st['dirt_frac'][0]:.3f} rough {st['roughness'][0]:.2f} float {st['floating'][0]:.1f}"
        print(f"| {kd} | {'/'.join(map(str, v))} | {sum(v)} | {'pass' if ok else 'fail'} | {'/'.join(map(str, m))} | {sum(x['changed_per_world']):.1f} | {x['worlds_valid']}/16 | {d}{' **disagree: ' + ', '.join(bad) + '**' if bad else ' (all agree)'} |")
    print(f"\ncells changed per guided step (mean over worlds): {np.round(r['step_changed'], 1).tolist()}; "
          f"time per world (GPU 1 unless noted): pass {r['time_per_world']['gen']*1e3:.1f} ms, repair {r['time_per_world']['repair']*1e3:.1f} ms, "
          f"jitter R=1 {r['time_per_world']['jitter1']*1e3:.1f} ms, R=2 {r['time_per_world']['jitter2']*1e3:.1f} ms, 8 guided steps {r['time_per_world']['guided']*1e3:.1f} ms ({r['time_per_world']['guided']/8*1e3:.1f} ms per step)")
