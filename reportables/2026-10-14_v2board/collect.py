import json,sys,collections
V2='ee11dda34556'
FSTAR={'toy_support':-0.4688,'toy_modes':0.2513}
rows=[json.loads(l) for l in open('../../results/bench.jsonl')]
def f(x,n=2):
    return 'n/a' if x is None else f'{x:.{n}f}'
sel={}
for r in rows:
    if r['train_budget_s']!=300 or 'blocks' not in r['per_seed'][0] or not r['per_seed'][0]['blocks']: continue
    k=(r['problem'],r['variant'],r['tag'])
    sel[k]=r
for (p,v,t),r in sorted(sel.items()):
    print(f"\n### {p} {v}  run {r['run']} harness {r['harness'][:8]} tag {r['tag']}")
    for i,ps in enumerate(r['per_seed']):
        b=ps['blocks']; fe=ps['free_energy']
        fs=FSTAR.get(p.split()[0])
        gap='' if fs is None else f' gap {fe-fs:+.4f}'
        vi=ps['viol']
        print(f" seed{i}: F {fe:.4f}{gap} updates {(r['info'][i] or {}).get('updates')} probe_flag {(ps['probes'] or {}).get('flag')} viol {json.dumps(vi)[:160]}")
        for L,d in sorted(b['levels'].items()):
            for w in ('direct','subtree'):
                x=d[w]
                print(f"   L{L} {w[:3]} rho pool/probe/own {f(x['rho_pool'])}/{f(x['rho_probe'])}/{f(x['rho_own'])} starved probe {100*x['starved_probe']:.1f}% own {100*x['starved_own']:.1f}% med_def {x['median_deficit_probe']:.1f}")
