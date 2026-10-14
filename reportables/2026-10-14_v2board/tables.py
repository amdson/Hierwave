import json
rows=[json.loads(l) for l in open('../../results/bench.jsonl')]
FS={'toy_support':-0.4688,'toy_modes':0.2513}
want=[('toy_modes','cbig256','v2board'),('toy_modes','cfast','v2board'),('toy_support','cbig256','v2board'),('toy_support','cfast','chot_gate3'),('ground','cfast','v2board'),('ground','creplay','v2board'),('ground','cstart','cstart'),('dimers','cfast','b300')]
sel={}
for r in rows:
    if r['train_budget_s']==300 and r['per_seed'][0].get('blocks'): sel[(r['problem'].split()[0],r['variant'],r['tag'])]=r
def f(x): return 'n/a' if x is None else f'{x:.2f}'
for p in ['toy_modes','toy_support','ground','dimers']:
    print(f'\n#### {p}\n')
    print('| variant (seed) | F (gap) | worst final viol | old probe flag | updates | level | w | rho pool | rho probes | rho own | starved % probe | own null % | median deficit (nats) |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for k in want:
        if k[0]!=p or k not in sel: continue
        r=sel[k]
        for i,ps in enumerate(r['per_seed']):
            fe=ps['free_energy']; gap=f" ({fe-FS[p]:+.4f})" if p in FS else ''
            wv=max(ps['viol']['final'].values())
            fl=(ps['probes'] or {}).get('flag')
            up=(r['info'][i] or {}).get('updates')
            first=True
            for L,d in sorted(ps['blocks']['levels'].items()):
                w='direct' if d['direct']['rho_pool'] is not None or p not in('toy_modes','toy_support') else 'subtree'
                x=d[w]
                head=f"{k[1]} s{i} | {fe:.4f}{gap} | {wv:.1e} | {'FLAG' if fl else fl} | {up}" if first else ' | | | |'
                first=False
                print(f"| {head} | {L} | {w[:3]} | {f(x['rho_pool'])} | {f(x['rho_probe'])} | {f(x['rho_own'])} | {100*x['starved_probe']:.1f} | {100*x['starved_own']:.1f} | {x['median_deficit_probe']:.1f} |")
