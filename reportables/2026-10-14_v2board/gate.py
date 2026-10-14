import json
rows=[json.loads(l) for l in open('../../results/bench.jsonl')]
sel={}
for r in rows:
    if r['train_budget_s']==300 and r['per_seed'][0].get('blocks'): sel[(r['problem'].split()[0],r['variant'],r['tag'])]=r
print('| problem | variant | seed | max starved% (sub) | worst level | max own null % | min rho_probe (levels w/ rho) | min rho_probe L0 | gate A starv<=max(3,null+2) | gate B rho_probe>=0.4 all levels | A and B |')
print('|---|---|---|---|---|---|---|---|---|---|---|')
for k,r in sorted(sel.items()):
    for i,ps in enumerate(r['per_seed']):
        L=ps['blocks']['levels']
        st=[(100*d['subtree']['starved_probe'],l,100*d['subtree']['starved_own']) for l,d in L.items()]
        mx=max(st); nul=max(s[2] for s in st)
        A=all(s[0]<=max(3,s[2]+2) for s in st)
        rp=[d['subtree']['rho_probe'] for d in L.values() if d['subtree']['rho_probe'] is not None]
        B=all(x>=0.4 for x in rp) if rp else None
        r0=L['0']['subtree']['rho_probe']
        f=lambda x:'n/a' if x is None else f'{x:.2f}'
        print(f"| {k[0]} | {k[1]} ({k[2]}) | {i} | {mx[0]:.1f} | L{mx[1]} | {nul:.1f} | {f(min(rp) if rp else None)} | {f(r0)} | {'pass' if A else 'FAIL'} | {'pass' if B else ('n/a' if B is None else 'FAIL')} | {'pass' if A and B else 'FAIL'} |")
