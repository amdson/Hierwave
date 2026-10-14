import json, sys
for r in sys.argv[1:]:
    L = [json.loads(l) for l in open(f'reportables/2026-10-14_cstart/runs/{r}/train_ground.jsonl')]
    U = [x for x in L if 'update' in x]
    print(r, 'updates', U[-1]['update'], 't', round(U[-1]['t'], 1))
    for x in [x for x in L if 'diag' in x][::2]:
        g = x['diag']
        print(f"{x['update']:5d} tau {x['tau']:.2f} dirt {g['dirt']:.3f} lm {g.get('line_match', float('nan')):.3f} H",
              [round(g[f'L{l}']['H'], 2) for l in range(5)], 'lam', [round(v, 1) for v in x['lam']])
