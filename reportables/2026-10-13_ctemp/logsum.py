"""Summarise training logs: python logsum.py <train_s*.jsonl>..."""
import json
import sys

for f in sys.argv[1:]:
    L = [json.loads(l) for l in open(f)]
    L = [d for d in L if 'update' in d]
    S = [d for d in L if any(k.startswith('st_') for k in d)]
    if S:
        L = S
    print(f, len(L))
    for d in L[::max(1, len(L) // 12)] + [L[-1]]:
        extra = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items() if k.startswith('st_')}
        print(round(d['t'], 1), d['update'], round(d['tau'], 2), [f'{v:.1e}' for v in d['raw_viol']],
              [round(v, 2) for v in d['lam']], round(d['cost_per_world'], 1), extra)
