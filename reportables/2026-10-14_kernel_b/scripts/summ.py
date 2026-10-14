import sys, json
for f in sys.argv[1:]:
    for l in open(f):
        if not l.startswith('{'): continue
        d = json.loads(l)
        print(d['problem'], d['q'], d['Ls'], d['lam'], 'K', d['K'], 'tau', d['tau'], d['start'], 'dropq' if d['drop_q'] else '', 'acc', {k: round(v, 3) for k, v in d['acc'].items()}, 'maxz', round(d['max_abs_z'], 1),
              {k: (round(v[0], 4), round(v[2], 4), round(v[3], 1)) for k, v in d['stats'].items()})
