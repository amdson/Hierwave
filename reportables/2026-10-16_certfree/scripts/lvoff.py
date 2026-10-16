import json, sys, numpy as np
for f in sys.argv[1:]:
    L = [json.loads(l) for l in open(f)]; L = [r for r in L if "own_raw" in r][-25:]
    o = lambda k, key: np.mean([r[k][key] for r in L], 0).round(2).tolist()
    d = lambda k: round(float(np.mean([r[k]["dirt"] for r in L])), 3)
    print(f, "\n  own_raw offset/level", o("own_raw", "offset"), "dirt", d("own_raw"), "\n  pop(chain-like) offset/level", o("pop", "offset"), "dirt", d("pop"))
