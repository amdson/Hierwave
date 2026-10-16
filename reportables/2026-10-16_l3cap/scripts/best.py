import json, sys, os
R = sys.argv[1]; best = ("none", 1e9)
for a in ("w2", "c2", "wc2"):
    f = f"{R}/{a}/classes_s0.json"
    if os.path.exists(f):
        d = json.load(open(f))[a]["probe/hi_cross"]["med_deficit"]
        if d < best[1]: best = (a, d)
print(best[0])
