"""How many cells, at which levels, must change together to free (part of) one column?
Fine states t -> t' differ by clearing the top k dirt tiles of one fine column; both
are lifted exactly (ground.lift: rho = counts, cert A = full columns, B = occupied
columns).  Counts changed cells per channel.  Starts: all-dirt (own-draw collapse)
and a probe (surface near the designed line).  numpy only."""
import json, sys
import numpy as np
from castlegen.bench.problems import get
from castlegen.bench.problems.ground import lift, T
from castlegen.bench.problem import Region
p = get("ground"); reg = Region(128)
out = {}
starts = {"all_dirt": np.ones((64, 128), np.int64), "probe": p.probes(7, reg, n=1)[0]["t"]}
for name, t in starts.items():
    L0 = lift(t)
    for x in (37, 64):
        col = t[:, x]
        top = int(np.argmax(col)) if col.any() else 64
        for k in (1, 2, 4, 8, 16):
            t2 = t.copy(); t2[top:top + k, x] = 0
            L1 = lift(t2)
            ch = {c: int((L0[c] != L1[c]).sum()) for c in L0}
            per_level = {l: ch["t"] if l == 0 else ch[f"rho{l}"] + ch[f"cert{l}"] for l in range(T + 1)}
            certB = {l: int((p.chan[f"cert{l}"].views["B"][L0[f"cert{l}"]] != p.chan[f"cert{l}"].views["B"][L1[f"cert{l}"]]).sum()) for l in range(1, T + 1)}
            certA = {l: int((p.chan[f"cert{l}"].views["A"][L0[f"cert{l}"]] != p.chan[f"cert{l}"].views["A"][L1[f"cert{l}"]]).sum()) for l in range(1, T + 1)}
            r = {"start": name, "x": x, "top": top, "k": k, "cells_changed_total": sum(ch.values()),
                 "per_level": per_level, "certB_changed": certB, "certA_changed": certA}
            out.setdefault("rows", []).append(r); print(json.dumps(r))
json.dump(out, open(sys.argv[1], "w"), indent=1)
