"""Where marian wildcards sit and why: per wildcard, its layer and, for each
of its 6 neighbours, whether some in-domain module fits all neighbours but
that one (the neighbour 'blocks' it).  Tallies blocking (direction,
neighbour prototype) and the wildcard's 6-neighbour prototype signature."""
import sys
from collections import Counter

import numpy as np

from castlegen import marian
from castlegen.infinite import BlockSchedule, generate

DN = ["W(col-1)", "down", "S(row+1)", "E(col+1)", "up", "N(row-1)"]
D3 = ((0, 0, -1), (-1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0), (0, -1, 0))


def main(n=48, seeds=(0, 1, 2), K=32):
    ms = marian.load()
    m = marian.tile_model(ms)
    T, L = m.T, m.L
    pname = np.array([ms.names[t].rsplit(" ", 1)[0] for t in range(T)])
    bs = BlockSchedule.anneal(B=8, Sb=3, K=K, shift=True, restart=False, beta0=1.0, gamma0=2.0)
    layers, block, sig, nblk = Counter(), Counter(), Counter(), Counter()
    total = 0
    for sd in seeds:
        g = generate(m, bs, sd, 0, 0, n, n)
        for k, i, j in zip(*np.nonzero(g == T)):
            if not (0 < i < n - 1 and 0 < j < n - 1):
                continue
            total += 1
            layers[k] += 1
            nbrs = []
            for d, (dk, di, dj) in enumerate(D3):
                kk, ii, jj = k + dk, i + di, j + dj
                nbrs.append((d, g[kk, ii, jj]) if 0 <= kk < L else (d, -1))
            cnt = np.zeros(T, np.int64)          # neighbours each candidate violates
            per = {}
            for d, nmod in nbrs:
                if nmod < 0 or nmod >= T:
                    per[d] = np.zeros(T, bool)
                    continue
                bad = m.F[d, nmod].astype(bool)
                per[d] = bad
                cnt += bad
            dom = m.dom[k]
            blockers = [d for d, _ in nbrs if np.any(dom & (cnt - per[d] == 0))]
            nblk[len(blockers)] += 1
            for d in blockers:
                nmod = dict(nbrs)[d]
                block[(DN[d], pname[nmod] if 0 <= nmod < T else "WILD")] += 1
            sig[tuple(pname[nm] if 0 <= nm < T else ("edge" if nm < 0 else "WILD") for _, nm in nbrs)] += 1
    print(f"{total} interior wildcards; by layer {dict(sorted(layers.items()))}")
    print("number of single neighbours whose removal makes it fit:", dict(sorted(nblk.items())))
    print("top blocking (direction, neighbour prototype):")
    for (d, p), c in block.most_common(15):
        print(f"  {c:4d}  {d:9s} {p}")
    print("top neighbour signatures (W, down, S, E, up, N):")
    for s, c in sig.most_common(8):
        print(f"  {c:4d}  {s}")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:2]))
