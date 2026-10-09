"""numba kernel for JointLevel._cond_batch (the batched E_c of one colour):
same inputs, same result up to float rounding, one parallel loop over cells.
JointLevel(fast="numba") uses it."""
import numpy as np
from numba import njit, prange


@njit(cache=True, inline="always")
def _cont(u, dy, dx, h, m, my, torus):
    uy, ux = u // m, u % m
    y = (uy + dy * h) % my if torus else min(max(uy + dy * h, 0), my - 1)
    return y * m + (ux + dx * h) % m


@njit(cache=True, parallel=True, fastmath=True)
def cond_batch(X, ey, ex, cand, G, c0, NE, OFF, h, m, my, periodic, torus):
    k, M = cand.shape
    NR, C = X.shape
    D = c0.shape[0]
    S = OFF.shape[0]
    out = np.zeros((k, M), np.float32)
    for i in prange(k):
        base = np.empty(D, np.float32)
        offs = np.empty(S, np.int64)
        for sh in range(S):
            qy, qx = ey[i] - OFF[sh, 0], ex[i] - OFF[sh, 1]
            if qy < 0 or qy >= NR:
                continue
            if not periodic and (qx < 0 or qx >= C):
                continue
            qx = qx % C
            uq = X[qy, qx]
            me = OFF[sh, 0] == 0 and OFF[sh, 1] == 0
            for d in range(D):
                base[d] = -c0[d]
            no = 0
            for s2 in range(S):
                if s2 == sh:
                    continue
                y2, x2 = qy + OFF[s2, 0], qx + OFF[s2, 1]
                off = y2 < 0 or y2 >= NR or (not periodic and (x2 < 0 or x2 >= C))
                if off:
                    if me:
                        offs[no] = s2
                        no += 1
                        continue
                    c = _cont(uq, OFF[s2, 0], OFF[s2, 1], h, m, my, torus)
                else:
                    c = X[y2, x2 % C]
                for d in range(D):
                    base[d] += G[s2, c, d]
            if me:
                for j in range(M):
                    c = cand[i, j]
                    acc = 0.0
                    for d in range(D):
                        v = base[d] + G[sh, c, d] - NE[c, d]
                        for t in range(no):
                            s2 = offs[t]
                            v += G[s2, _cont(c, OFF[s2, 0], OFF[s2, 1], h, m, my, torus), d]
                        acc += v * v
                    out[i, j] += acc
            else:                                                           # |a + g|^2 - |a|^2, a = base - NE[u_q]
                for d in range(D):
                    base[d] -= NE[uq, d]
                for j in range(M):
                    c = cand[i, j]
                    acc = 0.0
                    for d in range(D):
                        g = G[sh, c, d]
                        acc += g * (2.0 * base[d] + g)
                    out[i, j] += acc
    return out
