"""Per-L4-cell deviation d = rho4 - 256 f and the chain's empirical L4 marginal vs the direct weight.
usage: devs.py <pop_*.pkl> [more pkls]   (CPU, numpy only)
For stationary chain states: histogram of d by fill class; log occupancy + noise4(d) = the free energy of
everything below and beside the cell that the bench's direct weight does not contain (up to a constant)."""
import sys, pickle
import numpy as np
from castlegen.bench.problems import get
from castlegen.bench.problems import ground as G
from castlegen.bench.problem import Region
p = get("ground"); reg = Region(128)
D, F, S, SET = [], [], [], []
for f in sys.argv[1:]:
    P = pickle.load(open(f, "rb"))
    r4 = np.asarray(P["rho4"], float); fl = np.asarray(P["fill"], float); t = np.asarray(P["t"])
    for b in range(r4.shape[0]):
        L = G.lift(t[b])
        D.append((r4[b] - 16 * fl[b]).ravel()); F.append(fl[b].ravel()); S.append((L["rho4"] - 16 * fl[b]).ravel())
        SET.append(np.full(r4[b].size, 0 if b < 32 else 1))
D, F, S, SET = map(np.concatenate, (D, F, S, SET))
# probes (bench probe seeds): d of the exact lift
PD, PF = [], []
for i in range(32):
    seed = 1_000_000 + 600 + i
    for y in p.probes(seed, reg):
        fl = np.asarray(p.designed_values(seed, reg)["fill"], float)
        PD.append((np.asarray(y["rho4"], float) - 16 * fl).ravel()); PF.append(fl.ravel())
PD, PF = np.concatenate(PD), np.concatenate(PF)
def cls(f):
    return np.where(f == 0, 0, np.where(f == 16, 2, 1))
names = {0: "air (f=0)", 1: "mixed", 2: "dirt (f=16)"}
print("class  set        n     mean d   sd d   mean|d|  mean noise4/cell   tile-count dev (s-256f) mean/sd")
for c in (0, 1, 2):
    for sid, nm in ((0, "ownchain"), (1, "prbchain")):
        m = (cls(F) == c) & (SET == sid)
        if m.sum():
            print(f"{names[c]:11s} {nm:9s} {m.sum():5d} {D[m].mean():7.2f} {D[m].std():6.2f} {np.abs(D[m]).mean():7.2f} {(G.BETA*D[m]**2/256).mean():8.3f}   {S[m].mean():7.2f} {S[m].std():6.2f}")
    m = cls(PF) == c
    print(f"{names[c]:11s} {'probe':9s} {m.sum():5d} {PD[m].mean():7.2f} {PD[m].std():6.2f} {np.abs(PD[m]).mean():7.2f} {(G.BETA*PD[m]**2/256).mean():8.3f}")
# free energy profile for mixed cells (pooled chains): log occupancy(d) + noise4(d)
m = cls(F) == 1
bins = np.arange(-40, 41, 4)
h = np.histogram(D[m], bins)[0].astype(float); c = 0.5 * (bins[1:] + bins[:-1])
ok = h >= 5
prof = np.log(h[ok] / h.sum()) + G.BETA * c[ok] ** 2 / 256
print("mixed cells: d bin centre | chain occupancy | log occ + noise4(d) (= -F_rest(d) + const, relative to d=0 bin)")
z = prof[np.argmin(np.abs(c[ok]))]
for cc, hh, pp in zip(c[ok], h[ok], prof):
    print(f"  {cc:6.1f} {int(hh):5d} {pp - z:7.2f}")
pm = cls(PF) == 1
print("probe mixed-cell d quantiles 5/50/95:", np.quantile(PD[pm], [.05, .5, .95]).round(2),
      " chain:", np.quantile(D[m], [.05, .5, .95]).round(2))
