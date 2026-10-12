import time, numpy as np
from castlegen import marian
from castlegen.infinite import BlockSchedule, generate
from castlegen.plaquette import generate_mh

ms = marian.load()
m = marian.tile_model(ms)
T = m.T


def bs(K):
    return BlockSchedule.anneal(B=8, Sb=3, K=K, shift=True, restart=False, beta0=1.0, gamma0=2.0)


# exactness: overlapping requests agree; mh_from beyond the schedule equals plain generate
s = bs(4)
a = generate_mh(m, s, 5, -9, -6, 20, 18, mh_from=0)
b = generate_mh(m, s, 5, -3, 2, 20, 18, mh_from=0)
print("overlap equal", np.array_equal(a[:, 6:, 8:], b[:, :14, :10]))
print("no-MH equals generate", np.array_equal(generate_mh(m, s, 5, 0, 0, 16, 16, mh_from=10 ** 9), generate(m, s, 5, 0, 0, 16, 16)))

n, seeds = 48, (0, 1, 2)
for label, K, mh in (("base K32", 32, None), ("MH last quarter K32", 32, 72), ("MH all K32", 32, 0),
                     ("MH all K16", 16, 0), ("base K64", 64, None)):
    st, wl, t0 = {}, [], time.time()
    for sd in seeds:
        if mh is None:
            g = generate(m, bs(K), sd, 0, 0, n, n)
        else:
            g = generate_mh(m, bs(K), sd, 0, 0, n, n, mh_from=mh, stats=st)
        wl.append((g == T).mean())
        assert m.violations(g)[0] == 0
    tt = (time.time() - t0) / len(seeds)
    acc = ""
    if st:
        acc = (f"  accept finite {st['acc'] / max(st['prop'], 1):.3f} ({st['prop']}), "
               f"hard {st['acc_hard'] / max(st['prop_hard'], 1):.3f} ({st['prop_hard']})")
    print(f"{label:22s} wild {np.mean(wl):.4f} ({', '.join(f'{x:.4f}' for x in wl)})  {tt:5.1f}s/{n}^2{acc}", flush=True)
