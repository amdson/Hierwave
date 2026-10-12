import sys, time, numpy as np
from castlegen import marian
from castlegen.infinite import BlockSchedule, generate
from castlegen.plaquette import generate_mh

ms = marian.load()
m = marian.tile_model(ms)
T = m.T


def bs(K, Sb=3, B=8, **kw):
    a = dict(beta0=1.0, gamma0=2.0)
    a.update(kw)
    return BlockSchedule.anneal(B=B, Sb=Sb, K=K, shift=True, restart=False, **a)


def run(label, sched, shapes=None, mh_from=0, n=48, seeds=(0, 1, 2)):
    st, wl, t0 = {}, [], time.time()
    for sd in seeds:
        if shapes is None:
            g = generate(m, sched, sd, 0, 0, n, n)
        else:
            g = generate_mh(m, sched, sd, 0, 0, n, n, mh_from=mh_from, stats=st, shapes=shapes)
        wl.append((g == T).mean())
        assert m.violations(g)[0] == 0
    tt = (time.time() - t0) / len(seeds)
    acc = ""
    if st:
        acc = f"  acc {st['acc'] / max(st['prop'], 1):.2f}/{st['acc_hard'] / max(st['prop_hard'], 1):.2f}"
    print(f"{label:34s} wild {np.mean(wl):.4f} ({', '.join(f'{x:.4f}' for x in wl)})  {tt:5.1f}s/{n}^2{acc}",
          flush=True)


if __name__ == "__main__":
    exec(open(sys.argv[1]).read())
