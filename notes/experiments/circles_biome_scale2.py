"""The biome circles test, stage 7b: the stage 7 probe with the admitted list
(Sampler(adm=True): kernel.adm_lists at init, kernel.sweep_adm) beside the
old path (sweep_cap), same configs (circles_biome_scale.py: N random
families, 8 pair masks, nty = ntx = NT, "+bu+ou" thetas).

Per N: the cost probe at active fraction 1 (every biome non-none, obj from
empty, S_M sweeps, PROBE interleaved reps, medians), full rows and hard rows
only, K None / 8, old and adm paths (8 samplers), and a cache check (full,
K = 8, every table a strided view 10 x 10 apart); init time per path; the
admitted lists (count, total length, bytes); the forward (Forward with
use_sampler, the obj sampler switched to adm) us per active site, RUNS runs.

    NUMBA_NUM_THREADS=1 PYTHONPATH=. python notes/experiments/circles_biome_scale2.py

Env: NS (2,5,10,20), NT (6), RUNS (16), S_M (30), PROBE (40), SEED.
Output images/cbio_stage7b.json.
"""
import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from circles_biome_scale import thetas
from castlegen.channels.core import Model
from castlegen.channels.sampler import Sampler
from castlegen.channels.circles_biome import CirclesBiome, Forward, random_families, random_pair_masks

E = os.environ.get
NS = [int(v) for v in E("NS", "2,5,10,20").split(",")]
NT, RUNS, S_M, PROBE, SEED = int(E("NT", 6)), int(E("RUNS", 16)), int(E("S_M", 30)), int(E("PROBE", 40)), \
    int(E("SEED", 0))


class FwAdm(Forward):
    """Forward with the obj sampler on the admitted-list path (adm)."""
    adm = True

    def build(self):
        super().build()
        self.S["obj"].adm = self.adm


def forward(C, th, K, adm):
    F = FwAdm(C, th, seed=SEED + 555, use_sampler=True, K=K)
    F.adm = adm
    F.run(1, 1, 1)
    obj, act, init, conf = [], [], [], 0.0
    for _ in range(RUNS):
        F.run(30, S_M, 20)
        obj.append(F.times["obj"]); init.append(F.times["obj_init"]); act.append(1 - F.dormant_frac)
        conf = max(conf, float((C.dem_of(F.obj) == 3).mean()))
    n = C.nmy * C.nmx
    return dict(us=float(np.sum(obj) / (S_M * n * np.sum(act)) * 1e6), init_ms=float(np.median(init) * 1e3),
                conflict=conf, present=float((F.obj.grid > 0).mean()))


def spread(t, s=10):
    """t as a view with s x s times the footprint (same values, layout 'A')."""
    big = np.zeros((t.shape[0] * s, t.shape[1] * s + 1))
    v = big[::s, :t.shape[1] * s:s]
    v[:] = t
    return v


def probe(C, th):
    rng = np.random.default_rng(SEED + 9)
    F = Forward(C, th, seed=SEED + 3, use_sampler=True)
    F.build()
    hard = Model(C.H, C.W, list(F.chans), [f for f in C.designed_factors() if f.name != "pres"] + C.support_factors())
    S = {f"{n} K={K} {'adm' if a else 'old'}": Sampler(m, "obj", K=K, hb=C.BT, adm=a)
         for n, m in (("full", F.model), ("hard", hard)) for K in (None, 8) for a in (False, True)}
    for a in (False, True):                       # cache check: every table a strided view, 10 x 10 apart
        Sx = S[f"spread K=8 {'adm' if a else 'old'}"] = Sampler(F.model, "obj", K=8, hb=C.BT, adm=a)
        Sx.P.tabs = tuple(spread(t) for t in Sx.P.tabs)
    ts, ti = {k: [] for k in S}, {k: [] for k in S}
    for r in range(PROBE + 1):
        b = rng.integers(1, C.P, (C.nty, C.ntx))
        sd = int(rng.integers(1 << 30))
        for k, Sk in S.items():
            F.biome.grid[:] = b
            F.obj.grid[:] = 0
            F.obj.fixed[:] = False
            C.paint_allow(F.biome, F.allow)
            t = time.perf_counter()
            Sk.init()
            t1 = time.perf_counter()
            Sk.sweep(S_M, seed=sd)
            t2 = time.perf_counter()
            if r:
                ti[k].append(t1 - t)
                ts[k].append(t2 - t1)
            assert not (C.dem_of(F.obj) == 3).any()
    n = C.nmy * C.nmx * S_M
    out = {k: dict(us=float(np.median(ts[k])) / n * 1e6, us_min=float(np.min(ts[k])) / n * 1e6,
                   init_ms=float(np.median(ti[k])) * 1e3) for k in S}
    Sa = S["full K=8 adm"]
    lens = np.diff(Sa.adm_ptr)
    out["lists"] = dict(n=int(len(lens)), lens=lens.tolist(), sites=int(Sa.adm_id.size), sib=int(len(Sa.sib_rows)),
                        parent=int(len(Sa.parent_rows())), nhard=int(Sa.P.nhard), rows=int(Sa.P.fac.shape[0]),
                        bytes=int(Sa.adm_id.nbytes + Sa.adm_ptr.nbytes + Sa.adm_idx.nbytes + Sa.adm_e.nbytes),
                        dense_bytes=int(Sa.adm_id.size * C.D))
    return out


if __name__ == "__main__":
    T0 = time.time()
    res = dict(config=dict(NS=NS, NT=NT, RUNS=RUNS, S_M=S_M, PROBE=PROBE, SEED=SEED), N={})
    for N in NS:
        rng = np.random.default_rng(SEED + 100 + N)                 # the stage 7 configs
        C = CirclesBiome(NT, NT, families=random_families(N, rng), masks=random_pair_masks(N, rng))
        th = thetas(C)
        r = dict(N=N, D=C.D, probe=probe(C, th))
        for K in (None, 8):
            for a in (False, True):
                r[f"fw K={K} {'adm' if a else 'old'}"] = forward(C, th, K, a)
        res["N"][str(N)] = r
        p = r["probe"]
        print(f"N {N:2d} D {C.D:3d} lists {p['lists']} | probe "
              + " ".join(f"{k}: {v['us']:.2f} ({v['init_ms']:.2f} ms)" for k, v in p.items() if k != "lists")
              + " | fw " + " ".join(f"{k[3:]}: {r[k]['us']:.2f} ({r[k]['init_ms']:.2f} ms)" for k in r
                                    if k.startswith("fw")), flush=True)
    res["total_s"] = time.time() - T0
    json.dump(res, open("images/cbio_stage7b.json", "w"), indent=1)
    print(f"total {res['total_s']:.0f} s")
