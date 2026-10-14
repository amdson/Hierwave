"""Calibration of the bench v2 block check.  usage: calib.py exact|<variant> <problem> <train_s> [tag]
Saves <tag>.json (block_check result) and <tag>.npz (pooled arrays per level) next to this file."""
import sys, json, time, numpy as np
from pathlib import Path
from castlegen.bench import run as R, shares
from castlegen.bench.problem import Region
from castlegen.bench.problems import get as get_problem
from castlegen.bench.problems import toys
from castlegen.bench.variant import load_variants

HERE = Path(__file__).resolve().parent
mode, pname, train_s = sys.argv[1], sys.argv[2], float(sys.argv[3])
tag = sys.argv[4] if len(sys.argv) > 4 else f"{mode}_{pname}"
pr = get_problem(pname)
region = Region(pr.W_eval)


class ExactToy:
    """Exact samples of toy_modes; score_sites = exact conditional log p of each draw in the
    sampler's order (rho2, then rho1 raster, then t raster in each 4 x 4 block)."""
    def __init__(self, problem):
        self.p = problem
        blocks = toys.all_blocks()
        E = toys.modes_block_energy(blocks)
        self.prob = toys._probs(E)
        t = blocks.reshape(-1, 16)
        r1 = np.stack([t[:, 0] + t[:, 1] + t[:, 4] + t[:, 5], t[:, 2] + t[:, 3] + t[:, 6] + t[:, 7],
                       t[:, 8] + t[:, 9] + t[:, 12] + t[:, 13], t[:, 10] + t[:, 11] + t[:, 14] + t[:, 15]], 1)
        r2 = r1.sum(1)
        vars_ = [r2] + [r1[:, i] for i in range(4)] + [t[:, i] for i in range(16)]
        bases = [17] + [5] * 4 + [2] * 16
        code = np.zeros(len(t), np.int64)
        prev = np.zeros(len(t))                      # log marginal of the prefix, per state
        self.cond = []
        for v_, b_ in zip(vars_, bases):
            code = code * b_ + v_
            _, inv = np.unique(code, return_inverse=True)
            m = np.bincount(inv, weights=self.prob)[inv]
            with np.errstate(divide="ignore"):
                lm = np.log(m)
            self.cond.append(lm - prev)
            prev = lm
        self.cond = np.stack(self.cond, 1)           # (65536, 21)
        self.rng = np.random.default_rng(123)

    def generate(self, seed, region):
        rng = np.random.default_rng(seed)
        t = toys.sample_t(rng, self.prob, self.p.H, region.W)
        return {"raw": toys.lift(t), "final": toys.lift(t)}

    def score_sites(self, x, region):
        t = np.asarray(x["t"])
        H, W = t.shape
        bl = toys.blocks_of(t).reshape(-1, 16)
        idx = (bl * (1 << np.arange(16))).sum(1)
        c = self.cond[idx]                            # (nb, 21)
        nby, nbx = H // 4, W // 4
        c = c.reshape(nby, nbx, 21)
        s2 = c[:, :, 0]
        s1 = np.zeros((H // 2, W // 2))
        s0 = np.zeros((H, W))
        for i in range(4):
            s1[(i >> 1)::2, (i & 1)::2] = c[:, :, 1 + i]
        for j in range(16):
            s0[(j >> 2)::4, (j & 3)::4] = c[:, :, 5 + j]
        return {"rho2": s2, "rho1": s1, "t": s0}

    def score(self, x, region):
        return float(sum(a.sum() for a in self.score_sites(x, region).values()))


if mode == "exact":
    v = ExactToy(pr)
    info = {}
else:
    V = load_variants()
    v = V[mode](pr, seed=0, workdir="/tmp/probe2")
    t0 = time.time()
    v.train(train_s, lambda d: None)
    info = {"train_s": time.time() - t0, "updates": getattr(v, "updates", None)}
    print(info, flush=True)
pool = []
t0 = time.time()
res = R.block_check(pr, v, 0, region, pool_out=pool)
print("block check s", time.time() - t0, flush=True)
if res is None:
    print("n/a"); sys.exit(0)
res["info"] = info
# free energy / entropy for reference
S, LQ = [], []
from castlegen.bench import energy as en
for i in range(16):
    seed = R.EVAL_SEED0 + 77 + i
    x = R._with_designed(pr, v.generate(seed, region)["raw"], seed, region)
    S.append(en.energy(pr, x, region).soft); q = v.score(x, region); LQ.append(q)
res["free_energy"] = float(np.mean(np.array(S) + np.array(LQ)) / (pr.H * region.W))
json.dump(res, open(HERE / f"{tag}.json", "w"), indent=1)
P = pool[0]
np.savez_compressed(HERE / f"{tag}.npz", **{f"{n}{l}": a for l in range(pr.T + 1)
                    for n, a in zip(("lq", "dw", "sw", "kind", "wid", "bid"), P.arrays(l))})
for l, d in res["levels"].items():
    for w in ("direct", "subtree"):
        r = d[w]
        print(l, w, {k: (None if r[k] is None else round(r[k], 3)) for k in ("rho_pool", "rho_probe", "rho_own", "starved_probe", "starved_own")}, flush=True)
