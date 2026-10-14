"""Per-move acceptance, pure-dirt -> partial rate and cost of tempered-transition variants at
L3 / L4 on (a) own draws of the start params and (b) P2's final population.
lam-only ladders (tau = 1, lam 20 -> lam_hi) temper only the hard factors; on ground every hard
factor is a certificate factor (support B <= A below, row honour), the count prior and smoothness
stay at full strength.  usage (repo root): diag_tt.py out.json"""
import json, pickle, sys, time
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.problems import get
from castlegen.bench.problem import Region
from castlegen.variants.ckernel_own import CKernelOwn
from castlegen.kernels import explore as ex
p = get("ground"); reg = Region(p.W_train)
V = CKernelOwn(p, 0, "/tmp"); V._build(reg); lay = V.lay_tr
V.ctx_seed = 7_000_000
own = V._own_ctx(32, reg)
pop = {k: jnp.asarray(v, jnp.int32) for k, v in pickle.load(open("reportables/2026-10-14_wave3_explore/runs/P2_pop_r03/population.pkl", "rb")).items()}
starts = {"own_start": own, "P2_population": pop}
cfgs = [("tt_tau2", 2.0, 16, None), ("lamonly_20to5", 1.0, 32, 5.0), ("lamonly_20to2", 1.0, 32, 2.0), ("lamonly_20to2_n128", 1.0, 128, 2.0)]
R = []
rng = np.random.default_rng(0)
for L in (4, 3):
    h, w = lay.hw[L]
    for name, th, n, lh in cfgs:
        f = ex.make_tempered_transition(lay, L, th, n, 20.0, lh)
        for sn, st in starts.items():
            cells = np.stack([rng.integers(h, size=32), rng.integers(w, size=32)], 1)
            f(jax.random.PRNGKey(0), st, jnp.asarray(cells, jnp.int32))[0]["t"].block_until_ready()
            acc_all, p2p, npure, las = [], 0, 0, []
            t0 = time.time()
            for rep in range(8):
                cells = np.stack([rng.integers(h, size=32), rng.integers(w, size=32)], 1)
                new, la, acc, _ = f(jax.random.PRNGKey(rep + 1), st, jnp.asarray(cells, jnp.int32))
                acc = np.asarray(acc); to = np.asarray(st["t"]); tn = np.asarray(new["t"])
                fo = to.reshape(32, 64 >> L, 1 << L, 128 >> L, 1 << L).sum((2, 4))[np.arange(32), cells[:, 0], cells[:, 1]]
                fn = tn.reshape(32, 64 >> L, 1 << L, 128 >> L, 1 << L).sum((2, 4))[np.arange(32), cells[:, 0], cells[:, 1]]
                fm = (1 << L) ** 2
                pure = fo == fm
                npure += int(pure.sum()); p2p += int((pure & acc & (fn < fm)).sum())
                acc_all.append(acc.mean()); las.append(np.median(np.asarray(la)))
            dt = (time.time() - t0) / (8 * 32)
            r = {"L": L, "cfg": name, "tau_hi": th, "n": n, "lam_hi": lh, "start": sn, "acc": float(np.mean(acc_all)),
                 "med_log_a": float(np.median(las)), "pure_props": npure, "pure2part": p2p,
                 "pure2part_rate": p2p / max(npure, 1), "ms_per_move": 1000 * dt}
            R.append(r); print(json.dumps(r), flush=True)
json.dump(R, open(sys.argv[1], "w"), indent=1)
