"""Weighted-ML (here: uniform weights) fit of the fine levels 0-2 to exact
block-Gibbs refined ground states, by teacher forcing (castlegen/kernels/logq.py,
kernel-c's make_ml_update; levels 3-4 frozen).  Saves params_ground.pkl in the
cstart2 format so reportables/2026-10-14_cstart2/eval_ground.py runs on it.
usage: train_ml.py <data.npz>[,<data2.npz>...] <budget_s> <outdir> <init: untrained|trained> [B] [lr]"""
import json, os, pickle, sys, time
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.kernels.exact_block import Flat
from castlegen.kernels.logq import make_ml_update, ml_opt_init, logq_batch
from castlegen.variants import _cpass_core as core
from castlegen.variants._cln_core import window_of

files, budget, out, init = sys.argv[1].split(","), float(sys.argv[2]), sys.argv[3], sys.argv[4]
B = int(sys.argv[5]) if len(sys.argv) > 5 else 8
lr = float(sys.argv[6]) if len(sys.argv) > 6 else 1e-3
os.makedirs(out + "/images", exist_ok=True)
p = get("ground"); reg = Region(p.W_train); fl = Flat(p, reg)
S, src = [], []
for f in files:
    d = np.load(f)
    st = d["states"]                                   # (N, NS, n)
    S.append(st.reshape(-1, st.shape[-1])); src.append(np.repeat(d["src"], st.shape[1]))
S = np.concatenate(S); src = np.concatenate(src)
rng = np.random.default_rng(0)
perm = rng.permutation(len(S)); nho = max(len(S) // 10, min(16, len(S) // 2))
ho, tr = perm[:nho], perm[nho:]
print("data", S.shape, "train", tr.size, "held-out", ho.size, "own share", float((src == -1).mean()), flush=True)
chans = [c.name for c in p.chans]


def batch(idx):
    xs = [fl.unflatten(S[i]) for i in idx]
    return {k: jnp.asarray(np.stack([x[k] for x in xs]), jnp.int32) for k in chans}


ck = pickle.load(open("reportables/2026-10-14_cstart2/runs/cstart_f1L3_ln6/params_ground.pkl", "rb"))
lay = core.Layout(p, reg.W, True, 0, win=window_of((-2, 3)))
params = jax.tree_util.tree_map(jnp.asarray, ck[init]["params"])
p0 = jax.device_get(params)
opt = ml_opt_init(params)
step = make_ml_update(lay, tau=1.0, clip=10.0)
on = jnp.asarray([1, 1, 1, 0, 0], jnp.float32)
lqf = jax.jit(lambda pr, st: logq_batch(lay, pr, st)["per_level"])


def heldout(params):
    r = []
    for i in range(0, ho.size, B):
        idx = ho[i:i + B]
        if len(idx) < B:
            break
        r.append(np.asarray(lqf(params, batch(idx))))
    r = np.concatenate(r) / (p.H * reg.W)
    return r.mean(0).tolist()


# warm compile
_ = step(params, opt, batch(tr[:B]), jnp.ones(B), jnp.float32(lr), on)
h0 = heldout(params)
print("held-out log q per fine cell by level (init)", [round(x, 4) for x in h0], flush=True)
logf = open(f"{out}/train.jsonl", "w")
t0 = time.time(); u = 0; ninf = 0
while time.time() - t0 < budget:
    idx = rng.choice(tr, B, replace=False)
    params, opt, info = step(params, opt, batch(idx), jnp.ones(B), jnp.float32(lr), on)
    u += 1
    if u % 50 == 0:
        ninf += int(info["n_infeasible"])
        rec = {"t": time.time() - t0, "update": u, "loss_per_cell": float(info["loss"]) / (p.H * reg.W),
               "n_feasible": int(info["n_feasible"]), "n_infeasible": int(info["n_infeasible"])}
        if u % 500 == 0:
            rec["heldout"] = heldout(params)
            print(json.dumps(rec), flush=True)
        logf.write(json.dumps(rec) + "\n"); logf.flush()
hf = heldout(params)
print("updates", u, "held-out log q per fine cell by level (final)", [round(x, 4) for x in hf], flush=True)
lam = ck["trained"]["lam"]
pickle.dump({"untrained": {"params": p0, "lam": lam}, "trained": {"params": jax.device_get(params), "lam": lam},
             "updates": u, "cfg": {**ck["cfg"], "ml_init": init, "ml_B": B, "ml_lr": lr, "data": files},
             "train_s": time.time() - t0, "heldout0": h0, "heldout": hf, "devices": str(jax.devices())},
            open(f"{out}/params_ground.pkl", "wb"))
print("done", flush=True)
