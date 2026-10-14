"""Own-draw-only acceptance diagnostic at the start params (sweep2a arm B):
coarse flips at L3/L4 (tracked full->partial) for several K / flip modes,
after one exact coarse Gibbs sweep at L4 then L3; annealed L4/L3 subtree
candidates (validity, ESS, full->partial).  usage: diag_accept.py out.json"""
import json, sys, time
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.problems import get
from castlegen.bench.problem import Region
from castlegen.variants.ckernel_own import CKernelOwn
from castlegen.kernels import explore as ex

out = sys.argv[1]
p = get("ground"); reg = Region(p.W_train)
V = CKernelOwn(p, 0, "/tmp", m_flip=0)
V._build(reg); lay = V.lay_tr
V.ctx_seed = 31_000_000
st = V._own_ctx(32, reg)
R = {"own_raw": V._ground_stats(st)}
v = V._to_flat(st); key = jax.random.PRNGKey(0)
for L, G in V.Gc.items():
    key, s = jax.random.split(key); v, _ = G.sweep(v, s)
stg = V._from_flat(v)
R["after_gibbs"] = V._ground_stats(stg)
print(R, flush=True)
full = {L: lay.chans[L][0].D - 1 for L in (3, 4)}
for name, sst in (("gibbs", stg), ("raw", st)):
    for K, mode, tc in ((8, "q", 2.0), (32, "q", 2.0), (8, "uniform", 1.0), (32, "uniform", 1.0)):
        if name == "raw" and K != 8:
            continue
        t = time.time()
        f = ex.make_flip_chain_tracked(lay, [3, 4], 1.0, tc, mode, 20.0, K)
        key, s = jax.random.split(key)
        _, rec = f(V.params, jax.random.split(s, 32), sst, 8)
        rec = jax.device_get(rec)
        for L in (3, 4):
            m = rec["L"] == L
            r = {"start": name, "K": K, "mode": mode, "tau_c": tc, "L": L, "n": int(m.sum()),
                 "acc": float(rec["acc"][m].mean()), "med_log_r": float(np.median(rec["log_r"][m])),
                 "med_dE": float(np.median(rec["dE"][m])), "med_dlogq": float(np.median(rec["dlogq"][m])),
                 "full_old": float((rec["v_old"][m] == full[L]).mean()),
                 "full2part": int((m & rec["acc"] & (rec["v_old"] == full[L]) & (rec["v_new"] < full[L])).sum()),
                 "prop_part_from_full_medlogr": float(np.median(rec["log_r"][m & (rec["v_old"] == full[L])])) if (m & (rec["v_old"] == full[L])).any() else None,
                 "sec": time.time() - t}
            R.setdefault("flip", []).append(r); print(json.dumps(r), flush=True)
for L, steps in ((4, 64), (4, 128), (3, 64)):
    t = time.time()
    A = ex.make_annealed_candidates(lay, L, 8, 8.0, steps, 20.0, "const")
    h, w = lay.hw[L]
    rng = np.random.default_rng(L)
    cells = np.stack([rng.integers(h, size=32), rng.integers(w, size=32)], 1)
    key, s = jax.random.split(key)
    cand, logw, valid, info = A(V.params, s, stg, jnp.asarray(cells, jnp.int32))
    wn, ess = ex.snis(jax.device_get(logw)); valid = np.asarray(valid)
    rn = lay.chans[L][0].name
    vo = np.asarray(stg[rn])[np.arange(32), cells[:, 0], cells[:, 1]]
    vn = np.asarray(cand[rn]).reshape((32, 8) + cand[rn].shape[1:])[np.arange(32), :, cells[:, 0], cells[:, 1]]
    lw = np.asarray(logw)
    r = {"anneal_L": L, "steps": steps, "valid": float(valid.mean()), "ctx_any_valid": float(valid.any(1).mean()),
         "ess_mean_valid_ctx": float(ess[ess > 0].mean()) if (ess > 0).any() else 0,
         "logw_std_within_ctx": float(np.nanmean([lw[i][np.isfinite(lw[i])].std() for i in range(32) if np.isfinite(lw[i]).sum() > 1])) if (np.isfinite(lw).sum(1) > 1).any() else None,
         "full_old": float((vo == full[L]).mean()), "cand_partial_frac": float((vn < full[L]).mean()),
         "w_full2part": float((((vo[:, None] == full[L]) & (vn < full[L])) * wn).sum() / max((vo == full[L]).sum(), 1)),
         "row_of_cells": cells[:, 0].tolist(), "sec": time.time() - t}
    R.setdefault("anneal", []).append(r); print(json.dumps(r), flush=True)
json.dump(R, open(out, "w"), indent=1)
