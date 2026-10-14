"""Clamp diagnostic on toy_modes, L = 1: rho2 and rho1 clamped to one fixed
2 x 2 quadrant-count context c in every block (all 85 contexts with sum 8),
same noise seeds for every context.  Exact fine-level KL per block:
KL(c) = E_q[S_fine + log q_fine] + log Z(c), Z(c) = sum over fine blocks
lifting to c of exp(-S_fine).  Plus own-draw context frequencies vs p*(c)."""
import itertools, json, pickle, sys
import jax, jax.numpy as jnp, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from castlegen.bench.problem import Region
from castlegen.bench.problems import get as get_problem
from castlegen.bench.problems import toys
from castlegen.variants import _cpass_core as core
from clamp_tools import FIELDS, clamped_pass_stats

D = sys.argv[1]; NS = int(sys.argv[2]) if len(sys.argv) > 2 else 16
p = get_problem("toy_modes"); T = p.T
reg = Region(p.W_train); lay = core.Layout(p, reg.W, True, 0)
ck = pickle.load(open(f"{D}/params_toy_modes.pkl", "rb"))
H, W = p.H, reg.W; nblk = (H // 4) * (W // 4)

# exact per-context partition functions
blk = toys.all_blocks()
E = toys.modes_block_energy(blk)
ok = np.isfinite(E)
r1 = np.stack([blk[:, 0:2, 0:2].sum((1, 2)), blk[:, 0:2, 2:4].sum((1, 2)),
               blk[:, 2:4, 0:2].sum((1, 2)), blk[:, 2:4, 2:4].sum((1, 2))], 1)
ctxs = [c for c in itertools.product(range(5), repeat=4) if sum(c) == 8]
cid = {c: i for i, c in enumerate(ctxs)}
logZ = np.full(len(ctxs), -np.inf); pstar = np.zeros(len(ctxs))
Ztot = np.sum(np.exp(-E[ok]))
for i, c in enumerate(ctxs):
    m = ok & np.all(r1 == np.array(c), 1)
    if m.any():
        logZ[i] = np.log(np.sum(np.exp(-E[m]))); pstar[i] = np.exp(logZ[i]) / Ztot
modes = {"L": (4, 0, 4, 0), "R": (0, 4, 0, 4), "T": (4, 4, 0, 0), "B": (0, 0, 4, 4)}
is_mode = np.array([c in modes.values() for c in ctxs])

seeds = np.arange(3_000_000, 3_000_000 + NS)
des = {k: jnp.asarray(v) for k, v in p.designed_values(0, reg).items()}   # none on toy_modes
f = jax.jit(jax.vmap(lambda pr, lm, s, c: clamped_pass_stats(lay, pr, s, {}, 1.0, lm, c),
                     in_axes=(None, None, 0, 0)))
f_own = jax.jit(jax.vmap(lambda pr, lm, s: clamped_pass_stats(lay, pr, s, {}, 1.0, lm, {}),
                         in_axes=(None, None, 0)))
out = {}
for model in ("untrained", "trained"):
    params = jax.tree_util.tree_map(jnp.asarray, ck[model]["params"]); lam = jnp.asarray(ck[model]["lam"], jnp.float32)
    kl, viol, forced, noopt = (np.zeros(len(ctxs)) for _ in range(4))
    for i, c in enumerate(ctxs):
        q = np.array(c).reshape(2, 2)
        r1w = np.tile(q, (H // 4, W // 4)); r2w = np.full((H // 4, W // 4), 8)
        cl = {"rho1": jnp.asarray(np.broadcast_to(r1w, (NS,) + r1w.shape), jnp.int32),
              "rho2": jnp.asarray(np.broadcast_to(r2w, (NS,) + r2w.shape), jnp.int32)}
        st, acc = f(params, lam, jnp.asarray(seeds, jnp.int32), cl)
        acc = np.asarray(acc)[:, 0]
        n = acc[:, FIELDS.index("n")]
        kl[i] = np.mean((acc[:, FIELDS.index("soft")] + acc[:, FIELDS.index("logq")]) / nblk) + logZ[i]
        viol[i] = np.mean(acc[:, FIELDS.index("nv")] / nblk)
        forced[i] = np.mean(acc[:, FIELDS.index("forced")] / n)
        noopt[i] = np.mean(acc[:, FIELDS.index("noopt")] / n)
        if model == "trained" and c in ((4, 0, 4, 0), (2, 2, 2, 2), (3, 1, 1, 3), (4, 0, 0, 4)):
            out.setdefault("img", {})[str(c)] = np.asarray(st["t"][0])
    # own draws: context frequencies (all 4 * NS worlds) and fine KL per block in own contexts
    st, acc = f_own(params, lam, jnp.asarray(np.arange(4_000_000, 4_000_000 + 4 * NS), jnp.int32))
    r1o = np.asarray(st["rho1"]); qown = np.zeros(len(ctxs)); nbad = 0
    for b in range(r1o.shape[0]):
        for by in range(H // 4):
            for bx in range(W // 4):
                c = tuple(r1o[b, 2 * by:2 * by + 2, 2 * bx:2 * bx + 2].ravel())
                if c in cid: qown[cid[c]] += 1
                else: nbad += 1
    qown /= max(qown.sum() + nbad, 1)
    w_mode = pstar * is_mode; w_non = pstar * ~is_mode
    out[model] = {"kl": kl.tolist(), "viol": viol.tolist(), "forced": forced.tolist(), "noopt": noopt.tolist(),
                  "qown": qown.tolist(), "own_bad_ctx_frac": nbad / (r1o.shape[0] * nblk),
                  "kl_mode_mean_pstar": float(np.sum(w_mode * kl) / w_mode.sum()),
                  "kl_nonmode_mean_pstar": float(np.sum(w_non * kl) / w_non.sum()),
                  "kl_fine_under_pstar": float(np.sum(pstar * kl)),
                  "kl_fine_under_qown": float(np.sum(qown * kl) / max(qown.sum(), 1e-12)),
                  "viol_mode": float(np.sum(w_mode * viol) / w_mode.sum()),
                  "viol_nonmode": float(np.sum(w_non * viol) / w_non.sum()),
                  "forced_mode": float(np.sum(w_mode * forced) / w_mode.sum()),
                  "forced_nonmode": float(np.sum(w_non * forced) / w_non.sum()),
                  "pstar_mode_mass": float(w_mode.sum()), "qown_mode_mass": float(np.sum(qown * is_mode)),
                  "lam": ck[model]["lam"]}
    print(model, {k: v for k, v in out[model].items() if not isinstance(v, list)}, flush=True)
json.dump({"ctxs": ctxs, "pstar": pstar.tolist(), "logZ": logZ.tolist(), "is_mode": is_mode.tolist(),
           "updates": ck["updates"], **{m: out[m] for m in ("untrained", "trained")}},
          open(f"{D}/modes_summary.json", "w"), indent=1)
# figure: fine KL per context vs p*(c)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for j, model in enumerate(("untrained", "trained")):
    kl = np.array(out[model]["kl"])
    a = ax[j]
    a.scatter(pstar[~is_mode], kl[~is_mode], s=12, c="#888", label="non-mode context")
    a.scatter(pstar[is_mode], kl[is_mode], s=40, c="#c33", label="mode context (L/R/T/B)")
    a.set_xscale("log"); a.set_xlabel("p*(context)"); a.set_ylabel("fine-level KL per block (nats)")
    a.set_title(f"toy_modes, {model}: fine KL given clamped rho1"); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{D}/images/modes_fine_kl_by_context.png", dpi=110)
fig, ax = plt.subplots(1, 4, figsize=(12, 2.4))
for a, (c, t) in zip(ax, out["img"].items()):
    a.imshow(p.render({"t": t}), interpolation="nearest"); a.set_title(f"trained, rho1 = {c}", fontsize=8); a.axis("off")
plt.tight_layout(); plt.savefig(f"{D}/images/modes_clamped_examples.png", dpi=110)
