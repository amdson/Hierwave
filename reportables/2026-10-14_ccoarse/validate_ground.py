"""ccoarse validation on ground: coarse states at levels >= L from
(a) lifted probes, (b) the trained cfast's own draws (clamp's saved params,
unclamped pass), (c) untrained model draws, (d) _coarse_prior at several sweep
counts.  Metrics per source and L (64 worlds)."""
import json, pickle, sys, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from castlegen.bench.problem import Region
from castlegen.bench.problems import get as get_problem
from castlegen.bench.problems.ground import cert_domain
from castlegen.variants import _coarse_prior as cpr

OUT = "reportables/2026-10-14_ccoarse"
p = get_problem("ground"); T = p.T
reg = Region(p.W_train)
NS = 64
seeds = np.arange(3_000_000, 3_000_000 + NS)
des = [p.designed_values(int(s), reg) for s in seeds]
sampled = [c.name for c in p.sampled()]


def model_draws():
    import jax, jax.numpy as jnp
    from castlegen.variants import _cpass_core as core
    from castlegen.variants._creplay_clamp import clamped_pass
    lay = core.Layout(p, reg.W, True, 0)
    ck = pickle.load(open("reportables/2026-10-13_clamp/params_ground.pkl", "rb"))
    out = {}
    for model in ("untrained", "trained"):
        params = jax.tree_util.tree_map(jnp.asarray, ck[model]["params"])
        lam = jnp.asarray(ck[model]["lam"], jnp.float32)
        f = jax.jit(jax.vmap(lambda s, d: clamped_pass(lay, params, s, d, 1.0, lam, {})[0]))
        ds = {k: jnp.asarray(np.stack([d[k] for d in des]), jnp.int32) for k in des[0]}
        st = jax.device_get(f(jnp.asarray(seeds, jnp.int32), ds))
        out[model] = {k: np.asarray(v, np.int64) for k, v in st.items()}
    return out


def probe_states():
    ps = [p.probes(int(s), reg)[i % 4] for i, s in enumerate(seeds)]
    ps0 = [p.probes(int(s), reg)[0] for s in seeds]               # jitter 0: the designed line
    st = lambda L: {k: np.stack([x[k] for x in L]) for k in L[0]}
    return st(ps), st(ps0)


def metrics(x, L, line):
    h = 1 << L
    rho, cert = x[f"rho{L}"], x[f"cert{L}"]
    A, B = cert_domain(h)
    a, b = A[cert], B[cert]
    fr = rho / h ** 2
    fl = line[f"rho{L}"] / h ** 2
    edge = (fl > 0) & (fl < 1)
    near = np.zeros_like(edge)
    for d in (-1, 0, 1):
        near |= np.roll(edge, d, 1) if d else edge
        if d:
            sh = np.zeros_like(edge); 
            if d > 0: sh[:, d:] = edge[:, :-d]
            else: sh[:, :d] = edge[:, -d:]
            near |= sh
    m = {
        "dirt_frac": float(fr.mean()),
        "abs_err_line": float(np.abs(fr - fl).mean()),
        "line_match_L": float(((fr > 0.5) == (fl > 0.5))[near].mean()),
        "A/h": float(a.mean() / h), "B/h": float(b.mean() / h),
        "A==B": float((a == b).mean()),
        "B-A (edge)": float((b - a)[edge].mean()),
        "rho_cert_consistent": float(((rho >= a * h) & (rho <= b * h)).mean()),
    }
    if L < T:
        r2 = x[f"rho{L + 1}"]
        s = rho[:, 0::2, 0::2] + rho[:, 0::2, 1::2] + rho[:, 1::2, 0::2] + rho[:, 1::2, 1::2]
        m[f"|sum rho{L}-rho{L+1}|"] = float(np.abs(s - r2).mean())
    m["hard_viol/world"] = float(cpr.hard_violations(p, x, reg, L).mean())
    return m


if __name__ == "__main__":
    t0 = time.time()
    md = model_draws() if "--nomodel" not in sys.argv else {}
    print("model draws", time.time() - t0, flush=True)
    pr, line = probe_states()
    res, timing, keep = {}, {}, {}
    for L in (4, 3):
        srcs = {"probe lift": pr, **{f"model {k}": v for k, v in md.items()}}
        cp = cpr.CoarsePrior(p, reg, L)
        for sw in (2, 8, 32):
            t = time.time()
            x = cpr.sample_coarse_batch(p, reg, L, seeds, sweeps=sw, prior=cp)
            timing[f"L{L} sweeps{sw}"] = (time.time() - t) / NS
            srcs[f"prior {sw} sw"] = x
            keep[(L, sw)] = x
        for name, x in srcs.items():
            xs = {k: v for k, v in x.items() if k in p.chan and p.chan[k].level >= L}
            res[f"L{L} | {name}"] = metrics(xs, L, line)
            print(L, name, res[f"L{L} | {name}"], flush=True)
    json.dump({"metrics": res, "sec_per_world": timing}, open(f"{OUT}/ground_metrics.json", "w"), indent=1)
    # images: rho_L as a grey map with the line, cert A/B maps; 4 worlds
    for L in (4, 3):
        h = 1 << L
        A, B = cert_domain(h)
        rows = [("probe lift", pr)] + [(f"model {k}", v) for k, v in md.items()] + [("prior 8 sw", keep[(L, 8)])]
        fig, ax = plt.subplots(len(rows), 4, figsize=(14, 2.0 * len(rows)))
        for r, (nm, x) in enumerate(rows):
            for j in range(2):
                w = j
                ax[r, 2 * j].imshow(x[f"rho{L}"][w] / h ** 2, cmap="copper_r", vmin=0, vmax=1)
                ax[r, 2 * j].set_title(f"{nm}: rho{L}/h² (world {w})", fontsize=8)
                c = x[f"cert{L}"][w]
                img = np.stack([A[c] / h, B[c] / h, np.zeros_like(c, float)], -1)
                ax[r, 2 * j + 1].imshow(img)
                ax[r, 2 * j + 1].set_title(f"{nm}: cert{L} (R=A/h, G=B/h)", fontsize=8)
        for a in ax.ravel():
            a.set_xticks([]); a.set_yticks([])
        fig.tight_layout(); fig.savefig(f"{OUT}/images/ground_L{L}.png", dpi=90); plt.close(fig)
    print("done", time.time() - t0)
