"""Step 3 driver: recursion arms K = 1, 4, 16 and the control (cfitpen_ml2 continued), all from
arm_cfitpen_ml2/params.pkl, BUDGET s each, back to back.  usage: sweep_recur.py <outdir> <budget_s> [arm,...] [smoke]
Per arm: <outdir>/<arm>/train.jsonl (every iteration: own8 line_match/dirt/offset per level, raw viol per level,
lambda, recursion diagnostics), snapshots params_<min>.pkl every SNAP_S, params.pkl at the end, eval.json
(snapshots: reduced block check + raw stats; final: eval_seed with the stage (R 30, F 150) = bench eval incl. full
block check, plus per-level offsets of 16 raw/staged worlds).  Prints DONE."""
import json, os, pickle, sys, time
import jax, jax.numpy as jnp, numpy as np
from castlegen.bench.variant import load_variants
from castlegen.bench.problems import get
from castlegen.bench.problems import ground as G
from castlegen.bench import run as R
from castlegen.bench.problem import Region

out, budget = sys.argv[1], float(sys.argv[2])
ARMS = {"K16": dict(rec_K=16), "control": dict(rec_n=0), "K16_probe": dict(rec_K=16, rec_probe_frac=0.5), "K4": dict(rec_K=4), "K1": dict(rec_K=1)}
sel = sys.argv[3].split(",") if len(sys.argv) > 3 else list(ARMS)
smoke = len(sys.argv) > 4
SNAP_S = 600 if not smoke else 40
INIT = "reportables/2026-10-16_fitpen/runs/arm_cfitpen_ml2/params.pkl"
p = get("ground"); V = load_variants(); reg = Region(p.W_eval); T, H, W = p.T, p.H, p.W_eval
KEEP = ("iter", "update", "t", "t_train", "loss_per_cell", "own8", "lv_own", "own_raw", "lam", "rkl_raw_viol",
        "rkl_updates", "time_refine", "time_ml", "time_rkl", "setup_done")


def nvs(f):
    open(f, "w").write(os.popen("nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active,"
                                "memory.used --format=csv").read())


def offsets(xs, gs):
    """reportables/2026-10-16_offset/scripts/offset.py measure(): implied surface offset per level, |c1|, first-dirt."""
    res = {}
    for l in range(T + 1):
        h = 1 << l; ds = []
        for x, g in zip(xs, gs):
            r = np.asarray(x["t" if l == 0 else f"rho{l}"], np.float64)
            ds.append(H - r.sum(0) / h - g.reshape(-1, h).mean(1))
        ds = np.stack(ds); fz = np.fft.rfft(ds, axis=-1) / ds.shape[-1]
        res[f"L{l}"] = {"off": float(ds.mean()), "c1": float(np.abs(fz[:, 1]).mean())}
    hs, dd = [], []
    for x, g in zip(xs, gs):
        t = np.asarray(x["t"]).astype(bool)
        hh = np.where(t.any(0), t.argmax(0), H).astype(float); hs.append(hh); dd.append(hh - np.floor(g + .5))
    hs = np.stack(hs); dd = np.stack(dd); fz = np.fft.rfft(dd, axis=-1) / W
    res["first_dirt"] = {"off": float(dd.mean()), "rough": float(np.abs(np.diff(hs, axis=-1, append=hs[:, :1])).mean()),
                         "c1": float(np.abs(fz[:, 1]).mean())}
    return res


def blocks_short(b):
    if not b:
        return None
    lv = b["levels"]
    return {"rho": [lv[str(l)]["direct"]["rho_pool"] for l in range(T + 1)],
            "starved": [lv[str(l)]["direct"]["starved_probe"] for l in range(T + 1)],
            "rho_sub": [lv[str(l)]["subtree"]["rho_pool"] for l in range(T + 1)],
            "starved_sub": [lv[str(l)]["subtree"]["starved_probe"] for l in range(T + 1)],
            "n_own": b["n_own"], "n_probe": b["n_probe"]}


def evaluate(params, lam, tag, full, adir):
    t = time.time()
    m = {"tag": tag}
    if not full:
        R.N_POOL_OWN, R.N_POOL_PROBE_SEEDS = (24, 6) if not smoke else (4, 1)
        v = V["crecur"](p, 0, adir + "/ev", rec_n=0)
        v.params, v.lam = jax.tree_util.tree_map(jnp.asarray, params), np.asarray(lam, float)
        m["blocks"] = blocks_short(R.block_check(p, v, 0, reg))
    else:
        R.N_POOL_OWN, R.N_POOL_PROBE_SEEDS = (128, 32) if not smoke else (4, 1)
        S = V["crecur_stage"](p, 0, adir + "/ev", rec_n=0)
        S.params, S.lam = jax.tree_util.tree_map(jnp.asarray, params), np.asarray(lam, float)
        r = R.eval_seed(p, S, 0, 16 if not smoke else 2, None)
        m.update({"viol": r["viol"], "stats": r["stats"], "free_energy": r["free_energy"], "probes": r["probes"],
                  "blocks": blocks_short(r["blocks"]), "gen_s_per_world": r["gen_s_per_world"]})
        seeds = [R.EVAL_SEED0 + i for i in range(16 if not smoke else 2)]
        gs = [G._line_from_fill(np.asarray(S.designed(s, reg)["fill"], np.int64), W) for s in seeds]
        gens = [S.generate(s, reg) for s in seeds]
        m["offset_raw"] = offsets([g["raw"] for g in gens], gs)
        m["offset_staged"] = offsets([g["final"] for g in gens], gs)
        m["stage_changed_frac"] = float(np.mean([(g["raw"]["t"] != g["final"]["t"]).mean() for g in gens]))
    m["eval_s"] = time.time() - t
    return m


for arm in sel:
    adir = f"{out}/{arm}"; os.makedirs(adir, exist_ok=True)
    if os.path.exists(f"{adir}/eval.json"):
        print("skip", arm, flush=True); continue
    nvs(f"{adir}/nvidia_before.txt")
    v = V["crecur"](p, 0, adir, init_params=INIT, **ARMS[arm])
    lf = open(f"{adir}/train.jsonl", "w")
    st = dict(t0=time.time(), last_snap=0.0, snaps=[])

    def log(rec):
        row = {k: rec[k] for k in KEEP if k in rec}
        row["wall"] = time.time() - st["t0"]
        row["rec"] = v.rec_last
        lf.write(json.dumps(row, default=float) + "\n"); lf.flush()
        now = time.time() - st["t0"]
        if "iter" in rec and now - st["last_snap"] > SNAP_S:
            st["last_snap"] = now
            f = f"{adir}/params_{int(now / 60):03d}min.pkl"
            pickle.dump({"params": jax.device_get(v.params), "lam": np.asarray(v.lam), "updates": v.updates,
                         "rec_updates": v.rec_updates}, open(f, "wb"))
            st["snaps"].append(f)

    v.train(budget, log)
    lf.close()
    info = {"timing": v.timing, "updates": v.updates, "rkl_updates": v.rkl_updates, "rec_updates": v.rec_updates,
            "cfg": {k: v.cfg[k] for k in ("rec_l0", "rec_K", "rec_n", "rec_B", "rec_lr", "rec_mode", "rkl_levels", "rkl_lam0", "upi")}}
    pickle.dump({"params": jax.device_get(v.params), "lam": np.asarray(v.lam), "updates": v.updates,
                 "rec_updates": v.rec_updates}, open(f"{adir}/params.pkl", "wb"))
    print("trained", arm, json.dumps(info, default=float), flush=True)
    nvs(f"{adir}/nvidia_after_train.txt")
    ev = {"info": info, "snapshots": []}
    for f in st["snaps"][1:]:                       # the first snapshot is taken ~at setup end
        ck = pickle.load(open(f, "rb"))
        e = evaluate(ck["params"], ck["lam"], os.path.basename(f), False, adir)
        e["updates"], e["rec_updates"] = ck["updates"], ck["rec_updates"]
        ev["snapshots"].append(e)
        print("snap", arm, e["tag"], json.dumps(e["blocks"]), f"{e['eval_s']:.0f}s", flush=True)
    ev["final"] = evaluate(jax.device_get(v.params), v.lam, "final", True, adir)
    print("final", arm, json.dumps({k: ev["final"][k] for k in ("viol", "blocks", "offset_staged")}, default=float), flush=True)
    json.dump(ev, open(f"{adir}/eval.json", "w"), indent=1, default=float)
    del v
print("DONE", flush=True)
