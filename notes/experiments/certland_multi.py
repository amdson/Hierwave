"""Does more varied training data fix the held-out h16 / h8 deaths of the
learned twist?  All arms use the widek layout (known flags, the pass's own
contexts recorded) and the certland_twist schedule: World(64, 128), levels
(0, 1, 2, 3), K = {0: 8, 1: 4, 2: 1, 3: 1}, 60 updates, 50 Adam steps per
(level, channel) per update.

Compute matching: every arm runs 60 updates, each moving exactly ONE
64 x 128 world (all four root levels + a tile sweep, recorded) followed by
the same fit, so total recorded moves, examples per update, Adam steps and
memory per update are identical.  Arms differ only in which world an update
moves:
  base     one world (noise 7), every update
  multi    four worlds (noise 7, 8, 9, 10), round-robin (update u moves
           world u % 4: each chain gets 15 updates)
  restart  like multi, and every R updates (u % R == 0, u > 0) the chain
           about to be moved is replaced by a fresh forward generation
           (generate_pred with the current predictors) from a new noise
           seed (11, 12, ...), with its burn-in moves not recorded
The only extra compute of multi / restart is unrecorded init (generation +
burn-in) of the extra chains; it is timed and reported separately.

Evaluation (excluded from the budget): as certland_twist.py `heldout`:
fresh chains from theta = 0 generations on noise seeds not used in
training (21, 22, 23; rng default_rng(100 + noise), burn 0), 24 rounds of
moves with the trained predictors (not recorded): dead / ESS / changed per
root level; theta = 0 reference on the same chains (`zero`).  Held-out
generate_pred on 1001..1004 (fill_err, overhangs, stuck, pictures).  During
training, every 10 updates, a short held-out probe (noise 24, 4 rounds).

  CERTLAND_PRED_FEATS=widek NUMBA_NUM_THREADS=1 python notes/experiments/certland_multi.py run --arm base
  (likewise --arm multi, --arm restart), then `zero`, then `compare`.
Logs images/certland_multi_<arm>_log.txt; figures images/certland_multi_*.png."""
import argparse
import json
import os
import resource
import sys
import time

sys.path.insert(0, "/Users/amdson/dev/Hierwave/.claude/worktrees/certland")
os.environ.setdefault("CERTLAND_PRED_FEATS", "widek")

import numpy as np                                                       # noqa: E402

from castlegen.channels import certland as cl                           # noqa: E402
from castlegen.channels import certland_pred as cp                      # noqa: E402
from castlegen.channels import certland_chain as cc                     # noqa: E402
from castlegen.channels import certland_train as ct                     # noqa: E402

assert cp.LAYOUT == "widek", cp.LAYOUT
IMG = "/Users/amdson/dev/Hierwave/images"
TMP = os.environ.get("CERTLAND_TMP", "/Users/amdson/.claude/jobs/885763c6/tmp")
os.makedirs(TMP, exist_ok=True)
K = {0: 8, 1: 4, 2: 1, 3: 1}
LEVELS = (0, 1, 2, 3)
EVAL_SEEDS = (1001, 1002, 1003, 1004)
HELD_NOISES = (21, 22, 23)
PROBE_NOISE = 24
ARMS = ("base", "multi", "restart")
LOGF = None


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    if LOGF is not None:
        LOGF.write(s + "\n")
        LOGF.flush()


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20     # bytes on macOS


def heldout(preds, noise, rounds, H=64, W=128):
    """certland_twist.cmd_heldout for one noise seed: per root level the
    list of per-round move stats."""
    rng = np.random.default_rng(100 + noise)
    w = cl.World(H, W)
    cfg = ct.Cfg(levels=LEVELS, K=K, burn=0)
    ct.init_chain(w, cp.Preds(), noise, 0, rng, cfg)
    mover = (getattr(preds, "q", None) or preds) if preds is not None else None
    per = {i: [] for i in LEVELS}
    for _ in range(rounds):
        for i in LEVELS:
            per[i].append(cc.move_level(w, mover, i, K[i], rng))
        cc.tile_sweep(w, rng)
    return per


def cmd_ownstart(a):
    """Diagnostic: the held-out chains of `heldout` but started from a
    generation with the arm's TRAINED predictors (generate_pred, then burn
    0) instead of theta = 0: is the held-out / in-chain gap the start
    state rather than the noise field?"""
    global LOGF
    LOGF = open(os.path.join(IMG, "certland_multi_ownstart_log.txt"), "a")
    for arm in a.arms:
        preds, upd = ct.load_ckpt(os.path.join(TMP, f"certland_multi_{arm}_preds.npz"))
        res = {}
        for noise in HELD_NOISES:
            rng = np.random.default_rng(100 + noise)
            w = cl.World(64, 128)
            ct.init_chain(w, preds, noise, 0, rng, ct.Cfg(levels=LEVELS, K=K, burn=0))
            per = {i: [] for i in LEVELS}
            for _ in range(a.rounds):
                for i in LEVELS:
                    per[i].append(cc.move_level(w, preds, i, K[i], rng))
                cc.tile_sweep(w, rng)
            res[noise] = summarize(per)
        log(f"{arm:8s} trained-pred start, {a.rounds} rounds: " + " | ".join(
            f"h{cl.HS[i]} dead {np.mean([res[n][i]['dead'] for n in HELD_NOISES]):.3f} "
            f"ess {np.mean([res[n][i]['ess'] for n in HELD_NOISES]):.2f} "
            f"chg {np.mean([res[n][i]['changed'] for n in HELD_NOISES]):.3f}" for i in LEVELS[:3])
            + "  per noise h16: " + " ".join(f"{res[n][0]['dead']:.3f}" for n in HELD_NOISES))


def summarize(per):
    return {i: {k: float(np.mean([r[k] for r in v])) for k in ("dead", "ess", "changed")} for i, v in per.items()}


def cmd_run(a):
    global LOGF
    arm = a.arm
    LOGF = open(os.path.join(IMG, f"certland_multi_{arm}_log.txt"), "w")
    seeds = {"base": [7], "multi": [7, 8, 9, 10], "restart": [7, 8, 9, 10]}[arm]
    log(f"certland_multi run: arm {arm}, layout {cp.LAYOUT} NF {cp.NF}; world {a.H}x{a.W}; train noise {seeds}; "
        f"levels {LEVELS} K {K}; n {a.n} steps {a.steps} burn {a.burn} seed {a.seed}"
        + (f"; restart every {a.R} updates, new noise from 11" if arm == "restart" else ""))
    preds = cp.Preds(nh=64)
    cfg = ct.Cfg(levels=LEVELS, K=dict(K), steps=a.steps, burn=a.burn, eval_every=0, tag=f"multi_{arm}", seed=a.seed)
    rng_init = np.random.default_rng(a.seed + 7)
    t_init = 0.0
    worlds = []
    for s in seeds:
        w = cl.World(a.H, a.W)
        info = ct.init_chain(w, preds, s, cfg.burn, rng_init, cfg)
        t_init += info["time"]
        log(f"init noise {s}: stuck {info['bad']} viol {info['viol']} ({info['time']:.1f}s)")
        worlds.append(w)
    archive = ct.Archive(cfg.archive_cap, cfg.archive_floats)
    rng = np.random.default_rng(cfg.seed)          # as ct.train
    rows, probes, restarts = [], [], []
    next_noise = 11
    t_upd = 0.0
    for u in range(a.n):
        k = u % len(worlds)
        if arm == "restart" and u > 0 and u % a.R == 0:
            w = cl.World(a.H, a.W)
            try:
                info = ct.init_chain(w, preds, next_noise, cfg.burn, rng_init, cfg)
            except RuntimeError as e:        # no valid generation with the current preds: theta = 0 generation
                log(f"  restart noise {next_noise}: {e}; falling back to a theta = 0 generation")
                info = ct.init_chain(w, cp.Preds(), next_noise, 0, rng_init, cfg)
                for _ in range(cfg.burn):
                    for i in LEVELS:
                        cc.move_level(w, preds, i, K[i], rng_init)
                    cc.tile_sweep(w, rng_init)
            t_init += info["time"]
            restarts.append(dict(upd=u, slot=k, noise=next_noise, stuck=info["bad"], time=info["time"]))
            log(f"  restart at u{u}: chain {k} <- noise {next_noise}, gen stuck {info['bad']} viol {info['viol']} "
                f"({info['time']:.1f}s, not recorded)")
            worlds[k] = w
            next_noise += 1
        t0 = time.time()
        lg = ct.update(worlds[k], preds, archive, cfg, u, rng)
        dt = time.time() - t0
        t_upd += dt
        log(f"[w{k}] " + ct.format_log(lg) + f" | {dt:.1f}s rss {rss_mb():.0f}MB")
        r = dict(upd=u, world=k, time=dt)
        for i, d in lg["levels"].items():
            r[f"dead{i}"], r[f"ess{i}"], r[f"chg{i}"] = d["dead"], d["ess"], d["changed"]
        for (i, ch), d in lg["fit"].items():
            r[f"loss{i}{ch}"] = d["loss"]
        rows.append(r)
        if a.probe and (u + 1) % a.probe == 0:
            t0 = time.time()
            pr = summarize(heldout(preds, PROBE_NOISE, 4, a.H, a.W))
            probes.append(dict(upd=u + 1, **{f"dead{i}": pr[i]["dead"] for i in LEVELS},
                               **{f"ess{i}": pr[i]["ess"] for i in LEVELS}))
            log(f"  probe u{u + 1} (noise {PROBE_NOISE}, 4 rounds): "
                + " ".join(f"h{cl.HS[i]} dead {pr[i]['dead']:.3f} ess {pr[i]['ess']:.2f}" for i in LEVELS[:3])
                + f" ({time.time() - t0:.0f}s)")
    log(f"train: updates {t_upd:.0f}s + init/restarts (unrecorded) {t_init:.0f}s; peak rss {rss_mb():.0f}MB; "
        f"archive sizes " + " ".join(f"{cl.HS[i]}{'rc'[ch]}:{archive.size(i, ch)}" for (i, ch) in sorted(archive.buf)))
    ct.save_ckpt(os.path.join(TMP, f"certland_multi_{arm}_preds.npz"), preds, a.n)
    del archive
    # held-out generation
    t0 = time.time()
    ev = []
    for s in EVAL_SEEDS:
        ew = cl.World(a.H, a.W)
        ct.set_noise(ew, s)
        bad = cp.generate_pred(ew, preds, sweeps=20, seed=int(s))
        mk = cl.metrics(ew)
        viol = ct.count_violations(ew)
        ev.append(dict(seed=s, stuck=int(sum(bad)), overhangs=int(mk["overhangs"]), fill_err=float(mk["fill_err"]),
                       solid=float(ew.rho[cl.NL - 1].mean()), viol=int(sum(viol))))
        np.save(os.path.join(TMP, f"certland_multi_{arm}_tiles_{s}.npy"), ew.rho[cl.NL - 1])
        log(f"eval seed {s}: stuck {sum(bad)} {bad} overhangs {mk['overhangs']} fill_err {mk['fill_err']:.3f} "
            f"solid {ev[-1]['solid']:.3f} viol {sum(viol)}")
    log(f"eval {time.time() - t0:.0f}s")
    # held-out chains (certland_twist heldout protocol)
    t0 = time.time()
    held = {}
    for noise in HELD_NOISES:
        per = heldout(preds, noise, a.rounds, a.H, a.W)
        held[noise] = summarize(per)
        for i in LEVELS[:3]:
            d = [r["dead"] for r in per[i]]
            q = max(1, len(d) // 4)
            log(f"held-out noise {noise} h={cl.HS[i]:2d}: dead by round-quarter "
                + " ".join(f"{np.mean(d[j:j + q]):.3f}" for j in range(0, len(d), q))
                + f" | mean dead {held[noise][i]['dead']:.3f} ess {held[noise][i]['ess']:.2f} "
                  f"chg {held[noise][i]['changed']:.4f}")
    log(f"held-out {time.time() - t0:.0f}s; peak rss {rss_mb():.0f}MB")
    json.dump(dict(arm=arm, rows=rows, probes=probes, restarts=restarts, eval=ev,
                   held={str(n): {str(i): v for i, v in d.items()} for n, d in held.items()},
                   t_upd=t_upd, t_init=t_init, rss=rss_mb()),
              open(os.path.join(TMP, f"certland_multi_{arm}.json"), "w"), default=float)


def cmd_zero(a):
    """theta = 0 reference on the same held-out chains."""
    global LOGF
    LOGF = open(os.path.join(IMG, "certland_multi_zero_log.txt"), "w")
    held = {}
    for noise in HELD_NOISES:
        held[noise] = summarize(heldout(None, noise, a.rounds))
        log(f"theta=0 noise {noise}: " + " ".join(
            f"h{cl.HS[i]} dead {held[noise][i]['dead']:.3f} ess {held[noise][i]['ess']:.2f} "
            f"chg {held[noise][i]['changed']:.4f}" for i in LEVELS[:3]))
    json.dump({str(n): {str(i): v for i, v in d.items()} for n, d in held.items()},
              open(os.path.join(TMP, "certland_multi_zero.json"), "w"), default=float)


def cmd_compare(a):
    global LOGF
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    LOGF = open(os.path.join(IMG, "certland_multi_compare_log.txt"), "w")
    R = {arm: json.load(open(os.path.join(TMP, f"certland_multi_{arm}.json"))) for arm in a.arms
         if os.path.exists(os.path.join(TMP, f"certland_multi_{arm}.json"))}
    zp = os.path.join(TMP, "certland_multi_zero.json")
    if os.path.exists(zp):
        R0 = json.load(open(zp))
    else:
        R0 = None
    lv = LEVELS[:3]

    def hmean(held, i, k):
        return np.mean([held[str(n)][str(i)][k] for n in HELD_NOISES])

    log("== table: in-chain = mean over the last 20 of 60 updates; held-out = mean over noise "
        f"{HELD_NOISES} x 24 rounds; dead / ESS / changed")
    log("arm      | " + " | ".join(f"in-chain h{cl.HS[i]}" for i in lv) + " | "
        + " | ".join(f"held-out h{cl.HS[i]}" for i in lv) + " | fill_err overh stuck | t_upd t_init rss")
    for arm, r in R.items():
        last = r["rows"][-20:]
        ev = r["eval"]
        log(f"{arm:8s} | " + " | ".join(
            f"{np.mean([x[f'dead{i}'] for x in last]):.3f}/{np.mean([x[f'ess{i}'] for x in last]):.2f}/"
            f"{np.mean([x[f'chg{i}'] for x in last]):.3f}" for i in lv) + " | "
            + " | ".join(f"{hmean(r['held'], i, 'dead'):.3f}/{hmean(r['held'], i, 'ess'):.2f}/"
                         f"{hmean(r['held'], i, 'changed'):.3f}" for i in lv)
            + f" | {np.mean([e['fill_err'] for e in ev]):.3f} {sum(e['overhangs'] for e in ev)} "
              f"{sum(e['stuck'] for e in ev)} | {r['t_upd']:.0f}s {r['t_init']:.0f}s {r['rss']:.0f}MB")
    if R0:
        log("theta=0  | " + " | ".join("-" for _ in lv) + " | "
            + " | ".join(f"{hmean(R0, i, 'dead'):.3f}/{hmean(R0, i, 'ess'):.2f}/{hmean(R0, i, 'changed'):.3f}"
                         for i in lv))
    log("\n== held-out dead per noise seed (h16 h8 h4)")
    for arm, r in R.items():
        log(f"{arm:8s} " + "  ".join(f"n{n}: " + " ".join(f"{r['held'][str(n)][str(i)]['dead']:.3f}" for i in lv)
                                     for n in HELD_NOISES))
    log("\n== probes (noise 24, 4 rounds) dead h16/h8/h4 by update")
    for arm, r in R.items():
        log(f"{arm:8s} " + "  ".join(f"u{p['upd']}: " + "/".join(f"{p[f'dead{i}']:.2f}" for i in lv)
                                     for p in r["probes"]))
    log("\n== eval per seed")
    for arm, r in R.items():
        log(f"{arm:8s} " + "; ".join(f"{e['seed']}: fill_err {e['fill_err']:.3f} overh {e['overhangs']} "
                                     f"stuck {e['stuck']} solid {e['solid']:.3f}" for e in r["eval"]))
    # curves
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    sty = {"base": "-", "multi": "--", "restart": ":"}
    for arm, r in R.items():
        rows = r["rows"]
        u = np.array([x["upd"] for x in rows])
        for i, c in zip(lv, ("C0", "C1", "C2")):
            y = np.array([x[f"dead{i}"] for x in rows])
            ys = np.convolve(y, np.ones(4) / 4, mode="valid")
            ax[0].plot(u[3:], ys, sty[arm], color=c, label=f"{arm} h{cl.HS[i]}")
            ax[1].plot(u[3:], np.convolve([x[f"ess{i}"] for x in rows], np.ones(4) / 4, mode="valid"),
                       sty[arm], color=c)
            pr = r["probes"]
            ax[2].plot([p["upd"] for p in pr], [p[f"dead{i}"] for p in pr], sty[arm], marker="o", color=c,
                       label=f"{arm} h{cl.HS[i]}")
        for rs in r.get("restarts", []):
            ax[0].axvline(rs["upd"], color="0.85", lw=0.8, zorder=0)
    if R0:
        for i, c in zip(lv, ("C0", "C1", "C2")):
            ax[2].axhline(hmean(R0, i, "dead"), color=c, lw=0.6, alpha=0.5)
    for k, t in enumerate(("in-chain dead per proposal (4-update mean)", "in-chain ESS (of K+1, 4-update mean)",
                           "held-out probe dead (noise 24, 4 rounds; thin: theta=0)")):
        ax[k].set_title(t, fontsize=10)
        ax[k].set_xlabel("update")
    ax[0].legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "certland_multi_curves.png"), dpi=110)
    # pictures
    seeds = EVAL_SEEDS[:2]
    fig, ax = plt.subplots(len(seeds), len(R), figsize=(7 * len(R), 2.2 * len(seeds) + 0.6), squeeze=False)
    for c, (arm, r) in enumerate(R.items()):
        evd = {e["seed"]: e for e in r["eval"]}
        for k, s in enumerate(seeds):
            t = np.load(os.path.join(TMP, f"certland_multi_{arm}_tiles_{s}.npy"))
            col = np.where(t[..., None] == 1, np.array([138, 98, 66], np.uint8), np.array([169, 212, 240], np.uint8))
            ax[k, c].imshow(col, interpolation="nearest")
            e = evd[s]
            ax[k, c].set_title(f"{arm}, seed {s}: fill_err {e['fill_err']:.3f}, overhangs {e['overhangs']}, "
                               f"stuck {e['stuck']}", fontsize=9)
            ax[k, c].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "certland_multi_heldout.png"), dpi=110)
    log("figures: images/certland_multi_curves.png images/certland_multi_heldout.png")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--burn", type=int, default=3)
    p.add_argument("--R", type=int, default=10)
    p.add_argument("--H", type=int, default=64)
    p.add_argument("--W", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=24)
    p.add_argument("--probe", type=int, default=10)
    p = sub.add_parser("zero")
    p.add_argument("--rounds", type=int, default=24)
    p = sub.add_parser("compare")
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p = sub.add_parser("ownstart")
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--rounds", type=int, default=24)
    a = ap.parse_args()
    {"run": cmd_run, "zero": cmd_zero, "compare": cmd_compare, "ownstart": cmd_ownstart}[a.cmd](a)


if __name__ == "__main__":
    main()
