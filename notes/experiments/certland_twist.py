"""Learned twist for certland's subtree proposals: does a predictor with a
wider generic context cut the i-SIR deaths?  Arms (certland_pred layouts,
chosen by the env CERTLAND_PRED_FEATS read when certland_pred is imported,
so each runs in its own process), same chain, seeds and schedule:
  (a) v0     the old features (parent, 3 x 3 same-level window)
  (b) wide   4 x 4 same-level window over the parent block and one ring,
             the parent's 3 x 3
  (c) widek  wide + a known flag per same-level slot: the pass feeds its
             real known mask (placeholders flagged and zeroed), generation
             all ones; each recorded cell gives a Gibbs example and the
             proposal's own example (certland_chain.record_pass)

  CERTLAND_PRED_FEATS=v0 python notes/experiments/certland_twist.py run --n 60
  (likewise wide, widek)
  python notes/experiments/certland_twist.py compare

run: World(64, 128), levels (0, 1, 2, 3), K = {0: 8, 1: 4, 2: 1, 3: 1};
per update dead / ESS / changed per root level; then held-out generation
(generate_pred) on eval seeds (fill_err, overhangs, stuck) with a picture,
and the dead rate of the trained predictors on a held-out chain (theta
trained vs theta = 0, the same start state in both layouts).
Results to tmp (npz/json); log to images/certland_twist_<layout>_log.txt."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, "/Users/amdson/dev/Hierwave/.claude/worktrees/certland")

import numpy as np                                                       # noqa: E402

from castlegen.channels import certland as cl                           # noqa: E402
from castlegen.channels import certland_pred as cp                      # noqa: E402
from castlegen.channels import certland_chain as cc                     # noqa: E402
from castlegen.channels import certland_train as ct                     # noqa: E402

IMG = "/Users/amdson/dev/Hierwave/images"
TMP = os.environ.get("CERTLAND_TMP", "/Users/amdson/.claude/jobs/885763c6/tmp")
os.makedirs(TMP, exist_ok=True)
K = {0: 8, 1: 4, 2: 1, 3: 1}
LEVELS = (0, 1, 2, 3)
EVAL_SEEDS = (1001, 1002, 1003, 1004)
LOGF = None


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    if LOGF is not None:
        LOGF.write(s + "\n")
        LOGF.flush()


def _mover(preds):
    """The proposal-only set preds.q when present (stage 2), else preds."""
    return getattr(preds, "q", None) or preds


def heldout_dead(preds, rounds, seed=21, noise=21, H=64, W=128):
    """Dead / ESS / changed per root level on a held-out chain: the start
    state is generated with theta = 0 (identical across layouts), then
    `rounds` rounds of moves with `preds` (not recorded)."""
    rng = np.random.default_rng(seed)
    w = cl.World(H, W)
    cfg = ct.Cfg(levels=LEVELS, K=K, burn=3)
    ct.init_chain(w, cp.Preds(), noise, cfg.burn, rng, cfg)
    out = {i: dict(dead=[], ess=[], changed=[]) for i in LEVELS}
    for _ in range(rounds):
        for i in LEVELS:
            s = cc.move_level(w, _mover(preds), i, K[i], rng)
            for k in ("dead", "ess", "changed"):
                out[i][k].append(s[k])
        cc.tile_sweep(w, rng)
    return {i: {k: float(np.mean(v)) for k, v in d.items()} for i, d in out.items()}


def cmd_run(a):
    global LOGF
    lay = cp.LAYOUT + ("_q" if a.proposal else "")
    LOGF = open(os.path.join(IMG, f"certland_twist_{lay}_log.txt"), "w")
    log(f"certland_twist run: layout {lay} NF {cp.NF}; world {a.H}x{a.W} noise {a.noise}; levels {LEVELS} K {K}; "
        f"n {a.n} steps {a.steps} seed {a.seed} proposal-only predictor {a.proposal} (NFQ {cp.NFQ})")
    preds = cp.Preds(nh=64)
    cfg = ct.Cfg(levels=LEVELS, K=dict(K), steps=a.steps, burn=a.burn, eval_every=0, tag=f"twist_{lay}",
                 seed=a.seed, proposal=bool(a.proposal))
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    world = cl.World(a.H, a.W)
    info = ct.init_chain(world, preds, a.noise, cfg.burn, rng, cfg)
    log(f"init: stuck {info['bad']} viol {info['viol']} ({info['time']:.1f}s)")
    hist = ct.train(world, preds, cfg, a.n, log=log, eval_every=0)
    ttrain = time.time() - t0
    rows = []
    for lg in hist:
        r = dict(upd=lg["upd"], time=float(sum(v for v in lg["time"].values())))
        for i, d in lg["levels"].items():
            r[f"dead{i}"], r[f"ess{i}"], r[f"chg{i}"] = d["dead"], d["ess"], d["changed"]
        rows.append(r)
    log(f"train {ttrain:.0f}s")
    ck = os.path.join(TMP, f"certland_twist_{lay}_preds.npz")
    ct.save_ckpt(ck, preds, a.n)
    # held-out generation
    t0 = time.time()
    ev = []
    for k, s in enumerate(EVAL_SEEDS):
        ew = cl.World(a.H, a.W)
        ct.set_noise(ew, s)
        bad = cp.generate_pred(ew, preds, sweeps=20, seed=int(s))
        mk = cl.metrics(ew)
        viol = ct.count_violations(ew)
        ev.append(dict(seed=s, stuck=int(sum(bad)), overhangs=int(mk["overhangs"]), fill_err=float(mk["fill_err"]),
                       solid=float(ew.rho[cl.NL - 1].mean()), viol=int(sum(viol))))
        np.save(os.path.join(TMP, f"certland_twist_{lay}_tiles_{s}.npy"), ew.rho[cl.NL - 1])
        log(f"eval seed {s}: stuck {sum(bad)} {bad} overhangs {mk['overhangs']} fill_err {mk['fill_err']:.3f} "
            f"solid {ev[-1]['solid']:.3f} viol {sum(viol)}")
    log(f"eval {time.time() - t0:.0f}s")
    # held-out deaths
    t0 = time.time()
    hd = heldout_dead(preds, a.rounds)
    h0 = heldout_dead(None, a.rounds) if a.zero else None
    for i in LEVELS:
        s = f"held-out h={cl.HS[i]:2d}: trained dead {hd[i]['dead']:.3f} ess {hd[i]['ess']:.2f} chg {hd[i]['changed']:.3f}"
        if h0:
            s += f" | theta=0 dead {h0[i]['dead']:.3f} ess {h0[i]['ess']:.2f} chg {h0[i]['changed']:.3f}"
        log(s)
    log(f"held-out {time.time() - t0:.0f}s")
    json.dump(dict(layout=lay, NF=cp.NF, rows=rows, eval=ev, heldout=hd, heldout0=h0, ttrain=ttrain),
              open(os.path.join(TMP, f"certland_twist_{lay}.json"), "w"), default=float)


def cmd_heldout(a):
    """Dead rate per round of a held-out chain under the trained predictors
    of this layout (checkpoint of `run`), from a theta = 0 start, for
    several noise seeds (the training seed included): is the in-chain dead
    rate a property of the predictor or of the training world / state?"""
    global LOGF
    lay = cp.LAYOUT + ("_q" if a.proposal else "")
    LOGF = open(os.path.join(IMG, f"certland_twist_heldout_{lay}_log.txt"), "w")
    preds, upd = ct.load_ckpt(os.path.join(TMP, f"certland_twist_{lay}_preds.npz"))
    log(f"held-out chains: layout {lay}, preds after {upd} updates, {a.rounds} rounds, noise {a.noises}")
    res = {}
    for noise in a.noises:
        rng = np.random.default_rng(100 + noise)
        w = cl.World(a.H, a.W)
        cfg = ct.Cfg(levels=LEVELS, K=K, burn=0)
        ct.init_chain(w, cp.Preds(), noise, 0, rng, cfg)
        per = {i: [] for i in LEVELS}
        for _ in range(a.rounds):
            for i in LEVELS:
                per[i].append(cc.move_level(w, _mover(preds), i, K[i], rng))
            cc.tile_sweep(w, rng)
        res[noise] = per
        for i in LEVELS[:3]:
            d = [r["dead"] for r in per[i]]
            q = max(1, len(d) // 4)
            log(f"noise {noise:3d} h={cl.HS[i]:2d}: dead by round-quarter "
                + " ".join(f"{np.mean(d[k:k + q]):.3f}" for k in range(0, len(d), q))
                + f" | ess last half {np.mean([r['ess'] for r in per[i][len(d) // 2:]]):.2f}"
                + f" chg last half {np.mean([r['changed'] for r in per[i][len(d) // 2:]]):.4f}")


def cmd_compare(a):
    global LOGF
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    LOGF = open(os.path.join(IMG, "certland_twist_compare_log.txt"), "w")
    R = {lay: json.load(open(os.path.join(TMP, f"certland_twist_{lay}.json"))) for lay in a.layouts}
    for lay, r in R.items():
        rows = r["rows"]
        n = len(rows)
        last = rows[n - n // 3:]
        log(f"== layout {lay} (NF {r['NF']}), {n} updates, train {r['ttrain']:.0f}s; mean over the last {len(last)}:")
        for i in LEVELS:
            log(f"   h={cl.HS[i]:2d}: dead {np.mean([x[f'dead{i}'] for x in last]):.3f} "
                f"ess {np.mean([x[f'ess{i}'] for x in last]):.2f} chg {np.mean([x[f'chg{i}'] for x in last]):.4f}"
                f" | held-out dead {r['heldout'][str(i)]['dead']:.3f} ess {r['heldout'][str(i)]['ess']:.2f}"
                + (f" (theta=0 dead {r['heldout0'][str(i)]['dead']:.3f})" if r["heldout0"] else ""))
        ev = r["eval"]
        log(f"   eval: fill_err {np.mean([e['fill_err'] for e in ev]):.3f} overhangs {[e['overhangs'] for e in ev]} "
            f"stuck {[e['stuck'] for e in ev]} viol {[e['viol'] for e in ev]} solid "
            f"{np.mean([e['solid'] for e in ev]):.3f}")
        log("   per update dead (h16 h8 h4 h2):")
        for x in rows:
            log(f"     u{x['upd']:3d} " + " ".join(f"{x[f'dead{i}']:.2f}/{x[f'ess{i}']:.2f}/{x[f'chg{i}']:.3f}"
                                                  for i in LEVELS) + f"  ({x['time']:.0f}s)")
    # table
    log("\n== table (mean over the last third of updates; held-out generation over the eval seeds)")
    log("arm      | " + " | ".join(f"h{cl.HS[i]} dead/ESS/chg" for i in LEVELS) + " | fill_err overh stuck")
    for lay, r in R.items():
        rows = r["rows"]
        last = rows[len(rows) - len(rows) // 3:]
        ev = r["eval"]
        log(f"{lay:8s} | " + " | ".join(
            f"{np.mean([x[f'dead{i}'] for x in last]):.3f}/{np.mean([x[f'ess{i}'] for x in last]):.2f}/"
            f"{np.mean([x[f'chg{i}'] for x in last]):.4f}" for i in LEVELS)
            + f" | {np.mean([e['fill_err'] for e in ev]):.3f} {sum(e['overhangs'] for e in ev)} "
              f"{sum(e['stuck'] for e in ev)}")
    # curves
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for lay, r in R.items():
        rows = r["rows"]
        u = [x["upd"] for x in rows]
        for i, c in zip(LEVELS[:3], ("C0", "C1", "C2")):
            ls = {"v0": ":", "wide": "--", "widek_q": "-."}.get(lay, "-")
            ax[0].plot(u, [x[f"dead{i}"] for x in rows], ls, color=c, label=f"{lay} h{cl.HS[i]}")
            ax[1].plot(u, [x[f"ess{i}"] for x in rows], ls, color=c, label=f"{lay} h{cl.HS[i]}")
            ax[2].plot(u, [x[f"chg{i}"] for x in rows], ls, color=c, label=f"{lay} h{cl.HS[i]}")
    for k, t in enumerate(("dead per proposal", "ESS (of K+1)", "changed fraction")):
        ax[k].set_title(t)
        ax[k].set_xlabel("update")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "certland_twist_curves.png"), dpi=110)
    # pictures
    seeds = EVAL_SEEDS[:2]
    fig, ax = plt.subplots(len(seeds), len(R), figsize=(7 * len(R), 2.2 * len(seeds) + 0.6), squeeze=False)
    for c, (lay, r) in enumerate(R.items()):
        evd = {e["seed"]: e for e in r["eval"]}
        for k, s in enumerate(seeds):
            t = np.load(os.path.join(TMP, f"certland_twist_{lay}_tiles_{s}.npy"))
            col = np.where(t[..., None] == 1, np.array([138, 98, 66], np.uint8), np.array([169, 212, 240], np.uint8))
            ax[k, c].imshow(col, interpolation="nearest")
            e = evd[s]
            ax[k, c].set_title(f"({'abcdef'[c]}) {lay} NF {r['NF']}, seed {s}: fill_err "
                               f"{e['fill_err']:.3f}, overhangs {e['overhangs']}, stuck {e['stuck']}", fontsize=9)
            ax[k, c].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "certland_twist_heldout.png"), dpi=110)
    log("figures: images/certland_twist_curves.png images/certland_twist_heldout.png")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--burn", type=int, default=3)
    p.add_argument("--H", type=int, default=64)
    p.add_argument("--W", type=int, default=128)
    p.add_argument("--noise", type=int, default=7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--zero", type=int, default=1)
    p.add_argument("--proposal", type=int, default=0)
    p = sub.add_parser("heldout")
    p.add_argument("--rounds", type=int, default=24)
    p.add_argument("--noises", type=int, nargs="+", default=[7, 21, 22])
    p.add_argument("--proposal", type=int, default=0)
    p.add_argument("--H", type=int, default=64)
    p.add_argument("--W", type=int, default=128)
    p = sub.add_parser("compare")
    p.add_argument("--layouts", nargs="+", default=["v0", "wide", "widek"])
    a = ap.parse_args()
    {"run": cmd_run, "compare": cmd_compare, "heldout": cmd_heldout}[a.cmd](a)


if __name__ == "__main__":
    main()
