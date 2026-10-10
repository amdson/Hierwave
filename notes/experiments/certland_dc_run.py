"""Conditional DC-SMC subtree moves (castlegen/channels/certland_dc.py) against
the top-down i-SIR subtree move (certland_chain.subtree_move) on certland's
designed p*, World(64, 128).

  base    theta = 0 (designed-only proposals both ways): per root level
          h16 / h8 / h4 / h2 and particle count, ESS, accept (moves that
          leave the reference), changed fraction, deaths (DC: per node level,
          q-deaths = no admissible value for the node, x-deaths = a hard term
          between two children's subtrees; TD: dead passes), time per move
  train   train the bottom-up predictor from the DC chain (UpCollector on the
          chosen states, weighted CE, archive), then the same table with it;
          checkpoint images/certland_dc_up.npz
  mixed   sweeps of a mixed schedule (DC at h16 / h8, top-down at h4 / h2)
          against all-top-down and all-DC at matched particle counts: changed
          per level per sweep, time per sweep, violations, overhangs

Logs: images/certland_dc_<cmd>_log.txt.
Usage: NUMBA_NUM_THREADS=1 python notes/experiments/certland_dc_run.py <cmd> [...]"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/amdson/dev/Hierwave/.claude/worktrees/certland")

from castlegen.channels import certland as cl                 # noqa: E402
from castlegen.channels import certland_chain as cc           # noqa: E402
from castlegen.channels import certland_dc as dc              # noqa: E402
from castlegen.channels import certland_pred as cp            # noqa: E402
from castlegen.channels import certland_train as ct           # noqa: E402

IMG = "/Users/amdson/dev/Hierwave/images"
TMP = "/Users/amdson/.claude/jobs/885763c6/tmp"
LOGF = None


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    if LOGF is not None:
        LOGF.write(s + "\n")
        LOGF.flush()


def make_world(H, W, seed, burn):
    """A burned-in valid chain state (forward generation by the shipped
    sampler, then `burn` rounds of designed-only top-down moves)."""
    path = os.path.join(TMP, f"dc_world_{H}x{W}_s{seed}_b{burn}.npz")
    w = cl.World(H, W)
    if os.path.exists(path):
        z = np.load(path)
        w.fill[:] = z["fill"]
        for i in range(cl.NL):
            w.rho[i][:] = z[f"r{i}"]
            if i < cl.NL - 1:
                w.cert[i][:] = z[f"c{i}"]
        return w
    t0 = time.time()
    cfg = ct.Cfg(burn=burn, K={0: 8, 1: 4, 2: 2, 3: 2})
    info = ct.init_chain(w, cp.Preds(), seed, burn, np.random.default_rng(seed), cfg)
    log(f"init world {H}x{W} seed {seed}: gen stuck {info['bad']} burn {burn} in {time.time() - t0:.1f}s")
    d = dict(fill=w.fill)
    for i in range(cl.NL):
        d[f"r{i}"] = w.rho[i]
        if i < cl.NL - 1:
            d[f"c{i}"] = w.cert[i]
    np.savez(path, **d)
    return w


def copy_world(w):
    v = cl.World(w.H, w.W)
    v.fill[:] = w.fill
    for i in range(cl.NL):
        v.rho[i][:] = w.rho[i]
        if i < cl.NL - 1:
            v.cert[i][:] = w.cert[i]
    return v


def roots(w, i):
    rows, cols = w.rho[i].shape
    for cy in range(2):
        for cx in range(2):
            for ry in range(cy, rows, 2):
                for rx in range(cx, cols, 2):
                    yield ry, rx


def sweep_td(w, i, K, rng, preds=None):
    bias = cc._bias_args(preds)
    out = dict(ess=0.0, acc=0, changed=0, tot=0, dead=0, n=0, t=0.0)
    for ry, rx in roots(w, i):
        t0 = time.time()
        r = cc.subtree_move(w, preds, i, ry, rx, K, rng, _bias=bias)
        out["t"] += time.time() - t0
        out["ess"] += r["ess"]
        out["acc"] += r["chosen"] != 0
        out["changed"] += r["changed"]
        out["tot"] += r["n"]
        out["dead"] += r["dead"]
        out["n"] += 1
    return out


def sweep_dc(w, i, N, rng, pu=None, collect=None):
    bias = dc.up_args(pu)
    out = dict(ess=0.0, acc=0, changed=0, tot=0, n=0, t=0.0, dq=np.zeros(4, np.int64), dx=np.zeros(4, np.int64),
               mg=np.zeros(4, np.int64))
    for ry, rx in roots(w, i):
        t0 = time.time()
        r = dc.dc_move(w, pu, i, ry, rx, N, rng, collect=collect, _bias=bias)
        out["t"] += time.time() - t0
        out["ess"] += r["ess"]
        out["acc"] += r["chosen"] != 0
        out["changed"] += r["changed"]
        out["tot"] += r["n"]
        out["dq"] += r["dead_q"]
        out["dx"] += r["dead_x"]
        out["mg"] += r["merges"]
        out["n"] += 1
    return out


def fmt_td(i, K, s):
    n = max(s["n"], 1)
    return (f"  TD h{cl.HS[i]:<2} K={K:<3} ess {s['ess'] / n:5.2f} acc {s['acc'] / n:.3f} chg {s['changed'] / s['tot']:.4f} "
            f"dead {s['dead'] / (n * K):.2f}  {1e3 * s['t'] / n:7.2f} ms/move")


def fmt_dc(i, N, s):
    n = max(s["n"], 1)
    where = " ".join(f"h{cl.HS[l]}:{s['dq'][l] / s['mg'][l]:.2f}/{s['dx'][l] / s['mg'][l]:.2f}"
                     for l in range(i, cl.NL - 1))
    tot = (s["dq"].sum() + s["dx"].sum()) / max(s["mg"].sum(), 1)
    return (f"  DC h{cl.HS[i]:<2} N={N:<3} ess {s['ess'] / n:5.2f} acc {s['acc'] / n:.3f} chg {s['changed'] / s['tot']:.4f} "
            f"dead {tot:.2f} [q/x by node level {where}]  {1e3 * s['t'] / n:7.2f} ms/move")


def table(w, levels, Ns, Ks, sweeps, seed, pu=None, tag=""):
    """Per root level: DC for each N, TD for each K, from the same state
    (copies), `sweeps` sweeps each.  Returns rows."""
    rows = []
    for i in levels:
        log(f"root level h{cl.HS[i]} ({w.rho[i].size} roots){tag}")
        for N in Ns:
            v = copy_world(w)
            rng = np.random.default_rng(seed)
            acc = None
            for _ in range(sweeps):
                s = sweep_dc(v, i, N, rng, pu)
                acc = s if acc is None else {k: acc[k] + s[k] for k in acc}
            log(fmt_dc(i, N, acc))
            assert sum(ct.count_violations(v)) == 0 and cl.overhangs(v.rho[cl.NL - 1]) == 0
            rows.append(("dc", i, N, acc))
        for K in Ks:
            v = copy_world(w)
            rng = np.random.default_rng(seed)
            acc = None
            for _ in range(sweeps):
                s = sweep_td(v, i, K, rng)
                acc = s if acc is None else {k: acc[k] + s[k] for k in acc}
            log(fmt_td(i, K, acc))
            rows.append(("td", i, K, acc))
    return rows


def plot_rows(rows, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
    for k, i in enumerate((0, 1, 2, 3)):
        ax = axs[k]
        for kind, col in (("dc", "C0"), ("td", "C1")):
            pts = [(1e3 * s["t"] / s["n"], s["changed"] / s["tot"], P) for (kk, ii, P, s) in rows if kk == kind and ii == i]
            if not pts:
                continue
            x, y, P = zip(*pts)
            ax.plot(x, y, "o-", color=col, label="DC (N)" if kind == "dc" else "top-down (K)")
            for a, b, p in zip(x, y, P):
                ax.annotate(str(p), (a, b), fontsize=7, textcoords="offset points", xytext=(3, 3))
        ax.set_xscale("log")
        ax.set_title(f"root h{cl.HS[i]}")
        ax.set_xlabel("ms per move")
        if k == 0:
            ax.set_ylabel("changed fraction per move")
            ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    log(f"figure {path}")


# ------------------------------------------------------------------ commands
def cmd_base(a):
    w = make_world(a.H, a.W, a.seed, a.burn)
    log(f"theta = 0, world {a.H}x{a.W}, {a.sweeps} sweeps per config; deaths per fresh merged particle")
    rows = table(w, a.levels, a.N, a.K, a.sweeps, a.seed)
    plot_rows(rows, os.path.join(IMG, "certland_dc_base.png"), "theta = 0: changed fraction vs time per move")


def train_up(w, pu, a, archive):
    rng = np.random.default_rng(a.seed + 1)
    for u in range(a.updates):
        t0 = time.time()
        fresh = ct.Examples()
        col = dc.UpCollector(fresh, u)
        line = [f"u{u}"]
        for i in a.levels:
            s = sweep_dc(w, i, a.Ntrain, rng, pu, collect=col)
            tot = (s["dq"].sum() + s["dx"].sum()) / max(s["mg"].sum(), 1)
            line.append(f"h{cl.HS[i]} ess {s['ess'] / s['n']:.2f} acc {s['acc'] / s['n']:.3f} dead {tot:.2f}")
        cc.tile_sweep(w, rng)
        tm = time.time() - t0
        t1 = time.time()
        fl = dc.fit_up(pu, fresh, archive, steps=a.steps, lr=a.lr, batch=a.batch, rng=rng, seed=a.seed)
        line.append(" ".join(f"[{cl.HS[l]}{'rc'[ch]} {d['loss0']:.3f}->{d['loss']:.3f}]" for (l, ch), d in sorted(fl.items())))
        line.append(f"t moves {tm:.1f} rec {col.time:.1f} fit {time.time() - t1:.1f}")
        log(" | ".join(line))
        assert sum(ct.count_violations(w)) == 0
    return pu


def save_up(pu, path):
    d = dict(nh=np.int64(pu.nh))
    for l in range(cl.NL - 1):
        for ch in (0, 1):
            if pu.th[l][ch] is not None:
                d[f"th_{l}_{ch}"] = pu.th[l][ch]
    np.savez(path, **d)


def load_up(path):
    z = np.load(path)
    pu = dc.UpPreds(int(z["nh"]))
    for l in range(cl.NL - 1):
        for ch in (0, 1):
            if f"th_{l}_{ch}" in z:
                pu.th[l][ch] = np.ascontiguousarray(z[f"th_{l}_{ch}"])
    return pu


def cmd_train(a):
    w = make_world(a.H, a.W, a.seed, a.burn)
    pu = dc.UpPreds(a.nh)
    archive = ct.Archive(cap=a.cap, floats=4_000_000)
    log(f"training the bottom-up predictor: {a.updates} updates, N={a.Ntrain}, levels {a.levels}, nh {a.nh}")
    train_up(w, pu, a, archive)
    path = os.path.join(IMG, "certland_dc_up.npz")
    save_up(pu, path)
    log(f"saved {path}")
    # fresh held-out state (the burned-in world of another seed) for the comparison
    w2 = make_world(a.H, a.W, a.seed + 100, a.burn)
    log("held-out world, trained predictor:")
    rows = table(w2, a.levels, a.N, [], a.sweeps, a.seed, pu=pu, tag=" (trained)")
    log("held-out world, theta = 0:")
    rows0 = table(w2, a.levels, a.N, a.K, a.sweeps, a.seed, tag=" (theta 0)")
    for r in rows0:
        if r[0] == "td":
            rows.append(r)
    plot_rows(rows, os.path.join(IMG, "certland_dc_train.png"),
              f"held-out world: DC with the trained bottom-up predictor ({a.updates} updates) vs top-down at theta 0")


def cmd_mixed(a):
    w = make_world(a.H, a.W, a.seed, a.burn)
    pu = load_up(a.ckpt) if a.ckpt else None
    sched = {"td": {0: "td", 1: "td", 2: "td", 3: "td"}, "dc": {0: "dc", 1: "dc", 2: "dc", 3: "dc"},
             "mixed": {0: "dc", 1: "dc", 2: "td", 3: "td"}}
    for name, kind in sched.items():
        v = copy_world(w)
        rng = np.random.default_rng(a.seed)
        tot = {i: [0, 0, 0.0] for i in range(4)}
        for s in range(a.sweeps):
            for i in (0, 1, 2, 3):
                P = a.Pm[i]
                r = sweep_dc(v, i, P, rng, pu) if kind[i] == "dc" else sweep_td(v, i, P, rng)
                tot[i][0] += r["changed"]
                tot[i][1] += r["tot"]
                tot[i][2] += r["t"]
            cc.tile_sweep(v, rng)
        assert sum(ct.count_violations(v)) == 0 and cl.overhangs(v.rho[cl.NL - 1]) == 0
        mk = cl.metrics(v)
        log(f"{name:6s} " + " ".join(f"h{cl.HS[i]}({kind[i]},{a.Pm[i]}): chg {tot[i][0] / tot[i][1]:.4f} "
                                     f"{tot[i][2] / a.sweeps:.2f}s/sweep" for i in range(4))
            + f" | fill_err {mk['fill_err']:.3f} overhangs {mk['overhangs']}")


def main():
    global LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["base", "train", "mixed"])
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--W", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--burn", type=int, default=3)
    ap.add_argument("--levels", type=lambda s: tuple(int(v) for v in s.split(",")), default=(0, 1, 2, 3))
    ap.add_argument("--N", type=lambda s: tuple(int(v) for v in s.split(",")), default=(4, 16, 64))
    ap.add_argument("--K", type=lambda s: tuple(int(v) for v in s.split(",")), default=(4, 16, 64))
    ap.add_argument("--sweeps", type=int, default=2)
    ap.add_argument("--updates", type=int, default=20)
    ap.add_argument("--Ntrain", type=int, default=16)
    ap.add_argument("--nh", type=int, default=32)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--cap", type=int, default=12000)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--Pm", type=lambda s: tuple(int(v) for v in s.split(",")), default=(16, 16, 4, 4))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(IMG, exist_ok=True)
    LOGF = open(os.path.join(IMG, f"certland_dc_{a.cmd}{a.tag}_log.txt"), "w")
    log(" ".join(sys.argv))
    t0 = time.time()
    dict(base=cmd_base, train=cmd_train, mixed=cmd_mixed)[a.cmd](a)
    log(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
