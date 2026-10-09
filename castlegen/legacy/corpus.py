"""Legacy (G1, promises and texture synthesis era); superseded by castlegen/channels. See notes/history/promises.md.

Promise corpus and empirical-count tables (notes/history/promises.md sections 7.2, 7.3).

Samples p on n x n tori with exemplar.gibbs at the exemplar layout's
temperature schedule (anneal T_hot -> T over `sweeps`, then sweeps/2 cold,
then the layout's `clean` repair), makes each sample one network with
exemplar.connect (global, torus), and mines the promise grid of every level
h = K .. n with abstract o summarize.  Samples and tables are cached under
cache/, keyed by the tile set spec, the layout's sampling settings, n, K and
the seed list.

Tables are promise.Tables fitted by counts with additive smoothing.  The
diagnostics are section 7.3's: per child level, the mean number of consistent
refinements per corpus parent (alone, and given the corpus halo), the share
of corpus blocks that satisfy their own abstraction (the projection is lossy
when a block holds several components), and the share of corpus parents whose
observed refinement is consistent and has low probability under the fit.

Per quantity (connectivity, flow) and per source: "p" samples p with
Gibbs as above; "synth" samples the exemplar pipeline instead (texture
synthesis from the layout's exemplar at the sample's seed, then a repair),
for tile sets whose p has no macrostructure at equilibrium (wilds: p holds
~0.01 % water, so a p corpus would carry no flux at all).  A corpus that
serves the flow promise is made valid with flow.make_valid (each K-block
fulfilled to the projection of its own seam fluxes) before the global
connect, which bridges rivers and deletes to the layout's "void".

    python -m castlegen.legacy.corpus --layout castle --seeds 24
    python -m castlegen.legacy.corpus --layout wilds --source synth --quantity flow --quantities flow,connectivity
    python -m castlegen.legacy.corpus --layout cliffs --quantity support

Side-view layouts ("ground": G) sample p with the bottom G rows clamped to
"ground_tile" (default stone) at "corpus_T" (default the layout's T), and a
support corpus is made valid by support.make_valid (global fill / clear,
then every K-block fulfilled to the top-down projection of its own
abstractions); "connect": false skips the global connect.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np

from castlegen.legacy import exemplar as ex
from castlegen.legacy import tileset
from castlegen.legacy.promise import Tables
from castlegen.legacy.quantities import connectivity as C
from castlegen.legacy.quantities import flow as F
from castlegen.legacy.quantities import support as SP

QUANTITIES = {"connectivity": C, "flow": F, "support": SP}

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "cache")
SEEDS = tuple(range(1000, 1024))


def _key(ts, layout, n, seeds, extra=""):
    spec_path = os.path.join(tileset.TILESET_DIR, layout.get("tileset", "demo") + ".json")
    spec = open(spec_path).read() if os.path.exists(spec_path) else ts.name
    samp = {k: layout.get(k) for k in ("sweeps", "T", "T_hot", "clean", "connect", "torus")}
    samp.update({k: layout[k] for k in ("ground", "corpus_T") if k in layout})       # side-view layouts only
    return hashlib.sha1((spec + json.dumps(samp, sort_keys=True) + f"{n}{tuple(seeds)}{extra}").encode()).hexdigest()[:12]


def _source_key(layout, source, quantities, K):
    """Extra cache key: empty for the original p / connectivity corpus."""
    if source == "p" and tuple(quantities) == ("connectivity",):
        return ""
    return json.dumps([source, list(quantities), K, layout.get("place") if source == "synth" else None, "v2"])


def sample(ts, layout, seed, n=128, source="p", quantities=("connectivity",), K=16, exemplar=None):
    """One corpus sample on an n x n torus: p (or the exemplar pipeline),
    made flow-valid if flow is among `quantities`, made one network."""
    G = layout.get("ground", 0)
    if source == "p":
        sweeps, T, T_hot = layout.get("sweeps", 300), layout.get("corpus_T", layout.get("T", 1.0)), layout.get("T_hot", 4.0)
        tiles = np.full((n, n), ts.WALL, np.int32)
        upd = None
        if G:                                            # side view: the band is clamped ground
            tiles[n - G:] = ex.sig_by_name(ts, layout.get("ground_tile", "stone"))
            upd = ~SP.exempt_rows(n, G)[:, None].repeat(n, 1)
        tiles = ex.gibbs(ts, tiles, sweeps, seed, T, T_hot, update=upd)
        tiles = ex.gibbs(ts, tiles, sweeps // 2, seed + 1, T, update=upd)
        clean = layout.get("clean")
        if clean:
            c = clean if isinstance(clean, dict) else {}
            tiles = ex.repair(ts, tiles, c.get("rounds", 30), seed + 2, T=c.get("T", 0.4), tau=c.get("tau", 1.9))
    else:
        from castlegen.legacy import texsyn
        an = texsyn.Analysis(ts, exemplar, n_pca=32, w_sock=0.5)
        S = texsyn.synthesize(an, n, [1, 1] + [0] * (an.L - 1), seed=seed, corrections=2, kappa=4.0)
        tiles = ex.repair(ts, exemplar[S[..., 0], S[..., 1]], 20, seed + 2, T=0.4, tau=1.9, clear_orphans=True)
    if "flow" in quantities:
        tiles, _ = F.make_valid(ts, tiles, K, seed)
    if "support" in quantities:
        tiles, _ = SP.make_valid(ts, tiles, G, K, seed)
    join = layout.get("connect", True)
    if not join:
        return tiles
    c = join if isinstance(join, dict) else {}
    void = ex.sig_by_name(ts, c["void"]) if "void" in c else None
    if c:
        tiles, _ = ex.connect(ts, tiles, seed=seed + 3, tag=c.get("tag", "hallway"), void=void)
    else:
        tiles, _ = ex.connect(ts, tiles, seed=seed + 3)
    return tiles


def samples(ts, layout, n=128, seeds=SEEDS, log=print, source="p", quantities=("connectivity",), K=16):
    path = os.path.join(CACHE_DIR, f"corpus-{_key(ts, layout, n, seeds, _source_key(layout, source, quantities, K))}.npy")
    if os.path.exists(path):
        return np.load(path)
    t0 = time.time()
    E = None
    if source == "synth":
        from castlegen.legacy import pipeline
        E = pipeline.exemplar_for(dict(layout, name=layout.get("name", "layout")), ts, log=log)
    out = []
    for i, s in enumerate(seeds):
        out.append(sample(ts, layout, s, n, source, quantities, K, E))
        log(f"corpus: sample {i + 1}/{len(seeds)} ({time.time() - t0:.0f}s)")
    out = np.stack(out)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(path, out)
    log(f"corpus: {len(seeds)} samples {n}x{n} in {time.time() - t0:.1f}s -> {path}")
    return out


def mine(ts, tiles, K=16, q=C, ground=0):
    """-> ({h: promise grid}, {h: exact-valid mask}) for h = K .. n, for
    the quantity module q (support: with the ground band of `ground` rows)."""
    n = tiles.shape[0]
    grids, valid = {}, {}
    g = n // K
    kw = {"ground": ground} if q is SP else {}
    S = [[q.summarize(ts, tiles, y * K, x * K, K, **kw) for x in range(g)] for y in range(g)]
    h = K
    while True:
        grids[h] = np.array([[q.abstract(s) for s in row] for row in S], np.int64)
        valid[h] = np.array([[q.satisfied(q.abstract(s), s) for s in row] for row in S])
        if g == 1:
            break
        g //= 2
        S = [[q.merge({(dy, dx): S[2 * y + dy][2 * x + dx] for dy in (0, 1) for dx in (0, 1)})
              for x in range(g)] for y in range(g)]
        h *= 2
    return grids, valid


def tables_for(ts, layout, K=16, n=128, seeds=SEEDS, alpha=0.5, rebuild=False, log=print, quantity="connectivity",
               source="p", quantities=None):
    """-> (Tables, mined grids {h: [grid per sample]}, valid masks) for one
    quantity, from the corpus made for `quantities` (default: that one); cached."""
    quantities = tuple(quantities or (quantity,))
    q = QUANTITIES[quantity]
    extra = _source_key(layout, source, quantities, K)
    qk = "" if quantity == "connectivity" and not extra else quantity + extra
    path = os.path.join(CACHE_DIR, f"promise-tables-{_key(ts, layout, n, seeds, f'K{K}a{alpha}' + qk)}.npz")
    tiles = samples(ts, layout, n, seeds, log, source, quantities, K)
    mined = [mine(ts, t, K, q, layout.get("ground", 0)) for t in tiles]
    grids = {h: [m[0][h] for m in mined] for h in mined[0][0]}
    valid = {h: [m[1][h] for m in mined] for h in mined[0][1]}
    if os.path.exists(path) and not rebuild:
        return Tables.load(path), grids, valid
    tab = Tables.fit(q.V, grids, alpha)
    tab.save(path)
    log(f"tables -> {path}")
    return tab, grids, valid


def _halo(grid, py, px):
    """{(dy, dx, d): bit} of the external half-seams of parent (py, px), read
    from the neighbouring children in `grid` (torus)."""
    g0, g1 = grid.shape
    out = {}
    for (dy, dx) in ((0, 0), (0, 1), (1, 0), (1, 1)):
        y, x = 2 * py + dy, 2 * px + dx
        for d, (sy, sx) in enumerate(C.DIRS):
            if (d == C.N and dy == 0) or (d == C.S and dy == 1) or (d == C.W and dx == 0) or (d == C.E and dx == 1):
                out[dy, dx, d] = int(C.bit(grid[(y + sy) % g0, (x + sx) % g1], C.OPP[d]))
    return out


HALO = {"connectivity": _halo, "flow": F.halo, "support": SP.halo}


def _group_energy(tab, h, kids, parent, grid, py, px):
    """Energy of a 2x2 refinement `kids` (c00, c01, c10, c11) of `parent`
    with the rest of `grid` as the halo: unary + parent terms + every seam
    touching a child."""
    u, pair, par = tab.level(h)
    g0, g1 = grid.shape
    over = {(2 * py + dy, 2 * px + dx): v for (dy, dx), v in zip(((0, 0), (0, 1), (1, 0), (1, 1)), kids)}
    val = lambda y, x: over.get((y % g0, x % g1), grid[y % g0, x % g1])
    e = 0.0
    seams = set()
    for (y, x), v in over.items():
        e += u[v] + par[2 * (y % 2) + x % 2, v, parent]
        seams |= {(y, x, 0), (y, (x - 1) % g1, 0), (y, x, 1), ((y - 1) % g0, x, 1)}
    for y, x, o in seams:
        e += pair[o, val(y, x), val(y, x + 1) if o == 0 else val(y + 1, x)]
    return e


def diagnostics(tab, grids, valid, low=0.05, quantity="connectivity"):
    """Per child level h (parents at 2h): see the module docstring."""
    q, halo_fn = QUANTITIES[quantity], HALO[quantity]
    n_refine = getattr(q, "n_refine", lambda P, halo=None: len(q.refine(P, halo)))
    out = {}
    for h in sorted(grids):
        if 2 * h not in grids:
            continue
        n_ref, n_halo, cons, lowp, probs = [], [], [], [], []
        for g, G in zip(grids[h], grids[2 * h]):
            for py, px in np.ndindex(*G.shape):
                P = int(G[py, px])
                kids = tuple(int(g[2 * py + dy, 2 * px + dx]) for dy in (0, 1) for dx in (0, 1))
                n_ref.append(n_refine(P))
                cands = q.refine(P, halo_fn(g, py, px))
                n_halo.append(len(cands))
                ok = kids in cands
                cons.append(ok)
                if ok:
                    e = np.array([_group_energy(tab, h, c, P, g, py, px) for c in cands])
                    p = np.exp(-(e - e.min()))
                    p /= p.sum()
                    pk = float(p[cands.index(kids)])
                    probs.append(pk * len(cands))
                    lowp.append(pk < low)
        out[h] = dict(parents=len(n_ref), refinements=float(np.mean(n_ref)), with_halo=float(np.mean(n_halo)),
                      single_option=float(np.mean(np.array(n_halo) == 1)),
                      valid_blocks=float(np.mean([v.mean() for v in valid[h]])),
                      consistent=float(np.mean(cons)), low_prob=float(np.mean(lowp)) if lowp else float("nan"),
                      p_over_uniform=float(np.mean(probs)) if probs else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--layout", default="castle")
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--quantity", default="connectivity")
    ap.add_argument("--quantities", default=None, help="comma list the corpus is made valid for")
    ap.add_argument("--source", default="p", choices=["p", "synth"])
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    layout = dict(ex.load_layout(args.layout), name=args.layout)
    ts = tileset.load(layout.get("tileset", "demo"))
    seeds = tuple(range(1000, 1000 + args.seeds))
    qs = tuple(args.quantities.split(",")) if args.quantities else None
    tab, grids, valid = tables_for(ts, layout, args.K, args.size, seeds, rebuild=args.rebuild, quantity=args.quantity,
                                   source=args.source, quantities=qs)
    for h, d in diagnostics(tab, grids, valid, quantity=args.quantity).items():
        print(f"h={h:4d} " + "  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in d.items()))
    for h in sorted(grids):
        allv = np.concatenate([g.ravel() for g in grids[h]])
        cnt = np.bincount(allv, minlength=QUANTITIES[args.quantity].V)
        top = np.argsort(-cnt)[:6]
        print(f"h={h:4d} blocks {allv.size} distinct {int((cnt > 0).sum())} top " +
              ", ".join(f"{int(v)}:{int(cnt[v])}" for v in top if cnt[v]))


if __name__ == "__main__":
    main()
