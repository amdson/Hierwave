"""Legacy (G1, promises and texture synthesis era); superseded by castlegen/channels. See notes/history/promises.md.

Generation pipeline: layout JSON -> exemplar -> synthesis -> repairs.

A pipeline spec names a tile set, an exemplar layout (castlegen/legacy/exemplars)
and a list of stages, each a registered function

    stage(ctx, tiles, **params) -> tiles

run in order on the output grid (the first stage receives None).  After every
stage the pipeline records the same metrics (violations, connectivity,
structures, cells changed, time), so stages can be swapped and compared.

    {"name": "castle", "tileset": "demo", "exemplar": "castle",
     "size": 128, "seed": 0,
     "stages": [
       {"stage": "synthesize", "kappa": 4.0, "r": [1, 1, 0, 0, 0, 0, 0]},
       {"stage": "repair", "rounds": 20, "T": 0.4, "tau": 1.9},
       {"stage": "connect", "mode": "all", "K": 16, "r": 4, "max_cost": 4, "tag": "hallway"}
     ]}

Stages: synthesize, repair, relax (plain Gibbs sweeps), connect, ground
(paint the side-view ground band), and the promise stages hier (coordinates
+ the listed promises, connectivity, flow and/or support, through hier.run),
tiles (E[S]) and fulfil (per K-block contract enforcement, quantity by
quantity), see castlegen/legacy/pipelines/castle_promise.json, wilds_promise.json
and cliffs_promise.json.  Add one with @stage("name").  The exemplar is built once per layout content and
cached under cache/, keyed by the layout and tile set contents (--rebuild
forces a fresh build).

    python -m castlegen.legacy.pipeline castle --seed 1 --png images/castle.png
    python -m castlegen.legacy.pipeline wilds --size 256 --stage-pngs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

from castlegen.legacy import connmetrics as cm
from castlegen.legacy import exemplar as ex
from castlegen.legacy import hier, texsyn, tileset

PIPELINE_DIR = os.path.join(os.path.dirname(__file__), "pipelines")
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "cache")
STAGES = {}


def stage(name):
    def reg(fn):
        STAGES[name] = fn
        return fn
    return reg


@dataclass
class Context:
    ts: object
    exemplar: np.ndarray
    seed: int
    size: int
    torus: bool = True
    cache: dict = field(default_factory=dict)      # per-run scratch shared by stages (e.g. the coord map)
    layout: dict = None                            # the exemplar layout (the promise corpus samples with it)

    @property
    def ground(self):
        """Side view: rows of the ground band at the bottom (layout "ground", 0 if none)."""
        return int((self.layout or {}).get("ground", 0))

    def analysis(self, **kw):
        key = ("analysis", tuple(sorted(kw.items())))
        if key not in self.cache:
            self.cache[key] = texsyn.Analysis(self.ts, self.exemplar, **kw)
        return self.cache[key]


# ---------------------------------------------------------------------- stages
@stage("synthesize")
def _synthesize(ctx, tiles, kappa=4.0, r=(1, 1, 0, 0, 0, 0, 0), corrections=2, n_pca=32, w_sock=0.5):
    """Texture synthesis from the exemplar (texsyn); keeps the coordinate map
    in ctx.cache["S"] for seam metrics."""
    an = ctx.analysis(n_pca=n_pca, w_sock=w_sock)
    r = r if np.ndim(r) == 0 else list(r) + [0.0] * (an.L + 1 - len(r))
    S = texsyn.synthesize(an, ctx.size, r, seed=ctx.seed, corrections=corrections, kappa=kappa)
    ctx.cache["S"] = S
    return an.E[S[..., 0], S[..., 1]]


_FIT_CACHE = {}                                   # fitted predictors, per process
QDEFAULTS = {"connectivity": {}, "flow": {}, "support": {}, "envelope": {}, "average": {}}


def _promise(name, K, h_top, tab, T, sweeps, top_sweeps, extra):
    from castlegen.legacy.quantities.connectivity import ConnectivityPromise
    from castlegen.legacy.quantities.flow import FlowPromise
    if name == "connectivity":
        return ConnectivityPromise(K, h_top, tab, T=T, sweeps=sweeps, top_sweeps=top_sweeps,
                                   **{k: extra[k] for k in ("tag", "cross_socket", "void") if k in extra})
    if name == "flow":
        return FlowPromise(K, h_top, tab, T=T, sweeps=sweeps, top_sweeps=top_sweeps)
    if name == "support":
        from castlegen.legacy.quantities.support import SupportPromise
        return SupportPromise(K, h_top, tab, T=T, sweeps=sweeps, top_sweeps=top_sweeps, ground=extra.get("ground", 0),
                              I=extra.get("I", 2))
    if name == "average":
        from castlegen.legacy.average_var import AveragePromise
        clamps = {(int(h), int(y), int(x)): int(v) for h, y, x, v in extra.get("clamps", [])}
        return AveragePromise(K, h_top, tab, T=T, sweeps=sweeps, top_sweeps=top_sweeps, ground=extra.get("ground", 0),
                              clamps=clamps or None)
    if name == "envelope":
        from castlegen.legacy.envelope_var import EnvelopePromise
        clamps = {(int(h), int(y), int(x)): int(v) for h, y, x, v in extra.get("clamps", [])}
        return EnvelopePromise(K, h_top, tab, T=T, sweeps=sweeps, top_sweeps=top_sweeps, ground=extra.get("ground", 0),
                               I=extra.get("I", 2), Q=extra.get("Q", 2), clamps=clamps or None)
    raise KeyError(f"unknown quantity {name!r}")


@stage("hier")
def _hier(ctx, tiles, K=16, h_top=64, tables="corpus", corpus_seeds=24, corpus_size=128, alpha=0.5, lam=1.0,
          T=1.0, sweeps=3, top_sweeps=20, kappa=4.0, r=(1, 1, 0, 0, 0, 0, 0), corrections=2, first_corrected=3,
          coord_T=0.0, n_pca=32, w_sock=0.5, quantities=("connectivity",), corpus_source="p", bounds=None,
          **per_quantity):
    """The level sampler (hier.run) with the coordinate variable, one promise
    variable per entry of `quantities` (K <= h <= h_top; "connectivity",
    "flow") and a coordinate <-> promise coupling per promise (weight lam).
    Per-quantity overrides go in a dict named after the quantity, e.g.
    "connectivity": {"tag": "road", "cross_socket": "land", "void": "lake",
    "lam": 1.0, "T": 1.0}.  tables: "corpus" (castlegen.legacy.corpus, cached; the
    corpus is made valid for every listed quantity and sampled from
    `corpus_source`, "p" or "synth") or "flat".  Keeps every level in
    ctx.cache["levels"], the finest coordinate map in ctx.cache["S"] and the
    promise variables (in fulfil order) in ctx.cache["promises"].  bounds: None
    (toroidal exemplar and map) or "edge" (texsyn.Analysis: the exemplar
    continues its top / bottom rows forever, the map has sky above and ground
    below).  Returns E[S]."""
    from castlegen.legacy import corpus
    from castlegen.legacy.promise import CoordPromiseCoupling, Tables
    an = ctx.analysis(n_pca=n_pca, w_sock=w_sock, **({"bounds": bounds} if bounds else {}))
    assert not bounds or set(quantities) <= {"average"}, "bounded coordinates are wired into the average promise only"
    r = r if np.ndim(r) == 0 else list(r) + [0.0] * (an.L + 1 - len(r))
    coord = texsyn.CoordVar(an, r, kappa=kappa, corrections=corrections, first_corrected=first_corrected, T=coord_T)
    coord.h_max = min(coord.h_max, ctx.size)       # an exemplar larger than the map: start at one map-sized cell
    seeds = tuple(range(1000, 1000 + corpus_seeds))
    proms, couplings = [], []
    for q in quantities:
        qp = dict(QDEFAULTS[q], **per_quantity.get(q, {}))
        if q in ("support", "envelope", "average"):
            qp.setdefault("ground", ctx.ground)
        if q == "average":
            from castlegen.legacy import average_var
            from castlegen.legacy.quantities import average as AV
            lgA = AV.lang(K, ctx.ground)
            if tables == "flat":
                tab = Tables.flat(lgA.V, [K * 2 ** j for j in range(8)])
            elif tables == "linear":       # linear predictor (castlegen.legacy.envpredict) on corpus_path
                key = ("avg-linear", K, h_top, qp["corpus_path"], qp.get("l2", 1e-2))
                if key not in _FIT_CACHE:
                    _FIT_CACHE[key] = average_var.linear_tables(ctx.ts, np.load(qp["corpus_path"]), K, ctx.ground, h_top,
                                                                qp.get("l2", 1e-2), lgA, log=lambda *a: None)
                tab = _FIT_CACHE[key]
            else:                          # counts fitted to a corpus of valid maps (corpus_path)
                tab = average_var.tables(ctx.ts, np.load(qp["corpus_path"]), K, ctx.ground, h_top, alpha, lgA)
        elif q == "envelope":
            from castlegen.legacy import refheights
            from castlegen.legacy.quantities import envelope as EN
            lg = EN.lang(qp.get("I", 2), qp.get("Q", 2))
            if tables == "flat":
                tab = Tables.flat(lg.V, [K * 2 ** j for j in range(8)])
            elif tables == "linear":                # linear predictor (castlegen.legacy.envpredict) on the reference corpus
                from castlegen.legacy import envpredict
                ref = dict(qp.get("reference", {}))
                key = ("envpredict", K, h_top, json.dumps(ref, sort_keys=True), qp.get("corpus_path"))
                if key not in _FIT_CACHE:
                    maps = (np.load(qp["corpus_path"]) if qp.get("corpus_path") else
                            refheights.corpus(ctx.ts, n=corpus_size, seeds=seeds, ground=ctx.ground, **ref))
                    _FIT_CACHE[key] = envpredict.tables(ctx.ts, maps, K, ctx.ground, h_top, lg=lg, log=lambda *a: None)
                tab = _FIT_CACHE[key]
            else:                          # fitted to the exact p_G reference corpus (castlegen.legacy.refheights)
                ref = dict(qp.get("reference", {}))
                maps = np.load(qp["corpus_path"]) if qp.get("corpus_path") else None     # any (N, n, n) valid maps
                tab = refheights.tables(ctx.ts, maps, K=K, ground=ctx.ground, h_top=h_top, alpha=alpha, lg=lg,
                                        n=corpus_size, seeds=seeds, **ref)
        elif tables == "corpus":
            tab = corpus.tables_for(ctx.ts, ctx.layout, K, corpus_size, seeds, alpha, quantity=q, source=corpus_source,
                                    quantities=quantities)[0]
        else:
            V = 4 ** qp.get("I", 2) if q == "support" else corpus.QUANTITIES[q].V
            tab = Tables.flat(V, [K * 2 ** j for j in range(8)])
        prom = _promise(q, K, h_top, tab, qp.get("T", T), qp.get("sweeps", sweeps), qp.get("top_sweeps", top_sweeps), qp)
        proms.append(prom)
        lam_q = qp.get("lam", lam)
        if lam_q:
            key = ("window_costs", q, K, h_top, qp.get("chi_path"))
            if key not in ctx.cache:
                hs = [h for h in (K * 2 ** j for j in range(8)) if h <= min(h_top, an.m)]
                wkw = {"bounded": True} if an.bounded else {}
                ctx.cache[key] = {h: prom.window_costs(ctx.ts, ctx.exemplar, h, **wkw) for h in hs}
                if qp.get("chi_path"):                  # learned chi tables {h: (m*m, V)} override the distance cost
                    z = np.load(qp["chi_path"])
                    ctx.cache[key].update({int(k[1:]): z[k] for k in z.files})
            couplings.append(CoordPromiseCoupling(coord, prom, ctx.cache[key], lam_q, an.m, an.bounded))
    hctx = hier.Ctx(ctx.seed)
    S = hier.run(proms + [coord], couplings, ctx.size, hctx).vars[coord.name]
    ctx.cache.update(levels=hctx.levels, S=S, promises=proms)
    return an.E[S[..., 0], S[..., 1]]


@stage("tiles")
def _tiles(ctx, tiles, phase=1):
    """The tile level.  Phase 1: E[S] from the coordinate map of `hier`."""
    assert phase == 1, "phase 2 (tilted Gibbs) is not implemented"
    S = ctx.cache["S"]
    return an.E[S[..., 0], S[..., 1]]


@stage("fulfil")
def _fulfil(ctx, tiles, protect_nodes=False):
    """Every promise variable's `fulfil` on every K-block (blocks that
    already keep their contract are left alone).  Edited cells and crossing
    cells (and with protect_nodes, every node cell) are added to
    ctx.cache["protect"]; per-block edit counts go to ctx.cache["fulfil"]."""
    tiles = np.array(tiles, np.int32)
    protect = ctx.cache.get("protect")
    protect = np.zeros(tiles.shape, bool) if protect is None else protect.copy()
    edits, per_q = [], {}
    keep = None                                    # what earlier quantities need kept (flow: water, structures)
    for v in ctx.cache["promises"]:
        if getattr(v, "chain", False):             # exact tile level instead (stage "heights")
            continue
        K = v.K
        O = next(lv.vars[v.name] for lv in ctx.cache["levels"] if lv.h == K and v.name in lv.vars)
        qe = []
        for by, bx in np.ndindex(*O.shape):
            kw = {} if keep is None else {"protect": keep}
            new, e, cross = v.fulfil(ctx.ts, tiles, by * K, bx * K, K, int(O[by, bx]), seed=ctx.seed, **kw)
            protect |= new != tiles
            for y, x in cross:
                protect[y, x] = True
            tiles = new
            edits.append(e)
            qe.append(e)
        per_q[v.name] = int(max(qe))
        if hasattr(v, "protect_mask"):
            m = v.protect_mask(ctx.ts, tiles)
            keep = m if keep is None else keep | m
    if protect_nodes:
        protect |= cm._node(ctx.ts)[tiles]
    ctx.cache["protect"] = protect
    ctx.cache.setdefault("fulfil", []).append(dict(max_edits=int(max(edits)), mean_edits=float(np.mean(edits)),
                                                   blocks_edited=float(np.mean(np.array(edits) > 0)),
                                                   max_edits_by=per_q))
    return tiles


@stage("heights")
def _heights(ctx, tiles, T=1.0, w_tex=1.0, sweeps=10, ground_tile="stone", **kw):
    """The tile level of the envelope promise (castlegen.legacy.heights): project
    E[S_1] onto the maps whose K-blocks keep the level-K promise grid, then
    constrained MCMC on the exact tile conditional.  No fulfil, no repair."""
    from castlegen.legacy import heights
    v = next(p for p in ctx.cache["promises"] if p.name == "envelope")
    P = next(lv.vars[v.name] for lv in ctx.cache["levels"] if lv.h == v.K and v.name in lv.vars)
    out, st = heights.sample_tiles(ctx.ts, np.asarray(tiles, np.int32), P, v.K, ctx.ground, I=v.lang.I, Q=v.lang.Q,
                                   seed=ctx.seed, T=T, w_tex=w_tex, sweeps=sweeps, ground_tile=ground_tile, **kw)
    ctx.cache.setdefault("heights", []).append(st)
    return out


@stage("exheights")
def _exheights(ctx, tiles, w, T=0.9, sweeps=50):
    """Tile level for the exemplar-defined target (castlegen.legacy.exchain): the
    level-K promises honoured exactly, tile energy plus exemplar patch energy
    with weights w {h: weight}."""
    from castlegen.legacy import exchain
    v = next(p for p in ctx.cache["promises"] if p.name == "envelope")
    P = next(lv.vars[v.name] for lv in ctx.cache["levels"] if lv.h == v.K and v.name in lv.vars)
    return exchain.tile_stage(ctx.ts, np.asarray(tiles, np.int32), P, v.K, ctx.ground, ctx.exemplar,
                              {int(h): float(x) for h, x in w.items()}, T, sweeps, ctx.seed, v.lang.I, v.lang.Q)


@stage("avgexheights")
def _avgexheights(ctx, tiles, w, T=0.9, sweeps=50, feat="solid"):
    """Tile level of the average-height promise for the exemplar-defined
    target (castlegen.legacy.average_var.tile_stage).  feat: the patch energy's
    features, "solid" or "tiles" (castlegen.legacy.exchain)."""
    from castlegen.legacy import average_var
    v = next(p for p in ctx.cache["promises"] if p.name == "average")
    P = next(lv.vars[v.name] for lv in ctx.cache["levels"] if lv.h == v.K and v.name in lv.vars)
    return average_var.tile_stage(ctx.ts, np.asarray(tiles, np.int32), P, v.K, ctx.ground, ctx.exemplar,
                                  {int(h): float(x) for h, x in w.items()}, v.lang, T, sweeps, ctx.seed, feat)


@stage("ground")
def _ground(ctx, tiles, tile="stone", rows=None):
    """Side view: paint the bottom `rows` (default the layout's "ground")
    rows with `tile` and protect them from later repairs."""
    G = ctx.ground if rows is None else rows
    tiles = np.array(tiles, np.int32)
    if G:
        tiles[-G:] = ex.sig_by_name(ctx.ts, tile)
        band = np.zeros(tiles.shape, bool)
        band[-G:] = True
        protect = ctx.cache.get("protect")
        ctx.cache["protect"] = band if protect is None else protect | band
    return tiles


@stage("repair")
def _repair(ctx, tiles, rounds=20, T=0.4, tau=1.9, radius=1, clear_orphans=True, no_new_nodes=False,
            no_new_water=False, keep_solidity=False):
    """Masked Gibbs on socket violations (exemplar.repair); cells an earlier
    stage put in ctx.cache["protect"] are held.  no_new_nodes: never propose
    a node signature (with every node protected, repair then cannot change
    connectivity).  no_new_water: hold every water cell and never propose
    water (repair then cannot change the flow).  keep_solidity: a cell only
    moves between signatures of its own solidity (ts.solid), so repair cannot
    change support."""
    keep_class = ctx.ts.solid.astype(np.int32) if keep_solidity else None
    protect = ctx.cache.get("protect")
    forbid = cm._node(ctx.ts) if no_new_nodes else None
    if no_new_water:
        wet = ctx.ts.water[tiles]
        protect = wet if protect is None else protect | wet
        forbid = ctx.ts.water if forbid is None else forbid | ctx.ts.water
    if clear_orphans and protect is not None:           # exemplar.repair would clear protected orphans too
        tiles = np.where(ex.orphan_cells(ctx.ts, tiles) & ~protect, ctx.ts.WALL, tiles).astype(np.int32)
        clear_orphans = False
    return ex.repair(ctx.ts, tiles, rounds, ctx.seed, T=T, tau=tau, radius=radius, torus=ctx.torus,
                     clear_orphans=clear_orphans, update=None if protect is None else ~protect, forbid=forbid,
                     keep_class=keep_class)


@stage("relax")
def _relax(ctx, tiles, sweeps=5, T=0.6):
    """Plain heat-bath sweeps over the whole grid."""
    return ex.gibbs(ctx.ts, tiles, sweeps, ctx.seed + 17, T=T, torus=ctx.torus)


@stage("connect")
def _connect(ctx, tiles, mode="all", K=16, r=4, max_cost=4, tag="hallway", w_energy=1.0, passes=1):
    """Bounded windowed connectivity repair (exemplar.local_connect)."""
    for p in range(passes):
        tiles, st = ex.local_connect(ctx.ts, tiles, K=K, r=r, max_cost=max_cost, mode=mode,
                                     seed=ctx.seed + 101 * p, w_energy=w_energy, tag=tag)
        ctx.cache.setdefault("connect", []).append(st)
    return tiles


# -------------------------------------------------------------------- exemplar
def exemplar_for(layout, ts, rebuild=False, log=print):
    """Build (or load the cached) exemplar for a layout dict.  The cache key
    covers the layout and the tile set's spec file, so editing either rebuilds."""
    spec_path = os.path.join(tileset.TILESET_DIR, layout.get("tileset", "demo") + ".json")
    spec_text = open(spec_path).read() if os.path.exists(spec_path) else ts.name
    key = hashlib.sha1((json.dumps(layout, sort_keys=True) + spec_text).encode()).hexdigest()[:12]
    path = os.path.join(CACHE_DIR, f"exemplar-{layout.get('name', 'layout')}-{key}.npy")
    if os.path.exists(path) and not rebuild:
        return np.load(path)
    if layout.get("npy"):                          # a pre-built exemplar (e.g. exemplars/make_cliffs_big.py)
        return np.load(os.path.join(os.path.dirname(ex.__file__), "exemplars", layout["npy"])).astype(np.int32)
    t0 = time.time()
    _, E, _ = ex.build(layout, ts)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(path, E)
    log(f"built exemplar {path} in {time.time() - t0:.1f}s")
    return E


# ---------------------------------------------------------------------- driver
def load_spec(name):
    path = name if os.path.exists(name) else os.path.join(PIPELINE_DIR, name + ".json")
    with open(path) as f:
        return json.load(f)


def metrics(ts, tiles, torus=True, ground=0):
    g = cm.global_stats(ts, tiles, torus)
    lab, _ = cm.labels(ts, tiles, torus)
    sizes = np.bincount(lab[lab >= 0]) if (lab >= 0).any() else np.zeros(1)
    census = tileset.structure_census(ts, tiles)
    out = dict(bad=float(ex.bad_cells(ts, tiles, 1.9, torus).mean()),
               comps_per_1k=1000.0 * g["components"] / tiles.size,
               main_share=float(sizes.max() / max(sizes.sum(), 1)),
               rooms=g["node_frac"],
               structures={k: v[0] for k, v in census.items()},
               orphans=sum(v[1] for v in census.values()))
    from castlegen.legacy.quantities import flow as F
    if F.has_flow(ts):
        out.update(F.flow_violations(ts, tiles, torus))
        out["water"] = float(ts.water[tiles].mean())
    from castlegen.legacy.quantities import support as SP
    if SP.has_support(ts):
        out["unsupported"] = SP.support_violations(ts, tiles, ground)
        out["solid"] = float(ts.solid[tiles].mean())
    return out


def run(spec, seed=None, size=None, rebuild=False, log=print, on_stage=None, ctx_out=None):
    """-> (tiles, report).  report: one row of metrics per stage.
    on_stage(name, tiles) is called after every stage (e.g. to save images).
    ctx_out: a list that receives the run's Context."""
    spec = dict(spec)
    ts = tileset.load(spec.get("tileset", "demo"))
    layout = spec["exemplar"]
    layout = ex.load_layout(layout) if isinstance(layout, str) else layout
    layout = dict(layout, name=layout.get("name", spec["exemplar"] if isinstance(spec["exemplar"], str) else "inline"))
    E = exemplar_for(layout, ts, rebuild, log)
    ctx = Context(ts, E, spec.get("seed", 0) if seed is None else seed, spec.get("size", 128) if size is None else size,
                  layout.get("torus", True), layout=layout)
    if ctx_out is not None:
        ctx_out.append(ctx)
    report = [dict(stage="exemplar", **metrics(ts, E, ctx.torus, ctx.ground))]
    tiles = None
    for st in spec["stages"]:
        params = {k: v for k, v in st.items() if k != "stage"}
        t0 = time.time()
        new = np.asarray(STAGES[st["stage"]](ctx, tiles, **params), np.int32)
        row = dict(stage=st["stage"], seconds=time.time() - t0, **metrics(ts, new, ctx.torus, ctx.ground))
        row["changed"] = float((new != tiles).mean()) if tiles is not None else 1.0
        if st["stage"] == "synthesize":
            row["seams"] = float(texsyn.seam_mask(ctx.cache["S"], E.shape[0]).mean())
        if st["stage"] == "connect":
            row["max_window_edits"] = max(s["max_edits"] for s in ctx.cache["connect"])
        if st["stage"] == "fulfil":
            row.update(ctx.cache["fulfil"][-1])
        if "promises" in ctx.cache:
            row["sat"] = {f"{v.name}{h}": f for v in ctx.cache["promises"]
                          for h, f in v.satisfaction(ts, new, ctx.cache["levels"]).items()}
        report.append(row)
        tiles = new
        if on_stage:
            on_stage(st["stage"], tiles)
    return ts, tiles, report


def format_report(report):
    cols = ["stage", "seconds", "bad", "comps_per_1k", "main_share", "changed", "orphans", "structures"]
    cols += [c for c in ("water", "dead_ends", "unfed", "solid", "unsupported") if any(c in r for r in report)]
    cols += [c for c in ("max_edits", "mean_edits") if any(c in r for r in report)]
    lines = ["  ".join(f"{c:>13s}" for c in cols)]
    for r in report:
        vals = []
        for c in cols:
            v = r.get(c, "")
            vals.append(f"{v:13.3f}" if isinstance(v, float) else f"{str(v):>13s}")
        lines.append("  ".join(vals))
    sat = [(r["stage"], r["sat"]) for r in report if "sat" in r]
    if sat:
        lines.append("promise satisfaction (fraction of blocks per level):")
        for name, d in sat:
            lines.append(f"  {name:>11s}  " + "  ".join(f"{k} {v:.3f}" for k, v in d.items()))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("spec", help="pipeline name (castlegen/legacy/pipelines) or JSON path")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--png", default=None, help="final image (default images/<name>_s<seed>.png)")
    ap.add_argument("--px", type=int, default=8)
    ap.add_argument("--stage-pngs", action="store_true", help="also save an image after every stage")
    ap.add_argument("--out", default=None, help="save the final signature grid (.npy)")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the exemplar even if cached")
    args = ap.parse_args()
    spec = load_spec(args.spec)
    name = spec.get("name", os.path.splitext(os.path.basename(args.spec))[0])
    seed = spec.get("seed", 0) if args.seed is None else args.seed
    png = args.png or os.path.join("images", f"{name}_s{seed}.png")
    os.makedirs(os.path.dirname(png) or ".", exist_ok=True)
    k = [0]

    def on_stage(st, tiles):
        if args.stage_pngs:
            k[0] += 1
            ex.to_png(ts_[0], tiles, png.replace(".png", f"_{k[0]}{st}.png"), args.px)
    ts_ = [tileset.load(spec.get("tileset", "demo"))]
    ts, tiles, report = run(spec, seed, args.size, args.rebuild, on_stage=on_stage)
    print(format_report(report))
    ex.to_png(ts, tiles, png, args.px)
    print("wrote", png)
    if args.out:
        np.save(args.out, tiles)


if __name__ == "__main__":
    main()
