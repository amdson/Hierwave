"""Connectivity metrics for exemplars and synthesized grids (torus-aware).

Nodes are room cells and the gate (as in castlegen.conn); two edge-adjacent
nodes are joined when their facing sockets connect (Dh / Dv).  The main
component is the gate's, or the largest one when there is no gate (toroidal
exemplars).

  global      components, unreached components and cells
  blocks(K)   the local rule at block size K ("option 2"): a block is
              connected by its own cells; a neighbouring block pair has a door
              crossing; a side is open when some node on it has a non-wall
              socket facing out (somewhere a neighbour could connect).
              Enclosed = block-local components touching no block edge.
  repair cost per unreached component, the min number of cells a repair must
              change to join the main component: 0-1 shortest path where
              entering a cell through a door is free and anything else costs 1
  seams       seam edges by type, and whether unreached components touch one
  bias        E per cell, kind fractions (total variation to a reference),
              door density, split into seam / non-seam cells when S is given

    python -m castlegen.connmetrics images/ex_seed.npy
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from castlegen import tileset


def _node(ts):
    node = np.array(ts.np_tables["is_room"], bool)
    node[ts.GATE] = True
    return node


def _pairs(ts, tiles, torus):
    """[(a, b, joined, horizontal)] over right and down neighbour pairs, as flat indices."""
    H, W = tiles.shape
    idx = np.arange(H * W).reshape(H, W)
    node = _node(ts)[tiles]
    out = []
    for horiz, D in ((True, ts.np_tables["Dh"]), (False, ts.np_tables["Dv"])):
        ax = 1 if horiz else 0
        if torus:
            a, b = idx, np.roll(idx, -1, ax)
            ta, tb, na, nb = tiles, np.roll(tiles, -1, ax), node, np.roll(node, -1, ax)
        else:
            sl_a = (slice(None), slice(None, -1)) if horiz else (slice(None, -1), slice(None))
            sl_b = (slice(None), slice(1, None)) if horiz else (slice(1, None), slice(None))
            a, b, ta, tb, na, nb = idx[sl_a], idx[sl_b], tiles[sl_a], tiles[sl_b], node[sl_a], node[sl_b]
        joined = na & nb & D[ta, tb].astype(bool)
        out.append((a.ravel(), b.ravel(), joined.ravel(), horiz))
    return out


def labels(ts, tiles, torus=True, block=None):
    """-> (label grid, -1 off nodes; number of components).  block: cut every
    join that crosses a block boundary of that size (block-local components)."""
    tiles = np.asarray(tiles)
    H, W = tiles.shape
    node = _node(ts)[tiles].ravel()
    rows, cols = [], []
    for a, b, j, horiz in _pairs(ts, tiles, torus):
        if block:
            ca = a % W if horiz else a // W
            j = j & ((ca + 1) % block != 0)
        rows.append(a[j]); cols.append(b[j])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(H * W, H * W))
    _, lab = connected_components(g, directed=False)
    lab = np.where(node, lab, -1)
    uniq, lab_c = np.unique(lab[node], return_inverse=True)
    out = -np.ones(H * W, int)
    out[node] = lab_c
    return out.reshape(H, W), len(uniq)


def main_label(ts, tiles, lab, n):
    gate = np.asarray(tiles) == ts.GATE
    if gate.any():
        return int(lab[gate][0])
    return int(np.argmax(np.bincount(lab[lab >= 0], minlength=n))) if n else -1


def global_stats(ts, tiles, torus=True):
    lab, n = labels(ts, tiles, torus)
    m = main_label(ts, tiles, lab, n)
    sizes = np.bincount(lab[lab >= 0], minlength=n)
    return dict(components=n, unreached=max(n - 1, 0),
                unreached_cells=float((sizes.sum() - (sizes[m] if n else 0)) / max(sizes.sum(), 1)),
                node_frac=float((lab >= 0).mean()))


def block_stats(ts, tiles, K, torus=True):
    """The local rule at block size K (grid side must be a multiple of K)."""
    tiles = np.asarray(tiles)
    H, W = tiles.shape
    nby, nbx = H // K, W // K
    lab, n = labels(ts, tiles, torus, block=K)
    ys, xs = np.nonzero(lab >= 0)
    l = lab[ys, xs]
    bid = (ys // K) * nbx + xs // K
    on_edge = (ys % K == 0) | (ys % K == K - 1) | (xs % K == 0) | (xs % K == K - 1)
    touches = np.zeros(n, bool)
    np.logical_or.at(touches, l, on_edge)
    comp_block = np.zeros(n, int)
    comp_block[l] = bid
    per_block = np.bincount(comp_block, minlength=nby * nbx)
    enclosed = np.bincount(comp_block[~touches], minlength=nby * nbx)
    # crossings between neighbouring blocks
    cross_r = np.zeros((nby, nbx), bool); cross_d = np.zeros((nby, nbx), bool)
    for a, b, j, horiz in _pairs(ts, tiles, torus):
        ay, ax_ = a // W, a % W
        c = (ax_ % K == K - 1) if horiz else (ay % K == K - 1)
        hit = j & c
        (cross_r if horiz else cross_d)[ay[hit] // K, ax_[hit] // K] = True
    if not torus:
        cross_r, cross_d = cross_r[:, :-1], cross_d[:-1, :]
    # open sides: a node on the side with a non-wall socket facing out
    wall = ts.sockets.index("wall")
    node = _node(ts)[tiles]
    so = ts.sig_sockets[tiles]
    open_side = np.zeros((nby, nbx, 4), bool)
    edge_masks = (np.arange(H)[:, None] % K == 0, np.arange(W)[None, :] % K == K - 1,
                  np.arange(H)[:, None] % K == K - 1, np.arange(W)[None, :] % K == 0)
    for d in range(4):
        m = np.broadcast_to(edge_masks[d], (H, W)) & node & (so[..., d] != wall)
        yy, xx = np.nonzero(m)
        open_side[yy // K, xx // K, d] = True
    conn = (per_block <= 1).reshape(nby, nbx)
    ok = conn.copy()
    if torus:                                            # all four crossings present
        ok &= cross_r & cross_d & np.roll(cross_r, 1, 1) & np.roll(cross_d, 1, 0)
    return dict(K=K, blocks=nby * nbx,
                connected=float(conn.mean()),
                enclosed_per_block=float(enclosed.mean()),
                blocks_with_enclosed=float((enclosed > 0).mean()),
                crossing=float(np.concatenate([cross_r.ravel(), cross_d.ravel()]).mean()),
                closed_sides=float((~open_side).mean()),
                ok=float(ok.mean()))


def window_stats(ts, E, K, stride=4):
    """block_stats averaged over K x K windows of a toroidal exemplar at every
    offset on a stride grid (what synthesis could copy)."""
    acc = []
    for oy in range(0, K, stride):
        for ox in range(0, K, stride):
            acc.append(block_stats(ts, np.roll(E, (-oy, -ox), (0, 1)), K))
    return {k: float(np.mean([a[k] for a in acc])) if k != "K" else K for k in acc[0]}


def repair_cost(ts, tiles, torus=True):
    """-> (cost, size) per unreached component: min cells to change to reach the main component."""
    tiles = np.asarray(tiles)
    H, W = tiles.shape
    lab, n = labels(ts, tiles, torus)
    m = main_label(ts, tiles, lab, n)
    if n <= 1:
        return np.zeros(0, int), np.zeros(0, int)
    rows, cols, w = [], [], []
    for a, b, j, _ in _pairs(ts, tiles, torus):
        c = np.where(j, 1e-6, 1.0)
        rows += [a, b]; cols += [b, a]; w += [c, c]            # weight = cost of entering the target
    g = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(H * W, H * W)).tocsr()
    dist = dijkstra(g, directed=True, indices=np.flatnonzero(lab.ravel() == m), min_only=True)
    others = [c for c in range(n) if c != m]
    flat = lab.ravel()
    cost = np.array([int(round(dist[flat == c].min())) for c in others])
    size = np.bincount(flat[flat >= 0], minlength=n)[others]
    return cost, size


def seam_stats(ts, tiles, S, m, torus=True):
    """Seam edges by type, and the fraction of unreached components/cells touching a seam."""
    tiles = np.asarray(tiles)
    H, W = tiles.shape
    node = _node(ts)[tiles].ravel()
    Sf = S.reshape(-1, 2)
    types = dict(ff_joined=0, ff_cut=0, fw=0, ww=0)
    seam_cell = np.zeros(H * W, bool)
    n_seam = 0
    for a, b, j, horiz in _pairs(ts, tiles, torus):
        step = np.array([0, 1]) if horiz else np.array([1, 0])
        seam = ((Sf[b] - Sf[a]) % m != step).any(1)
        n_seam += seam.sum()
        seam_cell[a[seam]] = True; seam_cell[b[seam]] = True
        na, nb = node[a], node[b]
        types["ff_joined"] += (seam & j).sum()
        types["ff_cut"] += (seam & na & nb & ~j).sum()
        types["fw"] += (seam & (na ^ nb)).sum()
        types["ww"] += (seam & ~na & ~nb).sum()
    lab, n = labels(ts, tiles, torus)
    mm = main_label(ts, tiles, lab, n)
    flat = lab.ravel()
    un = (flat >= 0) & (flat != mm)
    touch = np.zeros(n, bool)
    np.logical_or.at(touch, flat[un], seam_cell[un])
    others = np.array([c for c in range(n) if c != mm], int)
    out = {k: float(v / max(n_seam, 1)) for k, v in types.items()}
    out.update(seam_edges=int(n_seam), unreached_touching_seam=float(touch[others].mean()) if len(others) else 0.0)
    return out


def bias(ts, tiles, ref, mask=None):
    """E per cell, door density and kind fractions vs a reference grid; mask
    restricts the per-cell statistics of `tiles` (e.g. seam cells)."""
    tiles, ref = np.asarray(tiles), np.asarray(ref)
    mask = np.ones(tiles.shape, bool) if mask is None else mask
    Eh, Ev, logz = ts.np_tables["Eh"], ts.np_tables["Ev"], ts.np_tables["logz"]
    Dh, Dv = ts.np_tables["Dh"], ts.np_tables["Dv"]

    def cellwise(t):
        r, d = np.roll(t, -1, 1), np.roll(t, -1, 0)
        e = Eh[t, r] + Ev[t, d] - logz[t]
        door = Dh[t, r] + Dv[t, d]
        return e, door

    K = len(ts.kinds)
    e, door = cellwise(tiles)
    er, doorr = cellwise(ref)
    kf = np.bincount(ts.sig_kind[tiles[mask]], minlength=K) / max(mask.sum(), 1)
    kr = np.bincount(ts.sig_kind[ref].ravel(), minlength=K) / ref.size
    return dict(E=float(e[mask].mean()), E_ref=float(er.mean()),
                doors=float(door[mask].mean()), doors_ref=float(doorr.mean()),
                kind_tv=float(0.5 * np.abs(kf - kr).sum()))


def report(ts, tiles, torus=True, Ks=(8, 32), S=None, m=None, ref=None):
    out = dict(global_stats(ts, tiles, torus))
    for K in Ks:
        if tiles.shape[0] % K == 0:
            out[f"block{K}"] = block_stats(ts, tiles, K, torus)
    cost, size = repair_cost(ts, tiles, torus)
    out["repair"] = dict(n=len(cost), cheap=int((cost <= 2).sum()), expensive=int((cost >= 3).sum()),
                         total_cost=int(cost.sum()), hist={int(c): int((cost == c).sum()) for c in np.unique(cost)},
                         median_size=float(np.median(size)) if len(size) else 0.0)
    if S is not None:
        out["seams"] = seam_stats(ts, tiles, S, m, torus)
    if ref is not None:
        out["bias"] = bias(ts, tiles, ref)
    return out


def fmt(r, indent=""):
    lines = []
    for k, v in r.items():
        if isinstance(v, dict) and k != "hist":
            lines.append(f"{indent}{k}:")
            lines.append(fmt(v, indent + "  "))
        else:
            lines.append(f"{indent}{k}: {v:.3f}" if isinstance(v, float) else f"{indent}{k}: {v}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid", help="signature grid .npy")
    ap.add_argument("--tileset", default="demo")
    ap.add_argument("--no-torus", action="store_true")
    ap.add_argument("--windows", type=int, nargs="*", default=[], help="also average K x K exemplar windows")
    args = ap.parse_args()
    ts = tileset.load(args.tileset)
    t = np.load(args.grid)
    print(fmt(report(ts, t, not args.no_torus)))
    for K in args.windows:
        print(f"windows{K}:")
        print(fmt(window_stats(ts, t, K), "  "))


if __name__ == "__main__":
    main()
