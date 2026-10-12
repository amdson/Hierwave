"""WFC on the city tile set (castlegen/citytiles.py): how often does an
unguided run contradict, and what guidance does it take to finish?

WFC: min-entropy cell (weighted Shannon, tiny noise), draw by tile weight,
arc-consistent propagation through sockets, open borders.  Three modes:
  plain         stop at the first contradiction
  plain+repair  on a contradiction un-collapse a small square about it
                (radius grows with repeated failures nearby) and resume,
                at most 50 times per run
  plan+repair   the same with a coarse district plan: ordered zone bands on
                the level sets of one smooth field of scale ~G (D|C|R|P|I,
                neighbours compatible, never three at a point) plus a river,
                multiplying a tile's weight by exp(beta * corners that agree
                with the plan).  Soft: the tiles still decide.

    NUMBA_NUM_THREADS=1 uv run python notes/experiments/city_wfc.py
"""
import os
import sys
import time

import numpy as np
from numba import njit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from castlegen import citytiles as ct  # noqa: E402

IMG = os.path.join(os.path.dirname(__file__), "..", "..", "images")
CLASSES = ["P", "R", "C", "D", "I", "W", "A"]


@njit(cache=True)
def _propagate(dom, cnt, sw, swl, weff, sN, sE, sS, sW, mark, stamp, stack, sp):
    """Arc consistency from the cells on the stack.  Returns (stamp, fy, fx);
    fy >= 0 is a cell whose domain emptied."""
    H, W, T = dom.shape
    while sp > 0:
        sp -= 1
        y = stack[sp, 0]
        x = stack[sp, 1]
        for d in range(4):
            ny = y + (-1 if d == 0 else 1 if d == 2 else 0)
            nx = x + (1 if d == 1 else -1 if d == 3 else 0)
            if ny < 0 or ny >= H or nx < 0 or nx >= W:
                continue
            stamp += 1
            for t in range(T):
                if dom[y, x, t]:
                    s = sN[t] if d == 0 else sE[t] if d == 1 else sS[t] if d == 2 else sW[t]
                    mark[s] = stamp
            changed = False
            for t in range(T):
                if dom[ny, nx, t]:
                    s = sS[t] if d == 0 else sW[t] if d == 1 else sN[t] if d == 2 else sE[t]
                    if mark[s] != stamp:
                        dom[ny, nx, t] = False
                        cnt[ny, nx] -= 1
                        v = weff[ny, nx, t]
                        sw[ny, nx] -= v
                        swl[ny, nx] -= v * np.log(v)
                        changed = True
            if changed:
                if cnt[ny, nx] == 0:
                    return stamp, ny, nx
                if sw[ny, nx] < 1e-300:
                    sw[ny, nx] = 1e-300
                stack[sp, 0] = ny
                stack[sp, 1] = nx
                sp += 1
    return stamp, -1, -1


@njit(cache=True)
def _rebuild(grid, dom, cnt, sw, swl, weff):
    """Domains from the collapsed cells alone: full where grid < 0."""
    H, W, T = dom.shape
    for y in range(H):
        for x in range(W):
            g = grid[y, x]
            sw[y, x] = 0.0
            swl[y, x] = 0.0
            for t in range(T):
                on = g < 0 or t == g
                dom[y, x, t] = on
                if on:
                    v = weff[y, x, t]
                    sw[y, x] += v
                    swl[y, x] += v * np.log(v)
            cnt[y, x] = T if g < 0 else 1


@njit(cache=True)
def wfc(sN, sE, sS, sW, nsock, w0, cls, zmap, beta, H, W, seed, repairs):
    """Returns (grid, completed, repairs used, last failure y, x).  Each
    contradiction un-collapses a square of radius 2 + (failures so far
    within 8 cells of it) // 2 about the failure and resumes, up to
    `repairs` times (0: plain WFC, stop at the first contradiction)."""
    np.random.seed(seed)
    T = w0.shape[0]
    dom = np.ones((H, W, T), np.bool_)
    cnt = np.full((H, W), T, np.int64)
    weff = np.empty((H, W, T))
    sw = np.zeros((H, W))
    swl = np.zeros((H, W))
    grid = np.full((H, W), -1, np.int64)
    cy = np.array([0, 0, 1, 1])
    cx = np.array([0, 1, 1, 0])
    for y in range(H):
        for x in range(W):
            for t in range(T):
                m = 0
                for k in range(4):
                    if zmap[y + cy[k], x + cx[k]] == cls[t, k]:
                        m += 1
                weff[y, x, t] = w0[t] * np.exp(beta * m)
    _rebuild(grid, dom, cnt, sw, swl, weff)
    mark = np.zeros(nsock, np.int64)
    stamp = 0
    stack = np.empty((H * W * 8, 2), np.int64)
    fails = np.empty((repairs + 1, 2), np.int64)
    used = 0
    while True:
        best = 1e300
        by = -1
        bx = -1
        for y in range(H):
            for x in range(W):
                if cnt[y, x] > 1:
                    e = np.log(sw[y, x]) - swl[y, x] / sw[y, x] + 1e-6 * np.random.random()
                    if e < best:
                        best = e
                        by = y
                        bx = x
        if by < 0:
            for y in range(H):
                for x in range(W):
                    if grid[y, x] < 0:
                        for t in range(T):
                            if dom[y, x, t]:
                                grid[y, x] = t
                                break
            return grid, True, used, -1, -1
        r = np.random.random() * sw[by, bx]
        pick = -1
        for t in range(T):
            if dom[by, bx, t]:
                r -= weff[by, bx, t]
                pick = t
                if r <= 0:
                    break
        for t in range(T):
            if dom[by, bx, t] and t != pick:
                dom[by, bx, t] = False
        grid[by, bx] = pick
        cnt[by, bx] = 1
        sw[by, bx] = weff[by, bx, pick]
        swl[by, bx] = sw[by, bx] * np.log(sw[by, bx])
        stack[0, 0] = by
        stack[0, 1] = bx
        stamp, fy, fx = _propagate(dom, cnt, sw, swl, weff, sN, sE, sS, sW, mark, stamp, stack, 1)
        while fy >= 0:
            fails[used, 0] = fy
            fails[used, 1] = fx
            if used == repairs:
                return grid, False, used, fy, fx
            near = 0
            for i in range(used):
                if abs(fails[i, 0] - fy) <= 8 and abs(fails[i, 1] - fx) <= 8:
                    near += 1
            used += 1
            R = 2 + near // 2
            for y in range(max(0, fy - R), min(H, fy + R + 1)):
                for x in range(max(0, fx - R), min(W, fx + R + 1)):
                    grid[y, x] = -1
            _rebuild(grid, dom, cnt, sw, swl, weff)
            sp = 0
            for y in range(H):
                for x in range(W):
                    if grid[y, x] >= 0:
                        stack[sp, 0] = y
                        stack[sp, 1] = x
                        sp += 1
            stamp, fy, fx = _propagate(dom, cnt, sw, swl, weff, sN, sE, sS, sW, mark, stamp, stack, sp)


def smooth_field(H, W, scale, rng):
    """Sum of random low-frequency cosines, roughly unit variance."""
    yy, xx = np.mgrid[0:H + 1, 0:W + 1]
    f = np.zeros((H + 1, W + 1))
    for _ in range(12):
        k = rng.normal(0, 1, 2) / scale
        f += np.cos(k[0] * yy + k[1] * xx + rng.uniform(0, 2 * np.pi))
    return f / np.sqrt(6)


BANDS = ["D", "C", "R", "P", "I"]     # a chain of compatible neighbours


def district_map(H, W, G, seed):
    """(H+1, W+1) zone-class ids: level sets of one smooth field cut into the
    ordered bands D | C | R | P | I (adjacent bands are compatible and three
    never meet), with a river band from a second field over it."""
    rng = np.random.default_rng(seed)
    f = smooth_field(H, W, G, rng)
    cuts = np.quantile(f, [0.08, 0.28, 0.62, 0.78])
    zm = np.array([CLASSES.index(b) for b in BANDS])[np.searchsorted(cuts, f)]
    g = smooth_field(H, W, 2 * G, rng)
    zm[np.abs(g) < 0.08] = CLASSES.index("W")
    return zm.astype(np.int64)


def run(ts, H, W, seed, guided=False, beta=1.5, G=12, repairs=0):
    """-> grid, completed, repairs used, failure cell, plan."""
    cls = np.array([[CLASSES.index(ct.zclass(ct.ZONES[z])) for z in row] for row in ts.corners], np.int64)
    zm = district_map(H, W, G, seed) if guided else np.full((H + 1, W + 1), -1, np.int64)
    g, ok, used, fy, fx = wfc(ts.sN, ts.sE, ts.sS, ts.sW, ts.n_sock, ts.w, cls, zm,
                              beta if guided else 0.0, H, W, seed, repairs)
    return g, ok, used, (fy, fx), zm


MODES = [("plain", False, 0), ("plain+repair", False, 50), ("plan+repair", True, 50)]


def main():
    from PIL import Image
    ts = ct.TileSet()
    print(ts.T, "tiles")
    run(ts, 4, 4, 0)                              # compile
    os.makedirs(IMG, exist_ok=True)
    N = int(os.environ.get("N", 12))
    for size in (32, 64, 96):
        for tag, guided, repairs in MODES:
            t0 = time.time()
            ok = 0
            used = []
            for s in range(N):
                g, success, u, (fy, fx), zm = run(ts, size, size, s, guided, repairs=repairs)
                ok += success
                used.append(u)
                if s == 0 and size == 64:
                    fail = np.zeros(g.shape, bool)
                    if not success:
                        fail[fy, fx] = True
                    Image.fromarray(ct.render(ts, g, 16, fail)).save(
                        os.path.join(IMG, f"city_wfc_{tag.replace('+', '_')}.png"))
            print(f"size {size:3d} {tag:13s}: {ok:2d}/{N} complete, "
                  f"repairs/run {np.mean(used):5.1f} (max {max(used)}), "
                  f"{(time.time() - t0) / N:.1f}s/run", flush=True)


if __name__ == "__main__":
    main()
