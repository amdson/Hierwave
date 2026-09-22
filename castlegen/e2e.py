"""End-to-end check of the base level at full size.

Runs one 192 x 192 castle of a tile set (default: tilesets/demo.json) from a
random initial state with a random projection and a flat level-1 plan
(100 annealed sweeps, no repair), reports the validity numbers against an
exact BFS, draws the concrete types, and times a vmapped batch.  This is the smoke test the Colab notebook
runs; it does not use coarse levels or the offline pipeline (not
implemented yet).

    python -m castlegen.e2e [--tileset demo] [--png castle.png]
"""
from __future__ import annotations

import argparse
import collections
import time

import jax
import jax.numpy as jnp
import numpy as np

from castlegen import base, core, render, schedule, tileset


def make_inputs(ts, size=192, seed=0, room_density=0.33):
    rng = np.random.default_rng(seed)
    P = jnp.asarray(rng.normal(size=(8, ts.n_sig)) / np.sqrt(ts.n_sig), jnp.float32)
    n_blocks = (size // schedule.BLOCK) ** 2
    # flat plan: the projected histogram of a uniform mix of room signatures at the target density
    rooms = np.asarray(ts.jt.is_room, np.float32)
    h_flat = P @ jnp.asarray(room_density * rooms / rooms.sum())
    h_plan = jnp.broadcast_to(h_flat, (n_blocks, 8))
    Lam = 4.0 * jnp.eye(8, dtype=jnp.float32)
    gate = jnp.array([0, size // 2])
    return P, h_plan, Lam, gate


def check(ts, tiles, d):
    """Validity numbers of the final state (not guaranteed zero: there is no repair)."""
    tiles_np = np.asarray(tiles)
    is_room = np.asarray(ts.jt.is_room)[tiles_np]
    v = base.violations(ts, d, tiles)
    n_rooms = int(is_room.sum())
    got = np.asarray(d).astype(np.int32)
    exact = exact_bfs(ts, tiles_np)
    kinds = collections.Counter(ts.kinds[k].name for k in ts.sig_kind[tiles_np].ravel())
    return dict(
        violations=int(v.sum()),
        gateways=int((tiles_np == ts.GATE).sum()),
        rooms=n_rooms,
        room_density=n_rooms / tiles_np.size,
        unreached_rooms=int((is_room & (got == int(core.INF))).sum()),
        d_mismatch_vs_bfs=int((got[is_room] != exact[is_room]).sum()),
        doors_per_room=float(sum(np.asarray(x) for x in base.door_field(ts, tiles))[is_room].mean()) if n_rooms else 0.0,
        max_d=int(got[is_room].max()) if n_rooms else 0,
        kinds=dict(kinds.most_common()),
    )


def exact_bfs(ts, tiles):
    tiles = np.asarray(tiles)
    S = tiles.shape[0]
    Dh, Dv = ts.np_tables["Dh"], ts.np_tables["Dv"]
    is_room = ts.np_tables["is_room"][tiles]
    d = np.full(tiles.shape, int(core.INF), np.int32)
    gy, gx = np.argwhere(tiles == ts.GATE)[0]
    d[gy, gx] = 0
    q = collections.deque([(gy, gx)])
    while q:
        y, x = q.popleft()
        for dy, dx in core.DIRS:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < S and 0 <= nx < S) or not is_room[ny, nx] or d[ny, nx] != int(core.INF):
                continue
            a, b = tiles[y, x], tiles[ny, nx]
            door = {(-1, 0): Dv[b, a], (1, 0): Dv[a, b], (0, 1): Dh[a, b], (0, -1): Dh[b, a]}[(dy, dx)]
            if door:
                d[ny, nx] = d[y, x] + 1
                q.append((ny, nx))
    return d


def _castle_fn(ts, size, seed):
    P, h_plan, Lam, gate = make_inputs(ts, size, seed)
    pre = schedule.Presets()
    def f(cid):
        tiles0, d0 = schedule.init_random(ts, cid, size, gate, pre)
        tiles, d = schedule.run_base(ts, cid, tiles0, d0, gate, P, h_plan, Lam, pre)
        return tiles, d, tileset.decorate(ts, cid, tiles)
    return f


def run_one(ts=None, size=192, castle_id=0, seed=0):
    """Returns signature grid, d, type grid, compile+run time, second-run time."""
    ts = ts or tileset.load("demo")
    f = jax.jit(_castle_fn(ts, size, seed))
    t0 = time.time()
    tiles, d, types = f(castle_id)
    types.block_until_ready()
    t_first = time.time() - t0
    t0 = time.time()
    f(castle_id + 1)[2].block_until_ready()
    t_second = time.time() - t0
    return tiles, d, types, t_first, t_second


def time_batch(ts=None, size=192, batch=16, seed=0):
    ts = ts or tileset.load("demo")
    f = jax.jit(jax.vmap(_castle_fn(ts, size, seed)))
    ids = jnp.arange(batch)
    f(ids)[2].block_until_ready()
    t0 = time.time()
    f(ids + batch)[2].block_until_ready()
    return (time.time() - t0) / batch


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tileset", default="demo")
    ap.add_argument("--png", default=None, help="write a rendered castle here")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    print("devices:", jax.devices())
    ts = tileset.load(args.tileset)
    print(f"tile set {ts.name}: {ts.n_sig} signatures, {ts.n_types} types")
    tiles, d, types, t1, t2 = run_one(ts, size=20)
    stats = check(ts, tiles, d)
    print("first call (compile + run): %.2fs, second call: %.3fs" % (t1, t2))
    for k, v in stats.items():
        print(f"  {k}: {v}")
    assert stats["gateways"] == 1
    gx = int(np.argwhere(np.asarray(tiles) == ts.GATE)[0][1])
    print(render.ascii_grid(ts, np.asarray(types)[:24, gx - 30: gx + 30]))
    if args.png:
        render.save_png(render.image(ts, types), args.png)
        print("wrote", args.png)
    print("per-castle time in a batch of %d: %.3fs" % (args.batch, time_batch(ts, batch=args.batch)))
