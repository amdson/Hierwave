"""End-to-end check of the base level at full size.

Runs one 192 x 192 castle from a wall-only initial state with a random
adjacency table and a flat level-1 plan, then asserts the validity
invariants, reports rooms lost and reachability, and times a vmapped
batch.  This is the smoke test the Colab notebook runs; it does not use
coarse levels or the offline pipeline (not implemented yet).
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from castlegen import base, core, schedule


def make_inputs(size=192, seed=0, room_density=0.33):
    rng = np.random.default_rng(seed)
    A = rng.choice([-2.0, 0.0, 0.0, 0.0, 2.0], (core.N_TYPES, core.N_TYPES))
    A = jnp.asarray((A + A.T) / 2, jnp.float32)
    P = jnp.asarray(rng.normal(size=(8, core.N_TYPES)) / np.sqrt(core.N_TYPES), jnp.float32)
    n_blocks = (size // schedule.BLOCK) ** 2
    # flat plan: the projected histogram of a uniform mix at the target density
    h_flat = P @ jnp.full((core.N_TYPES,), room_density / core.N_TYPES, jnp.float32)
    h_plan = jnp.broadcast_to(h_flat, (n_blocks, 8))
    Lam = 4.0 * jnp.eye(8, dtype=jnp.float32)
    gate = jnp.array([0, size // 2])
    return A, P, h_plan, Lam, gate


def check(tiles, d):
    """Validity invariants after the fallback.  Returns a dict of numbers."""
    typ, mask, is_room, is_wall, is_gate = core.split_tile(tiles)
    v = base.violations(d, tiles)
    n_rooms = int(is_room.sum())
    unreached = int((is_room & (d == core.INF)).sum())
    exact = _exact_bfs(np.asarray(tiles))
    got = np.asarray(d).astype(np.int32)
    mism = int((got[np.asarray(is_room)] != exact[np.asarray(is_room)]).sum())
    return dict(
        violations=int(v.sum()),
        half_doors=int(base.half_doors(tiles).sum()),
        gateways=int(is_gate.sum()),
        rooms=n_rooms,
        room_density=n_rooms / tiles.size,
        unreached_rooms=unreached,
        d_mismatch_vs_bfs=mism,
        doors_per_room=float(core.POP[np.asarray(mask)][np.asarray(is_room)].mean()) if n_rooms else 0.0,
        max_d=int(np.asarray(d)[np.asarray(is_room)].max()) if n_rooms else 0,
    )


def _exact_bfs(tiles):
    import collections
    S = tiles.shape[0]
    typ, mask, is_room, _, is_gate = [np.asarray(a) for a in core.split_tile(jnp.asarray(tiles))]
    d = np.full((S, S), int(core.INF), np.int32)
    gy, gx = np.argwhere(is_gate)[0]
    d[gy, gx] = 0
    q = collections.deque([(gy, gx)])
    while q:
        y, x = q.popleft()
        for dy, dx, own, facing in core.DIRS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < S and 0 <= nx < S and is_room[ny, nx] and d[ny, nx] == int(core.INF):
                if (mask[y, x] >> own) & 1 and (mask[ny, nx] >> facing) & 1:
                    d[ny, nx] = d[y, x] + 1
                    q.append((ny, nx))
    return d


def run_one(size=192, castle_id=0, seed=0):
    A, P, h_plan, Lam, gate = make_inputs(size, seed)
    pre = schedule.Presets()
    def f_(cid):
        tiles0, d0 = schedule.init_random(cid, size, gate, pre)
        return schedule.run_base(cid, tiles0, d0, gate, A, P, h_plan, Lam, pre)
    f = jax.jit(f_)
    t0 = time.time()
    tiles, d = f(castle_id)
    tiles.block_until_ready()
    t_first = time.time() - t0
    t0 = time.time()
    tiles, d = f(castle_id + 1)
    tiles.block_until_ready()
    t_second = time.time() - t0
    return tiles, d, t_first, t_second


def time_batch(size=192, batch=16, seed=0):
    A, P, h_plan, Lam, gate = make_inputs(size, seed)
    pre = schedule.Presets()
    def f_(cid):
        tiles0, d0 = schedule.init_random(cid, size, gate, pre)
        return schedule.run_base(cid, tiles0, d0, gate, A, P, h_plan, Lam, pre)
    f = jax.jit(jax.vmap(f_))
    ids = jnp.arange(batch)
    tiles, d = f(ids); tiles.block_until_ready()
    t0 = time.time()
    tiles, d = f(ids + batch); tiles.block_until_ready()
    return (time.time() - t0) / batch


if __name__ == "__main__":
    print("devices:", jax.devices())
    tiles, d, t1, t2 = run_one()
    stats = check(tiles, d)
    print("first call (compile + run): %.2fs, second call: %.3fs" % (t1, t2))
    for k, v in stats.items():
        print(f"  {k}: {v}")
    assert stats["violations"] == 0
    assert stats["gateways"] == 1
    assert stats["unreached_rooms"] == 0
    assert stats["d_mismatch_vs_bfs"] == 0
    print("per-castle time in a batch of 16: %.3fs" % time_batch())
