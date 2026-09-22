import collections

import jax
import jax.numpy as jnp
import numpy as np

from castlegen import base, core, schedule


def test_noise_is_chunk_invariant():
    # the same absolute coordinate gives the same number whatever array it sits in
    full = core.noise(7, 0, 3, 1, jnp.arange(64)[:, None], jnp.arange(64)[None, :], 0)
    chunk = core.noise(7, 0, 3, 1, jnp.arange(16, 32)[:, None], jnp.arange(40, 48)[None, :], 0)
    assert np.allclose(np.asarray(full[16:32, 40:48]), np.asarray(chunk))
    u = np.asarray(full)
    assert u.min() > 0 and u.max() < 1
    assert abs(u.mean() - 0.5) < 0.02


def _exact_bfs(tiles, gate_yx):
    S = tiles.shape[0]
    typ, mask, is_room, _, is_gate = [np.asarray(a) for a in core.split_tile(tiles)]
    d = np.full((S, S), int(core.INF), np.int32)
    q = collections.deque([tuple(gate_yx)])
    d[tuple(gate_yx)] = 0
    while q:
        y, x = q.popleft()
        for dy, dx, own, facing in core.DIRS:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < S and 0 <= nx < S):
                continue
            if not is_room[ny, nx]:
                continue
            if (mask[y, x] >> own) & 1 and (mask[ny, nx] >> facing) & 1 and d[ny, nx] == int(core.INF):
                d[ny, nx] = d[y, x] + 1
                q.append((ny, nx))
    return d


def _random_consistent_tiles(rng, S):
    # random rooms/walls with door bits made consistent across every edge
    is_room = rng.random((S, S)) < 0.8
    typ = rng.integers(0, core.N_TYPES, (S, S))
    east = (rng.random((S, S)) < 0.8) & is_room & np.roll(is_room, -1, 1)
    east[:, -1] = False
    south = (rng.random((S, S)) < 0.8) & is_room & np.roll(is_room, -1, 0)
    south[-1, :] = False
    mask = (east.astype(int) << 1) | (south.astype(int) << 2)
    mask |= np.roll(east, 1, 1).astype(int) << 3
    mask |= np.roll(south, 1, 0).astype(int) << 0
    tiles = np.where(is_room, typ * core.N_MASKS + mask, core.WALL)
    gy, gx = 0, S // 2
    tiles[gy, gx] = core.GATE
    # give the gate a south door and its neighbour a north door
    if is_room[gy + 1, gx]:
        tiles[gy + 1, gx] |= 1      # north door toward the gate
    return jnp.asarray(tiles, jnp.int32), (gy, gx)


def test_d_step_converges_to_bfs():
    rng = np.random.default_rng(0)
    tiles, gate = _random_consistent_tiles(rng, 32)
    # the gateway tile carries no mask; treat it as having a south door in d_step
    d = jnp.full((32, 32), core.INF, jnp.int16)
    for _ in range(32 * 32):
        d_new = base.d_step(d, tiles)
        if bool(jnp.all(d_new == d)):
            break
        d = d_new
    exact = _exact_bfs(tiles, gate)
    got = np.asarray(d).astype(np.int32)
    _, _, is_room, _, _ = [np.asarray(a) for a in core.split_tile(tiles)]
    assert np.array_equal(got[is_room], exact[is_room])
    assert (exact < int(core.INF)).sum() > 10   # the gate actually reaches something


def test_base_run_smoke():
    S = 32
    schedule.SIZE = S
    rng = np.random.default_rng(1)
    A = jnp.asarray(rng.choice([-2.0, 0.0, 0.0, 0.0, 2.0], (50, 50)), jnp.float32)
    A = (A + A.T) / 2
    P = jnp.asarray(rng.normal(size=(8, 50)) / np.sqrt(50), jnp.float32)
    n_blocks = (S // schedule.BLOCK) ** 2
    h_plan = jnp.zeros((n_blocks, 8), jnp.float32)
    Lam = 4.0 * jnp.eye(8, dtype=jnp.float32)
    tiles0 = jnp.full((S, S), core.WALL, jnp.int32).at[0, S // 2].set(core.GATE)
    d0 = jnp.full((S, S), core.INF, jnp.int16).at[0, S // 2].set(0)
    tiles, d = schedule.run_base(3, tiles0, d0, jnp.array([0, S // 2]), A, P,
                                 h_plan, Lam, schedule.Presets())
    v = base.violations(d, tiles)
    assert int(v.sum()) == 0
    assert int((tiles == core.GATE).sum()) == 1
