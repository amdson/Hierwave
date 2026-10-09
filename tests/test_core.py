import jax
import jax.numpy as jnp
import numpy as np

from castlegen import base, core, e2e, schedule, tileset


def test_noise_is_chunk_invariant():
    # the same absolute coordinate gives the same number whatever array it sits in
    full = core.noise(7, 0, 3, 1, jnp.arange(64)[:, None], jnp.arange(64)[None, :], 0)
    chunk = core.noise(7, 0, 3, 1, jnp.arange(16, 32)[:, None], jnp.arange(40, 48)[None, :], 0)
    assert np.allclose(np.asarray(full[16:32, 40:48]), np.asarray(chunk))
    u = np.asarray(full)
    assert u.min() > 0 and u.max() < 1
    assert abs(u.mean() - 0.5) < 0.02


def _random_tiles(ts, rng, S):
    # mostly four-door tiles so the gate reaches a sizeable region
    rooms = np.nonzero(ts.np_tables["is_room"])[0]
    open_ = [s for s in rooms if ts.kinds[ts.sig_kind[s]].name == "hall_cross"]
    pick = np.where(rng.random((S, S)) < 0.7, rng.choice(open_, (S, S)), rng.choice(rooms, (S, S)))
    tiles = np.where(rng.random((S, S)) < 0.9, pick, ts.WALL)
    gy, gx = 0, S // 2
    tiles[gy, gx] = ts.GATE
    tiles[gy + 1, gx] = open_[0]                # doors on all sides: joins the gate
    return jnp.asarray(tiles, jnp.int32)


def test_d_step_converges_to_bfs():
    ts = tileset.load("demo")
    rng = np.random.default_rng(0)
    tiles = _random_tiles(ts, rng, 32)
    d = jnp.full((32, 32), core.INF, jnp.int16)
    for _ in range(32 * 32):
        d_new = base.d_step(ts, d, tiles)
        if bool(jnp.all(d_new == d)):
            break
        d = d_new
    exact = e2e.exact_bfs(ts, tiles)
    got = np.asarray(d).astype(np.int32)
    is_room = ts.np_tables["is_room"][np.asarray(tiles)]
    assert np.array_equal(got[is_room], exact[is_room])
    assert (exact < int(core.INF)).sum() > 20   # the gate actually reaches a region


def test_base_run_smoke():
    # no repair step: only the gateway and the enclosure are guaranteed
    ts = tileset.load("demo")
    S = 32
    P, h_plan, Lam, gate = e2e.make_inputs(ts, size=S, seed=1)
    pre = schedule.Presets(n_sweeps=20)
    tiles0, d0 = schedule.init_random(ts, 3, S, gate, pre)
    tiles, d = schedule.run_base(ts, 3, tiles0, d0, gate, P, h_plan, Lam, pre)
    stats = e2e.check(ts, tiles, d)
    assert stats["gateways"] == 1
    t = np.asarray(tiles)
    perim = np.concatenate([t[0], t[-1], t[:, 0], t[:, -1]])
    assert not ts.np_tables["is_room"][perim].any()
