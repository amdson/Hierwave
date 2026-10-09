import itertools

import numpy as np

from castlegen.legacy import conn, refsampler as rs, tileset

TINY = {
    "sockets": ["wall", {"name": "door", "draw": "gap"}, "open"],
    "connects": [["door", "door"], ["door", "open"], ["open", "open"]],
    "dangling": {"door": 1.0},
    "kinds": [
        {"name": "straight", "tags": ["hall"], "sides": {"N": "door", "E": "wall", "S": "door", "W": "wall"}, "rotations": [0, 90]},
        {"name": "corner", "tags": ["hall"], "unary": 0.3, "sides": {"N": "door", "E": "door", "S": "wall", "W": "wall"}, "rotations": "all"},
        {"name": "chamber", "sides": {"N": "door", "E": "wall", "S": "wall", "W": "wall"}, "rotations": "all"},
        {"name": "openroom", "unary": 1.0, "sides": {"N": "open", "E": "open", "S": "open", "W": "open"}},
    ],
    "rules": [{"a": "chamber", "b": "chamber", "when": "door", "energy": 2.0},
              {"a": "chamber", "b": "hall", "when": "door", "energy": -1.0},
              {"a": "bed", "b": "bed", "when": "door", "energy": -1.5}],
    "structures": [{"name": "bed", "size": [1, 2], "unary": 0.5, "seam": 3.0,
                    "doors": [{"piece": [0, 0], "side": "W", "socket": "door"}]}],
}


def test_matches_exact_enumeration():
    """4 x 4 castle, 2 x 2 free interior: sampled marginals and room-count
    distribution under tempering with site + window moves vs exact enumeration
    of the connected states (fixed seed; tolerances ~4 standard errors)."""
    ts = tileset.compile_spec(TINY)
    base = np.full((4, 4), ts.WALL, np.int32)
    base[0, 1] = ts.GATE
    free = [(1, 1), (1, 2), (2, 1), (2, 2)]
    opts = [s for s in range(ts.n_sig) if s != ts.GATE]
    states, logw = [], []
    for combo in itertools.product(opts, repeat=4):
        g = base.copy()
        for c, v in zip(free, combo):
            g[c] = v
        if conn.components_bfs(ts, g)[1] == 0:
            states.append(combo)
            logw.append(-rs.energy(ts, g))
    states, logw = np.array(states), np.array(logw)
    p = np.exp(logw - logw.max())
    p /= p.sum()
    room = np.asarray(ts.np_tables["is_room"])
    exact_rooms = np.bincount(room[states].sum(1), weights=p, minlength=5)
    exact_marg = np.stack([np.bincount(states[:, i], weights=p, minlength=ts.n_sig) for i in range(4)])

    _, samples, swaps, _ = rs.run_pt(ts, base, rs.ladder(3, 3.0, 3.0, 0.3), sweeps=6000, seed=3, burn=200,
                                     window=2, n_window=2, observe=lambda c: [c.t[f] for f in free])
    st = np.array(samples)
    w = np.full(len(st), 1.0 / len(st))
    rooms = np.bincount(room[st].sum(1), weights=w, minlength=5)
    marg = np.stack([np.bincount(st[:, i], weights=w, minlength=ts.n_sig) for i in range(4)])
    assert len(st) > 4000
    assert np.abs(rooms - exact_rooms).max() < 0.03
    assert np.abs(marg - exact_marg).max() < 0.04
    assert (swaps[:, 0] > 0).all()
