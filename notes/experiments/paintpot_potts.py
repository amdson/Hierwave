"""Stage 4 of notes/paintpot_test.md: the paint potential on the Potts toy.

    NUMBA_NUM_THREADS=1 PYTHONPATH=. python notes/experiments/paintpot_potts.py

Exact window free energies by a site-by-site transfer matrix (frontier =
the last W tiles, q^W states): 2 x 2 mid windows (8 x 8 tiles at BM = 4),
free boundary, every one of the q^4 = 256 window colourings; target F(z) -
F(all colour 0).  Painted channel = the block colour broadcast to its tiles
(V = q = 4); paint potential unary + OFF8 (+ OFF2) over painted tiles;
materialised mid_h / mid_v / diagonals vs BM J d (kappa -> inf) and the
self-play tables of images/potts_results.json.  Results in
images/paintpot_potts.json."""
import itertools, json, time
import numpy as np

from castlegen.channels.potts import Potts, dist
from castlegen.channels import paintpot as pp

DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
SETTINGS = [("k8_J0.1", 8.0, 0.1), ("k1_J0.3", 1.0, 0.3), ("k8_J1", 8.0, 1.0)]


def logZ_grid(mcol, q, J, kappa):
    """log sum over tile colourings of an (H, W) grid of exp(-E), E = J d on
    4-neighbour pairs + kappa [c != mcol] (free boundary).  Frontier
    transfer: state = the colours of the last W tiles."""
    H, W = mcol.shape
    cc = np.arange(q)
    eJ = np.exp(-J * dist(cc[:, None], cc[None, :], q))                 # (old / left, new)
    T = np.ones((q,) * W)
    lz = 0.0
    for r in range(H):
        for c in range(W):
            Tm = np.moveaxis(T, c, -1)                                   # (..., old)
            Tn = Tm @ eJ if r > 0 else np.repeat(Tm.sum(-1, keepdims=True) / q, q, -1)
            Tn = Tn * np.exp(-kappa * (cc != mcol[r, c]))
            if c > 0:
                shp = [1] * W
                shp[c - 1], shp[-1] = q, q
                Tn = Tn * eJ.reshape(shp)
            T = np.moveaxis(Tn, -1, c)
            s = T.max()
            lz += np.log(s)
            T = T / s
    # row 0: each frontier entry started as one state summed over q dummy colours / q: no factor
    return lz + np.log(T.sum())


def brute(mcol, q, J, kappa):
    H, W = mcol.shape
    best = []
    for cols in itertools.product(range(q), repeat=H * W):
        g = np.array(cols).reshape(H, W)
        e = J * (dist(g[:, :-1], g[:, 1:], q).sum() + dist(g[:-1], g[1:], q).sum()) + kappa * (g != mcol).sum()
        best.append(-e)
    best = np.array(best)
    return best.max() + np.log(np.exp(best - best.max()).sum())


def broadcast(z, BM):
    return np.repeat(np.repeat(np.asarray(z), BM, -2), BM, -1)


def canvas(BM, v, vp=None, d=None, ref=0):
    g = np.full((5, 5), ref, np.int64)
    g[2, 2] = v
    if vp is not None:
        g[2 + d[0], 2 + d[1]] = vp
    return broadcast(g, BM)


def by_dist(t, q):
    """Double-centred table (gauge-free) averaged by cyclic distance, and the
    max deviation of its entries from those averages."""
    t = np.asarray(t, float)
    t = t - t.mean(0, keepdims=True) - t.mean(1, keepdims=True) + t.mean()
    cc = np.arange(q)
    D = dist(cc[:, None], cc[None, :], q)
    return [float(t[D == k].mean()) for k in range(q // 2 + 1)], float(
        np.abs(t - np.array([[t[D == D[a, b]].mean() for b in range(q)] for a in range(q)])).max())


if __name__ == "__main__":
    # check the transfer against brute force on 3 x 3 tiles
    rng = np.random.default_rng(0)
    for _ in range(3):
        m = rng.integers(4, size=(3, 3))
        a, b = logZ_grid(m, 4, 0.7, 1.3), brute(m, 4, 0.7, 1.3)
        assert abs(a - b) < 1e-9, (a, b)
    print("transfer = brute force on 3 x 3 (3 random fields)")
    PR = json.load(open("images/potts_results.json"))
    out = {}
    for key, kappa, J in SETTINGS:
        P = Potts(1, 1, J=J, kappa=kappa)
        q, BM = P.q, P.BM
        t = time.time()
        Z = np.array(list(itertools.product(range(q), repeat=4))).reshape(-1, 2, 2)
        F = np.array([-logZ_grid(broadcast(z, BM), q, J, kappa) for z in Z])
        y = F - F[0]
        t_F = time.time() - t
        # exact 1 x 2 window pair term (free boundary) for reference
        F12 = np.array([[-logZ_grid(broadcast(np.array([[a, b]]), BM), q, J, kappa) for b in range(q)]
                        for a in range(q)])
        pair12 = F12 - np.diag(F12)[:, None] / 2 - np.diag(F12)[None, :] / 2
        res = dict(t_F=t_F, target_rms=float(np.sqrt(np.mean(y ** 2))), pair12=by_dist(pair12, q))
        Pz, Pr = broadcast(Z, BM), broadcast(np.zeros((1, 2, 2), int), BM)
        for oname, offs in (("unary", []), ("OFF8", pp.OFF8), ("OFF2", pp.OFF2)):
            X = pp.features(Pz, q, offs) - pp.features(Pr, q, offs)
            th, d = pp.fit(X, y, 1e-6)
            u, g = pp.materialise(th, lambda v: canvas(BM, v), lambda v, w, dd: canvas(BM, v, w, dd), q, DIRS, q, offs)
            res[oname] = dict(rel=d["rel"], resid=d["resid_rms"], mid_h=g[0].tolist(), mid_v=g[1].tolist(),
                              diag=[g[2].tolist(), g[3].tolist()], mid_h_dist=by_dist(g[0], q),
                              mid_v_dist=by_dist(g[1], q), diag_dist=by_dist(g[2], q), u=u.tolist())
        res["BMJd"] = by_dist(BM * P.Jd, q)
        if key in PR:
            res["selfplay"] = {s: by_dist(PR[key]["schemes"][s]["theta"]["mid_h"], q) for s in ("S0", "S1")}
        out[key] = res
        print(f"## {key}: exact F of {len(Z)} windows {t_F:.1f}s, target RMS {res['target_rms']:.3f}")
        print(f"  BM J d (kappa -> inf) by d = 0/1/2: {np.round(res['BMJd'][0], 3).tolist()}")
        print(f"  exact 1 x 2 window pair (free boundary): {np.round(res['pair12'][0], 3).tolist()}")
        for oname in ("unary", "OFF8", "OFF2"):
            r = res[oname]
            print(f"  paint {oname}: rel {r['rel']:.2e}; mid_h by d {np.round(r['mid_h_dist'][0], 3).tolist()} "
                  f"(spread within d {r['mid_h_dist'][1]:.1e}); mid_v {np.round(r['mid_v_dist'][0], 3).tolist()}; "
                  f"diag SE {np.round(r['diag_dist'][0], 3).tolist()}")
        if "selfplay" in res:
            for s, v in res["selfplay"].items():
                print(f"  self-play {s} mid_h by d: {np.round(v[0], 3).tolist()}")
    json.dump(out, open("images/paintpot_potts.json", "w"))
