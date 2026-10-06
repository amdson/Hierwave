"""The sugar test: a free-energy correction learned for a coarse channel.

Fine level (tiles): water, sugar, stone.  Sugar needs water on all four
sides (hard: no sugar beside sugar or stone), and carries a unary mu
(activity z = exp(-mu)); nothing else.  Stone comes only from the objects.
Coarse level (blocks of B): one R x R stone rectangle per block at one of
(B - R + 1)^2 offsets, or absent; rectangles never leave their block, so
they never overlap, and touch only across block edges.

Given the objects, the fine level is a hard-square lattice gas with the
stone and its one-cell ring excluded.  Two rectangles that abut share ring
cells, so the union excludes fewer sites and the lattice gas has more
configurations: an attraction at contact mediated purely by counting.  The
chain generator (objects first, then tiles) has no object-object term and
places objects independently; the correction it needs is the pair table
over adjacent blocks' offsets that reproduces the ideal joint's contact
statistics.  This module provides
  - the forward model: object channel + tile channel on the generic kernel,
  - the ideal: a two-way sampler, tiles by Gibbs, objects by collapsed
    Gibbs (the block's tiles integrated out exactly by a column transfer
    matrix, then redrawn by forward-filter backward-sample),
  - the statistics phi (adjacent offset pairs, presence, sugar density)
    and the moment-matching step."""
import numpy as np
from numba import njit

from .core import Channel, Factor, Kinds, Model

INF = np.inf
WATER, SUGAR, STONE = 0, 1, 2
KINDS = Kinds(["water", "sugar", "stone"], [frozenset({"water"}), frozenset({"sugar"}), frozenset({"stone"})],
              [(60, 120, 220), (255, 255, 255), (110, 110, 110)])


# ------------------------------------------------------------------ model
class Sugar:
    def __init__(self, nby, nbx, B=8, R=4):
        self.nby, self.nbx, self.B, self.R = nby, nbx, B, R
        self.H, self.W = nby * B, nbx * B
        self.noff = B - R + 1
        self.D = 1 + self.noff ** 2                                  # 0 absent, 1 + oy * noff + ox
        self.oy = np.array([-1] + [o // self.noff for o in range(self.noff ** 2)])
        self.ox = np.array([-1] + [o % self.noff for o in range(self.noff ** 2)])

    def channels(self):
        obj = Channel("obj", self.B, self.D).add_view("self", np.arange(self.D), self.D)
        obj.add_view("present", (np.arange(self.D) > 0).astype(np.int64), 2)
        tile = Channel("tile", 1, 3).add_view("kind", np.arange(3), 3)
        return obj, tile

    def tile_factors(self, mu):
        """Hard: sugar beside sugar or stone forbidden; unary mu on sugar."""
        E = np.zeros((3, 3))
        E[SUGAR, SUGAR] = E[SUGAR, STONE] = E[STONE, SUGAR] = INF
        return [Factor.pair(("tile", "kind"), ("tile", "kind"), (0, 1), E, pad_b=WATER, pad_a=WATER, name="h"),
                Factor.pair(("tile", "kind"), ("tile", "kind"), (1, 0), E, pad_b=WATER, pad_a=WATER, name="v"),
                Factor.unary(("tile", "kind"), np.array([0.0, mu, 0.0]), name="mu")]

    def obj_factors(self, b, theta_h, theta_v):
        """Presence bonus b (energy -b on present) and the learned pair tables."""
        return [Factor.unary(("obj", "present"), np.array([0.0, -b]), name="presence"),
                Factor.pair(("obj", "self"), ("obj", "self"), (0, 1), theta_h, name="theta_h"),
                Factor.pair(("obj", "self"), ("obj", "self"), (1, 0), theta_v, name="theta_v")]

    def paint(self, obj, tile):
        """Stone from the objects; everything else left as it is (water if
        it was stone).  Returns the fixed mask for the tile kernel."""
        B, R = self.B, self.R
        g = tile.grid
        g[g == STONE] = WATER
        fixed = np.zeros(g.shape, bool)
        for i in range(self.nby):
            for j in range(self.nbx):
                o = obj.grid[i, j]
                if o == 0:
                    continue
                y0, x0 = i * B + self.oy[o], j * B + self.ox[o]
                g[y0:y0 + R, x0:x0 + R] = STONE
                fixed[y0:y0 + R, x0:x0 + R] = True
        # sugar left beside new stone is invalid: clear it (the kernel never proposes it)
        s = g == SUGAR
        st = g == STONE
        bad = np.zeros_like(s)
        bad[1:] |= st[:-1]; bad[:-1] |= st[1:]; bad[:, 1:] |= st[:, :-1]; bad[:, :-1] |= st[:, 1:]
        g[s & bad] = WATER
        return fixed

    # ------------------------------------------------------- statistics
    def stats(self, obj, tile):
        """phi: (h pairs (D, D), v pairs (D, D), presence, sugar density off stone)."""
        g = obj.grid
        D = self.D
        ph = np.zeros((D, D)); pv = np.zeros((D, D))
        np.add.at(ph, (g[:, :-1].ravel(), g[:, 1:].ravel()), 1)
        np.add.at(pv, (g[:-1, :].ravel(), g[1:, :].ravel()), 1)
        ph /= max(ph.sum(), 1); pv /= max(pv.sum(), 1)
        t = tile.grid
        return dict(h=ph, v=pv, present=float((g > 0).mean()),
                    sugar=float((t == SUGAR).sum() / max((t != STONE).sum(), 1)))

    def contact(self, stats):
        """Summary of a pair table: probability that two present neighbours
        share a full side, a partial side, touch at a corner, or not."""
        out = {}
        for key, dy, dx in (("h", 0, 1), ("v", 1, 0)):
            P = stats[key]
            both = P[1:, 1:]
            tot = both.sum()
            full = partial = corner = 0.0
            for a in range(1, self.D):
                for b in range(1, self.D):
                    if dx:                                           # a left of b: a's right edge at ox = noff-1, b's at 0
                        touch = self.ox[a] == self.noff - 1 and self.ox[b] == 0
                        overlap = self.R - abs(self.oy[a] - self.oy[b])
                    else:
                        touch = self.oy[a] == self.noff - 1 and self.oy[b] == 0
                        overlap = self.R - abs(self.ox[a] - self.ox[b])
                    if not touch:
                        continue
                    if overlap == self.R:
                        full += P[a, b]
                    elif overlap > 0:
                        partial += P[a, b]
                    elif overlap == 0:
                        corner += P[a, b]
            out[key] = dict(full=full / max(tot, 1e-12), partial=partial / max(tot, 1e-12),
                            corner=corner / max(tot, 1e-12), apart=1 - (full + partial + corner) / max(tot, 1e-12))
        return out


# ---------------------------------------------------- the ideal: two-way
@njit(cache=True)
def _block_logZ_and_sample(tiles, y0, x0, B, R, oy, ox, z, do_sample, out_cols, rng_u):
    """Column transfer matrix over the B x B block at (y0, x0) with a
    rectangle at offset (oy, ox) (oy < 0: absent).  Cells outside the block
    are read from `tiles` as fixed boundary.  Returns log Z (-inf if the
    rectangle would sit beside an outside sugar); if do_sample, draws the
    block's sugar columns into out_cols (bitmasks) by backward sampling."""
    H, W = tiles.shape
    NS = 1 << B
    # capable[y, x]: cell may hold sugar (not stone, not beside stone or outside sugar); stone[y, x]
    capable = np.ones((B, B), np.bool_)
    stone = np.zeros((B, B), np.bool_)
    if oy >= 0:
        for y in range(oy, oy + R):
            for x in range(ox, ox + R):
                stone[y, x] = True
    for y in range(B):
        for x in range(B):
            if stone[y, x]:
                capable[y, x] = False
                continue
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < B and 0 <= nx < B:
                    if stone[ny, nx]:
                        capable[y, x] = False
                else:
                    gy, gx = y0 + ny, x0 + nx
                    if 0 <= gy < H and 0 <= gx < W and tiles[gy, gx] != WATER:
                        capable[y, x] = False
    # an outside sugar beside the rectangle makes the candidate invalid
    if oy >= 0:
        for y in range(oy, oy + R):
            for x in range(ox, ox + R):
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < B and 0 <= nx < B):
                        gy, gx = y0 + ny, x0 + nx
                        if 0 <= gy < H and 0 <= gx < W and tiles[gy, gx] == SUGAR:
                            return -np.inf
    # column masks of capable cells and column weights
    cap = np.zeros(B, np.int64)
    for x in range(B):
        m = 0
        for y in range(B):
            if capable[y, x]:
                m |= 1 << y
        cap[x] = m
    wts = np.zeros(NS)
    pc = np.zeros(NS, np.int64)
    for s in range(NS):
        c = 0
        for y in range(B):
            if (s >> y) & 1:
                c += 1
        pc[s] = c
        wts[s] = z ** c
    alpha = np.zeros((B, NS))
    scale = np.zeros(B)
    for s in range(NS):
        if (s & (s >> 1)) == 0 and (s & ~cap[0]) == 0:
            alpha[0, s] = wts[s]
    scale[0] = alpha[0].sum()
    if scale[0] <= 0:
        return -np.inf
    alpha[0] /= scale[0]
    for x in range(1, B):
        for s in range(NS):
            if (s & (s >> 1)) != 0 or (s & ~cap[x]) != 0:
                continue
            acc = 0.0
            for sp in range(NS):
                if alpha[x - 1, sp] > 0 and (sp & s) == 0:
                    acc += alpha[x - 1, sp]
            alpha[x, s] = acc * wts[s]
        scale[x] = alpha[x].sum()
        if scale[x] <= 0:
            return -np.inf
        alpha[x] /= scale[x]
    logZ = 0.0
    for x in range(B):
        logZ += np.log(scale[x])
    if do_sample:
        # backward sample
        x = B - 1
        u = rng_u[x] * alpha[x].sum()
        acc = 0.0
        pick = 0
        for s in range(NS):
            acc += alpha[x, s]
            if acc >= u:
                pick = s
                break
        out_cols[x] = pick
        for x in range(B - 2, -1, -1):
            tot = 0.0
            for sp in range(NS):
                if (sp & out_cols[x + 1]) == 0:
                    tot += alpha[x, sp]
            u = rng_u[x] * tot
            acc = 0.0
            pick = 0
            for sp in range(NS):
                if (sp & out_cols[x + 1]) == 0:
                    acc += alpha[x, sp]
                    if acc >= u:
                        pick = sp
                        break
            out_cols[x] = pick
    return logZ


class Ideal:
    """Two-way sampler over objects and tiles: tile Gibbs sweeps (the
    generic kernel) alternate with collapsed object moves, each block's
    object drawn from p(o | everything else) with its tiles integrated out
    exactly, then its tiles redrawn exactly."""

    def __init__(self, S: Sugar, mu, b, seed=0):
        self.S, self.mu, self.b, self.z = S, mu, b, float(np.exp(-mu))
        self.obj, self.tile = S.channels()
        self.m = Model(S.H, S.W, [self.tile, self.obj], S.tile_factors(mu))
        self.rng = np.random.default_rng(seed)
        self.cols = np.zeros(S.B, np.int64)

    def object_move(self, i, j):
        S = self.S
        B, R = S.B, S.R
        y0, x0 = i * B, j * B
        tiles = self.tile.grid
        # the block's own tiles must not constrain the candidates: blank them to water for the boundary reads
        saved = tiles[y0:y0 + B, x0:x0 + B].copy()
        tiles[y0:y0 + B, x0:x0 + B] = WATER
        logw = np.empty(S.D)
        for o in range(S.D):
            lz = _block_logZ_and_sample(tiles, y0, x0, B, R, S.oy[o], S.ox[o], self.z, False, self.cols, np.zeros(B))
            logw[o] = lz + (self.b if o > 0 else 0.0)
        fin = np.isfinite(logw)
        if not fin.any():
            tiles[y0:y0 + B, x0:x0 + B] = saved
            return
        p = np.exp(logw - logw[fin].max()) * fin
        o = int(self.rng.choice(S.D, p=p / p.sum()))
        self.obj.grid[i, j] = o
        _block_logZ_and_sample(tiles, y0, x0, B, R, S.oy[o], S.ox[o], self.z, True, self.cols, self.rng.random(B))
        blk = np.full((B, B), WATER, np.int32)
        for x in range(B):
            for y in range(B):
                if (self.cols[x] >> y) & 1:
                    blk[y, x] = SUGAR
        if o > 0:
            blk[S.oy[o]:S.oy[o] + R, S.ox[o]:S.ox[o] + R] = STONE
        tiles[y0:y0 + B, x0:x0 + B] = blk

    def sweep(self, n=1, tile_sweeps=2):
        S = self.S
        for _ in range(n):
            order = self.rng.permutation(S.nby * S.nbx)
            for k in order:
                self.object_move(k // S.nbx, k % S.nbx)
            fixed = S.paint(self.obj, self.tile)
            self.tile.fixed[:] = fixed
            self.m.sweep("tile", tile_sweeps, seed=int(self.rng.integers(1 << 30)))


# ------------------------------------------------------- forward model
class Forward:
    """Objects by the generic kernel with the learned tables, then tiles."""

    def __init__(self, S: Sugar, mu, b, theta_h=None, theta_v=None, seed=0):
        self.S, self.mu, self.b = S, mu, b
        D = S.D
        self.theta_h = np.zeros((D, D)) if theta_h is None else theta_h.copy()
        self.theta_v = np.zeros((D, D)) if theta_v is None else theta_v.copy()
        self.obj, self.tile = S.channels()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def run(self, obj_sweeps=30, tile_sweeps=20, fresh=True):
        S = self.S
        m = Model(S.H, S.W, [self.tile, self.obj], S.tile_factors(self.mu) + S.obj_factors(self.b, self.theta_h, self.theta_v))
        if fresh:
            self.obj.grid[:] = 0
            self.tile.grid[:] = WATER
        m.sweep("obj", obj_sweeps, seed=int(self.rng.integers(1 << 30)))
        fixed = S.paint(self.obj, self.tile)
        self.tile.fixed[:] = fixed
        m.sweep("tile", tile_sweeps, seed=int(self.rng.integers(1 << 30)))
        return S.stats(self.obj, self.tile)


def cond_pairs(P):
    """The pair table restricted to both-present, normalised."""
    Q = P[1:, 1:]
    return Q / max(Q.sum(), 1e-12)


def moment_step(theta, model_P, target_P, eta):
    """Energy table step on the both-present entries: theta += eta (Q_model -
    Q_target) with Q the conditional pair distribution given both present.
    Too many of a pair in the model raises its energy.  Rows and columns
    of the absent value stay zero: presence is the unary's business."""
    out = theta.copy()
    out[1:, 1:] += eta * (cond_pairs(model_P) - cond_pairs(target_P))
    out[1:, 1:] -= out[1:, 1:].mean()
    return out
