"""Hierarchical connectivity probe: blocks of 4 x 4 at every level, e.g.
20 x 20 = 5 x 5 blocks (two levels) or 128 x 128 -> 32 x 32 -> 8 x 8 (three).

Top level: one crossing mask per block (bits N, E, S, W; a link exists iff
both blocks have the facing bit, exactly like doors at the base).  Sampled by
Gibbs with a global connectivity check (every block with a link is connected
to the gate's block) -- fine at the top level, which is small.

Middle levels: each block's crossing mask is a tile of a generated 16-mask
tile set (sockets "link"/"wall"), refined under the same contract as the base
inside its parent, with the same sampler.

Contract for each block with links: for every link, one crossing on a chosen
child edge of the shared boundary; the crossing cells are rooms with a door to
each other; every room of the block is connected to every other through doors
inside the block.  Blocks without links hold no rooms.  If the coarse level is
connected and every block keeps its contract, all rooms reach the gate.

Base level: start from a feasible fill (an open tile along an L path from each
crossing to the block's hub cell), then Gibbs with the demo energy where any
candidate that breaks its block's contract is infeasible.  One cell per block
per step (16 colours), so the check for one block never races another update
in the same block.  No d, no reachability term.

    python -m castlegen.hier_probe [--size 20] [--seed 0] [--sweeps 1000] [--png out.png]
"""
from __future__ import annotations

import argparse
import collections

import jax
import jax.numpy as jnp
import numpy as np

from castlegen import base, e2e, render, tileset
from castlegen.core import INF, noise

B = 4
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))          # bit order N, E, S, W
BOX = "·╵╶└╷│┌├╴┘─┴┐┤┬┼"                            # indexed by matched-link mask


# --------------------------------------------------------------------------
# coarse level
# --------------------------------------------------------------------------
def matched(m):
    """(nb, nb) mask of links that exist (both facing bits set)."""
    nb = m.shape[0]
    out = np.zeros_like(m)
    for b, (dy, dx) in enumerate(DIRS):
        for y in range(nb):
            for x in range(nb):
                ny, nx = y + dy, x + dx
                if 0 <= ny < nb and 0 <= nx < nb and (m[y, x] >> b) & 1 and (m[ny, nx] >> ((b + 2) % 4)) & 1:
                    out[y, x] |= 1 << b
    return out


def coarse_connected(m, gate_block):
    mm = matched(m)
    nb = m.shape[0]
    seen = {gate_block}
    q = collections.deque([gate_block])
    while q:
        y, x = q.popleft()
        for b, (dy, dx) in enumerate(DIRS):
            if (mm[y, x] >> b) & 1 and (y + dy, x + dx) not in seen:
                seen.add((y + dy, x + dx)); q.append((y + dy, x + dx))
    return all((y, x) in seen for y in range(nb) for x in range(nb) if mm[y, x])


def sample_coarse(nb, gate_block, rng, sweeps=200, J=1.0, dangle=1.0,
                  u_pop=(1.0, 0.0, -0.3, 0.0, 0.5), T0=3.0, T1=1.0):
    """Gibbs over the 16 masks per block; candidates that disconnect are skipped."""
    m = np.zeros((nb, nb), int)
    pop = [bin(i).count("1") for i in range(16)]
    def local_energy(m, y, x):
        e = u_pop[pop[m[y, x]]]
        for b, (dy, dx) in enumerate(DIRS):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < nb and 0 <= nx < nb):
                continue
            own, other = (m[y, x] >> b) & 1, (m[ny, nx] >> ((b + 2) % 4)) & 1
            e += -J if own and other else dangle * (own != other)
        return e
    for s in range(sweeps):
        T = T0 + (T1 - T0) * min(1.0, s / (0.7 * sweeps))
        for i in rng.permutation(nb * nb):
            y, x = divmod(int(i), nb)
            cands, es = [], []
            for v in range(16):
                if any((v >> b) & 1 and not (0 <= y + dy < nb and 0 <= x + dx < nb)
                       for b, (dy, dx) in enumerate(DIRS)):
                    continue                          # no crossing out of the castle
                old, m[y, x] = m[y, x], v
                if coarse_connected(m, gate_block):
                    cands.append(v); es.append(local_energy(m, y, x))
                m[y, x] = old
            p = np.exp(-(np.array(es) - min(es)) / T); p /= p.sum()
            m[y, x] = cands[rng.choice(len(cands), p=p)]
    return matched(m)


# --------------------------------------------------------------------------
# crossings and the initial fill
# --------------------------------------------------------------------------
def place_crossings(links, size, rng, avoid_perimeter):
    """need[y, x]: bitmask of directions this cell must have a door towards,
    one crossing per parent link on a random child edge of the shared side."""
    nb = links.shape[0]
    need = np.zeros((size, size), int)
    ok = (lambda i: 0 < i < size - 1) if avoid_perimeter else (lambda i: True)
    for by in range(nb):
        for bx in range(nb):
            if (links[by, bx] >> 1) & 1:              # E link: pick a row
                r = rng.choice([r for r in range(by * B, by * B + B) if ok(r)])
                need[r, bx * B + B - 1] |= 1 << 1
                need[r, bx * B + B] |= 1 << 3
            if (links[by, bx] >> 2) & 1:              # S link: pick a column
                c = rng.choice([c for c in range(bx * B, bx * B + B) if ok(c)])
                need[by * B + B - 1, c] |= 1 << 2
                need[by * B + B, c] |= 1 << 0
    return need


def initial_fill(ts, active, anchor, size):
    """An all-sides-open tile along an L path from each anchor to its block's hub."""
    Dh, Dv = ts.np_tables["Dh"], ts.np_tables["Dv"]
    opens = [s for s in range(ts.n_sig) if ts.np_tables["is_room"][s] and Dh[s, s] and Dv[s, s] and Dv[ts.GATE, s]]
    if not opens:
        raise ValueError("the probe's initial fill needs a room signature open on all sides")
    o = opens[0]
    tiles = np.full((size, size), ts.WALL, np.int32)
    for by, bx in zip(*np.nonzero(active)):
        hy, hx = by * B + 1, bx * B + 1
        ys, xs = np.nonzero(anchor[by * B:(by + 1) * B, bx * B:(bx + 1) * B])
        for ay, ax in zip(ys + by * B, xs + bx * B):
            for x in range(min(ax, hx), max(ax, hx) + 1):
                tiles[ay, x] = o
            for y in range(min(ay, hy), max(ay, hy) + 1):
                tiles[y, hx] = o
    return tiles


def mask_tileset(J=1.0, dangle=1.0, u_pop=(1.0, 0.0, -0.3, 0.0, 0.5)):
    """A coarse level as a tile set: one kind per non-empty crossing mask."""
    kinds = []
    for m in range(1, 16):
        kinds.append({"name": f"m{m}", "tags": ["blk"], "unary": u_pop[bin(m).count("1")], "glyph": BOX[m],
                      "sides": {side: ("link" if (m >> b) & 1 else "wall") for b, side in enumerate("NESW")}})
    spec = {"name": "masks", "sockets": ["wall", "link"], "connects": [["link", "link"]],
            "dangling": {"link": dangle}, "wall": {"unary": u_pop[0]}, "gate": {"socket": "link"},
            "kinds": kinds, "rules": [{"a": "blk", "b": "blk", "when": "door", "energy": -J}]}
    ts = tileset.compile_spec(spec)
    link = ts.sockets.index("link")
    sig_mask = np.array([sum(1 << b for b in range(4) if ts.sig_sockets[s][b] == link) for s in range(ts.n_sig)])
    return ts, sig_mask


# --------------------------------------------------------------------------
# one level under the contract
# --------------------------------------------------------------------------
def make_sampler(ts, size, active, need, anchor, sweeps, gate_yx=None, enclose=True,
                 T0=4.0, T1=1.0, seed=0, level=0):
    """Gibbs over a (size, size) grid of ts signatures, blocks of B x B, where a
    candidate that breaks its block's contract is infeasible:
      inactive block : no rooms
      active block   : anchors are rooms, need[] doors are kept, and all rooms of
                       the block form one component through in-block doors."""
    t = ts.jt
    nb = size // B
    blocks = [(by, bx) for by in range(nb) for bx in range(nb)]
    blk_cells = jnp.asarray([[(by * B + i // B) * size + bx * B + i % B for i in range(B * B)] for by, bx in blocks])
    coords = jnp.asarray(np.stack([np.array([[by * B + c // B, bx * B + c % B] for by, bx in blocks])
                                   for c in range(B * B)]))                    # (16, nblk, 2)
    act = jnp.asarray(np.asarray(active).reshape(-1))
    anc_flat = jnp.asarray(np.asarray(anchor).reshape(-1))
    need_j = jnp.asarray(need)
    d_inf = jnp.full((size, size), INF, jnp.int16)
    zeros_block = jnp.zeros((size, size), jnp.int32)
    g0 = jnp.zeros((1, 8), jnp.float32)
    P0 = jnp.zeros((8, ts.n_sig), jnp.float32)
    gate = jnp.asarray(gate_yx if gate_yx is not None else (-10, -10))
    S = ts.n_sig
    cand = jnp.arange(S)

    def feasible(tiles, c, co):
        bt = tiles.reshape(-1)[blk_cells]                                      # (nblk, 16)
        g = jnp.where(jnp.arange(B * B)[None, None, :] == c, cand[None, :, None], bt[:, None, :])
        g = g.reshape(-1, S, B, B)                                             # (nblk, S, 4, 4)
        room = t.is_room[g]
        hl = (t.Dh[g[..., :, :-1], g[..., :, 1:]] > 0) & room[..., :, :-1] & room[..., :, 1:]
        vl = (t.Dv[g[..., :-1, :], g[..., 1:, :]] > 0) & room[..., :-1, :] & room[..., 1:, :]
        big = 99
        lab = jnp.where(room, jnp.arange(B * B).reshape(B, B), big)
        for _ in range(B * B - 1):
            l = lab
            l = l.at[..., :, :-1].min(jnp.where(hl, lab[..., :, 1:], big))
            l = l.at[..., :, 1:].min(jnp.where(hl, lab[..., :, :-1], big))
            l = l.at[..., :-1, :].min(jnp.where(vl, lab[..., 1:, :], big))
            l = l.at[..., 1:, :].min(jnp.where(vl, lab[..., :-1, :], big))
            lab = l
        one_comp = jnp.where(room, lab, -1).max((-2, -1)) == jnp.where(room, lab, big).min((-2, -1))
        one_comp = one_comp | ~room.any((-2, -1))
        anchors_ok = (room | ~anc_flat[blk_cells].reshape(-1, 1, B, B)).all((-2, -1))
        # doors this cell must keep towards cells outside its block (or the gate)
        tp = jnp.pad(tiles, 1, constant_values=ts.WALL)
        ys, xs = co[:, 0] + 1, co[:, 1] + 1
        up, rt, dn, lf = tp[ys - 1, xs], tp[ys, xs + 1], tp[ys + 1, xs], tp[ys, xs - 1]
        doors = (t.Dv[up, :], t.Dh[:, rt].T, t.Dv[:, dn].T, t.Dh[lf, :])
        nbits = need_j[co[:, 0], co[:, 1]]
        door_ok = jnp.ones((co.shape[0], S), bool)
        for b, dr in enumerate(doors):
            door_ok = door_ok & ((((nbits >> b) & 1)[:, None] == 0) | (dr > 0))
        return jnp.where(act[:, None], one_comp & anchors_ok & door_ok, ~t.is_room[None, :])

    n_steps = sweeps * B * B
    @jax.jit
    def run(tiles):
        def step(tiles, s):
            c = s % (B * B)
            co = coords[c]
            T = T0 + (T1 - T0) * jnp.minimum(s / (0.7 * n_steps), 1.0)
            p = base.BaseParams(M_reach=jnp.float32(0.0), P=P0, lam=jnp.float32(0.0), T=T)
            L = base.base_logits(ts, tiles, d_inf, co, zeros_block, g0, p, gate, enclose=enclose)
            ok = feasible(tiles, c, co)
            L = jnp.where(ok, L, -1e9)
            u = noise(seed, 1 + level, s, 0, co[:, 0][:, None], co[:, 1][:, None], cand[None, :])
            new = jnp.where(ok.any(1), base.sample_sites(L, u), tiles[co[:, 0], co[:, 1]])
            return base.scatter_tiles(tiles, co, new), None
        tiles, _ = jax.lax.scan(step, tiles, jnp.arange(n_steps))
        return tiles
    return run


def contract_ok(ts, tiles, active, need, anchor):
    """Numpy recheck of every block's contract; returns a list of failures."""
    Dh, Dv, is_room = ts.np_tables["Dh"], ts.np_tables["Dv"], ts.np_tables["is_room"]
    bad = []
    for by, bx in np.ndindex(*active.shape):
        ys, xs = slice(by * B, by * B + B), slice(bx * B, bx * B + B)
        t = tiles[ys, xs]; room = is_room[t]
        if not active[by, bx]:
            if room.any(): bad.append((by, bx, "rooms in an inactive block"))
            continue
        if not room[anchor[ys, xs]].all():
            bad.append((by, bx, "anchor is not a room")); continue
        cells = list(zip(*np.nonzero(room)))
        seen = {cells[0]}; q = [cells[0]]
        while q:
            y, x = q.pop()
            for (dy, dx) in DIRS:
                ny, nx = y + dy, x + dx
                if 0 <= ny < B and 0 <= nx < B and room[ny, nx] and (ny, nx) not in seen:
                    a, b = t[y, x], t[ny, nx]
                    door = Dv[b, a] if dy < 0 else Dv[a, b] if dy > 0 else Dh[a, b] if dx > 0 else Dh[b, a]
                    if door: seen.add((ny, nx)); q.append((ny, nx))
        if len(seen) != len(cells):
            bad.append((by, bx, "block not connected"))
    return bad


def run(seed=0, sweeps=1000, size=20, tileset_name="demo", log=print):
    """Top level, then refine level by level down to the base.  Returns the base
    tile set, the base tiles, and per level (links, active, need, anchor)."""
    ts = tileset.load(tileset_name)
    mts, sig_mask = mask_tileset()
    rng = np.random.default_rng(seed)
    sizes = [size]
    while sizes[-1] % B == 0 and sizes[-1] // B > 8:
        sizes.append(sizes[-1] // B)
    sizes.append(sizes[-1] // B)                      # the top grid of blocks
    gx0 = size // 2
    gate_at = lambda lvl: (0, gx0 // B ** lvl)       # the gate's cell / block at a level
    top = len(sizes) - 1
    links = sample_coarse(sizes[top], gate_at(top), rng)
    log(f"level {top}: {sizes[top]} x {sizes[top]} blocks, {int((links != 0).sum())} linked")
    info = {top: (links,)}
    for lvl in range(top - 1, -1, -1):
        n = sizes[lvl]
        active = links != 0
        active[gate_at(lvl + 1)] = True
        need = place_crossings(links, n, rng, avoid_perimeter=(lvl == 0))
        if lvl == 0:
            need[1, gx0] |= 1 << 0                    # the cell under the gate
        anchor = need != 0
        anchor[gate_at(lvl)] = lvl > 0 or anchor[gate_at(lvl)]
        lts = ts if lvl == 0 else mts
        tiles = initial_fill(lts, active, anchor, n)
        if lvl == 0:
            tiles[0, gx0] = ts.GATE
        bad = contract_ok(lts, tiles, active, need, anchor)
        assert not bad, f"initial fill breaks the contract at level {lvl}: {bad[:3]}"
        sampler = make_sampler(lts, n, active, need, anchor, sweeps, gate_yx=(0, gx0) if lvl == 0 else None,
                               enclose=(lvl == 0), seed=seed, level=lvl)
        tiles = np.asarray(sampler(jnp.asarray(tiles)))
        bad = contract_ok(lts, tiles, active, need, anchor)
        info[lvl] = (links, active, need, anchor, bad)
        if lvl > 0:
            links = matched(sig_mask[tiles])
        log(f"level {lvl}: {n} x {n} {'cells' if lvl == 0 else 'blocks'}, contract failures {len(bad)}")
    return ts, tiles, info


def block_overlay(img, px, size, every=B, colour=(200, 60, 60)):
    img = img.copy()
    for k in range(every, size, every):
        img[k * px, :] = colour
        img[:, k * px] = colour
    return img


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweeps", type=int, default=1000)
    ap.add_argument("--tileset", default="demo")
    ap.add_argument("--png", default=None)
    ap.add_argument("--px", type=int, default=0, help="pixels per cell (default: 16 up to 32 cells, else 8)")
    args = ap.parse_args()
    ts, tiles, info = run(args.seed, args.sweeps, args.size, args.tileset)
    top = max(info)
    print("top-level links:")
    print("\n".join("".join(BOX[v] for v in row) for row in info[top][0]))
    room = ts.np_tables["is_room"][tiles]
    bfs = e2e.exact_bfs(ts, tiles)
    print(f"rooms {room.sum()}, reachable from gate {(room & (bfs < int(INF))).sum()}, "
          f"base contract failures {info[0][4]}")
    k = ts.sig_kind[tiles].ravel()
    print({ts.kinds[i].name: round(c / k.size, 3) for i, c in enumerate(np.bincount(k, minlength=len(ts.kinds))) if c})
    types = np.asarray(tileset.decorate(ts, args.seed, jnp.asarray(tiles)))
    if args.size <= 64:
        print(render.ascii_grid(ts, types))
    if args.png:
        px = args.px or (16 if args.size <= 32 else 8)
        every = B if top == 1 else B * B              # outline the level-1 or level-2 blocks
        render.save_png(block_overlay(render.image(ts, types, px), px, tiles.shape[0], every), args.png)
        print("wrote", args.png)
