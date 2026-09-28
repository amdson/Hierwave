"""Build the 256 x 256 side-view exemplar cliffs_big.npy: a horizontally
periodic heightmap made of varied segments (rolling hills, plains, mesas,
terraces, a canyon, a spire, ramps), layered materials (turf on top, soil of
varying depth, stone below with sparse ore, some bare-rock tops) and the
stone ground band.  Valid for the support rule by construction.
    python -m castlegen.exemplars.make_cliffs_big [SEED]"""
from __future__ import annotations

import os
import sys

import numpy as np

from castlegen import tileset
from castlegen.exemplar import sig_by_name

M, G = 256, 8
LO, HI = G + 4, 118                        # terrain stays within a 128-row map's vertical range
HERE = os.path.dirname(os.path.abspath(__file__))


def segment(kind, w, h0, rng):
    """Heights of one segment of width w starting near h0 -> (heights, bare_top)."""
    x = np.arange(w) / max(w - 1, 1)
    if kind == "hills":
        H = h0 + 10 * np.sin(2 * np.pi * (x * rng.uniform(1, 2.5) + rng.uniform())) + 5 * np.sin(2 * np.pi * x * 5)
    elif kind == "plain":
        H = h0 + rng.normal(0, 0.8, w).cumsum() * 0.3
    elif kind == "mesa":
        top = h0 + rng.uniform(25, 55)
        e = max(2, w // 6)
        H = np.full(w, top)
        H[:e] = h0
        H[-e:] = h0
    elif kind == "terraces":
        steps = rng.integers(3, 5)
        H = h0 + np.floor(x * steps) * rng.uniform(7, 12) * rng.choice([-1, 1])
    elif kind == "canyon":
        H = np.full(w, h0)
        c, cw = w // 2, max(3, w // 4)
        H[c - cw // 2:c + cw // 2] = LO
    elif kind == "spire":
        H = np.full(w, h0) + rng.normal(0, 1, w)
        c = w // 2
        H[c - 2:c + 2] = h0 + rng.uniform(45, 70)
    else:                                                   # ramp
        H = h0 + x * rng.uniform(20, 40) * rng.choice([-1, 1])
    return H, kind == "mesa" and rng.random() < 0.5


def build(seed=0):
    rng = np.random.default_rng(seed)
    ts = tileset.load("cliffs")
    kinds = ["hills", "plain", "mesa", "terraces", "canyon", "spire", "ramp", "hills", "mesa", "plain"]
    rng.shuffle(kinds)
    widths = rng.dirichlet(np.ones(len(kinds)) * 4) * M
    widths = np.maximum(np.round(widths).astype(int), 12)
    widths[-1] += M - widths.sum()
    H, bare, h0 = [], [], 40.0
    for k, w in zip(kinds, widths):
        seg, b = segment(k, int(w), h0, rng)
        H.append(seg)
        bare.append(np.full(int(w), b))
        h0 = float(np.clip(seg[-1] + rng.normal(0, 6), 25, 80))
    H = np.clip(np.round(np.concatenate(H)), LO, HI).astype(int)
    bare = np.concatenate(bare)
    air, stone, soil, turf, ore = (sig_by_name(ts, k) for k in ("air", "stone", "soil", "turf", "ore"))
    t = np.full((M, M), air, np.int32)
    depth = np.clip(np.round(3 + 2 * np.sin(np.arange(M) / 13.0) + rng.normal(0, 0.7, M)), 1, 6).astype(int)
    for c in range(M):
        top = M - H[c]
        t[top:, c] = stone
        if not bare[c]:
            t[top:top + depth[c], c] = soil
            t[top, c] = turf
    body = (t == stone) & (rng.random((M, M)) < 0.03)
    body &= ~np.roll(body, 1, 0) & ~np.roll(body, 1, 1)             # ore specks, no ore-ore contact
    t[body] = ore
    t[M - G:] = stone
    return t


if __name__ == "__main__":
    E = build(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    np.save(os.path.join(HERE, "cliffs_big.npy"), E)
    from castlegen import exemplar as ex
    from castlegen.quantities import support as SP
    ts = tileset.load("cliffs")
    print("unsupported:", SP.support_violations(ts, E, G), " solid fraction:", float(ts.solid[E].mean()))
