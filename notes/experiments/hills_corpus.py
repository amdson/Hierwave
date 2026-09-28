"""Synthetic corpus of rolling-hill heightmaps (periodic multi-octave 1D
noise, plus occasional flat-topped mesas), saved as valid tile maps.
    .venv/bin/python notes/experiments/hills_corpus.py OUT.npy [N]"""
import sys
import numpy as np
from castlegen import refheights as RH
from castlegen import tileset

G, N_SIDE = 8, 128


def heights(rng, n=N_SIDE):
    x = np.arange(n) / n
    H = np.zeros(n)
    for k, amp in ((1, 22), (2, 12), (3, 8), (5, 5), (8, 3), (13, 1.5)):       # periodic octaves
        H += amp * rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * (k * x + rng.uniform()))
    H = H + rng.uniform(35, 75)
    for _ in range(rng.integers(0, 3)):                                          # a mesa or a pit
        c, w, d = rng.integers(n), rng.integers(6, 24), rng.uniform(-25, 25)
        H[(np.arange(n) - c) % n < w] += d
    return np.clip(np.round(H), G + 1, n - 4).astype(int)


def surface_map(ts, H, n=N_SIDE):
    t = RH.from_heights(ts, H, n, G, "stone", "stone")
    soil, turf = RH._sig(ts, "soil"), RH._sig(ts, "turf")
    for c, h in enumerate(H):                                                    # turf over a few rows of soil
        top = n - h
        t[top:top + 3, c] = soil
        t[top, c] = turf
    return t


if __name__ == "__main__":
    out, N = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 48
    ts = tileset.load("cliffs")
    rng = np.random.default_rng(0)
    maps = np.stack([surface_map(ts, heights(rng)) for _ in range(N)])
    assert all(RH.violations(ts, m, G) == 0 for m in maps)
    np.save(out, maps)
    print("saved", maps.shape, "H mean/std", np.mean([RH.heights(ts, m, G) for m in maps]).round(1))
