"""A sheet of city tiles (castlegen/citytiles.py): one tile per shape class up
to rotation and mirror, a random sample of the classes, grouped by zones.

    uv run python notes/experiments/city_catalogue.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from castlegen import citytiles as ct  # noqa: E402

IMG = os.path.join(os.path.dirname(__file__), "..", "..", "images")


def rot(cs, fs):   # 90 degrees clockwise
    return (cs[3], cs[0], cs[1], cs[2]), (fs[3], fs[0], fs[1], fs[2])


def mirror(cs, fs):   # left-right
    return (cs[1], cs[0], cs[3], cs[2]), (fs[0], fs[3], fs[2], fs[1])


def canon(cs, fs):
    cs = tuple(ct.zclass(c) if c[0] in "HV" else c for c in cs)
    cs = tuple("A" if c[0] in "HV" else c for c in cs)
    best = None
    t = (cs, fs)
    for m in range(2):
        for _ in range(4):
            best = t if best is None or t < best else best
            t = rot(*t)
        t = mirror(*t)
    return best


def main():
    from PIL import Image
    ts = ct.TileSet()
    classes = {}
    for i, (cs, fs) in enumerate(ts.tiles):
        classes.setdefault(canon(cs, fs), i)
    print(ts.T, "tiles,", len(classes), "classes up to symmetry")
    rng = np.random.default_rng(0)
    keys = sorted(classes)
    pick = sorted(rng.choice(len(keys), size=min(240, len(keys)), replace=False),
                  key=lambda k: (sorted(set(keys[k][0])), keys[k][1]))
    cols = 20
    rows = -(-len(pick) // cols)
    grid = np.full((rows * 2, cols * 2), -1, np.int64)   # tiles with a gap between
    for n, k in enumerate(pick):
        grid[(n // cols) * 2, (n % cols) * 2] = classes[keys[k]]
    img = ct.render(ts, grid, 24)
    img = img[: (rows * 2 - 1) * 24, : (cols * 2 - 1) * 24]
    Image.fromarray(img).save(os.path.join(IMG, "city_tile_catalogue.png"))


if __name__ == "__main__":
    main()
