"""Rendering of a type-id grid: ASCII glyphs, or a pixel image with one
px x px sprite per cell.

Sprite colour, most specific first: a variant value's "render", the kind's
"render", else a random colour per (kind, variant) that is stable across
runs.  A render is {"color": "#rrggbb"} or {"image": "file.png"} (path
relative to the tile-set file; resized nearest-neighbour and rotated with the
tile).  Room sprites get a border drawn per side from the socket's "draw"
style: "line", "gap" (a doorway) or "none"; {"borders": false} turns it off.
Images need Pillow (pip install castlegen[render]).
"""
from __future__ import annotations

import colorsys
import os
from functools import lru_cache

import numpy as np

from .tileset import TileSet

BORDER = np.array([40, 34, 30], np.uint8)


def ascii_grid(ts: TileSet, types) -> str:
    types = np.asarray(types)
    k, r, _ = ts.decode(types)
    return "\n".join("".join(ts.kinds[kk].glyphs[rr] for kk, rr in zip(krow, rrow))
                     for krow, rrow in zip(k, r))


def image(ts: TileSet, types, px: int = 8) -> np.ndarray:
    """(H * px, W * px, 3) uint8."""
    types = np.asarray(types)
    H, W = types.shape
    uniq, inv = np.unique(types, return_inverse=True)
    sprites = np.stack([sprite(ts, int(t), px) for t in uniq])        # (U, px, px, 3)
    out = sprites[inv.reshape(H, W)]                                   # (H, W, px, px, 3)
    return out.transpose(0, 2, 1, 3, 4).reshape(H * px, W * px, 3)


def save_png(img: np.ndarray, path: str):
    from PIL import Image
    Image.fromarray(img).save(path)


def sprite(ts: TileSet, tid: int, px: int) -> np.ndarray:
    k, r, vals = ts.decode(tid)
    k, r = int(k), int(r)
    kind = ts.kinds[k]
    spec = kind.render
    for (_, values), v in zip(kind.axes, vals):
        vr = values[int(v)][2]
        if "color" in vr or "image" in vr:
            spec = vr
    if "image" in spec:
        img = _load_image(os.path.join(ts.base_dir, spec["image"]), px)
        img = np.rot90(img, k=-(kind.rotations[r] // 90)).copy()
    else:
        rgb = _hex(spec["color"]) if "color" in spec else _random_colour(int(ts.kind_offset[k]), int(tid - ts.kind_offset[k]) % int(ts.np_tables["kind_V"][k]))
        img = np.broadcast_to(rgb, (px, px, 3)).copy()
    if spec.get("borders", True) and not kind.solid and kind.name != "gate":
        sockets = ts.sig_sockets[ts.sig_of_type(tid)]
        _draw_borders(img, [ts.socket_draw[s] for s in sockets])
    return img


def _draw_borders(img, styles):
    px = img.shape[0]
    w = max(1, px // 8)
    gap = slice(px // 3, px - px // 3)
    edges = [(slice(0, w), slice(None)), (slice(None), slice(px - w, px)),
             (slice(px - w, px), slice(None)), (slice(None), slice(0, w))]   # N E S W
    for side, style in enumerate(styles):
        if style == "none":
            continue
        img[edges[side]] = BORDER
        if style == "gap":
            if side in (0, 2):
                img[edges[side][0], gap] = img[px // 2, px // 2]
            else:
                img[gap, edges[side][1]] = img[px // 2, px // 2]


def _hex(s):
    s = s.lstrip("#")
    return np.array([int(s[i:i + 2], 16) for i in (0, 2, 4)], np.uint8)


def _random_colour(kind_key: int, variant: int):
    h = (kind_key * 0x9E3779B1 + variant * 0x85EBCA6B + 0x1234567) & 0xFFFFFFFF
    for _ in range(2):                                  # murmur3 finaliser
        h ^= h >> 16; h = (h * 0x85EBCA6B) & 0xFFFFFFFF
        h ^= h >> 13; h = (h * 0xC2B2AE35) & 0xFFFFFFFF
        h ^= h >> 16
    hue, sat, val = (h & 0x3FF) / 1024, 0.35 + ((h >> 10) & 0xFF) / 800, 0.7 + ((h >> 18) & 0xFF) / 1100
    return np.array([round(c * 255) for c in colorsys.hsv_to_rgb(hue, sat, val)], np.uint8)


@lru_cache(maxsize=256)
def _load_image(path: str, px: int) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("image tiles need Pillow: pip install castlegen[render]") from e
    return np.asarray(Image.open(path).convert("RGB").resize((px, px), Image.NEAREST), np.uint8)
