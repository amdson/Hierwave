"""Prefab text files -> per-tile voxel templates (.hwt) for the Luanti backend.

Contract: notes/luanti_plan.md.  A template is (L=24, N=8, N=8) uint8 palette
ids, indexed [layer k, row i (north -> south), column j (west -> east)].

Orientation conventions (arrays are (L, rows, cols)):
  mirror_j  flips columns (west <-> east), mirror_i flips rows (north <-> south).
  rot90     rotates clockwise seen from above with north up: the north side of
            a piece becomes its east side (the row-0 end goes to the last column).
Quadrants: zone_<Z> is drawn as the NW quadrant; NE = mirror_j, SW = mirror_i,
SE = both.  Edge strips are drawn as the N half-edge (4 rows x 2 cols, row 0
outermost, row 3 at the tile centre); S = rot 180, E = rot90, W = rot 270.
The centre 2x2 takes row 3 of the highest-priority strip present, for both
centre rows (the strip's inner row repeated), so crossing roads look continuous.

CLI: python -m castlegen.prefabs
"""
from __future__ import annotations

import os
import re
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PALETTE_PATH = os.path.join(ROOT, "luanti", "palette.txt")
DATA_DIR = os.path.join(ROOT, "luanti", "games", "hierwave", "mods", "hierwave_gen", "data")
PREFAB_DIR = os.path.join(HERE, "prefabs")

L = 24
N = 8
KEEP = -1


class PrefabError(ValueError):
    pass


def palette(path=PALETTE_PATH):
    with open(path) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    return {n: i for i, n in enumerate(names)}


PAL = palette()


# ---------------------------------------------------------------- parsing

def parse_prefab(path, pal=None):
    """dict(name, size=(rows, cols), layers=(L, rows, cols) int16, -1 = keep)."""
    pal = PAL if pal is None else pal
    with open(path) as f:
        lines = f.read().splitlines()

    def err(n, msg):
        raise PrefabError(f"{path}:{n}: {msg}")

    def strip_comment(s):
        return re.split(r"(?:^|\s)#", s, maxsplit=1)[0].strip()

    name = os.path.splitext(os.path.basename(path))[0]
    size = None
    legend = {".": pal["air"], "_": KEEP}
    layers = None
    pending = None     # (layer indices, first line no, rows read so far)
    for n, raw in enumerate(lines, 1):
        s = raw.strip()
        if pending is not None:
            ks, rows = pending
            if not s:
                continue
            rows_expected, cols = size
            row = s
            if not (len(row) == cols and all(c in legend for c in row)):
                row = strip_comment(s)
                if not row:
                    continue
                if row.startswith(("layer", "name:", "size:", "legend:")):
                    err(n, f"layer block incomplete ({len(rows)} of {rows_expected} rows)")
                if len(row) != cols:
                    err(n, f"row has length {len(row)}, expected {cols}")
                bad = [c for c in row if c not in legend]
                err(n, f"unknown legend char {bad[0]!r}")
            rows.append([legend[c] for c in row])
            if len(rows) == rows_expected:
                block = np.array(rows, np.int16)
                for k in ks:
                    layers[k] = block
                pending = None
            continue
        s = strip_comment(raw)
        if not s:
            continue
        if s.startswith("name:"):
            name = s[5:].strip()
        elif s.startswith("size:"):
            try:
                r, c = s[5:].split()
                size = (int(r), int(c))
            except ValueError:
                err(n, "size needs two integers: rows cols")
            if size[0] < 1 or size[1] < 1:
                err(n, "size must be positive")
            layers = np.full((L, size[0], size[1]), KEEP, np.int16)
        elif s.startswith("legend:"):
            for tok in s[7:].split():
                if "=" not in tok or len(tok.split("=", 1)[0]) != 1:
                    err(n, f"bad legend entry {tok!r}, expected c=palette_name")
                ch, nm = tok.split("=", 1)
                if nm == "keep":
                    legend[ch] = KEEP
                elif nm not in pal:
                    err(n, f"unknown palette name {nm!r}")
                else:
                    legend[ch] = pal[nm]
        elif s.startswith("layer"):
            if size is None:
                err(n, "layer before size")
            m = re.match(r"layer\s+(\d+)(?:\s*-\s*(\d+))?\s*:\s*(\S*)$", s)
            if not m:
                err(n, f"bad layer line {s!r}")
            a = int(m.group(1))
            b = int(m.group(2)) if m.group(2) is not None else a
            if not (0 <= a <= b < L):
                err(n, f"layer range {a}-{b} outside 0..{L - 1}")
            ks = range(a, b + 1)
            fill = m.group(3)
            if fill:
                if len(fill) != 1:
                    err(n, "whole-layer fill takes one legend char")
                if fill not in legend:
                    err(n, f"unknown legend char {fill!r}")
                for k in ks:
                    layers[k] = legend[fill]
            else:
                pending = (ks, [])
        else:
            err(n, f"unrecognised line {s!r}")
    if size is None:
        raise PrefabError(f"{path}: no size line")
    if pending is not None:
        raise PrefabError(f"{path}: layer block at end of file is incomplete "
                          f"({len(pending[1])} of {size[0]} rows)")
    return dict(name=name, size=size, layers=layers)


# ------------------------------------------------------------- transforms

def mirror_j(a):
    return a[..., ::-1].copy()


def mirror_i(a):
    return a[..., ::-1, :].copy()


def rot90(a, k=1):
    """Clockwise (from above, north up) quarter turns k; north side -> east."""
    return np.rot90(a, -k, axes=(-2, -1)).copy()


def paste(dst, src, i0, j0):
    """Write src over dst at (i0, j0) where src is not keep."""
    r, c = src.shape[-2:]
    view = dst[:, i0:i0 + r, j0:j0 + c]
    np.copyto(view, src, where=src >= 0)


# ---------------------------------------------------------------- assembly

CITY_ZONES = ["P", "R", "C", "D", "I", "W", "A"]
CITY_FEATS = {1: "street", 2: "avenue", 3: "rail", 4: "highway"}   # FEATS index -> name
BAND_DIR_NAMES = ["deep", "water", "sand", "grass", "forest", "hill", "rock", "snow"]


def city_files():
    return (["zone_%s.txt" % z for z in CITY_ZONES]
            + ["edge_%s.txt" % f for f in CITY_FEATS.values()]
            + ["bridge_%s.txt" % f for f in CITY_FEATS.values()]
            + ["wildcard.txt"])


def biome_files(names=None):
    return ["band_%s.txt" % n for n in (names or BAND_DIR_NAMES)]


def missing_files(prefab_dir, files):
    return [f for f in files if not os.path.exists(os.path.join(prefab_dir, f))]


def ground():
    g = np.full((L, N, N), 0, np.int16)
    g[0:3] = PAL["dirt"]
    g[3] = PAL["grass"]
    return g


def _placeholder(fname):
    """A visible stand-in block for a missing prefab file (preview only)."""
    stem = os.path.splitext(fname)[0]
    if stem.startswith("zone_"):
        a = np.full((L, 4, 4), KEEP, np.int16)
        a[3] = PAL["paving"]
        a[4:6, 0:3, 0:3] = PAL[{"P": "leaves", "R": "house_wall", "C": "shop_wall",
                                "D": "tower_wall", "I": "shed_wall", "W": "water",
                                "A": "tarmac"}[stem[5:]]]
        if stem[5:] == "W":
            a[3] = PAL["water"]
            a[2] = PAL["water"]
            a[4:6] = KEEP
        return a
    if stem.startswith(("edge_", "bridge_")):
        a = np.full((L, 4, 2), KEEP, np.int16)
        f = stem.split("_", 1)[1]
        top = {"street": "asphalt", "avenue": "asphalt", "rail": "ballast", "highway": "asphalt"}[f]
        if stem.startswith("bridge_") or f == "highway":
            k = 8 if f == "highway" else 3
            a[k] = PAL["deck"]
        else:
            a[3] = PAL[top]
        return a
    if stem == "wildcard":
        a = np.full((L, N, N), KEEP, np.int16)
        a[3] = PAL["plaza"]
        return a
    a = np.full((L, N, N), KEEP, np.int16)   # band_*
    h = {"deep": 0, "water": 1, "sand": 2, "grass": 3, "forest": 3, "hill": 4, "rock": 5, "snow": 6}[stem[5:]]
    top = {"deep": "deep_water", "water": "water", "sand": "sand", "grass": "grass",
           "forest": "grass", "hill": "dirt", "rock": "rock", "snow": "snow"}[stem[5:]]
    a[0:h + 1] = PAL["dirt"]
    a[h] = PAL[top]
    if stem[5:] == "forest":
        a[4:7, 3:5, 3:5] = PAL["leaves"]
    return a


def load(prefab_dir, fname, allow_missing=False):
    path = os.path.join(prefab_dir, fname)
    if not os.path.exists(path):
        if allow_missing:
            return _placeholder(fname)
        raise FileNotFoundError(path)
    return parse_prefab(path)["layers"]


def _need(p, shape, fname):
    if p.shape[1:] != shape:
        raise PrefabError(f"{fname}: size {p.shape[1:]} but expected {shape}")
    return p


def assemble_city(ts, prefab_dir=PREFAB_DIR, allow_missing=False):
    """(T + 1, L, 8, 8) uint8; last template is the wildcard.  ts: citytiles.TileSet."""
    from castlegen import citytiles as ct
    if not allow_missing:
        miss = missing_files(prefab_dir, city_files())
        if miss:
            raise FileNotFoundError("missing city prefabs: " + ", ".join(miss))
    zq = {}
    for z in CITY_ZONES:
        q = _need(load(prefab_dir, f"zone_{z}.txt", allow_missing), (4, 4), f"zone_{z}")
        zq[z] = (q, mirror_j(q), mirror_i(q), mirror_j(mirror_i(q)))   # NW NE SW SE
    strips = {}
    for f, nm in CITY_FEATS.items():
        for kind in ("edge", "bridge"):
            s = _need(load(prefab_dir, f"{kind}_{nm}.txt", allow_missing), (4, 2), f"{kind}_{nm}")
            strips[kind, f] = (s, rot90(s, 1), rot90(s, 2), rot90(s, 3))   # N E S W
    wild = _need(load(prefab_dir, "wildcard.txt", allow_missing), (N, N), "wildcard")

    base = ground()
    out = np.zeros((ts.T + 1, L, N, N), np.uint8)
    # corner index c: 0 NW, 1 NE, 2 SE, 3 SW
    place = {0: (0, 0, 0), 1: (0, 4, 1), 2: (4, 4, 3), 3: (4, 0, 2)}   # (i0, j0, index in zq tuple)
    # strip placement per side N E S W: (i0, j0)
    spos = [(0, 3), (3, 4), (4, 3), (3, 0)]
    water = "W"
    for t in range(ts.T):
        a = base.copy()
        zc = [ct.zclass(ct.ZONES[c]) for c in ts.corners[t]]
        for c in range(4):
            i0, j0, qi = place[c]
            paste(a, zq[zc[c]][qi], i0, j0)
        fs = ts.feats[t]
        for f in sorted(set(int(x) for x in fs if x)):
            for side in range(4):
                if fs[side] != f:
                    continue
                ca, cb = ct.EDGE_CORNERS[side]
                kind = "bridge" if (zc[ca] == water and zc[cb] == water) else "edge"
                paste(a, strips[kind, f][side], *spos[side])
            # (centre applied after all strips, below)
        if fs.any():
            fmax = int(fs.max())
            side = int(np.nonzero(fs == fmax)[0][0])
            ca, cb = ct.EDGE_CORNERS[side]
            kind = "bridge" if (zc[ca] == water and zc[cb] == water) else "edge"
            row3 = strips[kind, fmax][0][:, 3:4, :]          # N strip inner row, (L, 1, 2)
            centre = np.concatenate([row3, row3], axis=1)
            paste(a, centre, 3, 3)
        out[t] = np.where(a < 0, 0, a)
    out[ts.T] = np.where(np.where(wild < 0, base, wild) < 0, 0, np.where(wild < 0, base, wild))
    return out


def assemble_biomes(prefab_dir=PREFAB_DIR, names=None, allow_missing=False):
    """(8, L, 8, 8) uint8 in the order of castlegen.infinite.biomes().names."""
    if names is None:
        from castlegen.infinite import biomes
        names = biomes().names
    files = biome_files(names)
    if not allow_missing:
        miss = missing_files(prefab_dir, files)
        if miss:
            raise FileNotFoundError("missing biome prefabs: " + ", ".join(miss))
    out = np.zeros((len(names), L, N, N), np.uint8)
    base = ground()
    for t, f in enumerate(files):
        p = _need(load(prefab_dir, f, allow_missing), (N, N), f)
        a = np.where(p < 0, base, p)
        out[t] = np.where(a < 0, 0, a)
    return out


# -------------------------------------------------------------------- hwt

def write_hwt(path, arr):
    arr = np.ascontiguousarray(arr, np.uint8)
    n, l, a, b = arr.shape
    if a != b:
        raise ValueError("templates must be square")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"HWT1" + struct.pack("<III", n, a, l))
        f.write(arr.tobytes())


def read_hwt(path):
    with open(path, "rb") as f:
        head = f.read(16)
        if head[:4] != b"HWT1":
            raise ValueError(f"{path}: bad magic {head[:4]!r}")
        n, a, l = struct.unpack("<III", head[4:])
        data = np.frombuffer(f.read(), np.uint8)
    if data.size != n * l * a * a:
        raise ValueError(f"{path}: size mismatch")
    return data.reshape(n, l, a, a).copy()


def main():
    from castlegen.citytiles import TileSet
    os.makedirs(DATA_DIR, exist_ok=True)
    cdir, bdir = os.path.join(PREFAB_DIR, "city"), os.path.join(PREFAB_DIR, "biomes")
    miss = missing_files(cdir, city_files())
    if miss:
        print("city: skipped, missing in %s: %s" % (cdir, ", ".join(miss)))
    else:
        arr = assemble_city(TileSet(), cdir)
        write_hwt(os.path.join(DATA_DIR, "city.hwt"), arr)
        print("city.hwt:", arr.shape)
    from castlegen.infinite import biomes
    miss = missing_files(bdir, biome_files(biomes().names))
    if miss:
        print("biomes: skipped, missing in %s: %s" % (bdir, ", ".join(miss)))
    else:
        arr = assemble_biomes(bdir)
        write_hwt(os.path.join(DATA_DIR, "biomes.hwt"), arr)
        print("biomes.hwt:", arr.shape)


if __name__ == "__main__":
    sys.exit(main())
