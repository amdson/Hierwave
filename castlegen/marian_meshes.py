"""Marian Kleineberg's 3D WFC city blocks as voxel templates for Luanti.

Source data (MIT, https://marian42.de/article/wfc/, downloaded to
cache/marian/Assets and treated as untrusted data: it is only parsed here):

  blocks.fbx          binary FBX, one mesh per block (pure-Python reader below)
  blocks.fbx.meta     Unity import settings; maps mesh fileIDs 43000nn to names
  Prototypes.prefab   one GameObject per module prototype (Unity YAML)
  ModuleData.asset    454 modules = prototypes x distinct rotations

Conventions (verified in tests/test_marian_meshes.py and by the neighbour
check in notes/experiments/marian_voxels.py):
  * The FBX stores each block mesh in a 2 x 2 x 2 box centred on the origin,
    y up (extents [-1, 1]; a few meshes poke out of it and are clipped).
    The importer scale is 1, so one Unity unit is one FBX unit and a block
    is BLOCK_SIZE = 2 units.
  * Unity is left handed: +x east, +y up, +z north.  FBX is right handed; the
    x axis is negated on import (X_SIGN = -1, picked by the same check:
    0.69 against 0.49 for x_sign +1 with the best rotation sense).
  * Each prototype GameObject in Prototypes.prefab carries a scene rotation
    (multiples of 90 degrees about y) but it must NOT be applied to the mesh:
    the neighbour check (marian_voxels.py check) gives boundary agreement
    0.69 without it against 0.49 with either sign (0.19 for random pairs).
    The meshes are authored in the frame the module faces refer to.
  * Module.Rotation r: the module's face in horizontal direction d is the
    prototype's face Orientations.Rotate(d, r), i.e. LEFT(-x) shows the
    prototype's BACK(-z), BACK shows RIGHT(+x), ...  The prototype's -z face
    lands on -x, so the mesh is rotated by +90 r degrees about +y,
    Quaternion.Euler(0, 90 r, 0):  (x, z) -> (x cos + z sin, -x sin + z cos),
    clockwise seen from above.

Template array: uint8 palette ids, shape (455, k, k, k), index
[module][y][i][j] with y up, i = row north -> south (i grows as Unity z
falls), j = column west -> east.  Module order is the order of Modules in
ModuleData.asset (0..453; 'Solid' is a full stone cube since its mesh is only
edge stubs, 'Empty' is air); template 454 is the wildcard filler, a solid
concrete cube (a wildcard then shows as an obvious block instead of a hole).

Written to luanti/games/hierwave/mods/hierwave_gen/data/marian.hwt in the
HWT1 format of castlegen.prefabs.write_hwt, with L = k layers and N = k, so
each template is k x k x k (the file header is n, N, L = 455, k, k).

The FBX has no materials (one grey Unity material), so palette names come
from the prototype name and the triangle normal (material_for).
"""

import hashlib
import os
import re
import struct
import zlib

import numpy as np
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "cache", "marian", "Assets")
PALETTE_FILE = os.path.join(ROOT, "luanti", "palette.txt")
HWT_PATH = os.path.join(ROOT, "luanti", "games", "hierwave", "mods",
                        "hierwave_gen", "data", "marian.hwt")
CACHE_DIR = os.path.join(ROOT, "cache", "marian")

BLOCK_SIZE = 2.0
X_SIGN = -1.0        # FBX (right handed) -> Unity (left handed): negate x
TURN_SIGN = 0        # 0: the prefab's own scene rotation is NOT applied (see docstring)
ROT_SIGN = 1         # module Rotation r turns the mesh by +90 r about +y
N_MODULES = 454


def palette():
    names = open(PALETTE_FILE).read().split()
    return {n: i for i, n in enumerate(names)}


# --------------------------------------------------------------------------
# FBX reader (binary, versions 7100-7700)
# --------------------------------------------------------------------------

def parse_fbx(path):
    """Return (version, [node]) with node = (name, props, children).  Array
    properties come back as numpy arrays."""
    d = open(path, "rb").read()
    if not d.startswith(b"Kaydara FBX Binary"):
        raise ValueError("not a binary FBX file")
    ver = struct.unpack_from("<I", d, 23)[0]
    wide = ver >= 7500
    dtypes = {"f": "<f4", "d": "<f8", "i": "<i4", "l": "<i8", "b": "u1"}
    scalar = {"Y": "<h", "C": "<?", "F": "<f", "D": "<d", "I": "<i", "L": "<q"}

    def node(pos):
        if wide:
            end, npr, _, nl = struct.unpack_from("<QQQB", d, pos)
            pos += 25
        else:
            end, npr, _, nl = struct.unpack_from("<IIIB", d, pos)
            pos += 13
        if end == 0:
            return None, pos
        name = d[pos:pos + nl].decode("latin1")
        pos += nl
        props = []
        for _ in range(npr):
            t = chr(d[pos])
            pos += 1
            if t in scalar:
                props.append(struct.unpack_from(scalar[t], d, pos)[0])
                pos += struct.calcsize(scalar[t])
            elif t in dtypes:
                n, enc, cl = struct.unpack_from("<III", d, pos)
                pos += 12
                raw = d[pos:pos + cl]
                pos += cl
                if enc:
                    raw = zlib.decompress(raw)
                props.append(np.frombuffer(raw, dtypes[t], count=n))
            elif t in "SR":
                n = struct.unpack_from("<I", d, pos)[0]
                pos += 4
                props.append(bytes(d[pos:pos + n]))
                pos += n
            else:
                raise ValueError(f"bad FBX property type {t!r}")
        kids = []
        while pos < end:
            k, pos = node(pos)
            if k is None:
                break
            kids.append(k)
        return (name, props, kids), end

    out, pos = [], 27
    while pos < len(d):
        n, pos = node(pos)
        if n is None:
            break
        out.append(n)
    return ver, out


def _child(n, name):
    for k in n[2]:
        if k[0] == name:
            return k
    return None


def _obj_name(props):
    return props[1].split(b"\x00")[0].decode("latin1")


def read_fbx_meshes(path=None):
    """dict model name -> (V (n,3) float64 raw FBX coords, F (m,3) int tris).

    The mesh is the Geometry connected to a Model of the same name.  Polygons
    are fan-triangulated."""
    path = path or os.path.join(ASSETS, "blocks.fbx")
    _, top = parse_fbx(path)
    objs = next(n for n in top if n[0] == "Objects")
    conns = next(n for n in top if n[0] == "Connections")
    byid = {k[1][0]: k for k in objs[2]}
    geo_of = {}
    for c in conns[2]:
        if c[0] == "C" and c[1][0] == b"OO":
            a, b = c[1][1], c[1][2]
            if a in byid and b in byid and byid[a][0] == "Geometry" \
                    and byid[b][0] == "Model":
                geo_of[b] = a
    meshes = {}
    for mid, gid in geo_of.items():
        name = _obj_name(byid[mid][1])
        g = byid[gid]
        v = np.asarray(_child(g, "Vertices")[1][0], np.float64).reshape(-1, 3)
        pi = np.asarray(_child(g, "PolygonVertexIndex")[1][0], np.int64)
        ends = np.nonzero(pi < 0)[0]
        idx = np.where(pi < 0, ~pi, pi)
        tris, s = [], 0
        for e in ends:
            poly = idx[s:e + 1]
            for t in range(1, len(poly) - 1):
                tris.append((poly[0], poly[t], poly[t + 1]))
            s = e + 1
        if name not in meshes:
            meshes[name] = (v, np.asarray(tris, np.int64).reshape(-1, 3))
    return meshes


def fbx_to_unity(v):
    """Raw FBX coordinates -> Unity block space (x east, y up, z north)."""
    v = np.array(v, np.float64)
    v[:, 0] *= X_SIGN
    return v


# --------------------------------------------------------------------------
# Unity YAML: prototypes and modules
# --------------------------------------------------------------------------

def _yaml_objects(path):
    t = open(path, encoding="utf-8-sig").read()
    parts = re.split(r"^--- !u!(\d+) &(\d+)[^\n]*\n", t, flags=re.M)
    return {int(parts[i + 1]): (int(parts[i]), parts[i + 2])
            for i in range(1, len(parts), 3)}


def mesh_names_by_file_id():
    """fileID 43000nn (mesh sub-asset) -> mesh name, from blocks.fbx.meta."""
    t = open(os.path.join(ASSETS, "blocks.fbx.meta")).read()
    return {int(a): b.strip()
            for a, b in re.findall(r"^    (43\d{5}): (.*)$", t, re.M)}


def _quat_to_y_steps(qx, qy, qz, qw):
    """World rotation quaternion -> number of 90 degree turns about +y."""
    if abs(qx) > 0.01 or abs(qz) > 0.01:
        raise ValueError("prototype rotation is not about y")
    ang = 2 * np.arctan2(qy, qw)
    steps = ang / (np.pi / 2)
    if abs(steps - round(steps)) > 0.02:
        raise ValueError("prototype rotation is not a multiple of 90 degrees")
    return int(round(steps)) % 4


def load_modules():
    """Return (protos, modules).

    protos: list of dict(name, mesh, turns) in order of first use; `turns` is
    the prototype's own world rotation (90 degree steps about +y).
    modules: list of dict(name, proto, rotation, neighbors) in asset order;
    neighbors is a (6, 454) bool array [direction][module] with directions
    LEFT -x, DOWN, BACK -z, RIGHT +x, UP, FORWARD +z.
    """
    objs = _yaml_objects(os.path.join(ASSETS, "Prototypes.prefab"))
    ids = mesh_names_by_file_id()
    md = open(os.path.join(ASSETS, "ModuleData.asset"),
              encoding="utf-8-sig").read()
    head, _, rest = md.partition("  Modules:\n")
    entries = re.split(r"^  - Name: ", rest, flags=re.M)[1:]
    protos, pindex, modules = [], {}, []
    for e in entries:
        name = e.split("\n", 1)[0].strip()
        pid = int(re.search(r"Prototype: \{fileID: (\d+)", e).group(1))
        rot = int(re.search(r"^    Rotation: (\d+)", e, re.M).group(1))
        nb = np.zeros((6, N_MODULES), bool)
        for d, h in enumerate(re.findall(r"- data: ([0-9a-f]*)", e)[:6]):
            bits = np.unpackbits(np.frombuffer(bytes.fromhex(h), np.uint8),
                                 bitorder="little")
            m = min(len(bits), N_MODULES)
            nb[d, :m] = bits[:m].astype(bool)
        if pid not in pindex:
            _, body = objs[pid]
            go = int(re.search(r"m_GameObject: \{fileID: (\d+)", body).group(1))
            _, gb = objs[go]
            comps = [int(x) for x in
                     re.findall(r"component: \{fileID: (\d+)", gb)]
            pname = re.search(r"m_Name: (.*)", gb).group(1).strip()
            tr = next(objs[c][1] for c in comps if objs[c][0] == 4)
            q = [float(x) for x in re.search(
                r"m_LocalRotation: \{x: (\S+),\s+y: (\S+),\s+z: (\S+),\s+w: (\S+)\}",
                tr).groups()]
            turns = _quat_to_y_steps(*q)
            mesh = None
            for c in comps:
                if objs[c][0] == 33:
                    fid = int(re.search(r"m_Mesh: \{fileID: (\d+)",
                                        objs[c][1]).group(1))
                    mesh = ids.get(fid)
            pindex[pid] = len(protos)
            protos.append(dict(name=pname, mesh=mesh, turns=turns))
        modules.append(dict(name=name, proto=pindex[pid], rotation=rot,
                            neighbors=nb))
    return protos, modules


# --------------------------------------------------------------------------
# materials (no materials in the FBX: by prototype name and triangle normal)
# --------------------------------------------------------------------------

def material_for(pname, normal):
    """Palette name for a triangle of prototype `pname` with unit `normal`
    (Unity axes)."""
    n = pname.lower().replace("_", " ")
    up = normal[..., 1]
    if n == "empty":
        return "air"
    if n.startswith("roof"):
        # slopes and tops are tiles, steep side faces (gables, walls) plaster
        return "house_roof" if abs(up) > 0.25 else "house_wall"
    if n.startswith("railing") or "railing" in n:
        return "fence"
    if n.startswith("pillars"):
        return "pillar"
    if n.startswith("water fountain") or "fountain" in n:
        return "fountain" if abs(up) < 0.9 else "water"
    if "window" in n:
        return "shop_wall" if n.startswith("interior") else "house_wall"
    if n.startswith("interior"):
        if up > 0.8:
            return "wood"
        return "shop_wall"
    if "bridge" in n or n.startswith("tunnel"):
        return "brick" if abs(up) < 0.9 else "concrete"
    if "stairs" in n:
        return "concrete"
    if n.startswith("floor") or n.startswith("balcony"):
        return "deck"
    if n.startswith("high wall"):
        return "tower_wall"
    if n.startswith("wall"):
        return "house_wall" if up < 0.9 else "concrete"
    return "concrete"


# --------------------------------------------------------------------------
# voxelisation
# --------------------------------------------------------------------------

def _tri_arrays(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(b - a, c - a)
    area2 = np.linalg.norm(cr, axis=1)
    keep = area2 > 1e-12
    n = np.zeros_like(cr)
    n[keep] = cr[keep] / area2[keep, None]
    return a, b, c, n, keep


def voxelize(V, F, pname, k, S=4):
    """Voxelise unity-space triangles of one block into a (k, k, k) uint8
    array [x][y][z] of palette ids (0 = air); +x east, +y up, +z north.

    Subsamples S per cell and axis.  Inside = nonzero winding of rays along
    +y (columns that do not close are ignored); every triangle is also
    rasterised so that thin, open parts (railings, plates) survive.  A cell is
    solid when at least half of its subcells are inside, or when thin surface
    (surface that is not next to a deep interior) covers 15%+ of it."""
    pal = palette()
    n_ = k * S
    h = BLOCK_SIZE / n_
    a, b, c, nrm, keep = _tri_arrays(V, F)
    a, b, c, nrm = a[keep], b[keep], c[keep], nrm[keep]
    centers = (np.arange(n_) + 0.5) * h - BLOCK_SIZE / 2
    # irrational jitter so rays avoid mesh edges lying on the sample lattice
    jx, jz = 0.00123456789 * h, 0.00234567891 * h

    # --- winding fill along y
    diff = np.zeros((n_, n_, n_ + 1), np.int32)     # [x][z][y-bin]
    for ta, tb, tc, tn in zip(a, b, c, nrm):
        if abs(tn[1]) < 1e-9:
            continue
        xs = np.array([ta[0], tb[0], tc[0]])
        zs = np.array([ta[2], tb[2], tc[2]])
        ix0 = max(int(np.ceil((xs.min() - jx + BLOCK_SIZE / 2) / h - 0.5)), 0)
        ix1 = min(int(np.floor((xs.max() - jx + BLOCK_SIZE / 2) / h - 0.5)), n_ - 1)
        iz0 = max(int(np.ceil((zs.min() - jz + BLOCK_SIZE / 2) / h - 0.5)), 0)
        iz1 = min(int(np.floor((zs.max() - jz + BLOCK_SIZE / 2) / h - 0.5)), n_ - 1)
        if ix1 < ix0 or iz1 < iz0:
            continue
        px = centers[ix0:ix1 + 1] + jx
        pz = centers[iz0:iz1 + 1] + jz
        PX, PZ = np.meshgrid(px, pz, indexing="ij")
        d = (tb[2] - tc[2]) * (ta[0] - tc[0]) + (tc[0] - tb[0]) * (ta[2] - tc[2])
        if abs(d) < 1e-14:
            continue
        l1 = ((tb[2] - tc[2]) * (PX - tc[0]) + (tc[0] - tb[0]) * (PZ - tc[2])) / d
        l2 = ((tc[2] - ta[2]) * (PX - tc[0]) + (ta[0] - tc[0]) * (PZ - tc[2])) / d
        l3 = 1 - l1 - l2
        inside = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        if not inside.any():
            continue
        y = l1 * ta[1] + l2 * tb[1] + l3 * tc[1]
        # first subcell centre at or above the hit
        yb = np.ceil((y + BLOCK_SIZE / 2) / h - 0.5).astype(np.int64)
        yb = np.clip(yb, 0, n_)
        sgn = 1 if tn[1] > 0 else -1
        ii, jj = np.nonzero(inside)
        np.add.at(diff, (ii + ix0, jj + iz0, yb[ii, jj]), sgn)
    wind = np.cumsum(diff, axis=2)[:, :, :n_]
    leak = np.cumsum(diff, axis=2)[:, :, n_] != 0
    inside_v = (wind != 0) & ~leak[:, :, None]            # [x][z][y]
    inside_v = inside_v.transpose(0, 2, 1)                 # [x][y][z]

    # --- surface raster with material labels
    names = ["air"] + sorted(set(pal) - {"air"})
    surf = np.zeros((n_, n_, n_), bool)
    lab = np.zeros((n_, n_, n_), np.uint8)
    cache = {}
    for ta, tb, tc, tn in zip(a, b, c, nrm):
        e = max(np.linalg.norm(tb - ta), np.linalg.norm(tc - tb),
                np.linalg.norm(ta - tc))
        m = max(int(np.ceil(e / (h * 0.5))), 1)
        if m not in cache:
            ij = [(i, j) for i in range(m + 1) for j in range(m + 1 - i)]
            ij = np.array(ij, np.float64) / m
            cache[m] = ij
        ij = cache[m]
        pts = (ta[None] + ij[:, :1] * (tb - ta)[None] + ij[:, 1:] * (tc - ta)[None])
        idx = np.floor((pts + BLOCK_SIZE / 2) / h).astype(np.int64)
        ok = ((idx >= 0) & (idx < n_)).all(axis=1)
        idx = idx[ok]
        if not len(idx):
            continue
        mat = pal[material_for(pname, tn)]
        surf[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        lab[idx[:, 0], idx[:, 1], idx[:, 2]] = mat

    deep = ndimage.binary_erosion(inside_v, iterations=1, border_value=1)
    near_deep = ndimage.binary_dilation(deep, structure=np.ones((3, 3, 3), bool))
    thin = surf & ~near_deep

    def cellmean(x):
        return x.reshape(k, S, k, S, k, S).mean(axis=(1, 3, 5))

    solid = (cellmean(inside_v | surf & near_deep) >= 0.5) | (cellmean(thin) >= 0.15)
    solid &= (cellmean(inside_v | surf) > 0)

    # --- material per cell: majority of surface labels in the cell, else
    # nearest labelled cell
    out = np.zeros((k, k, k), np.uint8)
    nmat = len(pal)
    counts = np.zeros((k, k, k, nmat), np.int32)
    l6 = lab.reshape(k, S, k, S, k, S).transpose(0, 2, 4, 1, 3, 5).reshape(k, k, k, -1)
    s6 = surf.reshape(k, S, k, S, k, S).transpose(0, 2, 4, 1, 3, 5).reshape(k, k, k, -1)
    for mid in range(1, nmat):
        counts[..., mid] = ((l6 == mid) & s6).sum(-1)
    has = counts.sum(-1) > 0
    out[has] = counts[has].argmax(-1).astype(np.uint8)
    out[~solid] = 0
    labelled = (out > 0)
    if solid.any():
        if not labelled.any():
            out[solid] = pal["concrete"]
        else:
            _, (ix, iy, iz) = ndimage.distance_transform_edt(
                ~labelled, return_indices=True)
            nearest = out[ix, iy, iz]
            out[solid & ~labelled] = nearest[solid & ~labelled]
    return out


def rotate_xyz(vox, steps):
    """Rotate an [x][y][z] voxel array by `steps` * 90 degrees about +y the
    way Quaternion.Euler(0, 90 steps, 0) rotates a Unity mesh: (x, z) ->
    (z, -x) per step."""
    return np.rot90(vox, -steps * ROT_SIGN, axes=(0, 2))


def to_template(vox):
    """[x][y][z] (z north) -> [y][i][j]: i = row north -> south, j = west -> east."""
    return np.ascontiguousarray(vox.transpose(1, 2, 0)[:, ::-1, :])


def from_template(t):
    """Inverse of to_template."""
    return np.ascontiguousarray(t[:, ::-1, :].transpose(2, 0, 1))


def rotate_unity_y(v, steps):
    """Rotate Nx3 Unity points by steps * 90 degrees about +y (same sense)."""
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][steps % 4]
    o = np.array(v, np.float64)
    o[:, 0] = v[:, 0] * c + v[:, 2] * s
    o[:, 2] = -v[:, 0] * s + v[:, 2] * c
    return o


def prototype_voxels(k=8, S=4):
    """dict prototype index -> (k,k,k) [x][y][z] voxels of the prototype in its
    own world orientation (prefab rotation applied), plus (protos, modules)."""
    protos, modules = load_modules()
    meshes = read_fbx_meshes()
    pal = palette()
    out = {}
    for pi, p in enumerate(protos):
        if p["mesh"] is None or p["mesh"] not in meshes:
            out[pi] = np.zeros((k, k, k), np.uint8)
            continue
        if p["name"] == "Solid":
            # the "Solid" mesh is only a few edge stubs (the block itself is
            # invisible underground fill in the original); make it a full cube
            out[pi] = np.full((k, k, k), pal["stone"], np.uint8)
            continue
        V, F = meshes[p["mesh"]]
        U = rotate_unity_y(fbx_to_unity(V), TURN_SIGN * p["turns"])
        out[pi] = voxelize(U, F if X_SIGN > 0 else F[:, ::-1], p["name"], k, S)
    return out, protos, modules


def _cache_key():
    h = hashlib.md5()
    h.update(open(__file__, "rb").read())
    return h.hexdigest()[:10]


def module_templates(k=8, S=4, use_cache=True):
    """uint8 (455, k, k, k) [module][y][i][j]; see the module docstring."""
    path = os.path.join(CACHE_DIR, f"vox_k{k}_s{S}_{_cache_key()}.npy")
    if use_cache and os.path.exists(path):
        return np.load(path)
    pv, protos, modules = prototype_voxels(k, S)
    pal = palette()
    out = np.zeros((len(modules) + 1, k, k, k), np.uint8)
    for mi, m in enumerate(modules):
        out[mi] = to_template(rotate_xyz(pv[m["proto"]], m["rotation"]))
    out[len(modules)] = pal["concrete"]
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(path, out)
    return out


def write_marian_hwt(k=8, path=HWT_PATH):
    from castlegen.prefabs import write_hwt
    arr = module_templates(k)
    write_hwt(path, arr)
    return arr


if __name__ == "__main__":
    import sys
    kk = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    a = write_marian_hwt(kk)
    print("marian.hwt", a.shape, "solid fraction", float((a > 0).mean()))
