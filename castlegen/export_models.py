"""Export the contract tile models (notes/luanti_plan.md section 0) to .hwm
files for the C sampler (castlegen/csrc/hwsampler.c documents the format).

    biomes  castlegen.infinite.biomes(), Schedule.anneal(S=64)
    city    castlegen.citytiles with BOUNDARY_W = 0.5 and ZONE_W below
            (FEAT_W unchanged), SocketModel u = -log w, wild_u = 20,
            BlockSchedule.anneal(B=8, Sb=3, K=32, shift=True, restart=False,
            beta0=1.0, gamma0=2.0)  (block Gibbs)
    city_flat  the same tiles on the old flat
            Schedule.anneal(S=64, beta0=1.0, gamma0=2.0)
    marian  castlegen.marian.tile_model(load()) (TileModel3D, 454 modules,
            L = 6, Empty's per-layer weight from marian.AIR_WEIGHTS read at
            export time), BlockSchedule.anneal(B=8, Sb=3, K=32, shift=True,
            restart=False, beta0=1.0, gamma0=2.0)

biomes_model() / city_model() / city_flat_model() / marian_model() return (model, schedule) so tests can run
castlegen.infinite.generate on the same thing.  Run as a script to write
luanti/games/hierwave/mods/hierwave_gen/data/{biomes,city,marian}.hwm."""
from __future__ import annotations

import os
import struct

import numpy as np

from castlegen.infinite import BlockSchedule, Schedule, SocketModel, TileModel, TileModel3D, biomes

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "luanti", "games", "hierwave", "mods", "hierwave_gen", "data")

CITY_BOUNDARY_W = 0.5
CITY_ZONE_W = {"P": 0.4, "R": 1.0, "C": 1.0, "D": 1.0, "I": 1.0, "W": 0.8, "A": 0.6}
CITY_WILD_U = 20.0


def biomes_model(S=64, **kw):
    return biomes(), Schedule.anneal(S=S, **kw)


def city_tiles():
    """citytiles.TileSet under the contract weights; module globals restored."""
    from castlegen import citytiles as ct
    old = ct.BOUNDARY_W, dict(ct.ZONE_W)
    try:
        ct.BOUNDARY_W = CITY_BOUNDARY_W
        ct.ZONE_W.clear()
        ct.ZONE_W.update(CITY_ZONE_W)
        return ct.TileSet()
    finally:
        ct.BOUNDARY_W = old[0]
        ct.ZONE_W.clear()
        ct.ZONE_W.update(old[1])


def city_socket_model():
    ts = city_tiles()
    return SocketModel(-np.log(ts.w), ts.sN, ts.sE, ts.sS, ts.sW, wild_u=CITY_WILD_U)


def city_flat_model(S=64, beta0=1.0, gamma0=2.0, **kw):
    return city_socket_model(), Schedule.anneal(S=S, beta0=beta0, gamma0=gamma0, **kw)


def city_model(B=8, Sb=3, K=32, shift=True, restart=False, beta0=1.0, gamma0=2.0, **kw):
    return city_socket_model(), BlockSchedule.anneal(B=B, Sb=Sb, K=K, shift=shift, restart=restart,
                                                     beta0=beta0, gamma0=gamma0, **kw)


def marian_model(B=8, Sb=3, K=32, shift=True, restart=False, beta0=1.0, gamma0=1.5, gamma1=5.0, **kw):
    """Hard-weight ramp 1.5 -> 5 (notes/marian_test.md): 0.62% wildcards at K32
    against 1.25% with the default 2 -> 8; K48 0.40%, K64 0.34%."""
    from castlegen import marian
    return marian.tile_model(marian.load()), BlockSchedule.anneal(B=B, Sb=Sb, K=K, shift=shift, restart=restart,
                                                                 beta0=beta0, gamma0=gamma0, gamma1=gamma1, **kw)


def socket_base(model: SocketModel, sched):
    """exactly the base table the numba kernel gets (rows: sweeps, or the
    raveled (Sb, K) block rows)."""
    beta = sched.beta.ravel()
    return np.exp(-beta[:, None] * (model.u - model.u.min())[None, :])


def write_hwm(path, model, sched):
    """Format HWM2, documented in castlegen/csrc/hwsampler.c."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    block = isinstance(sched, BlockSchedule)
    beta, gamma = sched.beta.ravel(), sched.gamma.ravel()
    R = beta.size
    bk = struct.pack("<IIII", sched.B, sched.Sb, sched.K, int(bool(sched.shift))) if block else bytes(16)
    f64 = lambda a: np.ascontiguousarray(a, "<f8").tobytes()
    i32 = lambda a: np.ascontiguousarray(a, "<i4").tobytes()
    if isinstance(model, TileModel):
        if block:
            raise TypeError("block schedules are for SocketModel and TileModel3D")
        head = struct.pack("<4sIIII", b"HWM2", 0, model.T, R, 0) + bk
        body = (f64(beta) + f64(gamma) + f64(model.u) + f64(model.Eh) + f64(model.Ev)
                + i32(model.Hh) + i32(model.Hv))
    elif isinstance(model, TileModel3D):
        if model.soft:
            raise TypeError("TileModel3D with soft E is not supported by the C sampler")
        head = struct.pack("<4sIIII", b"HWM2", 2, model.T, R, 1 if model.wild else 0) + bk
        head += struct.pack("<I", model.L)
        body = (f64(beta) + f64(gamma) + f64(model.u)
                + np.ascontiguousarray(model.F, np.uint8).tobytes()
                + f64(model.base(beta)))
    elif isinstance(model, SocketModel):
        wild = model.wild_u is not None
        head = struct.pack("<4sIIII", b"HWM2", 1, model.T, R, 1 if wild else 0) + bk
        body = (f64(beta) + f64(gamma) + f64(model.u)
                + struct.pack("<d", model.wild_u if wild else 0.0)
                + i32(model.sN) + i32(model.sE) + i32(model.sS) + i32(model.sW)
                + f64(socket_base(model, sched)))
    else:
        raise TypeError(type(model))
    with open(path, "wb") as fh:
        fh.write(head + body)
    return path


def main(flat=False, only=None):
    """flat=True (python -m castlegen.export_models --flat) also writes
    city_flat.hwm; only: names to write (python -m castlegen.export_models
    marian writes just marian.hwm)."""
    allm = [("biomes", biomes_model), ("city", city_model), ("marian", marian_model), ("city_flat", city_flat_model)]
    makers = [(n, f) for n, f in allm if (n in only if only else n != "city_flat" or flat)]
    for name, make in makers:
        m, s = make()
        p = write_hwm(os.path.join(DATA, f"{name}.hwm"), m, s)
        print(f"{name}: T = {m.T}, rows = {s.beta.size} -> {os.path.relpath(p, ROOT)} ({os.path.getsize(p)} bytes)")


if __name__ == "__main__":
    import sys
    main(flat="--flat" in sys.argv, only=[a for a in sys.argv[1:] if not a.startswith("-")])
