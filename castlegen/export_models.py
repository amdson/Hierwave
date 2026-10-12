"""Export the contract tile models (notes/luanti_plan.md section 0) to .hwm
files for the C sampler (castlegen/csrc/hwsampler.c documents the format).

    biomes  castlegen.infinite.biomes(), Schedule.anneal(S=64)
    city    castlegen.citytiles with BOUNDARY_W = 0.5 and ZONE_W below
            (FEAT_W unchanged), SocketModel u = -log w, wild_u = 20,
            Schedule.anneal(S=64, beta0=1.0, gamma0=2.0)

biomes_model() / city_model() return (model, schedule) so tests can run
castlegen.infinite.generate on the same thing.  Run as a script to write
luanti/games/hierwave/mods/hierwave_gen/data/{biomes,city}.hwm."""
from __future__ import annotations

import os
import struct

import numpy as np

from castlegen.infinite import Schedule, SocketModel, TileModel, biomes

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


def city_model(S=64, beta0=1.0, gamma0=2.0, **kw):
    ts = city_tiles()
    m = SocketModel(-np.log(ts.w), ts.sN, ts.sE, ts.sS, ts.sW, wild_u=CITY_WILD_U)
    return m, Schedule.anneal(S=S, beta0=beta0, gamma0=gamma0, **kw)


def socket_base(model: SocketModel, sched: Schedule):
    """exactly the base table SocketModel.run passes to the kernel."""
    return np.exp(-sched.beta[:, None] * (model.u - model.u.min())[None, :])


def write_hwm(path, model, sched: Schedule):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    S = sched.S
    f64 = lambda a: np.ascontiguousarray(a, "<f8").tobytes()
    i32 = lambda a: np.ascontiguousarray(a, "<i4").tobytes()
    if isinstance(model, TileModel):
        T = model.T
        head = struct.pack("<4sIIII", b"HWM1", 0, T, S, 0)
        body = (f64(sched.beta) + f64(sched.gamma) + f64(model.u) + f64(model.Eh) + f64(model.Ev)
                + i32(model.Hh) + i32(model.Hv))
    elif isinstance(model, SocketModel):
        T = model.T
        wild = model.wild_u is not None
        head = struct.pack("<4sIIII", b"HWM1", 1, T, S, 1 if wild else 0)
        body = (f64(sched.beta) + f64(sched.gamma) + f64(model.u)
                + struct.pack("<d", model.wild_u if wild else 0.0)
                + i32(model.sN) + i32(model.sE) + i32(model.sS) + i32(model.sW)
                + f64(socket_base(model, sched)))
    else:
        raise TypeError(type(model))
    with open(path, "wb") as fh:
        fh.write(head + body)
    return path


def main():
    for name, (m, s) in (("biomes", biomes_model()), ("city", city_model())):
        p = write_hwm(os.path.join(DATA, f"{name}.hwm"), m, s)
        print(f"{name}: T = {m.T}, S = {s.S} -> {os.path.relpath(p, ROOT)} ({os.path.getsize(p)} bytes)")


if __name__ == "__main__":
    main()
