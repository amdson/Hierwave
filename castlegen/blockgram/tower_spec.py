"""The tower grammar as a choice spec for compile.py: the same choices as
towers.py, written as instances with domains, anchors, reads of the ground,
and hard factors.  No level is named anywhere.

Per region r (window of NR regions; focus on the inner ones):
  tower.present, tower.n, tower.span      existence, floor count, width
  tower.x        (position) tower centre; reads ground flatness under it
  tower.base     (position, vertical) floor-0 slab story; reads ground min
                 under the footprint; base_clear: slab 2..14 rows above it
  bridge.story   to the east neighbour or none; reads ground under the gap;
                 bridge_ok: both towers hold the story above floor 0, the
                 gap is at least 4 wide, the deck clears the ground by 6
  tower.entrance none / west / east; reads the ground beside the tower;
                 never on a bridged side
  floor.state    per floor: (widths, stair room, direction); floor_compat
                 between consecutive floors; floor_doors: no stair in a
                 room behind an exterior door (bridge, entrance)
  room.kind      per room: free
  foot.style     per 8-column segment of the footprint: plinth or arch;
                 an arch needs 7 rows of clearance over the whole segment
  path.step      per 8-column segment outside the entrance: inactive,
                 descending, landing, ladder; a chain"""
from __future__ import annotations

import numpy as np

from .compile import Fac, Inst
from .towers import H, PITCH, RW, Y_LAT, floor_states, spans, y_slab
from .towers import surface as _surface

SPANS = sorted(spans())
NFL, NSEG, NPATH = 15, 7, 6


_X0 = -256
_SURF = _surface(np.arange(_X0, 16 * RW))                     # the terrain is pure: cache it over the windows used


def surface(x):
    return _SURF[np.asarray(x) - _X0]


def _surf(a, b):
    return _SURF[a - _X0:b - _X0]


class TowerSpec:
    def __init__(self, bridges=True, nr=6):
        self.bridges, self.nr = bridges, nr

    def focus(self, name):
        r = int(name[1:name.index(".")])
        return 1 <= r <= self.nr - 3 if name[0] == "g" else 1 <= r <= self.nr - 2

    def active(self, name, v):
        """Whether an instance exists in this assignment (geometry of absent
        things is not evidence)."""
        r = int(name[1:name.index(".")])
        if name[0] == "g":
            return v[name] >= 0
        if not v[f"t{r}.present"]:
            return name.endswith(".present")
        k = name.split(".")[1]
        if k.startswith("floor"):
            return v[name] >= 0
        if k.startswith("room"):
            return v[f"t{r}.floor{k[4:].split('_')[0]}"] >= 0
        if k.startswith("path"):
            return v[f"t{r}.entrance"] != 0 and v[name] != 0
        return True

    def build(self):
        insts, order, facs = {}, [], []

        def add(inst):
            insts[inst.name] = inst
            order.append(inst.name)

        def T(r, k):
            return f"t{r}.{k}"

        def geom(v, r):
            span = v[T(r, "span")]
            total = span + 2
            x_w = v[T(r, "x")] - total // 2
            return x_w, x_w + total - 1

        for r in range(self.nr):
            c = r * RW + RW // 2
            ac = (int(surface(c)), c)
            m = int(_surf(r * RW, (r + 1) * RW).min())
            b0 = -(-(Y_LAT - (m - 2)) // PITCH)
            p = T(r, "present")
            add(Inst(p, "tower.present", lambda v: [0, 1], lambda v, ac=ac: ac))
            add(Inst(T(r, "n"), "tower.n", lambda v, p=p: range(10, 16) if v[p] else [0],
                     lambda v, ac=ac: ac, deps=[p]))
            add(Inst(T(r, "span"), "tower.span", lambda v, p=p: SPANS if v[p] else [SPANS[0]],
                     lambda v, ac=ac: ac, deps=[p]))

            def xdom(v, r=r):
                total = v[T(r, "span")] + 2
                return [x_w + total // 2 for x_w in range(r * RW + 8, r * RW + RW - total - 8 + 1)]

            def footbox(v, r=r):
                x_w, x_e = geom(v, r)
                s = _surf(x_w - 1, x_e + 2)
                return int(s.min()), x_w - 1, int(s.max()) + 1, x_e + 2
            add(Inst(T(r, "x"), "tower.x", xdom, lambda v, r=r: (int(surface(v[T(r, "x")])), v[T(r, "x")]),
                     reads=[("ground.range", footbox)], deps=[T(r, "span")], position=True))
            add(Inst(T(r, "base"), "tower.base", lambda v, b0=b0: range(b0 - 6, b0 + 2),
                     lambda v, r=r: (y_slab(v[T(r, "base")]), v[T(r, "x")]),
                     reads=[("ground.min", footbox)], position=True))

            def base_clear(v, r=r):
                if not v[T(r, "present")]:
                    return True
                x_w, x_e = geom(v, r)
                fm = int(_surf(x_w - 1, x_e + 2).min())
                ys = y_slab(v[T(r, "base")])
                return fm - 14 <= ys <= fm - 2
            facs.append(Fac("base_clear", [T(r, "base"), T(r, "x"), T(r, "span"), T(r, "present")], base_clear))

        if self.bridges:
            for r in range(self.nr - 1):
                g = f"g{r}.bridge"
                lo = min(-(-(Y_LAT - (int(_surf(q * RW, (q + 1) * RW).min()) - 2)) // PITCH) for q in (r, r + 1)) - 6
                ab = lambda v, r=r: (int(surface((r + 1) * RW)), (r + 1) * RW)

                def gapbox(v, r=r):
                    xa, xb = geom(v, r)[1] + 1, geom(v, r + 1)[0]
                    s = v[f"g{r}.bridge"]
                    top = y_slab(s) if s >= 0 else int(_surf(xa, max(xb, xa + 1)).min())
                    return top, xa, int(_surf(xa, max(xb, xa + 1)).max()) + 1, max(xb, xa + 1)
                add(Inst(g, "bridge.story", lambda v, lo=lo: [-1] + list(range(lo + 1, lo + 22)), ab,
                         reads=[("ground.min", gapbox)]))

                def bridge_ok(v, r=r):
                    s = v[f"g{r}.bridge"]
                    if s < 0:
                        return True
                    if not (v[T(r, "present")] and v[T(r + 1, "present")]):
                        return False
                    for q in (r, r + 1):
                        if not 1 <= s - v[T(q, "base")] <= v[T(q, "n")] - 1:
                            return False
                    xa, xb = geom(v, r)[1] + 1, geom(v, r + 1)[0] - 1
                    return xb - xa + 1 >= 4 and int(_surf(xa, xb + 1).min()) >= y_slab(s) + 6
                facs.append(Fac("bridge_ok", [g, T(r, "n"), T(r + 1, "n"), T(r, "base"), T(r + 1, "base"),
                                              T(r, "x"), T(r + 1, "x"), T(r, "span"), T(r + 1, "span"),
                                              T(r, "present"), T(r + 1, "present")], bridge_ok))

        for r in range(self.nr):
            c = r * RW + RW // 2
            ac = (int(surface(c)), c)
            bw = f"g{r - 1}.bridge" if self.bridges and r >= 1 else None
            be = f"g{r}.bridge" if self.bridges and r < self.nr - 1 else None
            e = T(r, "entrance")
            add(Inst(e, "tower.entrance", lambda v, r=r: [0, 1, 2] if v[T(r, "present")] else [0],
                     lambda v, ac=ac: ac,
                     reads=[("ground.slope", lambda v, r=r: (0, r * RW - 40, H, (r + 1) * RW + 40))],
                     deps=[T(r, "present")]))

            def ent_free(v, e=e, bw=bw, be=be):
                return not ((v[e] == 1 and bw and v[bw] >= 0) or (v[e] == 2 and be and v[be] >= 0))
            facs.append(Fac("ent_free", [e] + [b for b in (bw, be) if b], ent_free))

            # floors
            for i in range(NFL):
                f = T(r, f"floor{i}")
                fa = lambda v, r=r, i=i: (y_slab(v[T(r, "base")] + i) - 3, v[T(r, "x")])

                def fdom(v, r=r, i=i):
                    if not v[T(r, "present")] or i >= v[T(r, "n")]:
                        return [-1]
                    return range(len(floor_states(v[T(r, "span")])[1]))
                add(Inst(f, "floor.state", fdom, fa, deps=[T(r, "present"), T(r, "n"), T(r, "span")]))
                if i > 0:
                    def compat(v, r=r, i=i):
                        a, b = v[T(r, f"floor{i - 1}")], v[T(r, f"floor{i}")]
                        if a < 0 or b < 0:
                            return True
                        return floor_states(v[T(r, "span")])[4][a, b] > 0
                    facs.append(Fac("floor_compat", [T(r, f"floor{i - 1}"), f, T(r, "span")], compat))

                def doors(v, r=r, i=i, f=f, e=e, bw=bw, be=be):
                    s = v[f]
                    if s < 0:
                        return True
                    room = floor_states(v[T(r, "span")])[1][s][1]
                    base = v[T(r, "base")]
                    west = (bw and v[bw] >= 0 and v[bw] - base == i) or (i == 0 and v[e] == 1)
                    east = (be and v[be] >= 0 and v[be] - base == i) or (i == 0 and v[e] == 2)
                    return not ((west and room == 0) or (east and room == 4))
                parts = [f, T(r, "span"), e] + ([T(r, "base")] + [b for b in (bw, be) if b] if self.bridges else [])
                facs.append(Fac("floor_doors", parts, doors))
                for k in range(5):
                    add(Inst(T(r, f"room{i}_{k}"), "room.kind", lambda v: range(7), fa, deps=[f]))

            # foundation segments
            for k in range(NSEG):
                fs = T(r, f"foot{k}")

                def seg(v, r=r, k=k):
                    x_w, x_e = geom(v, r)
                    return x_w - 1 + 8 * k, min(x_w - 1 + 8 * (k + 1), x_e + 2)

                def sdom(v, r=r, seg=seg):
                    a, b = seg(v)
                    return [0, 1] if v[T(r, "present")] and b > a else [0]

                def sanchor(v, r=r, seg=seg):
                    a, b = seg(v)
                    return y_slab(v[T(r, "base")]) + 4, (a + b) // 2

                def sbox(v, r=r, seg=seg):
                    a, b = seg(v)
                    ys = y_slab(v[T(r, "base")])
                    return ys + 1, a, ys + 8, max(b, a + 1)          # is there ground within 7 rows below the slab?
                add(Inst(fs, "foot.style", sdom, sanchor, reads=[("ground.min", sbox)],
                         deps=[T(r, "base"), T(r, "x"), T(r, "span")]))

                def foot_geo(v, r=r, fs=fs, seg=seg):
                    if v[fs] == 0:
                        return True
                    a, b = seg(v)
                    return b > a and int(_surf(a, b).min()) - y_slab(v[T(r, "base")]) >= 7
                facs.append(Fac("foot_geo", [fs, T(r, "base"), T(r, "x"), T(r, "span")], foot_geo))

            # outside stair from the entrance
            for k in range(NPATH):
                ps = T(r, f"path{k}")

                def cols(v, r=r, k=k, e=e):
                    x_w, x_e = geom(v, r)
                    if v[e] == 1:
                        return [x_w - 2 - j for j in range(8 * k, 8 * k + 8)]
                    return [x_e + 2 + j for j in range(8 * k, 8 * k + 8)]

                def panchor(v, r=r, k=k, cols=cols):
                    cs = cols(v)
                    return y_slab(v[T(r, "base")]) + 8 * k + 4, (cs[0] + cs[-1]) // 2

                def pbox(v, r=r, k=k, cols=cols):
                    cs = cols(v)
                    y0 = y_slab(v[T(r, "base")]) + 8 * k
                    return y0, min(cs), y0 + 9, max(cs) + 1         # does the ground meet the stair in my segment?
                add(Inst(ps, "path.step", lambda v, e=e: [0, 1, 2, 3] if v[e] else [0], panchor,
                         reads=[("ground.min", pbox)], deps=[e, T(r, "base"), T(r, "x"), T(r, "span")]))

                def path_geo(v, r=r, k=k, ps=ps, cols=cols, e=e):
                    p = v[ps]
                    if p in (0, 3) or not v[e]:
                        return p == 0 or bool(v[e])
                    yf = y_slab(v[T(r, "base")])
                    cs = cols(v)
                    reach = [yf + 8 * k + j + 1 >= int(surface(c)) for j, c in enumerate(cs)]
                    return (not any(reach)) if p == 1 else any(reach)
                facs.append(Fac("path_geo", [ps, T(r, "base"), T(r, "x"), T(r, "span"), e], path_geo))
                if k == 0:
                    facs.append(Fac("path_start", [e, ps], lambda v, e=e, ps=ps: (v[e] == 0) == (v[ps] == 0)))
                else:
                    prev = T(r, f"path{k - 1}")
                    facs.append(Fac("path_cont", [prev, ps],
                                    lambda v, a=prev, b=ps: (v[b] in (1, 2, 3)) if v[a] == 1 else v[b] == 0))
            facs.append(Fac("path_end", [T(r, f"path{NPATH - 1}")], lambda v, p=T(r, f"path{NPATH - 1}"): v[p] != 1))

        # sampling order: tower heads, bridges, then the rest per tower (already appended in that order)
        return insts, order, facs
