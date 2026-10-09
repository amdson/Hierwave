"""Legacy (G0, JAX castle sampler era); superseded by castlegen/channels. See the original README.md (git history).

Hierarchical connectivity: exact component counts under single-cell edits.

The grid is padded with wall to a side N = B^L and cut into a quadtree-like
hierarchy with fan-out B x B (leaves are B x B cells).  Every block keeps a
summary of its connectivity as seen from outside:

  ports       node cells on the block's boundary (nodes = rooms and the gate)
  labels      which ports are connected to each other inside the block
  closed      components entirely inside the block that touch no port and do
              not contain the gate (they can never join anything outside)
  gate_closed the gate's component is entirely inside the block
  gate_label  the class holding the gate, if it reaches a port

A parent is built from its children by union-find over the children's port
classes, joined wherever a door crosses between two children; a class with no
port on the parent's boundary becomes closed.  At the root there is no
outside, so

  K = root.closed + (root classes) + [gate exists]

counts every component (the gate's included).  A single-cell edit recomputes
its leaf and the L ancestors: O(L) merges whose size grows with the block
perimeter, independent of the rest of the grid.  Doors between cells of
different blocks are evaluated only in the merge of their common ancestor, so
sibling summaries never change on an edit.

Pure numpy / Python: this is the sequential structure for the offline
reference sampler.
"""
from __future__ import annotations

import numpy as np

from .tileset import TileSet

B = 4


class _Summary:
    __slots__ = ("ports", "labels", "n_classes", "closed", "gate_closed", "gate_label")

    def __init__(self, ports, labels, n_classes, closed, gate_closed, gate_label):
        self.ports = ports              # list of (y, x)
        self.labels = labels            # list of class ids, parallel to ports
        self.n_classes = n_classes
        self.closed = closed
        self.gate_closed = gate_closed
        self.gate_label = gate_label    # -1 if none


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        p = self.p
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[b] = a


class Connectivity:
    def __init__(self, ts: TileSet, tiles):
        self.ts = ts
        self.Dh = ts.np_tables["Dh"]
        self.Dv = ts.np_tables["Dv"]
        node = np.array(ts.np_tables["is_room"], bool)
        node[ts.GATE] = True
        self.node = node
        tiles = np.asarray(tiles)
        self.H, self.W = tiles.shape
        n = B
        self.L = 1
        while n < max(self.H, self.W):
            n *= B
            self.L += 1
        self.N = n
        self.t = np.full((n, n), ts.WALL, np.int32)
        self.t[:self.H, :self.W] = tiles
        # levels[l][by][bx]: summary of the block of side B**(l+1) at (by, bx)
        self.levels = []
        side = B
        for l in range(self.L):
            nb = n // side
            self.levels.append([[None] * nb for _ in range(nb)])
            side *= B
        for by in range(n // B):
            for bx in range(n // B):
                self.levels[0][by][bx] = self._leaf(by, bx)
        for l in range(1, self.L):
            nb = len(self.levels[l])
            for by in range(nb):
                for bx in range(nb):
                    self.levels[l][by][bx] = self._merge(l, by, bx)

    # ---------------------------------------------------------------- public
    @property
    def tiles(self):
        return self.t[:self.H, :self.W]

    def set(self, y, x, sig):
        """Change one cell and update the summaries on its path to the root."""
        self.t[y, x] = sig
        by, bx = y // B, x // B
        self.levels[0][by][bx] = self._leaf(by, bx)
        for l in range(1, self.L):
            by, bx = by // B, bx // B
            self.levels[l][by][bx] = self._merge(l, by, bx)

    def components(self):
        """Number of connected components of rooms + gate (the gate's included)."""
        r = self.levels[-1][0][0]
        gate_present = r.gate_closed or r.gate_label >= 0
        open_nongate = r.n_classes - (1 if r.gate_label >= 0 else 0)
        return r.closed + open_nongate + int(gate_present)

    def unreached_components(self):
        """Components not containing the gate (0 = every room reaches the gate)."""
        r = self.levels[-1][0][0]
        gate_present = r.gate_closed or r.gate_label >= 0
        return self.components() - int(gate_present)

    # --------------------------------------------------------------- summaries
    def _door(self, y0, x0, y1, x1):
        """A door joins two edge-adjacent cells (both assumed to be nodes)."""
        t = self.t
        if y0 == y1:
            a, b = (t[y0, x0], t[y1, x1]) if x0 < x1 else (t[y1, x1], t[y0, x0])
            return self.Dh[a, b]
        a, b = (t[y0, x0], t[y1, x1]) if y0 < y1 else (t[y1, x1], t[y0, x0])
        return self.Dv[a, b]

    def _leaf(self, by, bx):
        t, node, gate = self.t, self.node, self.ts.GATE
        y0, x0 = by * B, bx * B
        comp = -np.ones((B, B), int)
        n_comp = 0
        for sy in range(B):
            for sx in range(B):
                if comp[sy, sx] >= 0 or not node[t[y0 + sy, x0 + sx]]:
                    continue
                comp[sy, sx] = n_comp
                stack = [(sy, sx)]
                while stack:
                    cy, cx = stack.pop()
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < B and 0 <= nx < B and comp[ny, nx] < 0 and node[t[y0 + ny, x0 + nx]] \
                                and self._door(y0 + cy, x0 + cx, y0 + ny, x0 + nx):
                            comp[ny, nx] = n_comp
                            stack.append((ny, nx))
                n_comp += 1
        return self._close(n_comp, comp, y0, x0, B,
                           [(y0 + sy, x0 + sx, comp[sy, sx]) for sy in range(B) for sx in range(B)
                            if comp[sy, sx] >= 0],
                           [comp[sy, sx] for sy in range(B) for sx in range(B)
                            if comp[sy, sx] >= 0 and t[y0 + sy, x0 + sx] == gate])

    def _merge(self, l, by, bx):
        kids = self.levels[l - 1]
        side_c = B ** l                                  # child side in cells
        offs, members, closed, gate_classes, gate_closed = [], [], 0, [], False
        port_of = {}
        n = 0
        for cy in range(B):
            for cx in range(B):
                s = kids[by * B + cy][bx * B + cx]
                offs.append(n)
                for (py, px), lab in zip(s.ports, s.labels):
                    port_of[py, px] = n + lab
                    members.append((py, px, n + lab))
                closed += s.closed
                gate_closed |= s.gate_closed
                if s.gate_label >= 0:
                    gate_classes.append(n + s.gate_label)
                n += s.n_classes
        uf = _UF(n)
        y0, x0 = by * B * side_c, bx * B * side_c
        for cy in range(B):
            for cx in range(B):
                cy0, cx0 = y0 + cy * side_c, x0 + cx * side_c
                if cx + 1 < B:                           # shared vertical edge with the right child
                    xa = cx0 + side_c - 1
                    for y in range(cy0, cy0 + side_c):
                        a, b = port_of.get((y, xa)), port_of.get((y, xa + 1))
                        if a is not None and b is not None and self._door(y, xa, y, xa + 1):
                            uf.union(a, b)
                if cy + 1 < B:                           # shared horizontal edge with the child below
                    ya = cy0 + side_c - 1
                    for x in range(cx0, cx0 + side_c):
                        a, b = port_of.get((ya, x)), port_of.get((ya + 1, x))
                        if a is not None and b is not None and self._door(ya, x, ya + 1, x):
                            uf.union(a, b)
        roots = [uf.find(i) for i in range(n)]
        return self._close(n, None, y0, x0, B * side_c,
                           [(py, px, roots[c]) for py, px, c in members],
                           [roots[g] for g in gate_classes],
                           closed=closed, gate_closed=gate_closed, classes=set(roots))

    def _close(self, n, comp, y0, x0, side, members, gate_classes, closed=0, gate_closed=False, classes=None):
        """Turn (cell, class) memberships into a summary: classes touching the
        block boundary keep ports; the rest become closed components."""
        classes = set(range(n)) if classes is None else classes
        on_edge = lambda y, x: y == y0 or x == x0 or y == y0 + side - 1 or x == x0 + side - 1
        ported = {}
        ports, labels = [], []
        for y, x, c in members:
            if on_edge(y, x):
                if c not in ported:
                    ported[c] = len(ported)
                ports.append((y, x))
                labels.append(ported[c])
        gate_c = gate_classes[0] if gate_classes else None
        for c in classes:
            if c not in ported:
                if c == gate_c:
                    gate_closed = True
                else:
                    closed += 1
        gate_label = ported[gate_c] if gate_c is not None and gate_c in ported else -1
        return _Summary(ports, labels, len(ported), closed, gate_closed, gate_label)


def components_bfs(ts: TileSet, tiles):
    """Reference: (components of rooms + gate, components not containing the gate)."""
    tiles = np.asarray(tiles)
    H, W = tiles.shape
    node = np.array(ts.np_tables["is_room"], bool)
    node[ts.GATE] = True
    Dh, Dv = ts.np_tables["Dh"], ts.np_tables["Dv"]
    seen = np.zeros((H, W), bool)
    K, gate_comps = 0, 0
    for y in range(H):
        for x in range(W):
            if seen[y, x] or not node[tiles[y, x]]:
                continue
            K += 1
            has_gate = False
            seen[y, x] = True
            stack = [(y, x)]
            while stack:
                cy, cx = stack.pop()
                has_gate |= tiles[cy, cx] == ts.GATE
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if not (0 <= ny < H and 0 <= nx < W) or seen[ny, nx] or not node[tiles[ny, nx]]:
                        continue
                    a, b = tiles[cy, cx], tiles[ny, nx]
                    door = Dv[b, a] if dy < 0 else Dv[a, b] if dy > 0 else Dh[b, a] if dx < 0 else Dh[a, b]
                    if door:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            gate_comps += has_gate
    return K, K - gate_comps
