"""Compile a choice grammar into per-level channels.

A grammar is given as choice instances and factors, all plain functions of
an assignment `vals` (instance name -> value):
  Inst(name, kind, domain(vals), anchor(vals), reads, deps, position)
    domain   values allowed given the instances it depends on (deps)
    anchor   world (y, x) where the choice is stored; a position choice's
             anchor depends on its own value (it is stored where it lands)
    reads    [(summary, box(vals))]: summaries of other sets over a world box
  Fac(kind, parts, fn(vals) -> bool): a hard factor; parts[0] is its subject
The compiler only calls these functions.  It learns geometry and coupling
from sampled feasible derivations (contexts) and decides, per choice kind:

  external (E)  has reads: the finest level at which every read box stays
                within one block of the anchor's block (the 3 x 3 window
                a block's kernel sees);
  internal (I)  no reads but in factors or depended on: collapsed into the
                level of its finest partner (decided with it, exactly);
  free (F)      neither: stateless, hashed from the path.

Then, to a fixed point: a factor whose parts sit at different levels is
decided top-down; its finer parts are grouped into clusters of instances
local to each other at their level.  One cluster: the coarse side keeps
values for which a completion exists (an existence projection, honoured
below).  Several clusters (they cannot coordinate): either quantify (the
coarse side keeps values valid for every fine assignment) when that loses
at most TAU of the coarse values that have some completion, or lift the
fine kind whose lifting brings the loss under TAU.  A dependency on a finer
kind is resolved the same way.  Finally an honourability check: for coarse
values chosen under the projections, does every fine cluster still have a
joint completion?"""
from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

LEVELS = (8, 32, 96)                         # 96: column strips, vertical extent free
TAU = 0.25
CAP = 4096


@dataclass
class Inst:
    name: str
    kind: str
    domain: callable
    anchor: callable
    reads: list = field(default_factory=list)
    deps: list = field(default_factory=list)
    position: bool = False


@dataclass
class Fac:
    kind: str
    parts: list
    fn: callable


# ---------------------------------------------------------------- contexts
def sample_context(spec, seed, tries=50):
    """A feasible assignment, sampled top-down in the spec's order: each
    instance uniform among values that pass the factors already decidable."""
    insts, order, facs = spec.build()
    by = defaultdict(list)
    for f in facs:
        for p in f.parts:
            by[p].append(f)
    g = np.random.default_rng(seed)
    for _ in range(tries):
        vals, ok = {}, True
        for name in order:
            cand = list(insts[name].domain(vals))
            g.shuffle(cand)
            for v in cand:
                vals[name] = v
                if all(f.fn(vals) for f in by[name] if all(p in vals for p in f.parts)):
                    break
            else:
                ok = False
                break
        if ok:
            return vals
    raise RuntimeError("no feasible context")


def _local(level, dy, dx):
    """Anchors at most one block apart in every alignment."""
    return dx <= level and (level == 96 or dy <= level)


def _read_ok(level, lo_y, lo_x, hi_y, hi_x):
    okx = lo_x >= -level and hi_x <= level + 1
    oky = level == 96 or (lo_y >= -level and hi_y <= level + 1)
    return okx and oky


# ----------------------------------------------------------------- compile
class Compiled:
    def __init__(self, spec, ncontexts=24, seed=0):
        self.spec = spec
        self.insts, self.order, self.facs = spec.build()
        self.ctx = [sample_context(spec, seed + i) for i in range(ncontexts)]
        self.kinds = sorted({i.kind for i in self.insts.values()})
        self.kind_of = {n: i.kind for n, i in self.insts.items()}
        self.log = []
        self._closure()
        self._geometry()
        self._place()
        self._honour()

    def _closure(self):
        """Transitive dependents of every instance, in sampling order."""
        direct = defaultdict(set)
        for n, i in self.insts.items():
            for d in i.deps:
                direct[d].add(n)
        self.downstream = {}
        pos = self.pos = {n: j for j, n in enumerate(self.order)}
        for n in self.order:
            seen, stack = set(), [n]
            while stack:
                for q in direct[stack.pop()]:
                    if q not in seen:
                        seen.add(q)
                        stack.append(q)
            self.downstream[n] = sorted(seen, key=pos.get)
        self._g = np.random.default_rng(11)

    def refresh(self, w, changed):
        """After changing `changed`, redraw dependents whose value left their domain."""
        todo = sorted({q for n in changed for q in self.downstream[n]}, key=self.pos.get)
        for q in todo:
            d = self.insts[q].domain(w)
            if w[q] not in d:
                d = list(d)
                w[q] = d[self._g.integers(len(d))]

    # ------------------------------------------------------------ geometry
    def _geometry(self):
        """Read extents relative to the anchor; anchor distances between the
        parts of every factor kind; domain sizes."""
        inf = float("inf")
        self.read_ext = {}
        self.pair_dist = defaultdict(lambda: [0, 0])          # (fk, ka, kb) -> max |dy|, |dx|
        self.dom = defaultdict(int)
        for vals in self.ctx:
            anchors = {n: self.insts[n].anchor(vals) for n in self.order
                       if self.spec.focus(n) and self.spec.active(n, vals)}
            for n, a in anchors.items():
                inst = self.insts[n]
                self.dom[inst.kind] = max(self.dom[inst.kind], len(list(inst.domain(vals))))
                for summary, box in inst.reads:
                    y0, x0, y1, x1 = box(vals)
                    e = self.read_ext.setdefault(inst.kind, [inf, inf, -inf, -inf])
                    e[0], e[1] = min(e[0], y0 - a[0]), min(e[1], x0 - a[1])
                    e[2], e[3] = max(e[2], y1 - a[0]), max(e[3], x1 - a[1])
            for f in self.facs:
                if not all(p in anchors for p in f.parts):
                    continue
                for p, q in itertools.combinations(f.parts, 2):
                    dy = abs(anchors[p][0] - anchors[q][0])
                    dx = abs(anchors[p][1] - anchors[q][1])
                    for key in ((f.kind, self.kind_of[p], self.kind_of[q]), (f.kind, self.kind_of[q], self.kind_of[p])):
                        d = self.pair_dist[key]
                        d[0], d[1] = max(d[0], dy), max(d[1], dx)

    # ----------------------------------------------------------- placement
    def _finest(self, k):
        e = self.read_ext[k]
        for lv in LEVELS:
            if _read_ok(lv, *e):
                return lv
        return LEVELS[-1]

    def _fac_kinds(self):
        out = {}
        for f in self.facs:
            out.setdefault(f.kind, list(dict.fromkeys(self.kind_of[p] for p in f.parts)))
        return out

    def _clusters(self, f, fine_names, vals):
        """Union-find over a factor's fine parts: two are joined when local at the
        coarser of their two levels."""
        par = {n: n for n in fine_names}

        def find(n):
            while par[n] != n:
                n = par[n]
            return n
        lv = lambda n: self.level[self.kind_of[n]]
        for p, q in itertools.combinations(fine_names, 2):
            if lv(p) != lv(q):
                continue                      # a finer part reads its coarser ancestors through the broadcast frame
            if self.cat[self.kind_of[p]] == "I" and p.split(".")[0] == q.split(".")[0]:
                par[find(p)] = find(q)        # collapsed into the same owner's struct
                continue
            a, b = self.insts[p].anchor(vals), self.insts[q].anchor(vals)
            if _local(lv(p), abs(a[0] - b[0]), abs(a[1] - b[1])):
                par[find(p)] = find(q)
        per_level = defaultdict(set)
        for n in fine_names:
            per_level[lv(n)].add(find(n))
        return max(len(v) for v in per_level.values())

    def _loss(self, fk, quant_kinds):
        key = (fk, frozenset(quant_kinds), tuple(sorted(self.level.items())))
        cache = self.__dict__.setdefault("_loss_cache", {})
        if key not in cache:
            cache[key] = self._loss_raw(fk, quant_kinds)
        return cache[key]

    def _loss_raw(self, fk, quant_kinds):
        """1 - |values of the subject valid for every assignment of the quantified
        kinds| / |values valid for some|, over contexts (fine assignments
        sampled, the context's own always included)."""
        g = np.random.default_rng(7)
        num = den = 0
        for vals in self.ctx:
            for f in self.facs:
                if f.kind != fk or not all(self.spec.focus(p) or True for p in f.parts):
                    continue
                if not self.spec.focus(f.parts[0]):
                    continue
                subj = f.parts[0]
                qn = [p for p in f.parts if self.kind_of[p] in quant_kinds and p != subj]
                if not qn:
                    continue
                assigns = [tuple(vals[p] for p in qn)]
                doms = [list(self.insts[p].domain(vals)) for p in qn]
                for _ in range(48):
                    assigns.append(tuple(d[g.integers(len(d))] for d in doms))
                dom = list(self.insts[subj].domain(vals))
                if len(dom) > 32:
                    dom = [dom[j] for j in g.choice(len(dom), 32, replace=False)]
                for v in dom:
                    w = dict(vals)
                    w[subj] = v
                    res = []
                    for a in assigns:
                        w.update(zip(qn, a))
                        self.refresh(w, qn)
                        res.append(f.fn(w))
                    if any(res):
                        den += 1
                        num += all(res)
        return 1.0 - num / den if den else 0.0

    def _place(self):
        fk_parts = self._fac_kinds()
        in_fac = {k for ks in fk_parts.values() for k in ks}
        depended = {self.kind_of[d] for i in self.insts.values() for d in i.deps}
        self.cat, self.level = {}, {}
        for k in self.kinds:
            if k in self.read_ext:
                self.cat[k] = "E"
                self.level[k] = self._finest(k)
            elif k in in_fac or k in depended:
                self.cat[k] = "I"
            else:
                self.cat[k] = "F"
        self.quant = set()                                        # (fk, kind)
        self.exists = set()                                       # (fk, kind): projection honoured below
        self.floor = {}                                           # lifts: levels never drop below these
        kdeps = defaultdict(set)
        for i in self.insts.values():
            for d in i.deps:
                kdeps[i.kind].add(self.kind_of[d])
        for it in range(30):
            changed = False
            self.quant, self.exists = set(), set()               # recomputed each round; only lifts persist
            log_round = []
            # internal kinds: decided before every kind that depends on them (their dependents'
            # coarsest level); with no dependents, collapsed into their finest partner
            for k in self.kinds:
                if self.cat[k] != "I":
                    continue
                dependents = [q for q in self.kinds if k in kdeps[q] and q in self.level]
                if dependents:
                    lv = max(self.level[q] for q in dependents)
                else:
                    partners = set(kdeps[k])
                    for fk, ks in fk_parts.items():
                        if k in ks:
                            partners |= {q for q in ks if (fk, q) not in self.quant}
                    partners.discard(k)
                    lv = min([self.level[q] for q in partners if q in self.level] or [96])
                lv = max(lv, self.floor.get(k, 0))
                if self.level.get(k) != lv:
                    self.level[k] = lv
                    changed = True
            # dependencies on finer kinds
            for k in self.kinds:
                for d in kdeps[k]:
                    if k in self.level and self.level.get(d, 96) < self.level[k]:
                        self.log.append(f"lift {d}: {k} (level {self.level[k]}) depends on it")
                        self.level[d] = self.floor[d] = self.level[k]
                        if self.cat[d] == "E":
                            self.cat[d] = "E*"
                        changed = True
            # factors across levels
            for fk, ks in fk_parts.items():
                top = max(self.level.get(k, 96) for k in ks)
                fine = [k for k in ks if self.level.get(k, 96) < top and (fk, k) not in self.quant]
                if not fine:
                    continue
                nclu = 0
                for vals in self.ctx:
                    for f in self.facs:
                        if f.kind == fk and self.spec.focus(f.parts[0]) and all(self.spec.active(p, vals) for p in f.parts):
                            names = [p for p in f.parts if self.kind_of[p] in fine]
                            nclu = max(nclu, self._clusters(f, names, vals))
                if nclu <= 1:
                    for k in fine:
                        if (fk, k) not in self.exists:
                            self.exists.add((fk, k))
                    continue
                loss = self._loss(fk, set(fine))
                if loss <= TAU:
                    log_round.append(f"quantify {fk} over {fine}: {nclu} fine clusters cannot coordinate; "
                                     f"loss {loss:.2f} <= {TAU}")
                    self.quant |= {(fk, k) for k in fine}
                    continue
                best = None
                for k in fine:
                    rest = set(fine) - {k}
                    l2 = self._loss(fk, rest) if rest else 0.0
                    if best is None or l2 < best[1]:
                        best = (k, l2)
                k, l2 = best
                self.log.append(f"lift {k} to {top}: {fk} has {nclu} fine clusters; quantifying all loses "
                                f"{loss:.2f}, lifting {k} leaves {l2:.2f}")
                self.level[k] = self.floor[k] = top
                if self.cat[k] == "E":
                    self.cat[k] = "E*"
                changed = True
            if not changed:
                self.log += log_round
                break
        self.dep_kinds = kdeps

    # --------------------------------------------------------- honourability
    def _honour(self):
        """Per-factor projections at the coarse level against joint feasibility
        one level down.  For each context and each alternative value v of a
        factor's subject: v is allowed if every factor touching the subject
        passes its projection (existence over its finer parts, or all sampled
        assignments for quantified parts, or plainly when it has none).  Then
        the finer parts of all those factors, closed over same-level factor
        links, are searched jointly for a completion that passes every factor
        among them (factors reaching even finer levels are checked when their
        own subjects come up).  A failure means the projections are not enough."""
        self.honour = defaultdict(lambda: [0, 0])
        g = np.random.default_rng(3)
        by = defaultdict(list)
        for f in self.facs:
            for p in f.parts:
                by[p].append(f)
        lv = lambda n: self.level.get(self.kind_of[n], 0)

        def pool(names, w, n=CAP):
            doms = [list(self.insts[p].domain(w)) for p in names]
            size = float(np.prod([len(d) for d in doms], dtype=float))
            if size <= n:
                return list(itertools.product(*doms))
            return [tuple(d[g.integers(len(d))] for d in doms) for _ in range(n)]

        def ok_all(facs, names, assign, w):
            w.update(zip(names, assign))
            self.refresh(w, names)
            return all(h.fn(w) for h in facs)

        seen = set()
        for vals in self.ctx[:8]:
            for subj in self.order:
                if not self.spec.focus(subj) or (subj, id(vals)) in seen:
                    continue
                seen.add((subj, id(vals)))
                ks = lv(subj)
                hs = by[subj]
                if not any(lv(p) < ks for h in hs for p in h.parts):
                    continue
                alts = [v for v in self.insts[subj].domain(vals) if v != vals[subj]]
                if len(alts) > 24:
                    alts = [alts[j] for j in g.choice(len(alts), 24, replace=False)]
                for v in alts:
                    w = dict(vals)
                    w[subj] = v
                    self.refresh(w, [subj])
                    allowed = True
                    for h in hs:
                        fine = [p for p in h.parts if lv(p) < ks]
                        if not fine:
                            allowed = h.fn(w)
                        else:
                            quant = [p for p in fine if (h.kind, self.kind_of[p]) in self.quant]
                            res = [ok_all([h], fine, a, dict(w)) for a in pool(fine, w, 256)]
                            allowed = all(res) if quant else any(res)
                        if not allowed:
                            break
                    if not allowed:
                        continue
                    comp = {p for h in hs for p in h.parts if lv(p) < ks}
                    top_fine = max(lv(p) for p in comp)
                    comp = {p for p in comp if lv(p) == top_fine}
                    stack = list(comp)
                    while stack:
                        p = stack.pop()
                        for h in by[p]:
                            for q in h.parts:
                                if lv(q) == top_fine and q not in comp:
                                    comp.add(q)
                                    stack.append(q)
                    comp = sorted(comp)
                    facs = list({id(h): h for p in comp for h in by[p]
                                 if all(lv(q) >= top_fine for q in h.parts)}.values())
                    # independent components of the fine side are searched separately (exhaustively when small)
                    par = {p: p for p in comp}

                    def find(p):
                        while par[p] != p:
                            p = par[p]
                        return p
                    for h in facs:
                        ins = [q for q in h.parts if q in par]
                        for q in ins[1:]:
                            par[find(q)] = find(ins[0])
                    groups = defaultdict(list)
                    for p in comp:
                        groups[find(p)].append(p)
                    found = all(self._solve(sorted(grp, key=self.pos.get),
                                                [h for h in facs if any(q in grp for q in h.parts)], dict(w))
                                    for grp in groups.values())
                    rec = self.honour[self.kind_of[subj]]
                    rec[0] += 1
                    rec[1] += int(not found)

    def _solve(self, names, facs, w, cap=20000):
        """Backtracking search over `names` (in sampling order) for values that pass
        every factor in `facs` once its parts are assigned; True if found within cap."""
        by = defaultdict(list)
        for h in facs:
            for q in h.parts:
                if q in names:
                    by[q].append(h)
        todo = set(names)
        for q in names:
            w.pop(q, None)
        visits = [0]

        def rec(i):
            if i == len(names):
                return True
            q = names[i]
            dom = list(self.insts[q].domain(w))
            self._g.shuffle(dom)
            for v in dom:
                visits[0] += 1
                if visits[0] > cap:
                    return False
                w[q] = v
                if all(h.fn(w) for h in by[q] if all(p in w and (p not in todo or p in names[:i + 1]) for p in h.parts)):
                    if rec(i + 1):
                        return True
            w.pop(q, None)
            return False
        return rec(0)

    # --------------------------------------------------------------- report
    def table(self):
        node = lambda k: k.split(".")[0]
        rows = []
        for lv in sorted(set(self.level.values()), reverse=True):
            ks = [k for k in self.kinds if self.level.get(k) == lv and self.cat[k] != "F"]
            for nd in sorted({node(k) for k in ks}):
                group = [k for k in ks if node(k) == nd]
                rows.append((lv, nd, group))
        out = ["| level | node | sampled (E: reads other sets; E*: lifted) | collapsed (I) | head domain |",
               "|---|---|---|---|---|"]
        for lv, nd, group in rows:
            e = [f"{k.split('.', 1)[1]}{'*' if self.cat[k] == 'E*' else ''}{' (pos)' if self._pos(k) else ''}"
                 for k in group if self.cat[k] in ("E", "E*")]
            i = [k.split(".", 1)[1] for k in group if self.cat[k] == "I"]
            head = int(np.prod([self.dom[k] for k in group if self.cat[k] in ("E", "E*")] or [1], dtype=float))
            out.append(f"| {lv} | {nd} | {', '.join(e) or '-'} | {', '.join(i) or '-'} | {head} |")
        free = [k for k in self.kinds if self.cat[k] == "F"]
        out.append(f"\nstateless (hashed from the path): {', '.join(free) or '-'}")
        return "\n".join(out)

    def _pos(self, k):
        return any(i.position for i in self.insts.values() if i.kind == k)

    def report(self):
        lines = [self.table(), "", "decisions:"]
        lines += [f"  - {l}" for l in self.log] or ["  - none"]
        ex = sorted({(fk, k) for fk, k in self.exists if (fk, k) not in self.quant})
        lines.append("existence projections (coarse keeps values with a completion; honoured below): "
                     + ", ".join(f"{fk}/{k}" for fk, k in ex))
        lines.append("quantified (coarse keeps values valid for every fine value): "
                     + (", ".join(f"{fk}/{k}" for fk, k in sorted(self.quant)) or "-"))
        lines.append("read extents relative to the anchor (dy0, dx0, dy1, dx1):")
        for k, e in sorted(self.read_ext.items()):
            lines.append(f"  {k}: {tuple(int(v) for v in e)} -> finest local level {self._finest(k)}")
        lines.append("honourability (coarse values tried, with no fine completion):")
        for k, (t, f) in sorted(self.honour.items()):
            lines.append(f"  {k}: {t} tried, {f} failed")
        return "\n".join(lines)
