"""cfine: creplay's coupled replays extended to the fine levels.

csat found the fine count draws on ground deterministic and blind to their
parent; a deterministic policy gets no score-function gradient for actions it
never takes.  Enumerating A at a branched draw gives sum_a grad q(a)[ell(a) +
G^a] with real cone returns for every action, including unvisited ones, with
no change of objective (constitution: "branching at any level, enumerating
small candidate sets").  cfine branches every level (`blevs`), choosing R_l
non-forced draws per world and level (|A| > 1, inverse-inclusion weighted;
unbiased, see _cfine_core.py), and replays the fine-level draws (levels <=
`wlev`) only inside their dependency cone (windowed replay).  Generation is
cpass's / cfast's, unchanged.  creplay's own defaults are untouched.

Config on top of creplay's: blevs (levels to branch, any order), R (per entry
of blevs), nonforced (select among |A| > 1 only), wlev (levels <= wlev use the
windowed replay; -1 = full-world replays everywhere)."""
from __future__ import annotations

from castlegen.bench.variant import register

from . import _cfine_core as fcore
from .creplay import CReplay

CFG = dict(Kr=4, blevs=(4, 3, 2, 1, 0), R=(2, 2, 8, 16, 32), nonforced=True, wlev=2, chunk=4096)


@register
class CFine(CReplay):
    name = "cfine"

    def __init__(self, problem, seed, workdir, **cfg):
        cfg = {**CFG, **cfg}
        cfg.setdefault("blev", 0)
        super().__init__(problem, seed, workdir, **cfg)

    def train_fns(self, lay):
        c = self.cfg
        key = ("cfine", c["Kr"], tuple(c["blevs"]), tuple(c["R"]), c["nonforced"], c["wlev"], c["chunk"],
               c["critic"], c["sub"])
        if not hasattr(lay, "cfast_fns"):
            lay.cfast_fns = {}
        if key not in lay.cfast_fns:
            lay.cfast_fns[key] = fcore.build(self, lay)
        return lay.cfast_fns[key]


def _variant(name, **over):
    cls = type(f"CFine_{name}", (CFine,), {"name": name})
    orig = cls.__init__

    def init(self, problem, seed, workdir, **cfg):
        orig(self, problem, seed, workdir, **{**over, **cfg})
    cls.__init__ = init
    return register(cls)


_variant("cfine_fineonly", blevs=(2, 1, 0), R=(8, 16, 32))     # fine levels only; coarse = cfast's terms
