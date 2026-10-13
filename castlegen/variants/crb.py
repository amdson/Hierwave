"""crb: cfast with Rao-Blackwellised downstream returns (ideas.md #8).

Generation, pass, cone, critic, subsampled loss, schedules and optimiser are
cfast's.  The only change: in every cone return G_i, each later draw's
realised immediate cost l_j(a_j) = s_j(a_j) + lam n_j(a_j) + tau log q_j(a_j)
is replaced by its expectation under its own conditional,
lbar_j = sum_{a in A_j} q_j(a) l_j(a) (see _crb_core.py for the
unbiasedness argument).  Draws are still sampled; returns are stop-gradient;
the actor loss's direct all-action term is unchanged.  K = 0 only."""
from __future__ import annotations

from castlegen.bench.variant import register

from . import _crb_core
from .cfast import CFast


@register
class CRB(CFast):
    name = "crb"

    def __init__(self, problem, seed, workdir, **cfg):
        cfg = {"rb": True, **cfg}
        super().__init__(problem, seed, workdir, **cfg)

    def train_fns(self, lay):
        key = (self.cfg["rb"], self.cfg["critic"], self.cfg["sub"])
        if not hasattr(lay, "crb_fns"):
            lay.crb_fns = {}
        if key not in lay.crb_fns:
            lay.crb_fns[key] = _crb_core.build_train(self, lay, rb=self.cfg["rb"])
        return lay.crb_fns[key]

    def _build_train(self, lay):
        return _crb_core.build_train(self, lay, rb=self.cfg["rb"])


def _variant(name, **over):
    cls = type(f"CRB_{name}", (CRB,), {"name": name})
    orig = cls.__init__

    def init(self, problem, seed, workdir, **cfg):
        orig(self, problem, seed, workdir, **{**over, **cfg})
    cls.__init__ = init
    return register(cls)


_variant("crb_off", rb=False)        # control: cfast's realised returns through crb's code path
