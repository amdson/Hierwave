"""cnocq (P11-1 of reportables/2026-10-14_cmix/proposals.md; ideas #22; user-accepted behaviour change, recorded in
notes/constitution_change_requests.md): the reverse-KL cost without the tau log q term of selected channels.

    ckpop_hard_rkl_nocq   ckpop_hard_rkl (ckernel_own.py) whose 32 reverse-KL steps per iteration use cfast's fused
                          update with tau * log q dropped for the channels named in cfg["nocq_chans"] (default
                          cert1..cert4 on ground: the certificate channels).  Those channels are still sampled, and
                          their soft energies and lam * violations still count; only their entropy term is gone, both
                          from the per-draw cost (so it leaves every return) and from the immediate term ell of the
                          actor loss.  The ML (kernel-route) half of the iteration is unchanged.

The channel list is a cfg list chosen by name FOR THIS TEST ONLY: a generic rule for which channels are "derived"
(constraint 1) is being measured separately (P6).  With nocq_chans = [] the update is the parent's, bit for bit
(tests/variants/test_cnocq.py).

Implementation: cfast.CFast._build_train is copied at import by source transform (inspect + three asserted string
replacements), not edited; the shared file stays untouched and the copy follows it.
"""
from __future__ import annotations

import inspect
import textwrap

from castlegen.bench.variant import VARIANTS, register

from . import cfast as _cfast
from . import ckernel_own  # noqa: F401  (registers the ckpop_* family)

_REPL = [
    # 1. the channel set, resolved per build
    ("    use_cone = self.cfg[\"cone\"]\n",
     "    use_cone = self.cfg[\"cone\"]\n"
     "    _nocq = set(self.cfg.get(\"nocq_chans\") or [])\n"
     "    _isnocq = lambda key: key[key.index(\"_\") + 1:] in _nocq\n"),
    # 2. per-draw cost: subtract tau * log q(a) of the selected channels' draws (enters every return and tot)
    ("    def recost(recs, lam):\n",
     "    def recost(recs, lam, tau=None):\n"
     "        if _nocq and tau is not None:\n"
     "            recs = {k: ({**r, \"cost\": r[\"cost\"] - jnp.where(r[\"valid\"], tau * jnp.take_along_axis(\n"
     "                r[\"logq\"], r[\"a\"][..., None], -1)[..., 0], 0.0)} if _isnocq(k) else r) for k, r in recs.items()}\n"),
    # 3. immediate term of the actor loss
    ("else r[\"soft\"] + lam[l] * r[\"nv\"] + tau * lq_sg\n",
     "else (r[\"soft\"] + lam[l] * r[\"nv\"] if _isnocq(key) else r[\"soft\"] + lam[l] * r[\"nv\"] + tau * lq_sg)\n"),
]


def _make_build_train():
    src = textwrap.dedent(inspect.getsource(_cfast.CFast._build_train))
    for old, new in _REPL:
        assert src.count(old) == 1, ("cnocq: cfast._build_train changed, replacement not unique", old)
        src = src.replace(old, new)
    n = src.count("recost(recs, lam)")
    assert n == 2, ("cnocq: expected two recost call sites", n)
    src = src.replace("recost(recs, lam)", "recost(recs, lam, tau)")
    ns = dict(vars(_cfast))
    exec(compile(src, "<cnocq:_build_train>", "exec"), ns)
    return ns["_build_train"]


_build_train_nocq = _make_build_train()
CERTS = ["cert1", "cert2", "cert3", "cert4"]


class CKPopHardRklNocq(VARIANTS["ckpop_hard_rkl"]):
    name = "ckpop_hard_rkl_nocq"

    def __init__(self, problem, seed, workdir, **cfg):
        super().__init__(problem, seed, workdir, **{"nocq_chans": CERTS, **cfg})
        assert not self.cfg.get("pen_only"), "nocq is a change of the full reverse-KL cost"

    def train_fns(self, lay):
        c = self.cfg
        key = ("nocq", tuple(sorted(c.get("nocq_chans") or [])), c["K"], c["cone"], c["critic"], c["sub"],
               c.get("net", "tanh"))
        if not hasattr(lay, "nocq_fns"):
            lay.nocq_fns = {}
        if key not in lay.nocq_fns:
            lay.nocq_fns[key] = _build_train_nocq(self, lay)
        return lay.nocq_fns[key]

    def _build_train(self, lay):
        return _build_train_nocq(self, lay)


register(CKPopHardRklNocq)
