"""csat: cfast with saturation diagnostics in its train log (no change to
training or generation).  Every `diag_every` updates it runs the coloured
pass (train_pass, current tau and lambda, current parameters) on a fixed batch
of diagnostic worlds and logs, per level and per channel:

- H: mean entropy of q over non-forced draws (valid sites with |A| > 1);
- H0: the same with the learned bias removed (b = 0), i.e. what the
  conditional alone gives at this tau;
- forced: the fraction of valid draws with |A| = 1;
- absb: mean |b| over allowed values (A) of valid sites;
- spread: mean (max - min) of b over A at non-forced sites (b is shift-
  invariant, so this is the part that acts);
- dirt: the fraction of t = 1 (dirt) in the drawn worlds.

Variants differ only in engineering freedoms (constitution: lr, clipping, tau
schedule)."""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench.problem import Region

from castlegen.variants import _cpass_core as core
from castlegen.variants._cfast_core import train_pass
from castlegen.variants.cfast import CFast

_DIAG = {}


class CSat(CFast):
    name = "csat"

    def _diag_fn(self, lay):
        key = id(lay)
        if key in _DIAG:
            return _DIAG[key]
        keys = [(l, c.name, core.key_of(l, c.name)) for l in lay.levels for c in lay.chans[l]]

        def one(params, seed, des, tau, lam):
            st, recs = train_pass(lay, params, seed, des, tau, lam)
            out = {}
            for l, cn, k in keys:
                r = recs[k]
                valid, mask = r["valid"], r["mask"]
                nA = jnp.sum(mask, -1)
                nf = valid & (nA > 1)
                fo = valid & (nA == 1)
                logq = r["logq"]
                H = -jnp.sum(jnp.where(mask, jnp.exp(logq) * logq, 0.0), -1)
                lq0 = jax.nn.log_softmax(jnp.where(mask, -r["soft"] / tau, -jnp.inf), -1)
                H0 = -jnp.sum(jnp.where(mask, jnp.exp(lq0) * lq0, 0.0), -1)
                b = core.mlp(params[k], r["feat"])
                mv = mask & valid[..., None]
                absb = jnp.sum(jnp.where(mv, jnp.abs(b), 0.0))
                sp = jnp.max(jnp.where(mask, b, -jnp.inf), -1) - jnp.min(jnp.where(mask, b, jnp.inf), -1)
                out[k] = jnp.stack([jnp.sum(nf), jnp.sum(fo), jnp.sum(valid),
                                    jnp.sum(jnp.where(nf, H, 0.0)), jnp.sum(jnp.where(nf, H0, 0.0)),
                                    absb, jnp.sum(mv), jnp.sum(jnp.where(nf, sp, 0.0))])
            dirt = jnp.mean((st["k"] > 0).astype(jnp.float32))
            return out, dirt

        f = jax.jit(jax.vmap(one, in_axes=(None, 0, 0, None, None)))
        _DIAG[key] = (f, keys)
        return _DIAG[key]

    def diag(self, tau=None, lam=None):
        cfg = self.cfg
        region = Region(cfg["W_train"] or self.problem.W_train)
        lay = self.layout(region)
        f, keys = self._diag_fn(lay)
        nb = cfg.get("diag_B", 8)
        seeds = np.arange(900_000, 900_000 + nb)
        des = self._batch(seeds, region)
        tau = self.tau if tau is None else tau
        lam = self.lam if lam is None else lam
        out, dirt = f(self.params, jnp.asarray(seeds, jnp.int32), des, jnp.float32(tau),
                      jnp.asarray(lam, jnp.float32))
        out = {k: np.asarray(v).sum(0) for k, v in jax.device_get(out).items()}
        res = {"dirt": float(np.mean(np.asarray(dirt)))}
        per_level = {}
        for l, cn, k in keys:
            a = out[k]
            res[f"{k}"] = {"H": a[3] / max(a[0], 1), "H0": a[4] / max(a[0], 1),
                           "forced": a[1] / max(a[2], 1), "absb": a[5] / max(a[6], 1),
                           "spread": a[7] / max(a[0], 1), "nonforced": a[0] / max(a[2], 1)}
            pl = per_level.setdefault(l, np.zeros(8))
            pl += a
        for l, a in per_level.items():
            res[f"L{l}"] = {"H": a[3] / max(a[0], 1), "H0": a[4] / max(a[0], 1),
                            "forced": a[1] / max(a[2], 1), "absb": a[5] / max(a[6], 1),
                            "spread": a[7] / max(a[0], 1)}
        return {k: ({kk: float(vv) for kk, vv in v.items()} if isinstance(v, dict) else v)
                for k, v in res.items()}

    def train(self, budget_s, log):
        N = self.cfg.get("diag_every", 25)
        self._dtime = 0.0

        def wlog(d):
            u = d.get("update")
            if u is not None and (u == 1 or u % N == 0):
                ta = time.time()
                d = {**d, "diag": self.diag()}
                self._dtime += time.time() - ta
                d["diag_time"] = self._dtime
            log(d)
        super().train(budget_s, wlog)


