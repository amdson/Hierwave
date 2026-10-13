"""cbig: cfast on the GPU with large batches.

Training is cfast's estimator unchanged (dependency-cone returns, h1 critic,
subsampled loss, one fused jitted update): unbiased for the same objective.
What changes is how it is driven:

- B worlds per update (default 256; cfast 8).  Adam lr kept at 3e-3: with a
  larger batch the per-batch gradient is less noisy, so the same normalised
  step carries more signal.
- Pipelined host work.  The fused update is dispatched asynchronously; the
  next batch's designed inputs (numpy, per world) are built while the GPU
  runs, and the update's scalars are read back only after that.  Designed
  inputs go to the device in one transfer per batch.
- Baseline energies are measured on the first 64 worlds of the (full-B)
  baseline batch, so the baseline still uses the training shape (no extra
  compile) without B host-side energy evaluations.
- Generation in batches.  generate(seed, region) on a periodic region runs a
  vmapped jitted pass over GEN_BATCH consecutive seeds (seed, seed + 1, ...)
  and caches the results; later calls for those seeds are lookups.  Each
  world's output is the same deterministic function of (parameters, seed,
  region) as a single-world pass.  The pass's log q of each raw draw is kept,
  so score() of a cached raw draw is a lookup (other states are scored by the
  forced pass).  Compiled functions are cached per region shape and shared
  by every instance in the process (both bench seeds).
"""
from __future__ import annotations

import hashlib
import time

import jax
import jax.numpy as jnp
import numpy as np

from castlegen.bench.problem import Region
from castlegen.bench.variant import register

from . import _cpass_core as core
from ._cbig_core import build_train
from .cfast import CFast

CFG = dict(B=256)
GEN_BATCH = 16
BASE_WORLDS = 64


def _hash(problem, x):
    h = hashlib.sha1()
    for c in problem.chans:
        if c.name not in x:
            continue
        a = np.ascontiguousarray(np.asarray(x[c.name], np.int32))
        h.update(c.name.encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


@register
class CBig(CFast):
    name = "cbig"

    def __init__(self, problem, seed, workdir, **cfg):
        super().__init__(problem, seed, workdir, **{**CFG, **cfg})
        self._gen_cache = {}
        self._lq_cache = {}

    def train_fns(self, lay):
        key = ("cbig", self.cfg["sub"])
        if not hasattr(lay, "cfast_fns"):
            lay.cfast_fns = {}
        if key not in lay.cfast_fns:
            lay.cfast_fns[key] = build_train(self, lay)
        return lay.cfast_fns[key]

    # ------------------------------------------------------------- batches
    def _designed_vec(self, c, seeds, region):
        """c.designed over a batch of seeds in one numpy call (seed as a
        leading broadcast axis), or None if the declaration does not
        broadcast that way; checked once against per-seed calls."""
        h, w = self.problem.H >> c.level, region.W >> c.level
        Y, X = np.meshgrid(np.arange(h), np.arange(w) + (region.x0 >> c.level), indexing="ij")
        period = (region.W >> c.level) if region.periodic else None
        sa = np.asarray(seeds, np.int64).reshape((-1,) + (1,) * 6)
        try:
            with np.errstate(all="ignore"):
                v = np.asarray(c.designed(sa, Y, X, period), np.int64)
            if v.shape == (h, w):
                v = np.broadcast_to(v, (len(seeds), h, w))
            v = v.reshape(len(seeds), h, w)
        except Exception:
            return None
        return v

    def _batch_np(self, seeds, region):
        """Designed inputs of a batch as numpy (host work only)."""
        seeds = np.asarray(seeds)
        out = {}
        vec_ok = getattr(self, "_vec_ok", {})
        self._vec_ok = vec_ok
        for c in self.problem.chans:
            if c.designed is None:
                continue
            v = None
            if vec_ok.get(c.name, True):
                v = self._designed_vec(c, seeds, region)
                if c.name not in vec_ok:                  # first use: check against per-seed calls
                    ok = v is not None
                    for b in sorted({0, len(seeds) - 1}):
                        if not ok:
                            break
                        ref = self.problem.designed_values(int(seeds[b]), region)[c.name]
                        ok = np.array_equal(v[b], ref)
                    vec_ok[c.name] = ok
                    if not ok:
                        v = None
            if v is None:
                v = np.stack([self.problem.designed_values(int(s), region)[c.name] for s in seeds])
            out[c.name] = np.ascontiguousarray(v, np.int32)
        return out

    def _batch(self, seeds, region):
        return {k: jnp.asarray(v) for k, v in self._batch_np(seeds, region).items()}

    def _measure_baseline(self, lay, log):
        """cfast's baseline (fused at lr = 0 on a full-B batch, untrained tau =
        1 draws), with the host-side energies on the first BASE_WORLDS worlds."""
        from castlegen.bench import energy as en
        T, nb = self.problem.T, self.cfg["B"]
        region = Region(lay.W)
        seeds = np.arange(500_000 + 1000 * self.seed, 500_000 + 1000 * self.seed + nb)
        des = self._batch(seeds, region)
        lam = jnp.asarray(self.lam, jnp.float32)
        js = jnp.asarray(seeds, jnp.int32)
        ones = {k: jnp.float32(1.0) for k in self.params}
        out = self.fn["fused"](self.params, self.cparams, self.opt, self.copt, ones, js, des,
                               jnp.float32(1.0), lam, 0.0, 0.0, self.cfg["clip"], jax.random.PRNGKey(0))
        st, gstd = out[0], out[-1]
        self.gs = {k: jnp.float32(max(float(v), 1.0)) for k, v in gstd.items()}
        fin = self.fn["repair_b"](st, js)
        nm = min(nb, BASE_WORLDS)
        st = jax.device_get({k: v[:nm] for k, v in st.items()})
        fin = jax.device_get({k: v[:nm] for k, v in fin.items()})
        raw, ff_, n = np.zeros(T + 1), np.zeros(T + 1), np.zeros(T + 1)
        for b in range(nm):
            d = {k: np.asarray(v[b]) for k, v in des.items()}
            for kind, x, acc in (("raw", st, raw), ("final", fin, ff_)):
                xs = {c.name: np.asarray(x[c.name][b]) for c in self.problem.sampled()}
                E = en.energy(self.problem, {**d, **xs}, region)
                for l in range(T + 1):
                    acc[l] += E.viol[l]
                    if kind == "raw":
                        n[l] += E.nhard[l]
        self.nhard = n / nm
        rf, ff = raw / np.maximum(n, 1), ff_ / np.maximum(n, 1)
        rho = np.where(rf > 0, ff / np.maximum(rf, 1e-12), 1.0)
        self.eps = np.clip(1e-4 / np.maximum(rho, 1e-12), 1e-4, 1e-3)
        log({"baseline_raw": rf.tolist(), "baseline_final": ff.tolist(), "rho": rho.tolist(),
             "eps": self.eps.tolist()})
        self.baseline = {"raw": rf.tolist(), "final": ff.tolist(), "eps": self.eps.tolist()}

    # -------------------------------------------------------------- training
    def train(self, budget_s, log):
        t0 = time.time()
        cfg, T = self.cfg, self.problem.T
        region = Region(cfg["W_train"] or self.problem.W_train)
        lay = self.layout(region)
        self.fn = self.train_fns(lay)
        self._measure_baseline(lay, log)
        t_setup = time.time() - t0
        log({"t": t_setup, "setup_done": True})
        B = cfg["B"]
        tm = {"host": 0.0, "wait": 0.0}

        def next_batch():
            seeds = np.arange(self.train_seed, self.train_seed + B)
            self.train_seed += B
            return seeds, self._batch(seeds, region)

        seeds, des = next_batch()
        while True:
            if time.time() - t0 >= budget_s:
                break
            ftr = (time.time() - t0 - t_setup) / max(budget_s - t_setup, 1e-9)
            self.tau = cfg["tau1"] + (cfg["tau0"] - cfg["tau1"]) * max(0.0, 1 - ftr / cfg["anneal"])
            tau, lam = jnp.float32(self.tau), jnp.asarray(self.lam, jnp.float32)
            ta = time.time()
            out = self.fn["fused"](self.params, self.cparams, self.opt, self.copt, self.gs,
                                   jnp.asarray(seeds, jnp.int32), des, tau, lam, cfg["lr"], cfg["lr_critic"],
                                   cfg["clip"], jax.random.PRNGKey(self.updates + 7919 * self.seed))
            _, self.params, self.cparams, self.opt, self.copt, loss, closs, viol, tot, _ = out
            seeds, des = next_batch()                      # host work overlaps the device update
            tb = time.time()
            viol = np.asarray(viol)
            tc = time.time()
            tm["host"] += tb - ta
            tm["wait"] += tc - tb
            hf = np.where(self.nhard > 0, viol / np.maximum(self.nhard, 1), 0.0)
            self.vbar = (1 - cfg["alpha"]) * self.vbar + cfg["alpha"] * hf
            step = cfg["eta"] * np.clip(self.vbar / self.eps - 1, -1, 1)
            self.lam = np.where(self.nhard > 0, np.clip(self.lam + step, 0, cfg["lam_max"]), 0.0)
            self.updates += 1
            log({"t": time.time() - t0, "update": self.updates, "tau": self.tau, "loss": float(loss),
                 "critic_loss": float(closs), "cost_per_world": float(np.asarray(tot).mean()),
                 "raw_viol": hf.tolist(), "lam": self.lam.tolist(), "enum_frac": None,
                 "time_host": tm["host"], "time_wait": tm["wait"]})
        self.timing = {**tm, "setup": t_setup, "total": time.time() - t0, "B": B}
        self._gen_cache.clear()
        self._lq_cache.clear()

    # ------------------------------------------------------------ generation
    def _gen_fns(self, lay):
        if not hasattr(lay, "cbig_gen"):
            def gen(params, seed, des, lam):
                st, _, lq, _ = core.coloured_pass(lay, params, seed, des, 1.0, lam)
                return st, core.repair(lay, st, seed), lq
            lay.cbig_gen = jax.jit(jax.vmap(gen, in_axes=(None, 0, 0, None)))
        return lay.cbig_gen

    def generate(self, seed, region):
        if not region.periodic:                         # open windows: cpass's padded single-world pass
            return super().generate(seed, region)
        key = (region.W, seed)
        if key not in self._gen_cache:
            lay = self.layout(region)
            fn = self._gen_fns(lay)
            seeds = np.arange(seed, seed + GEN_BATCH)
            desn = self._batch_np(seeds, region)
            des = {k: jnp.asarray(v) for k, v in desn.items()}
            raw, fin, lq = jax.device_get(fn(self.params, jnp.asarray(seeds, jnp.int32), des,
                                             jnp.asarray(self.lam, jnp.float32)))
            for b, s in enumerate(seeds):
                out = {kind: {c.name: np.asarray(st[c.name][b]) for c in self.problem.sampled()}
                       for kind, st in (("raw", raw), ("final", fin))}
                self._gen_cache[(region.W, int(s))] = out
                full = {**{k: v[b] for k, v in desn.items()}, **out["raw"]}
                self._lq_cache[(region.W, _hash(self.problem, full))] = float(lq[b])
        return self._gen_cache.pop(key)

    def score(self, x, region):
        k = (region.W, _hash(self.problem, x))
        if region.periodic and k in self._lq_cache:
            return self._lq_cache[k]
        return super().score(x, region)

    def info(self):
        return {**super().info(), "gen_batch": GEN_BATCH}


def _variant(name, **over):
    cls = type(f"CBig_{name}", (CBig,), {"name": name})
    orig = cls.__init__

    def init(self, problem, seed, workdir, **cfg):
        orig(self, problem, seed, workdir, **{**over, **cfg})
    cls.__init__ = init
    return register(cls)
