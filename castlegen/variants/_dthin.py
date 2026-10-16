"""dthin: thin (kh x kw) level-0 blocks for FineBlockGibbs (p*-invariant; shapes only, no problem-specific move)."""
import numpy as np
from castlegen.kernels.fine_block import FineBlockGibbs, JaxFineBlockGibbs
from castlegen.kernels.amortised import internal_pruned
from castlegen.kernels.exact_block import Flat, build_instances


class ThinBlockGibbs(FineBlockGibbs):
    def __init__(self, problem, region, kh, kw, offset=(0, 0), lam=None, seed=0, max_out=20_000_000):
        self.p, self.region = problem, region
        self.L, self.depth, self.k, self.offset, self.lam = 0, 0, (kh, kw), tuple(offset), lam
        self.flat = Flat(problem, region); self.insts = build_instances(self.flat)
        self.rng = np.random.default_rng(seed)
        fl, W, H = self.flat, region.W, problem.H
        oy, ox = offset
        ny, nx = (H - oy) // kh, W // kw
        assert ny >= 1 and nx >= 1
        TY, TX = np.meshgrid(np.arange(ny) * kh + oy, (np.arange(nx) * kw + ox) % W, indexing="ij")
        TY, TX = TY.ravel(), TX.ravel()
        chans = [c for c in problem.level_chans(0) if c.designed is None]
        cols, doms = [], []
        for dy in range(kh):
            for dx in range(kw):
                for c in chans:
                    cols.append(fl.gid(c.name, TY + dy, TX + dx)); doms.append(c.D)
        self.BV, self.D = np.stack(cols, 1), np.array(doms)
        self._structure()
        self.joint = int(np.prod(self.D.astype(float)))
        self.comps = internal_pruned(self, max_out=max_out); self.shared = True; self._cache = {}


class ThinKernel:
    """sweep(v,key): alternate orientation each call (kh x kw, then kw x kh), random offset."""
    def __init__(self, problem, region, kh, kw, lam=20.0, seed=0, n_off=3, max_out=20_000_000):
        rng = np.random.default_rng(seed)
        self.sh = [(kh, kw), (kw, kh)] if kh != kw else [(kh, kw)]
        self.Js, self.nc = [], []
        for (a, b) in self.sh:
            offs = [(0, 0)] + [(int(rng.integers(a)), int(rng.integers(b))) for _ in range(n_off - 1)]
            Ks = [ThinBlockGibbs(problem, region, a, b, o, lam, seed, max_out) for o in offs]
            self.Js.append([JaxFineBlockGibbs(K) for K in Ks]); self.nc.append(Ks[0].comps.shape[0])
        self.flat = Ks[0].flat; self.rng = np.random.default_rng(seed + 17); self.t = 0

    def sweep(self, v, key):
        js = self.Js[self.t % len(self.Js)]; self.t += 1
        return js[int(self.rng.integers(len(js)))].sweep(v, key)
