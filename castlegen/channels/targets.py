"""Targets for the bootstrap (notes/dsl_updates.md C2, train.fit): p*'s
collapsed conditional at one site over the sampler's candidate set.

The protocol is duck-typed: at(model, home, y, x, cand) -> (len(cand),)
float64 summing to 1, computed from the model's current state.

ExactTargets(probs_fn)  a closed-form hook: probs_fn(model, home, y, x) -> (D,)
                        full conditional (e.g. Oracle.mid_probs), restricted to
                        cand and renormalised (a conditional restricted to a
                        subset is proportional to the same weights)
SampledTargets()        one-hot of the current value at (y, x) within cand:
                        for contexts that are joint samples of p* (the target is
                        then the sampled value; pseudo-likelihood)
AISTargets              package D (aistargets.py)."""
import numpy as np


class ExactTargets:
    def __init__(self, probs_fn):
        self.probs_fn = probs_fn

    def at(self, model, home, y, x, cand):
        p = np.asarray(self.probs_fn(model, home, y, x), np.float64)[np.asarray(cand, np.int64)]
        s = p.sum()
        assert s > 0, f"no target mass on the candidates at {home} ({y}, {x})"
        return p / s


class SampledTargets:
    def at(self, model, home, y, x, cand):
        cand = np.asarray(cand, np.int64)
        out = (cand == model.chan(home).grid[y, x]).astype(np.float64)
        assert out.sum() == 1, f"current value at {home} ({y}, {x}) not among the candidates"
        return out
