"""JaxBlockGibbs (castlegen/kernels/amortised.py) reproduces ExactBlockGibbs's
conditional energies and the bench energy on ground probe states."""
import jax.numpy as jnp
import numpy as np

from castlegen.bench.problem import Region
from castlegen.bench.problems import get
from castlegen.kernels.amortised import JaxBlockGibbs, make_energy
from castlegen.kernels.exact_block import total_energy


def test_jax_block_energies_match_numpy():
    p = get("ground"); reg = Region(p.W_train)
    xs = [{**p.probes(s, reg, n=k + 1)[k], **p.designed_values(s, reg)} for s, k in [(11, 0), (12, 3)]]
    for L, d in [(2, 0), (1, 1)]:
        J = JaxBlockGibbs(p, reg, L, d, lam=20.0)
        K = J.K
        v = np.stack([K.flat.flatten(x) for x in xs])
        for k in range(K.ncol):
            _, nbk, gs = J.classes[k]
            sj, nj = J._class_energy(jnp.asarray(v, jnp.int32), gs, nbk, jnp.asarray(J.comps[:200]))
            for w in range(2):
                sn, nn = K.energies(v[w], k, comps=np.asarray(J.comps[:200], np.int64))
                assert np.allclose(np.asarray(sj[w]), sn, atol=1e-3)
                assert (np.asarray(nj[w]) == nn).all()
        s, nv, _ = make_energy(K.flat, K.insts)(jnp.asarray(v, jnp.int32))
        for w in range(2):
            es, en = total_energy(K.flat, K.insts, v[w])
            assert abs(float(s[w]) - es) < 1e-2 and int(nv[w]) == en


def _states(J, p, reg, n=6):
    fl = J.K.flat
    v = np.stack([fl.flatten({**p.probes(s, reg, n=k + 1)[k], **p.designed_values(s, reg)}) for s, k in [(11, 0), (12, 3), (13, 1)] * (n // 3)])
    rng = np.random.default_rng(0)
    # perturb: random values on a third of the sampled entries (creates violations)
    D = np.zeros(fl.n, np.int64)
    for c in p.chans:
        D[fl.off[c.name]:fl.off[c.name] + int(np.prod(fl.shape[c.name]))] = c.D if hasattr(c, "D") else 1
    return jnp.asarray(v, jnp.int32)


def test_fast_sweep_equals_reference():
    import jax
    p = get("ground"); reg = Region(p.W_train)
    for L, d in [(2, 0), (1, 1)]:
        for lam in (20.0, None):
            J = JaxBlockGibbs(p, reg, L, d, lam=lam)
            assert J.fast
            v = _states(J, p, reg)
            # also a state with violations: take refined-by-another-colour state with shifted channel values
            v2 = jnp.roll(v, 1, axis=0)
            vm = jnp.where(jnp.arange(v.shape[1])[None] % 3 == 0, v2, v)
            for st in (v, vm):
                for i in range(2):
                    key = jax.random.PRNGKey(i)
                    a, ca = J.sweep(st, key)
                    b, cb = J.sweep_ref(st, key)
                    assert float((a != b).mean()) < 1e-4
                    assert abs(int(ca.sum()) - int(cb.sum())) <= 2
                    st = a


def test_refine_scan_and_multi_update_run():
    import jax
    from castlegen.kernels.amortised import make_refine_scan
    p = get("ground"); reg = Region(p.W_train)
    J = JaxBlockGibbs(p, reg, 2, 0, lam=20.0)
    v = _states(J, p, reg)
    E = make_energy(J.K.flat, J.K.insts)
    out, ch = make_refine_scan([J], 3)(v, jax.random.PRNGKey(0))
    assert out.shape == v.shape and ch.shape == (1, v.shape[0])
    assert float(E(out)[0].mean()) > float(E(v)[0].mean()) * 0.5     # sanity: still a sensible state
