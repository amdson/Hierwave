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
