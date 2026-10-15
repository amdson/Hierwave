import numpy as np, jax.numpy as jnp, sys
from castlegen.bench.problems import columns as C
from castlegen.bench.problem import Region
from castlegen.bench.energy import energy
from castlegen.kernels import colour_gibbs as cg
from castlegen.variants._cpass_core import Layout
W=128; pc=C.build_columns()
parts=C.chain_parts(C.allowed_table(), np.where(np.arange(C.D)>0,-C.MU,0.0))
ks=C.sample_chains(np.random.default_rng(0),W,parts); x=C.lift(ks)
print("bench", energy(pc,x,Region(W)).viol)
st={k:jnp.asarray(v[None].astype(np.int32)) for k,v in x.items()}
lay=Layout(pc,W,True)
print(cg.state_energy(lay,st))
for l in range(0,5):
    pass
# per-factor
from castlegen.kernels.colour_gibbs import _state_energy
