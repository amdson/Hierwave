import numpy as np, time
from castlegen.bench import energy as en
from castlegen.bench.problem import Region
from castlegen.variants._crevise_gradcheck import tiny_flat_problem
p=tiny_flat_problem(); r=Region(8); t=time.time(); E=[]
for i in range(1<<16):
    x=((i>>np.arange(16))&1).astype(np.int32).reshape(2,8)
    e=en.energy(p,{"t":x},r); E.append(e.soft+20.0*e.viol[0])
E=np.array(E); print("min E",E.min(),"logZ",np.log(np.exp(-(E-E.min())).sum())-E.min(),"n_min",(E<E.min()+1e-9).sum(),"secs",time.time()-t)
