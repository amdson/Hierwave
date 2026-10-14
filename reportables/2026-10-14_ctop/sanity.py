import pickle, numpy as np, jax, jax.numpy as jnp
import sys; sys.argv=["x","snaps.pkl","/tmp/x.pkl","2"]
exec(open("ctop_eval.py").read().split("seeds = np.arange")[0])
seeds = np.arange(4_000_000, 4_000_002)
dl = [p.designed_values(int(s), reg) for s in seeds]
des = {k: jnp.asarray(np.stack([d[k] for d in dl]), jnp.int32) for k in dl[0]}
sn = pickle.load(open("snaps.pkl","rb"))["u200"]; tau=sn["tau"]; lam=jnp.asarray(sn["lam"],jnp.float32)
params = jax.tree_util.tree_map(jnp.asarray, sn["params"])
_, recs = jax.vmap(lambda s,d: train_pass(lay, params, s, d, jnp.float32(tau), lam))(jnp.asarray(seeds,jnp.int32), des)
G = cone.returns({k: r["cost"] for k,r in recs.items()})
r = pickle.load(open("/tmp/ctop_smoke.pkl","rb"))[("u200", tau)]
tab=r["tab"]
errs=[]
for d,(ci,k,j,Y,X) in enumerate(tab):
    for b in range(2):
        errs.append(abs(float(G[topk[ci]][b,k,j]) - r["G"]["actual"][b,d].sum()))
print("max |G_cone - G_replay(actual)|", max(errs), "typical G", float(np.abs(G[topk[0]]).mean()))
