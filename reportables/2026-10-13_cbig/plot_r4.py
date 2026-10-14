import json,glob,numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="results/runs/"; NC=512; FS={"modes":0.2513,"support":-0.4688}
def load(v,p,s):
    d=glob.glob(f"{R}*_{v}_toy_{p}_r4")[0]
    rows=[json.loads(l) for l in open(f"{d}/train_s{s}.jsonl")]
    rows=[r for r in rows if "update" in r]
    return np.array([r["update"] for r in rows]),np.array([r["cost_per_world"] for r in rows])/NC,np.array([r["tau"] for r in rows])
def sm(x,y,n=40):
    k=max(1,len(y)//n); m=len(y)//k*k
    return x[:m].reshape(-1,k)[:,-1], y[:m].reshape(-1,k).mean(1)
fig,ax=plt.subplots(1,2,figsize=(12,4.5))
col={"cfast":"#888","cbig64":"#1f77b4","cbig256":"#d62728","cbig1024":"#2ca02c"}
out={}
for a,(p,vs) in zip(ax,[("modes",["cfast","cbig64","cbig256","cbig1024"]),("support",["cfast","cbig256"])]):
    for v in vs:
        for s in (0,1):
            u,f,t=load(v,p,s); x,y=sm(u,f)
            a.plot(x,y,color=col[v],ls="-" if s==0 else "--",label=v if s==0 else None)
            n=len(f); out[(v,p,s)]=(u[-1],f[int(.9*n):].mean(),f[int(.7*n):int(.8*n)].mean(),f[int(.8*n):int(.9*n)].mean())
    a.axhline(FS[p],color="k",lw=1,label="F* exact"); a.set_xscale("log"); a.set_xlabel("updates"); a.set_ylabel("training-batch F per cell (cost/512)")
    a.set_title(f"toy_{p} (solid seed 0, dashed seed 1)")
    lo=FS[p]; a.set_ylim(lo-0.01, lo+(0.12 if p=="modes" else 0.2)); a.legend()
plt.tight_layout(); plt.savefig("reportables/2026-10-13_cbig/images/r4_F_vs_updates.png",dpi=130)
for k,v in out.items(): print(k,"updates %d  last10%% %.4f  gap %.4f | 70-80%% %.4f 80-90%% %.4f"%(v[0],v[1],v[1]-FS[k[1]],v[2],v[3]))
