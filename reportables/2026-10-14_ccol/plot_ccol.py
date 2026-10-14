import json,numpy as np,sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D="reportables/2026-10-14_ccol"
L=[json.loads(x) for x in open(f"{D}/run/train_columns.jsonl")]
dg=[d for d in L if 'diag' in d]; u=np.array([d['update'] for d in dg])
cols=["#2a78d6","#eb6834","#1baf7a","#a855c9","#d4a514"]
def first(f):
    i=np.where(f)[0]; return int(u[i[0]]) if len(i) else None
for th in (0.3,):
    for l in range(5):
        H=np.array([d['diag'][f'L{l}']['H'] for d in dg]); print(f"L{l} H<{th} first at", first(H<th), " min", H.min().round(3), "last", H[-1].round(3))
dens=np.array([d['diag']['density'] for d in dg]); print("dens<0.1", first(dens<0.1), "dens<0.05", first(dens<0.05), "min", dens.min())
emp=np.array([d['diag']['empty_xcol'] for d in dg]); print("empty>0.3", first(emp>0.3))
fig,ax=plt.subplots(4,1,figsize=(8,11),sharex=True)
xl=(0,600)
for l in range(5):
    ax[0].plot(u,[d['diag'][f'L{l}']['H'] for d in dg],c=cols[l],label=f"level {l}")
    ax[1].plot(u,[d['diag'][f'L{l}']['forced'] for d in dg],c=cols[l])
    ax[2].plot(u,[d['diag'][f'L{l}']['absb'] for d in dg],c=cols[l])
ax[0].set_ylabel("entropy of q, non-forced (nats)");ax[1].set_ylabel("forced fraction");ax[2].set_ylabel("mean |b| over A")
ax[3].plot(u,dens,'k',label="density");ax[3].plot(u,emp,c='0.5',label="empty x-column frac");ax[3].axhline(0.05,ls=':',c='r')
ax[3].set_ylabel("density / empty frac");ax[3].legend(frameon=False);ax[0].legend(frameon=False)
for a in ax: a.grid(alpha=.3); a.set_xlim(*xl)
ax[3].set_xlabel("update (first 600 of %d; logged every 25)"%max(u))
plt.tight_layout();plt.savefig(f"{D}/images/ccol_timeline.png",dpi=110)
