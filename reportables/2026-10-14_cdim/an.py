import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
names=[("cfast","defaults"),("cfast_lam1","lam=1"),("cfast_lam4","lam=4"),("cfast_lam20","lam=20"),("cfast_tau2_lam4","tau=2, lam=4"),("cfast_lr1e3","lr 1e-3")]
fig,ax=plt.subplots(1,3,figsize=(15,4))
out=[]
for n,l in names:
    d=json.load(open(f"run_{n}.json")); R=d["rows"]; f=d["final"]
    u=[r["update"] for r in R]; dg=[r["raw"][2] for r in R]
    drop=next((r["update"] for r in R if r["raw"][2]<0.05),None)
    ax[0].plot(u,dg,label=l); ax[1].plot(u,[r["ent"] for r in R]); ax[2].plot(u,[max(r["raw_viol"],1e-5) for r in R])
    mid=[r for r in R if r["update"]>=100][:1]
    print(f"| {l} | {drop or 'survives'} | {f['update']} | {f['raw']} | {f['ent']:.3f} | {f['raw_viol']:.2e} | {f['final_viol']:.2e} | {'yes' if f['final_viol']<2e-4 else 'no'} | lam_end {R[-1]['lam']:.1f} tau_end {R[-1]['tau']:.2f} |")
    print("   u100:",[ (r['update'],r['raw'],round(r['ent'],3),round(r['raw_viol'],4)) for r in R if r['update'] in (1,20,100,200)])
ax[0].set_ylabel("diagonal fraction");ax[1].set_ylabel("entropy (nats/cell)");ax[2].set_ylabel("raw violation fraction");ax[2].set_yscale("log")
for a in ax:a.set_xlabel("update");a.set_xscale("symlog",linthresh=20)
ax[0].legend(fontsize=8);plt.tight_layout();plt.savefig("images/cdim_dimers.png",dpi=110)
