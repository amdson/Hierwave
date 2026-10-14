import json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
runs=[("lam11","cbound_lam seed 11"),("lam12","cbound_lam seed 12"),("cfast11","cfast seed 11 (control)")]
fig,ax=plt.subplots(1,3,figsize=(15,4),sharey=True)
for a,(t,l) in zip(ax,runs):
    d=json.load(open(f"dimers_{t}.json")); r=d["rows"]; u=[x["update"] for x in r]
    for i,(n,c) in enumerate([("horizontal","C0"),("vertical","C1"),("diagonal","C2")]): a.plot(u,[x["raw"][i] for x in r],c,label=n)
    a.set_title(l); a.set_xlabel("update"); a.axhline(.05,color="gray",ls=":")
    # diag recovery
    f=d["final"]; print(t,"updates",f["update"],"final raw",f["raw"],"ent",f["ent"],"raw viol",f["raw_viol"],"repaired",f["final_viol"])
    dg=[x["raw"][2] for x in r]; lo=[i for i,v in enumerate(dg) if v<0.05]
    print("  first diag<0.05 at",u[lo[0]] if lo else None,"min diag",min(dg),"at",u[dg.index(min(dg))])
    if lo:
        rec=[u[i] for i in range(lo[0],len(dg)) if dg[i]>=0.15]; print("  first diag>=0.15 after drop at",rec[0] if rec else None)
    print("  max diag after update 100:",max(v for x,v in zip(u,dg) if x>100))
ax[0].set_ylabel("orientation fraction (4 held-out worlds)"); ax[0].legend()
plt.tight_layout(); plt.savefig("images/orientation_curves.png",dpi=110)
