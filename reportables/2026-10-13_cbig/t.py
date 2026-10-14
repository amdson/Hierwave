import sys,time,tempfile
from castlegen.bench.problems import get
import castlegen.variants.cbig as cb
prob,B,bud=sys.argv[1],int(sys.argv[2]),float(sys.argv[3])
v=cb.CBig(get(prob),seed=0,workdir=tempfile.mkdtemp(),B=B)
logs=[]
t=time.time(); v.train(bud,logs.append)
s=[d for d in logs if d.get("setup_done")][0]
print("RESULT",prob,B,"setup_s",round(s["setup_s"],1),"compile_s",round(s["compile_s"],1),"updates",v.updates,"s/update",round(bud/max(v.updates,1),4),"wall",round(time.time()-t,1),flush=True)
