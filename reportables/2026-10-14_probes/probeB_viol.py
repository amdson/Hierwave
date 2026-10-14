import numpy as np
from castlegen.bench.problems import get as get_problem
from castlegen.bench.problems.dimers import DIRS, OPP, E, S, SE
from castlegen.bench.problem import Region
from castlegen.bench.run import EVAL_SEED0
from castlegen.bench.variant import load_variants
pr=get_problem("dimers"); reg=Region(pr.W_eval); g=load_variants()["greedy"](pr,seed=0,workdir="/tmp/pbv"); g.train(0,lambda d:None)
H,W=pr.H,pr.W_eval; cnt=np.zeros(6); bad=np.zeros(6); inst=np.zeros(3); badi=np.zeros(3)
for i in range(16):
    p=np.asarray(g.generate(EVAL_SEED0+i,reg)["raw"]["p"])
    for oi,o in enumerate((E,S,SE)):
        dy,dx=DIRS[o]
        q=np.full_like(p,-1); 
        r=np.roll(p,-dx,1); 
        if dy>=0: q[:H-dy]=r[dy:]
        else: q[-dy:]=r[:H+dy]
        viol=((p==o)!=(q==OPP[o]))                      # instance at cell c, offset o
        # rows where c+o is off-strip: ignore (hard factor not instantiated)
        valid=np.zeros_like(viol); valid[:H-dy]=True
        viol&=valid; inst[oi]+=valid.sum(); badi[oi]+=viol.sum()
        # attribute the violated instance to the orientations of both cells involved: p[c] orientation and q's
        for a in range(6):
            bad[a]+=(viol&(p==a)).sum()
    for a in range(6): cnt[a]+=(p==a).sum()
print("cells by dir",cnt/cnt.sum()); print("violated-instance heads (p[c]==a) per cell of dir a",(bad/cnt).round(4))
print("viol rate by instance offset E,S,SE",(badi/inst).round(4))
