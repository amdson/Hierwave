"""Exact edge-orientation fractions of uniform perfect matchings of the bench triangular strip (H rows, open top/bottom,
open x window of width W) via Kasteleyn/Pfaffian: P(edge uv)=|A_uv (A^-1)_vu|.  Fractions reported over the central columns."""
import itertools, sys, numpy as np
DIR = {'E':(0,1),'S':(1,0),'SE':(1,1)}
def find_orient():
    # ansatz: orientation bit of edge type t at anchor (y%2,x%2); tri faces need odd # clockwise edges
    for bits in itertools.product([1,-1], repeat=12):
        def o(t,y,x): return bits[t*4+(y%2)*2+(x%2)]   # +1 : u->v in forward direction
        ok=True
        for y in range(4):
            for x in range(4):
                # upper tri (y,x)->(y,x+1)->(y+1,x+1) is clockwise (y down): E fwd, S at x+1 fwd, SE traversed backward
                c = (o(0,y,x)==1) + (o(1,y,x+1)==1) + (o(2,y,x)==-1)
                # lower tri (y,x)->(y+1,x+1)->(y+1,x)->(y,x) clockwise? with y down: (y,x)->(y,x)... use ccw check below
                # lower tri vertices (y,x),(y+1,x),(y+1,x+1): clockwise order (y,x)->(y+1,x+1)->(y+1,x)->(y,x)
                d = (o(2,y,x)==1) + (o(0,y+1,x)==-1) + (o(1,y,x)==-1)
                if c%2==0 or d%2==0: ok=False;break
            if not ok:break
        if ok: return bits
def run(H,W):
    bits=find_orient(); assert bits
    n=H*W; A=np.zeros((n,n)); idx=lambda y,x:y*W+x
    edges=[]
    for y in range(H):
        for x in range(W):
            for t,(dy,dx) in enumerate(DIR.values()):
                y2,x2=y+dy,x+dx
                if y2<H and x2<W:
                    s=bits[t*4+(y%2)*2+(x%2)]
                    u,v=idx(y,x),idx(y2,x2); A[u,v]=s;A[v,u]=-s; edges.append((t,y,x,u,v))
    Ai=np.linalg.inv(A)
    P=np.zeros((3,H,W))
    for t,y,x,u,v in edges: P[t,y,x]=abs(A[u,v]*Ai[v,u])
    return P
if __name__=="__main__":
    for H,W in [(4,8),(8,32),(32,64),(32,128),(32,192)]:
        P=run(H,W)
        tot=P.sum()
        lo,hi=W//2-16 if W>=64 else 0, W//2+16 if W>=64 else W
        c=P[:,:,lo:hi].sum((1,2)); 
        a=P.sum((1,2))
        print(H,W,"sum P=",round(tot,3),"expected",H*W/2,"| all cols frac",np.round(a/tot,4),"| central",np.round(c/c.sum(),4), flush=True)
        if H==32 and W==128:
            rows=P[:,:,48:80].sum(2); print("per-row frac h,v,d rows 0,1,2,3,8,16:",[np.round(rows[:,r]/rows[:,r].sum(),3).tolist() for r in (0,1,2,3,8,16)])
