import numpy as np, sys
sys.setrecursionlimit(10000)
def run(H,W):
    D=[(0,1,0),(1,0,1),(1,1,2)]; cnt=np.zeros(3); N=0
    def nb(y,x):
        for dy,dx,t in D:
            if y+dy<H: yield (y+dy,(x+dx)%W,t)
    # also backward neighbours handled by used[]
    used=np.zeros((H,W),bool)
    def rec(i,c):
        nonlocal N,cnt
        while i<H*W and used[i//W,i%W]: i+=1
        if i==H*W: N+=1; cnt+=c; return
        y,x=divmod(i,W); used[y,x]=True
        for y2,x2,t in nb(y,x):
            if not used[y2,x2]:
                used[y2,x2]=True; c2=c.copy(); c2[t]+=1; rec(i+1,c2); used[y2,x2]=False
        used[y,x]=False
    rec(0,np.zeros(3)); return N,cnt/cnt.sum()
for H,W in [(4,8),(4,6),(4,4),(6,6)]: print(H,W,*run(H,W))
