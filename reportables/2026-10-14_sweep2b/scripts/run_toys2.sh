#!/bin/bash
. /workspace/hierwave_git/reportables/2026-10-14_sweep2b/scripts/env.sh
export JAX_PLATFORMS=cpu
S=reportables/2026-10-14_sweep2b/scripts/toy_flip.py
while read -r args; do
  echo "== $args"; taskset -c 52-55 timeout 600 python $S $args 2>&1 | grep '^{' | python3 -c "import sys,json
for l in sys.stdin:
  d=json.loads(l); print(d['problem'],d['kernel'],d['start'],'K',d['K'],'tc',d['tau_c'],'lam',d['lam'],'acc',round(d['acc'],3),'gibbs',d['gibbs_changed_frac'],'maxz',round(d['max_abs_z'],2),{k:round(v[3],2) for k,v in d['stats'].items()},'t',round(d['t_s']))"
done <<'L'
toy_support untrained flip_uniform 20 1 1 1 exact 10 100
toy_support untrained flip_uniform_dropq 20 1 1 1 exact 10 100
toy_support untrained flip_q 20 4 1 2 exact 10 100
toy_support untrained flip_uniform 20 4 1 1 own 10 100
L
