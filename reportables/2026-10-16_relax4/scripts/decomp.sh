#!/bin/bash
cd /workspace/Hierwave && source reportables/2026-10-14_cr2_penalty/scripts/env.sh 0
R=reportables/2026-10-16_relax4/runs; L=reportables/2026-10-16_longfit2/runs/big300
M=""; for s in 0 1; do M="$M s$s=clongfit:$L/params_s$s.pkl"; done; for s in 2 3; do M="$M s$s=clongfit:$R/train/params_s$s.pkl"; done
taskset -c 16-23 python reportables/2026-10-16_relax4/scripts/decomp.py $R/decomp.json $M > $R/decomp.log 2>&1
echo DONE > $R/DECOMP_DONE
