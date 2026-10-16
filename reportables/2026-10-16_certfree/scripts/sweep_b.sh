#!/bin/bash
# wave B driver: 2 seeds x (train 600 s cold, stage tables R 0/15, block check on ground_cf). usage: sweep_b.sh <gpu>
cd /workspace/Hierwave; R=reportables/2026-10-16_certfree/runs; S=reportables/2026-10-16_certfree/scripts
source reportables/2026-10-14_cr2_penalty/scripts/env.sh $1
for s in 0 1; do
  D=$R/b600_s$s; mkdir -p $D
  nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active,memory.used --format=csv > $D/nv_before.txt
  CF_SEED=$s taskset -c 0-3 python $S/train.py 600 $D > $D/train.log 2>&1
  taskset -c 0-3 python $S/tables.py $D/params.pkl 128 $D/tables.json 0,15 20 > $D/tables.log 2>&1
  taskset -c 0-3 python $S/blockcf.py $D/params.pkl $s $D/blockcf.json > $D/blockcf.log 2>&1
  nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active,memory.used --format=csv > $D/nv_after.txt
done
echo DONE
