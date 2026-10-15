#!/bin/bash
# usage: fast.sh <runname> <train_s> <seed> <variant>...   sequential arms on GPU 0, cores 16-23
N=$1; TS=$2; K=$3; shift 3
source reportables/2026-10-14_cr2_penalty/scripts/env.sh 0
R=reportables/2026-10-15_gpen_p11/runs
for V in "$@"; do
  nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,utilization.gpu,clocks_throttle_reasons.active --format=csv > $R/nv_before_${N}_${V}.txt
  taskset -c 16-23 timeout 1500 python reportables/2026-10-15_gpen_p11/scripts/arm_eval_cert.py $V $TS $K $R/${N}_${V}_s$K > $R/${N}_${V}_s$K.log 2>&1
  echo "EXIT $?" >> $R/${N}_${V}_s$K.log
  nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,utilization.gpu,clocks_throttle_reasons.active --format=csv > $R/nv_after_${N}_${V}.txt
done
echo CHAIN_DONE
