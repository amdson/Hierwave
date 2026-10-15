#!/bin/bash
# kblock: instrumented arm (copy of cr2_penalty/arm.sh, own folder).  usage: arm.sh <variant> <seed> <train_s> <tag>
V=$1; K=$2; TS=$3; TAG=$4
source /workspace/Hierwave/reportables/2026-10-14_cr2_penalty/scripts/env.sh 0
OUT=reportables/2026-10-15_kblock/runs/${TAG}_${V}_s${K}
mkdir -p $OUT
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $OUT/nvsmi_before.txt
taskset -c 16-23 timeout 1500 python reportables/2026-10-15_kblock/scripts/arm_eval.py $V $TS $K $OUT > $OUT.log 2>&1
echo "EXIT $?" >> $OUT.log
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $OUT/nvsmi_after.txt
