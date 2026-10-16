#!/bin/bash
# fitpen fast loop: 120 s instrumented arm + bench-style eval (cr2's arm_eval.py, unchanged), seed 0, back to back on GPU 0.
source /workspace/Hierwave/reportables/2026-10-14_cr2_penalty/scripts/env.sh 0
R=reportables/2026-10-16_fitpen/runs
for V in "$@"; do
  nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $R/smi_before_$V.txt
  taskset -c 16-23 timeout 900 python reportables/2026-10-14_cr2_penalty/scripts/arm_eval.py $V 120 0 $R/arm_$V > $R/arm_$V.log 2>&1
  echo "EXIT $?" >> $R/arm_$V.log
  nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $R/smi_after_$V.txt
done
