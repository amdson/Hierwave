#!/bin/bash
cd /workspace/Hierwave && source reportables/2026-10-14_cr2_penalty/scripts/env.sh 0
R=reportables/2026-10-16_relax4/runs
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active,memory.used --format=csv > $R/nvidia_before.txt
for s in 2 3; do
  taskset -c 16-23 python reportables/2026-10-16_longfit2/scripts/train.py clongfit 300 $s $R/train > $R/train_s$s.log 2>&1
  nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $R/nvidia_after_s$s.txt
done
echo DONE > $R/TRAIN_DONE
