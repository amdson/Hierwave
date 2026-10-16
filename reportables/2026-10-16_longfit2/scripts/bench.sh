#!/bin/bash
cd /workspace/Hierwave
D=reportables/2026-10-16_longfit2; R=$D/runs/bench
while [ $(ps -eo args | grep -E "^python .*reportables" | grep -vc grep) -gt 2 ]; do sleep 300; done
m0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0); m1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
if [ $m0 -le $m1 ]; then G=0; C=16-23; else G=1; C=8-15; fi
echo "GPU $G cores $C (mem $m0 $m1)" > $R/gpu.txt
source reportables/2026-10-14_cr2_penalty/scripts/env.sh $G
for v in clongfit_stage cfitpen_ml2_stage; do
  nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $R/nvidia_before_$v.txt
  taskset -c $C python reportables/2026-10-14_cr2_penalty/scripts/bench_warm.py bench $v 600 2 > $R/$v.log 2>&1
  nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $R/nvidia_after_$v.txt
done
echo DONE > $R/DONE
