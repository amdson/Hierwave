#!/bin/bash
cd /workspace/Hierwave && source reportables/2026-10-14_cr2_penalty/scripts/env.sh 1
D=reportables/2026-10-16_stage2
V=${1:-ckpop_hard_pen_stage}
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $D/runs/nvidia_before_bench_$V.txt
taskset -c 8-15 python -m castlegen.bench run $V ground --train 300 --seeds 2 --tag stage2
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $D/runs/nvidia_after_bench_$V.txt
echo ALL_DONE
