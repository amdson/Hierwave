#!/bin/bash
# warm 300 s x 2 bench of cfitpen_ml2_stage, tag combo, GPU 1, cores 8-15
cd /workspace/Hierwave && source reportables/2026-10-14_cr2_penalty/scripts/env.sh 1
D=reportables/2026-10-16_combo; V=cfitpen_ml2_stage
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $D/runs/nvidia_before_bench.txt
taskset -c 8-15 timeout 2400 python $D/scripts/bench_warm.py bench $V 300 2
echo "EXIT $?"
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $D/runs/nvidia_after_bench.txt
echo ALL_DONE
