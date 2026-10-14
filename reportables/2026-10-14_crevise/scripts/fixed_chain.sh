#!/bin/bash
# after the all-action fix: CPU smoke, then guided dimers 300 s on GPU 0 (cores 16-23).
D=reportables/2026-10-14_crevise/runs
source /workspace/Hierwave/reportables/2026-10-14_crevise/scripts/env_cpu.sh
taskset -c 0-7 timeout 900 python reportables/2026-10-14_crevise/scripts/smoke_fix.py > $D/smoke_fix.log 2>&1 || { echo SMOKE_FAILED >> $D/smoke_fix.log; exit 1; }
source /workspace/Hierwave/reportables/2026-10-14_crevise/scripts/env_gpu.sh
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $D/gpu_before_fixed.txt
taskset -c 16-23 timeout 400 python reportables/2026-10-14_crevise/scripts/dimers_run.py crevise 30 1000 2 $D/dimers_fixed_warm.json > $D/dimers_fixed_warm.log 2>&1
taskset -c 16-23 timeout 900 python reportables/2026-10-14_crevise/scripts/dimers_run.py crevise 300 25 4 $D/dimers_crevise_guided_fixed.json > $D/dimers_crevise_guided_fixed.log 2>&1
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $D/gpu_after_fixed.txt
echo CHAIN_DONE >> $D/dimers_crevise_guided_fixed.log
