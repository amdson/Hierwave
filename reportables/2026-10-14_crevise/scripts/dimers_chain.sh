#!/bin/bash
# dimers: warm compile (30 s) then 300 s crevise, then 180 s cfast control, back to back on GPU 0, cores 16-23.
source /workspace/Hierwave/reportables/2026-10-14_crevise/scripts/env_gpu.sh
D=reportables/2026-10-14_crevise/runs
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $D/gpu_before_dimers.txt
taskset -c 16-23 timeout 400 python reportables/2026-10-14_crevise/scripts/dimers_run.py crevise 30 1000 2 $D/dimers_warm.json > $D/dimers_warm.log 2>&1
taskset -c 16-23 timeout 900 python reportables/2026-10-14_crevise/scripts/dimers_run.py crevise 300 25 4 $D/dimers_crevise.json > $D/dimers_crevise.log 2>&1
taskset -c 16-23 timeout 600 python reportables/2026-10-14_crevise/scripts/dimers_run.py cfast 180 25 4 $D/dimers_cfast.json > $D/dimers_cfast.log 2>&1
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $D/gpu_after_dimers.txt
echo CHAIN_DONE >> $D/dimers_cfast.log
