#!/bin/bash
# guided crevise (default cfg: R = 2, rho = 0.5, guide = True): wait for the ground chain on GPU 0, then dimers 300 s
# and the ground patch 300 s, back to back on GPU 0, cores 16-23.
source /workspace/Hierwave/reportables/2026-10-14_crevise/scripts/env_gpu.sh
D=reportables/2026-10-14_crevise/runs
while ! grep -q CHAIN_DONE $D/ground_patch_cfast.log 2>/dev/null; do sleep 15; done
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $D/gpu_before_guided.txt
taskset -c 16-23 timeout 400 python reportables/2026-10-14_crevise/scripts/dimers_run.py crevise 30 1000 2 $D/dimers_guided_warm.json > $D/dimers_guided_warm.log 2>&1
taskset -c 16-23 timeout 900 python reportables/2026-10-14_crevise/scripts/dimers_run.py crevise 300 25 4 $D/dimers_crevise_guided.json > $D/dimers_crevise_guided.log 2>&1
taskset -c 16-23 timeout 900 python reportables/2026-10-14_crevise/scripts/ground_patch_train.py crevise 300 256 $D/ground_patch_crevise_guided.json > $D/ground_patch_crevise_guided.log 2>&1
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $D/gpu_after_guided.txt
echo CHAIN_DONE >> $D/ground_patch_crevise_guided.log
