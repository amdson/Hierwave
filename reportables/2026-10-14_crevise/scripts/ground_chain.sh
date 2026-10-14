#!/bin/bash
# ground patch (8 x 16, T = 2): wait for the dimers chain (GPU 0) and the MCMC reference, then crevise 300 s and
# cfast 180 s back to back on GPU 0, cores 16-23.
source /workspace/Hierwave/reportables/2026-10-14_crevise/scripts/env_gpu.sh
D=reportables/2026-10-14_crevise/runs
while ! grep -q CHAIN_DONE $D/dimers_cfast.log 2>/dev/null; do sleep 15; done
while [ ! -f $D/ground_patch_mcmc.json ]; do sleep 15; done
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $D/gpu_before_ground.txt
taskset -c 16-23 timeout 900 python reportables/2026-10-14_crevise/scripts/ground_patch_train.py crevise 300 256 $D/ground_patch_crevise.json > $D/ground_patch_crevise.log 2>&1
taskset -c 16-23 timeout 600 python reportables/2026-10-14_crevise/scripts/ground_patch_train.py cfast 180 256 $D/ground_patch_cfast.json > $D/ground_patch_cfast.log 2>&1
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $D/gpu_after_ground.txt
echo CHAIN_DONE >> $D/ground_patch_cfast.log
