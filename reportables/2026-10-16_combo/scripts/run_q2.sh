#!/bin/bash
# offline stage on a saved cfitpen_ml2 pass: usage run_q2.sh <label> <params.pkl>
cd /workspace/Hierwave && source reportables/2026-10-14_cr2_penalty/scripts/env.sh 1
D=reportables/2026-10-16_combo
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $D/runs/nvidia_before_$1.txt
taskset -c 8-15 timeout 1500 python reportables/2026-10-16_stage2/scripts/q2.py $1 cfitpen_ml2 $2 128 $D/runs/$1.json v1b:5:30:150:0,v1b:5:100:150:0,v1b:5:240:150:0
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $D/runs/nvidia_after_$1.txt
echo ALL_DONE
