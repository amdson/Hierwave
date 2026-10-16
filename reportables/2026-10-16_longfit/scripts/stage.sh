#!/bin/bash
# offline stage on the final clongfit params: R 15, 30, 60; 128 worlds (same q2.py as combo)
cd /workspace/Hierwave && source reportables/2026-10-14_cr2_penalty/scripts/env.sh 0
D=reportables/2026-10-16_longfit
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $D/runs/nvidia_before_stage.txt
taskset -c 6-7 python reportables/2026-10-16_stage2/scripts/q2.py longfit cfitpen_ml2 $D/runs/clongfit/params.pkl 128 $D/runs/q2_longfit.json v1b:5:15:150:0,v1b:5:30:150:0,v1b:5:60:150:0
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $D/runs/nvidia_after_stage.txt
echo ALL_DONE
