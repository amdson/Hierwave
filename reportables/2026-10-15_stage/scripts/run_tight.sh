#!/bin/bash
cd /workspace/Hierwave && source reportables/2026-10-14_cr2_penalty/scripts/env.sh 1
R=reportables/2026-10-15_stage
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $R/runs/nvidia_before_tight.txt
taskset -c 8-15 timeout 900 python $R/scripts/stage_q1.py hard_pen_tight ckpop_hard_pen reportables/2026-10-14_cr2_penalty/runs/arm_ckpop_hard_pen_s0/params.pkl 128 $R/runs/hard_pen_tight.json v1bt:5:30:20,v1b:5:30:20,v1bt:5:30:150
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $R/runs/nvidia_after_tight.txt
echo ALL_DONE
