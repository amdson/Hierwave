#!/bin/bash
cd /workspace/Hierwave && source reportables/2026-10-14_cr2_penalty/scripts/env.sh 1
R=reportables/2026-10-15_stage
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $R/runs/nvidia_before_q1.txt
taskset -c 8-15 timeout 1500 python $R/scripts/stage_q1.py hard_pen_q1 ckpop_hard_pen reportables/2026-10-14_cr2_penalty/runs/arm_ckpop_hard_pen_s0/params.pkl 128 $R/runs/hard_pen_q1.json v1b:5:15:150,v1b:5:30:150,v1b:5:60:150,v1b:5:120:150,v1b:5:240:150
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $R/runs/nvidia_after_q1.txt
echo ALL_DONE
