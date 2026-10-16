#!/bin/bash
cd /workspace/Hierwave && source reportables/2026-10-14_cr2_penalty/scripts/env.sh 1
D=reportables/2026-10-16_stage2
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $D/runs/nvidia_before_q2.txt
taskset -c 8-15 timeout 1800 python $D/scripts/q2.py q2 ckpop_hard_pen reportables/2026-10-14_cr2_penalty/runs/arm_ckpop_hard_pen_s0/params.pkl 128 $D/runs/q2.json v1b:5:30:150:0,v1b:5:30:150:1,v1b:5:30:150:2,v1b:5:100:150:0,v1b:5:100:150:1,v1b:5:100:150:2,v1b:5:240:150:0,v1b:5:240:150:1,v1b:5:240:150:2
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $D/runs/nvidia_after_q2.txt
echo ALL_DONE
