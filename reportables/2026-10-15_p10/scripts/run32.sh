#!/bin/bash
# 32 held-out worlds per saved model, the decisive arms (GPU 1, cores 8-15), sequential
cd /workspace/Hierwave && source reportables/2026-10-14_cr2_penalty/scripts/env.sh 1
R=reportables/2026-10-15_p10
A="v1:4:30:80,v1:4:60:80,v1:4:100:40,v1:4:100:80,v1:3:100:80"
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $R/runs/nvidia_before_run32.txt
taskset -c 8-15 timeout 900 python $R/scripts/p10.py pen16_32 ckpop_pen16 reportables/2026-10-14_cr2_penalty/runs/arm_ckpop_pen16_s0/params.pkl 32 $R/runs/pen16_32.json $A > $R/runs/pen16_32.log 2>&1
taskset -c 8-15 timeout 900 python $R/scripts/p10.py hard_pen_32 ckpop_hard_pen reportables/2026-10-14_cr2_penalty/runs/arm_ckpop_hard_pen_s0/params.pkl 32 $R/runs/hard_pen_32.json $A > $R/runs/hard_pen_32.log 2>&1
taskset -c 8-15 timeout 600 python $R/scripts/p10.py greedy_16 greedy - 16 $R/runs/greedy_16.json "v1:4:100:80,v1:4:300:80" > $R/runs/greedy_16.log 2>&1
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $R/runs/nvidia_after_run32.txt
echo ALL_DONE
