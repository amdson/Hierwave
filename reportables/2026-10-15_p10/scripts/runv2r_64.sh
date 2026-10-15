#!/bin/bash
# block-kernel arms (p5's FineBlockKernel + JaxBlockGibbs level 1 interleaved), 32 worlds per model, GPU 1
cd /workspace/Hierwave && source reportables/2026-10-14_cr2_penalty/scripts/env.sh 1
R=reportables/2026-10-15_p10
A="v2:5:60:80"
taskset -c 8-15 timeout 900 python $R/scripts/p10.py hard_pen_v2r_64 ckpop_hard_pen reportables/2026-10-14_cr2_penalty/runs/arm_ckpop_hard_pen_s0/params.pkl 64 $R/runs/hard_pen_v2r_64.json $A > $R/runs/hard_pen_v2r_64.log 2>&1
taskset -c 8-15 timeout 900 python $R/scripts/p10.py pen16_v2r_64 ckpop_pen16 reportables/2026-10-14_cr2_penalty/runs/arm_ckpop_pen16_s0/params.pkl 64 $R/runs/pen16_v2r_64.json $A > $R/runs/pen16_v2r_64.log 2>&1
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $R/runs/nvidia_after_v2r_64.txt
echo ALL_DONE
