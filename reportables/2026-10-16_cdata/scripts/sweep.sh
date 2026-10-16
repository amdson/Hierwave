#!/bin/bash
# cdata sweep: 6 arms back to back, 20 min training each from arm_cfitpen_ml2/params.pkl, GPU 1 cores 8-15 (GPU shared).
cd /workspace/Hierwave && source reportables/2026-10-14_cr2_penalty/scripts/env.sh 1
D=reportables/2026-10-16_cdata; TS=${TS:-1200}
declare -A CFG=(
 [ctrl]='{}'
 [a_pop2]='{"pop_mult":2}'
 [b_pop4]='{"pop_mult":4}'
 [c_w4]='{"w3":4.0}'
 [d_w4_pop2]='{"w3":4.0,"pop_mult":2}'
 [e_extra16]='{"extra_coarse":16}'
)
for A in ${ARMS:-ctrl a_pop2 b_pop4 c_w4 d_w4_pop2 e_extra16}; do
  while [ "$(cat /sys/fs/cgroup/pids.current)" -gt 1400 ]; do sleep 20; done
  nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active,memory.used --format=csv > $D/runs/nvidia_before_$A.txt
  taskset -c 8-15 python $D/scripts/arm.py $A "${CFG[$A]}" $TS $D/runs/arm_$A > $D/runs/arm_$A.log 2>&1
  echo "EXIT $?" >> $D/runs/arm_$A.log
  nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active,memory.used --format=csv > $D/runs/nvidia_after_$A.txt
  taskset -c 8-15 python reportables/2026-10-16_stage2/scripts/q2.py $A ccdata $D/runs/arm_$A/params.pkl 128 $D/runs/q2_$A.json v1b:5:30:150:0 > $D/runs/q2_$A.log 2>&1
done
python reportables/2026-10-16_combo/scripts/table.py $D/runs/q2_*.json > $D/runs/stage_table.md 2>&1
echo DONE
