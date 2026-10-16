#!/bin/bash
# usage: run_arm.sh <arm> <seed> <budget_s> <max_iters|0>
cd /workspace/Hierwave && source reportables/2026-10-14_cr2_penalty/scripts/env.sh 0 >/dev/null
arm=$1; s=$2; bud=$3; mi=$4; V=l3cap_$arm
R=reportables/2026-10-16_l3cap/runs/$arm; mkdir -p $R; S=reportables/2026-10-16_l3cap/scripts
waitslot() { while [ "$(ps -eo args | grep -c '[p]ython .*reportables')" -ge 3 ] || [ "$(cat /sys/fs/cgroup/pids.current)" -gt 1400 ]; do sleep 60; done; }
nv() { nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active,memory.used --format=csv > $R/nvidia_$1_s$s.txt; }
M="$arm=$V:$R/params_s$s.pkl"
waitslot; nv before
taskset -c 16-23 python $S/train_arm.py $V $bud $mi $s $R > $R/train_s$s.log 2>&1; nv after
waitslot; taskset -c 16-23 python reportables/2026-10-16_stage2/scripts/q2.py ${arm}_s$s $V $R/params_s$s.pkl 128 $R/q2_s$s.json v1b:5:15:150:0 > $R/q2_s$s.log 2>&1
waitslot; taskset -c 16-23 python reportables/2026-10-16_l4source/scripts/relaxcheck.py $R/relaxcheck_s$s.json $M > $R/relaxcheck_s$s.log 2>&1
waitslot; taskset -c 16-23 python reportables/2026-10-16_relax4/scripts/classes.py $R/classes_s$s.json $M > $R/classes_s$s.log 2>&1
waitslot; taskset -c 16-23 python reportables/2026-10-16_relax4/scripts/decomp.py $R/decomp_s$s.json $M > $R/decomp_s$s.log 2>&1
echo DONE > $R/DONE_s$s
