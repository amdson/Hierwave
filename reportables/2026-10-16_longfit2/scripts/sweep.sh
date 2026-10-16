#!/bin/bash
# usage: sweep.sh <gpu> <cores>.  Arms in order a,b,c; seeds 0,1 each.
G=$1; C=$2
cd /workspace/Hierwave && source reportables/2026-10-14_cr2_penalty/scripts/env.sh $G
D=reportables/2026-10-16_longfit2; R=$D/runs
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active,memory.used --format=csv > $R/nvidia_before.txt
# warm compile for both shapes (not counted)
taskset -c $C python $D/scripts/train.py clongfit 45 99 /tmp/lf2_warm_big > $R/warm_big.log 2>&1
taskset -c $C python $D/scripts/train.py clongfit_small 45 99 /tmp/lf2_warm_small > $R/warm_small.log 2>&1
run() { # arm variant budget
  arm=$1; var=$2; bud=$3
  for s in 0 1; do
    mkdir -p $R/$arm
    taskset -c $C python $D/scripts/train.py $var $bud $s $R/$arm > $R/$arm/train_s$s.log 2>&1
    nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $R/$arm/nvidia_after_s$s.txt
    taskset -c $C python reportables/2026-10-16_stage2/scripts/q2.py ${arm}_s$s cfitpen_ml2 $R/$arm/params_s$s.pkl 128 $R/$arm/q2_s$s.json v1b:5:15:150:0 > $R/$arm/q2_s$s.log 2>&1
    taskset -c $C python reportables/2026-10-16_longfit/scripts/blocks.py ${arm}_s$s=$R/$arm/params_s$s.pkl > $R/$arm/blocks_s$s.txt 2>$R/$arm/blocks_s$s.err
  done
}
run big300 clongfit 300
run big600 clongfit 600
run small600 clongfit_small 600
echo DONE > $R/DONE
