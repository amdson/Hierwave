#!/bin/bash
# Warm smoke (40 s, run_ck2.py through the bench's import path) then a 4-seed 300 s bench of a ckernel variant.
# usage: bench_ck.sh <variant> <gpu> <cores> <outdir>
V=$1; GPU=$2; CORES=$3; OUT=$4
source "$(dirname "$0")/env.sh" $GPU
mkdir -p $OUT
nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,clocks_throttle_reasons.active --format=csv > $OUT/nvidia_smi_before_$V.txt
taskset -c $CORES timeout 900 python reportables/2026-10-14_cr2_penalty/scripts/run_ck2.py $V 40 $OUT/smoke_$V > $OUT/smoke_$V.log 2>&1
taskset -c $CORES timeout 3600 python -m castlegen.bench run $V ground --seeds 4 --train 300 --tag cr2 > $OUT/bench_$V.log 2>&1
nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,clocks_throttle_reasons.active --format=csv > $OUT/nvidia_smi_after_$V.txt
echo BENCH_DONE $V >> $OUT/bench_$V.log
