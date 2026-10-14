#!/bin/bash
# Warm 4-seed 300 s bench of a ckpop variant through bench_warm.py (harness untouched).  usage: bench_pop.sh <variant> <gpu> <cores> [seeds]
V=$1; GPU=$2; CORES=$3; N=${4:-4}
source "$(dirname "$0")/env.sh" $GPU
OUT=reportables/2026-10-14_cr2_penalty/runs
nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,clocks_throttle_reasons.active --format=csv > $OUT/nvidia_smi_before_bench_$V.txt
taskset -c $CORES timeout 4000 python reportables/2026-10-14_cr2_penalty/scripts/bench_warm.py bench $V 300 $N > $OUT/bench_$V.log 2>&1
echo "EXIT $?" >> $OUT/bench_$V.log
nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,clocks_throttle_reasons.active --format=csv > $OUT/nvidia_smi_after_bench_$V.txt
