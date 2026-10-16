#!/bin/bash
# Warm 300 s bench (tag fitpen) of one variant on GPU 0, cores 16-23.  usage: bench.sh <variant> <seeds>
V=$1; N=${2:-2}
source /workspace/Hierwave/reportables/2026-10-14_cr2_penalty/scripts/env.sh 0
OUT=reportables/2026-10-16_fitpen/runs
nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,clocks_throttle_reasons.active --format=csv > $OUT/smi_before_bench_$V.txt
taskset -c 16-23 timeout 2400 python reportables/2026-10-16_fitpen/scripts/bench_warm.py bench $V 300 $N > $OUT/bench_$V.log 2>&1
echo "EXIT $?" >> $OUT/bench_$V.log
nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,clocks_throttle_reasons.active --format=csv > $OUT/smi_after_bench_$V.txt
