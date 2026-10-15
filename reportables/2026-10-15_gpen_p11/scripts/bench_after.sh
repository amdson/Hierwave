#!/bin/bash
# wait for the Q2 fast chain, then a warm 2-seed 300 s bench (bench_warm.py; harness untouched) of $1 on GPU 0
V=$1; R=reportables/2026-10-15_gpen_p11/runs
cd /workspace/Hierwave
until grep -q CHAIN_DONE $R/chain_q2.nohup; do sleep 10; done
source reportables/2026-10-14_cr2_penalty/scripts/env.sh 0
nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,utilization.gpu,clocks_throttle_reasons.active --format=csv > $R/nv_before_bench_$V.txt
taskset -c 16-23 timeout 2400 python reportables/2026-10-14_cr2_penalty/scripts/bench_warm.py bench $V 300 2 > $R/bench_$V.log 2>&1
echo "EXIT $?" >> $R/bench_$V.log
nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,utilization.gpu,clocks_throttle_reasons.active --format=csv > $R/nv_after_bench_$V.txt
echo BENCH_DONE >> $R/bench_$V.log
