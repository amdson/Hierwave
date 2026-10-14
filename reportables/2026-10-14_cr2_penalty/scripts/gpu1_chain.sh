#!/bin/bash
# GPU-1 queue: after the l2x2 bench, rerun l2x15 (wave5 part 2 on a cool GPU), then bench the upper-bound penalty arm.
R=/workspace/Hierwave/reportables/2026-10-14_cr2_penalty
until grep -q BENCH_DONE $R/runs/wave5_rerun/bench_ckernel_rkl4w3f_l2x2.log 2>/dev/null; do sleep 30; done
$R/scripts/bench_ck.sh ckernel_rkl4w3f_l2x15 1 8-15 $R/runs/wave5_rerun
$R/scripts/bench_ck.sh ckernel_pen4w3f 1 8-15 $R/runs/ub_pen
echo CHAIN_DONE
