#!/bin/bash
# Instrumented 300 s arm + bench-style eval of one seed (arm_eval.py), own log.  usage: arm.sh <variant> <seed> <gpu> <cores> [train_s]
V=$1; K=$2; GPU=$3; CORES=$4; TS=${5:-300}
source "$(dirname "$0")/env.sh" $GPU
OUT=reportables/2026-10-14_cr2_penalty/runs/arm_${V}_s${K}
mkdir -p $OUT
taskset -c $CORES timeout 1800 python reportables/2026-10-14_cr2_penalty/scripts/arm_eval.py $V $TS $K $OUT > $OUT.log 2>&1
echo "EXIT $?" >> $OUT.log
