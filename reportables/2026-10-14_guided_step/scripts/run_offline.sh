#!/bin/bash
# usage: run_offline.sh <cores> <base> <params> <fmt> <k> <n> <R> <out.json> [gpu]
source /workspace/Hierwave/reportables/2026-10-14_guided_step/scripts/env.sh $9
taskset -c $1 timeout 3000 python /workspace/Hierwave/reportables/2026-10-14_guided_step/scripts/offline_eval.py $2 $3 $4 $5 $6 $7 $8 2>&1 | grep -v "^WARNING"
