#!/bin/bash
# Run one exactopt job on the GPU box's CPU (the box's GPU client hit the shared
# pid limit: ptxas could not be launched), with few threads.  Usage on the box:
#   CORES=0-3 bash reportables/2026-10-13_exactopt/box.sh <problem> <log> [args...]
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
P=$1; LOG=$2; shift 2
nohup timeout 3000 taskset -c ${CORES:-0-3} python -m castlegen.variants._exactopt $P "$@" > $LOG 2>&1 &
