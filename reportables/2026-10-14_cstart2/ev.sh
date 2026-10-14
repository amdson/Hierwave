#!/bin/bash
# usage: ev.sh <script> <cores> args...
cd /workspace/hierwave_git && source /venv/main/bin/activate
export PYTHONPATH=.:reportables/2026-10-14_cstart2 XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 \
       OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
       JAX_COMPILATION_CACHE_DIR=/workspace/jax_cache
S=$1; C=$2; shift 2
taskset -c $C timeout 1200 python reportables/2026-10-14_cstart2/$S "$@"
