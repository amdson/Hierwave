#!/bin/bash
# usage: run.sh <cores> <secs> <timeout> <name> <train_s> <every> <NS> <seed> <tag>
cd /workspace/hierwave_git && source /venv/main/bin/activate
export PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" JAX_COMPILATION_CACHE_DIR=/workspace/jax_cache
taskset -c $1 timeout $2 python reportables/2026-10-14_cbound2/dimers2.py $3 $4 $5 $6 $7 $8
