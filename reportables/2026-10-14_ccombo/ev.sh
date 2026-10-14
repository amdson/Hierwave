#!/bin/bash
# usage: ev.sh <rundir> <cores>   (clamp gate, cstart2's window-aware eval_ground.py)
cd /workspace/hierwave_git && source /venv/main/bin/activate
export PYTHONPATH=.:reportables/2026-10-14_cstart2 XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 \
       OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
       JAX_COMPILATION_CACHE_DIR=/workspace/jax_cache
taskset -c $2 timeout 900 python reportables/2026-10-14_cstart2/eval_ground.py reportables/2026-10-14_ccombo/runs/$1 32
