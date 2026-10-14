#!/bin/bash
# usage: run_toys.sh <listfile>; each line = toy_chain.py args
cd /workspace/hierwave_git && source /venv/main/bin/activate
export PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" JAX_COMPILATION_CACHE_DIR=/workspace/jax_cache
while read -r line; do
  [ -z "$line" ] && continue
  taskset -c 48-55 timeout 900 python reportables/2026-10-14_kernel_b/scripts/toy_chain.py $line 2>&1 | grep -v Warn | grep '^{'
done < "$1"
