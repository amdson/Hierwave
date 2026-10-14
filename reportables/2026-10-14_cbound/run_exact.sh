#!/bin/bash
# cbound task 1: exact-gradient schedules on dimers 4x4 (cdim2 region), lam 20, 600 schedule steps + 400 post.
# usage: run_exact.sh <cores> <tag:cbound-spec> ...
cd /workspace/hierwave_git && source /venv/main/bin/activate
export PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" JAX_COMPILATION_CACHE_DIR=/workspace/jax_cache JAX_PLATFORMS=cpu
cores=$1; shift
for job in "$@"; do
  tag=${job%%:*}; spec=${job#*:}
  echo "=== $tag $spec"
  taskset -c $cores timeout 900 python -m castlegen.variants._exactopt dimers4 --runs exact_mlp --steps 600 --every 20 --lam 20 \
     --out reportables/2026-10-14_cbound/runs --tag $tag --cbound "$spec,post=400"
done
