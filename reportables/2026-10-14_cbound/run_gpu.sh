#!/bin/bash
# sequential GPU jobs: run_gpu.sh "<cmd args>" ...  (each a python invocation from the repo root)
cd /workspace/hierwave_git && source /venv/main/bin/activate
export PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" JAX_COMPILATION_CACHE_DIR=/workspace/jax_cache
for job in "$@"; do echo "=== $job"; taskset -c 20-23 timeout 400 python $job; done
