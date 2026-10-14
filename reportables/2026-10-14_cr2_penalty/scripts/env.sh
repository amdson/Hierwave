# This box (2026-10-14 evening session): Vast instance, 2x RTX A4000 16 GB, 24 cores, pids.max 1792.
# usage: source env.sh [gpu index]   (sets CUDA_VISIBLE_DEVICES when given)
cd /workspace/Hierwave && source /venv/main/bin/activate
export PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" JAX_COMPILATION_CACHE_DIR=/workspace/jax_cache
[ -n "$1" ] && export CUDA_VISIBLE_DEVICES=$1
