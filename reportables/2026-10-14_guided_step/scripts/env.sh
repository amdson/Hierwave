# usage: source env.sh [gpu index]   (CPU: JAX_PLATFORMS=cpu is added when no index is given)
cd /workspace/Hierwave && source /venv/main/bin/activate
export PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" JAX_COMPILATION_CACHE_DIR=/workspace/jax_cache
if [ -n "$1" ]; then export CUDA_VISIBLE_DEVICES=$1; else export JAX_PLATFORMS=cpu; fi
