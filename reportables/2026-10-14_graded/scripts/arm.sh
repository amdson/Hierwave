#!/bin/bash
# usage: arm.sh <variant> <train_s> <seed>   (GPU 1, cores 8-15; mandatory flags)
cd /workspace/Hierwave && source /venv/main/bin/activate
export PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" JAX_COMPILATION_CACHE_DIR=/workspace/jax_cache CUDA_VISIBLE_DEVICES=1
R=reportables/2026-10-14_graded
nvidia-smi --query-gpu=clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv -i 1 > $R/runs/nvidia_before_arm_$1_s$3.txt
taskset -c 8-15 timeout 1500 python $R/scripts/arm_eval.py $1 $2 $3 $R/runs/arm_$1_s$3
nvidia-smi --query-gpu=clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv -i 1 > $R/runs/nvidia_after_arm_$1_s$3.txt
echo ARM_DONE
