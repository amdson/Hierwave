#!/bin/bash
# GPU 1 chain after the arm: dimers 120 s with cgpen (gpen=1) then, if requested, the warm 4-seed bench.  usage: gpu1_chain.sh [bench]
cd /workspace/Hierwave && source /venv/main/bin/activate
export PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" JAX_COMPILATION_CACHE_DIR=/workspace/jax_cache CUDA_VISIBLE_DEVICES=1
R=reportables/2026-10-14_graded
taskset -c 8-15 timeout 600 python $R/scripts/dk_loop_gpen.py reportables/2026-10-14_dimers_kernel/runs/cfast60.pkl 120 $R/runs/dimers_gpen32.jsonl rkl_n=32 gpen=1 > $R/runs/dimers_gpen32.log 2>&1
echo DIMERS_DONE
if [ "$1" = "bench" ]; then
  nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,clocks_throttle_reasons.active --format=csv -i 1 > $R/runs/nvidia_before_bench.txt
  taskset -c 8-15 timeout 4000 python $R/scripts/bench_warm.py bench ckpop_hard_gpen 300 4 > $R/runs/bench_ckpop_hard_gpen.log 2>&1
  echo "EXIT $?" >> $R/runs/bench_ckpop_hard_gpen.log
  nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,clocks_throttle_reasons.active --format=csv -i 1 > $R/runs/nvidia_after_bench.txt
fi
echo CHAIN_DONE
