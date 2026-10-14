#!/bin/bash
# one CPU process per saved model (each well under 10 min), sequential; mandatory flags (box rules)
cd /workspace/Hierwave && source /venv/main/bin/activate
export PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" JAX_COMPILATION_CACHE_DIR=/workspace/jax_cache JAX_PLATFORMS=cpu
R=reportables/2026-10-14_graded
run() { taskset -c 0-7 timeout 600 python $R/scripts/diag1_gen.py "$1" "$2" "$3" $R/runs/diag1_$1.json 16; }
run cfast300            cfast               reportables/2026-10-14_cfine/runs/cfast300/params_ground.pkl
run cr2_hard_pen_s0     ckpop_hard_pen      reportables/2026-10-14_cr2_penalty/runs/arm_ckpop_hard_pen_s0/params.pkl
run w6_C_rkl123x2       ckpop_rkl123x2      reportables/2026-10-14_wave6_pop/runs/C_rkl123x2/params.pkl
run w6_D_rkl123x2_cg40  ckpop_rkl123x2_cg40 reportables/2026-10-14_wave6_pop/runs/D_rkl123x2_cg40/params.pkl
run w4_W1_hard          ckpop_hard          reportables/2026-10-14_wave4_pop/runs/W1_hard/params_ground.pkl
run w4_W2b_hard_rkl     ckpop_hard_rkl      reportables/2026-10-14_wave4_pop/runs/W2b_hard_rkl_300train/params_ground.pkl
echo ALL_DONE
