#!/bin/bash
# After the GPU-1 chain: two cheaper-pressure own-draws arms (300 s, seed 0, bench-style eval) on GPU 1.
R=/workspace/Hierwave/reportables/2026-10-14_cr2_penalty
until grep -q CHAIN_DONE $R/runs/gpu1_chain.nohup 2>/dev/null; do sleep 30; done
$R/scripts/arm.sh ckpop_pen16 0 1 8-15
$R/scripts/arm.sh ckpop_pen_lr5 0 1 8-15
echo FOLLOWUP_DONE
