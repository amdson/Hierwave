#!/bin/bash
# Waits for GPU 1 to be released (coordinator queue done + FOLLOWUP_DONE + 0 % utilisation), then runs the
# offline evaluations one JAX process at a time on cores 8-15.
S=/workspace/Hierwave/reportables/2026-10-14_guided_step
F=/workspace/Hierwave/reportables/2026-10-14_cr2_penalty/runs/gpu1_followup.nohup
while true; do
  q=$(pgrep -f "bench_ck.sh|arm.sh|bench_pop.sh" | wc -l)
  d=$(grep -c FOLLOWUP_DONE $F 2>/dev/null)
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 1 | tr -d ' ')
  echo "$(date -u +%H:%M:%S) queue=$q done=$d util=$u"
  if [ "$q" = "0" ] && [ "$d" != "0" ] && [ "$u" = "0" ]; then break; fi
  sleep 60
done
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $S/runs/nvidia_smi_before.txt
run() { echo "$(date -u +%H:%M:%S) start $5"; $S/scripts/run_offline.sh 8-15 $1 $2 $3 $4 16 8 $S/runs/$5.json 1 > $S/runs/$5.log 2>&1; echo "$(date -u +%H:%M:%S) done $5"; }
run ckpop_hard_pen /workspace/Hierwave/reportables/2026-10-14_cr2_penalty/runs/arm_ckpop_hard_pen_s0/params.pkl pen 0 pen_s0
run ckpop_hard_rkl /workspace/Hierwave/reportables/2026-10-14_wave4_pop/runs/W2b_hard_rkl_300train/params_ground.pkl w4 0 rkl_w2b
run ckpop_rkl123x2 /workspace/Hierwave/reportables/2026-10-14_wave6_pop/runs/C_rkl123x2/params.pkl pen 0 rkl123x2_C
run ckpop_hard /workspace/Hierwave/reportables/2026-10-14_wave4_pop/runs/W1_hard/params_ground.pkl w4 0 hard_w1
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv > $S/runs/nvidia_smi_after.txt
echo ALL_DONE
