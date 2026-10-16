#!/bin/bash
cd /workspace/Hierwave; D=reportables/2026-10-16_l3cap; S=$D/scripts
$S/run_arm.sh ctl 0 300 0
N=$(python3 -c "import json;print(json.load(open('$D/runs/ctl/updates_s0.json'))['iters'])")
echo "ctl iters $N" > $D/runs/iters.txt
for a in w2 c2 wc2; do $S/run_arm.sh $a 0 1500 $N; done
BEST=$(python3 $S/best.py $D/runs)
echo "best $BEST" >> $D/runs/iters.txt
$S/run_arm.sh ctl 1 1500 $N
[ "$BEST" != "none" ] && $S/run_arm.sh $BEST 1 1500 $N
echo DONE > $D/runs/DRIVER_DONE
