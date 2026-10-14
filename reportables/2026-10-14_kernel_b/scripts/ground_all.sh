#!/bin/bash
S=/workspace/hierwave_git/reportables/2026-10-14_kernel_b/scripts
$S/run_ground.sh reportables/2026-10-13_clamp/params_ground.pkl cfast 256 trained
$S/run_ground.sh reportables/2026-10-14_cstart2/runs/cstart_f1L3/params_ground.pkl cstart_f1L3 256 trained
$S/run_ground.sh reportables/2026-10-13_clamp/params_ground.pkl untrained 256 untrained
