#!/bin/bash
cd /workspace/Hierwave && source reportables/2026-10-14_cr2_penalty/scripts/env.sh 1
taskset -c 8-15 timeout 900 python reportables/2026-10-16_combo/scripts/smoke.py
echo ALL_DONE
