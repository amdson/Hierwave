#!/bin/bash
source /workspace/Hierwave/reportables/2026-10-14_guided_step/scripts/env.sh
taskset -c 0-3 timeout 1500 python -m pytest tests/variants/test_guided_step.py -x -q 2>&1 | grep -v "^WARNING"
