#!/bin/bash
# Sequential benches (one at a time) on the GPU box: cfast vs cmulti on dimers
# and columns.  taskset keeps XLA's CPU thread pool small (the box's pid
# limit is shared by every agent); a run that dies at start-up is retried.
for pr in dimers columns; do
  for v in cfast cmulti; do
    for try in 1 2 3; do
      taskset -c 0-15 python -m castlegen.bench run $v $pr --tag r2cmulti > /tmp/cmulti_${v}_${pr}.log 2>&1 && break
      sleep 30
    done
  done
done
