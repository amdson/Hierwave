#!/bin/bash
# Sequential benches (one at a time) on the GPU box: cfast vs cmulti.
for pr in dimers columns toy_support; do
  for v in cfast cmulti; do
    python -m castlegen.bench run $v $pr --tag r2cmulti > /tmp/cmulti_${v}_${pr}.log 2>&1
  done
done
