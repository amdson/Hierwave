#!/bin/bash
# GPU box helper (Vast.ai RTX 3090).  From the coloured-pass worktree root:
#   scripts/gpu.sh push                      copy code to the box (never results/ or images/)
#   scripts/gpu.sh run <cmd...>              run a command in /workspace/hierwave on the box
#   scripts/gpu.sh bench <variant> <problem> [bench args]   push, then run a bench there
#   scripts/gpu.sh pull                      fetch results into results/gpu/ and images/gpu/
# Results on the box live in its own results/bench.jsonl; pull copies it to
# results/gpu/bench.jsonl (GPU and Mac boards stay separate: the 150 s budget
# means different things on each).  Several processes may share the GPU.
set -e
HOST=root@137.175.22.196
PORT=26359
SSH="ssh -p $PORT -o BatchMode=yes -o ConnectTimeout=20"
DIR=/workspace/hierwave
ENV="source /venv/main/bin/activate >/dev/null && cd $DIR && export PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1"
# the container's pid limit (2304) is shared: pin each job to 8 of the 80 cores (~44 threads instead of ~316)
CORES=$(( (RANDOM % 10) * 8 ))
PIN="taskset -c $CORES-$((CORES + 7))"
quiet() { grep -v -E "^(Welcome to vast|Have fun|AI agents:)" || true; }

case "$1" in
  push)
    rsync -az --exclude .venv --exclude results --exclude images --exclude __pycache__ --exclude .pytest_cache \
      -e "$SSH" ./ $HOST:$DIR/ 2>&1 | quiet ;;
  run)
    shift
    $SSH $HOST "$ENV && $PIN $*" 2>&1 | quiet ;;
  bench)
    shift
    $0 push
    $SSH $HOST "$ENV && $PIN python -m castlegen.bench run $*" 2>&1 | quiet ;;
  pull)
    mkdir -p results/gpu images/gpu
    rsync -az -e "$SSH" $HOST:$DIR/results/bench.jsonl results/gpu/bench.jsonl 2>&1 | quiet
    rsync -az -e "$SSH" $HOST:$DIR/results/runs/ results/gpu/runs/ 2>&1 | quiet
    rsync -az -e "$SSH" $HOST:$DIR/images/bench/ images/gpu/ 2>&1 | quiet
    python -c "from castlegen.bench.board import board; from pathlib import Path; t = board(Path('results/gpu/bench.jsonl')); Path('results/gpu/board.md').write_text(t); Path('results/board.md').write_text(board())" ;;
  *)
    echo "usage: scripts/gpu.sh push|run <cmd>|bench <variant> <problem> [args]|pull"; exit 1 ;;
esac
