"""python -m castlegen.bench {run, board, list} (notes/bench.md).  FROZEN."""
import argparse
import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")


def main():
    ap = argparse.ArgumentParser(prog="castlegen.bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("variant")
    r.add_argument("problem")
    r.add_argument("--seeds", type=int, default=2)
    r.add_argument("--train", type=float, default=150.0, help="training budget per seed, seconds")
    r.add_argument("--eval", type=int, default=16, help="evaluation worlds per seed")
    r.add_argument("--tag", default="")
    sub.add_parser("board")
    sub.add_parser("list")
    a = ap.parse_args()
    if a.cmd == "run":
        from .run import run
        run(a.variant, a.problem, a.seeds, a.train, a.eval, a.tag)
        from .board import board
        board()
    elif a.cmd == "board":
        from .board import board
        print(board())
    else:
        from .problems import SUITE
        from .variant import load_variants
        print("problems:", ", ".join(SUITE))
        print("variants:", ", ".join(load_variants()))


main()
