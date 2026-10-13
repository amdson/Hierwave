"""Run one ctemp bench with a config override:
python reportables/2026-10-13_ctemp/runexp.py <problem> <tag> '<json cfg>' [seeds train eval]"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
os.environ["CTEMP_CFG"] = sys.argv[3] if len(sys.argv) > 3 else "{}"

from castlegen.bench.run import run  # noqa: E402

a = sys.argv
seeds = int(a[4]) if len(a) > 4 else 2
train = float(a[5]) if len(a) > 5 else 150.0
ev = int(a[6]) if len(a) > 6 else 16
rec = run("ctemp", a[1], seeds, train, ev, a[2], quiet=True)
for k, m in enumerate(rec["per_seed"]):
    print(f"seed {k}: final viol {m['viol']['final']} raw {m['viol']['raw']} F {m['free_energy']}")
    print("  stats", {n: round(v[0], 4) for n, v in m["stats"]["final"].items()})
    print("  probes", m["probes"])
v = rec["verdict"]
print("verdict:", {k: (v[k]["ok"] if isinstance(v[k], dict) and "ok" in v[k] else v[k]) for k in v})
print("run", rec["run"])
