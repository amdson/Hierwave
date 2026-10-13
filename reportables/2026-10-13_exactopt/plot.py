"""Training curves of the exactopt runs: python plot.py <runs dir> <out dir>.
One figure per problem (and tag): exact KL (log scale), coarse-marginal KL and
the key statistic vs optimiser step, exact-gradient runs solid, sampled dashed."""
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAT = {"support": "flat_block_frac", "modes": "mode_other", "tiny": "p_viol"}
COL = {"exact_mlp": "#1f77b4", "exact_table_feat": "#2ca02c", "exact_table_hist": "#9467bd",
       "cpass": "#d62728", "cfast": "#ff7f0e"}

runs_dir, out = sys.argv[1], sys.argv[2]
groups = defaultdict(list)
for f in sorted(glob.glob(os.path.join(runs_dir, "*.json"))):
    d = json.load(open(f))
    name = os.path.basename(f)[:-5]
    tag = name.split(d["run"], 1)[1].lstrip("_")
    groups[(d["meta"]["problem"], tag)].append(d)

for (prob, tag), ds in groups.items():
    st = STAT[prob]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for d in ds:
        c = d["curve"]
        x = [m["step"] for m in c]
        ls = "-" if d["run"].startswith("exact") else "--"
        lab = d["run"] + (f" B={d['info']['B']}" if "B" in d["info"] else "") + f" (lr {d['info']['lr']:g})"
        kw = dict(color=COL.get(d["run"]), ls=ls, label=lab)
        ax[0].semilogy(x, [max(m["kl"], 1e-5) for m in c], **kw)
        ax[1].semilogy(x, [max(m["kl_coarse"], 1e-6) for m in c], **kw)
        ax[2].plot(x, [m.get(st, float("nan")) for m in c], **kw)
    tgt = ds[0]["meta"].get("target", {}).get(st, ds[0]["meta"].get("target_p_viol") if st == "p_viol" else None)
    if tgt is not None:
        ax[2].axhline(tgt, color="k", lw=0.8, ls=":", label="target")
    ax[0].set_title(f"{prob}: KL(q || p) per block, tau = 1 (nats)")
    ax[1].set_title("KL of the coarse-level marginal")
    ax[2].set_title(st)
    for a in ax:
        a.set_xlabel("optimiser step")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"curves_{prob}{('_' + tag) if tag else ''}.png"), dpi=110)
    print("wrote", prob, tag)
