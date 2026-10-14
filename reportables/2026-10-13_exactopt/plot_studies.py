"""Round-3 study figures: python plot_studies.py <runs dir> <images dir>.
(1) toy_support representation study, (2) information study (actor window on
toy_modes / tiny, same-step independence on tiny).  KL per block, tau = 1."""
import glob, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs, out = sys.argv[1], sys.argv[2]
D = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(runs, "*.json"))}


def curve(ax, name, label, **kw):
    if name not in D:
        return
    c = D[name]["curve"]
    ax.semilogy([m["step"] for m in c], [max(m["kl"], 1e-5) for m in c], label=label, **kw)


fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
a = ax[0]
for n, lab, kw in [("support_exact_mlp", "tanh 64x2 (cpass MLP), lr 3e-3", dict(color="#1f77b4", lw=2)),
                   ("support_exact_mlp@0.01", "tanh 64x2, lr 1e-2", dict(color="#1f77b4", ls="--")),
                   ("support_exact_mlp@0.03", "tanh 64x2, lr 3e-2", dict(color="#1f77b4", ls=":")),
                   ("support_exact_mlp_3000", "tanh 64x2, 3000 steps", dict(color="#17becf")),
                   ("support_exact_net-tanh-64x2", "tanh 64x2 (fresh init)", dict(color="#aec7e8")),
                   ("support_exact_net-tanh-256x3", "tanh 256x3", dict(color="#8c564b")),
                   ("support_exact_net-ln-64x2", "LN+SiLU 64x2", dict(color="#e377c2", lw=2)),
                   ("support_exact_net-ln-256x3", "LN+SiLU 256x3", dict(color="#d62728")),
                   ("support_exact_table_feat", "feature table (same inputs)", dict(color="#2ca02c", lw=2)),
                   ("support_exact_table_hist", "history table", dict(color="#9467bd", lw=2)),
                   ("support_cfast", "sampled cfast B=8", dict(color="#ff7f0e", ls="--"))]:
    curve(a, n, lab, **kw)
a.set_title("toy_support block: function class, exact gradients")
a = ax[1]
for n, lab, kw in [("modes_exact_table_feat", "modes: feature table, window 4x4", dict(color="#2ca02c")),
                   ("modes_exact_table_feat_w6", "modes: feature table, window 6x6", dict(color="#2ca02c", ls="--")),
                   ("modes_exact_mlp", "modes: tanh MLP, window 4x4", dict(color="#1f77b4")),
                   ("modes_exact_mlp_w6", "modes: tanh MLP, window 6x6", dict(color="#1f77b4", ls="--")),
                   ("tiny_exact_table_feat", "tiny: feature table, window 4x4", dict(color="#bcbd22")),
                   ("tiny_exact_table_feat_w8", "tiny: feature table, window 8 wide", dict(color="#bcbd22", ls="--")),
                   ("tiny_exact_table_hist@0.3", "tiny: history table (lr 0.3)", dict(color="#9467bd")),
                   ("tiny_exact_table_seq", "tiny: history + same-step draws (diagnostic)", dict(color="#9467bd", ls="--"))]:
    curve(a, n, lab, **kw)
a.set_title("information: actor window and same-colour independence")
for a in ax:
    a.set_xlabel("optimiser step"); a.set_ylabel("KL(q || p) per block (nats)"); a.legend(fontsize=7)
    a.set_ylim(1e-4, 20)
fig.tight_layout()
fig.savefig(os.path.join(out, "studies.png"), dpi=110)
print("wrote studies.png")
