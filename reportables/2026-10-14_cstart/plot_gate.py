"""Gate-2 figure: dirt fraction, line_match and per-level entropy of q
(non-forced draws) against update, cstart vs cstart_off (= cfast).
usage: plot_gate.py <out.png> run:label ..."""
import json, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
out, runs = sys.argv[1], [a.split(":") for a in sys.argv[2:]]
fig, ax = plt.subplots(1, 7, figsize=(20, 3.2), sharex=True)
for i, (r, lab) in enumerate(runs):
    L = [json.loads(l) for l in open(f"runs/{r}/train_ground.jsonl")]
    d = [x for x in L if "diag" in x]
    u = [x["update"] for x in d]
    kw = dict(color=COL[i], lw=2, label=lab)
    ax[0].plot(u, [x["diag"]["dirt"] for x in d], **kw)
    ax[1].plot(u, [x["diag"].get("line_match", float("nan")) for x in d], **kw)
    for l in range(5):
        ax[2 + l].plot(u, [x["diag"][f"L{l}"]["H"] for x in d], **kw)
ax[0].set_title("dirt fraction (own draws)"); ax[0].axhline(0.5, color="#888", lw=1, ls=":")
ax[1].set_title("line_match (own draws)"); ax[1].axhline(0.5, color="#888", lw=1, ls=":")
for l in range(5):
    ax[2 + l].set_title(f"level {l}: entropy of q, nats")
for a in ax:
    a.set_xlabel("update"); a.grid(alpha=0.25); a.spines[["top", "right"]].set_visible(False)
ax[0].legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(out, dpi=110)
