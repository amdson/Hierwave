"""The sugar test (castlegen/channels/sugar.py): learn the coarse pair
table that makes the chain generator reproduce the ideal joint's object
clustering, which is a pure free-energy effect of the sugar lattice gas.

1. Ideal: two-way sampler on NB x NB blocks; collect phi (adjacent offset
   pair frequencies, presence, sugar density) over SWEEPS_I sweeps after
   burn-in.
2. Forward, untrained: objects independent (theta = 0) with the presence
   bonus; phi.
3. Fit: persistent moment matching, theta += eta (phi_forward - phi_ideal),
   the presence bonus and mu on their own statistics, STEPS steps.
4. Report contact summaries (full side / partial / corner / apart) for the
   ideal, the untrained forward and the trained forward; held-out: the
   nearest-neighbour statistic not in the fit.

    NUMBA_NUM_THREADS=1 PYTHONPATH=. python notes/experiments/learn_sugar.py
"""
import os, time, json
import numpy as np

from castlegen.channels import sugar as SG

NB = int(os.environ.get("NB", 6))                                   # ideal: blocks per side (expensive)
NBF = int(os.environ.get("NBF", 12))                                # forward: blocks per side (cheap)
MU, BPRES = float(os.environ.get("MU", 0.3)), float(os.environ.get("B", 6.0))
BURN, SWEEPS_I = int(os.environ.get("BURN", 100)), int(os.environ.get("SWEEPS_I", 300))
STEPS, ETA, RUNS = int(os.environ.get("STEPS", 300)), float(os.environ.get("ETA", 3.0)), int(os.environ.get("RUNS", 2))
SEED = int(os.environ.get("SEED", 0))
OUT = os.environ.get("OUT", "images/learn_sugar.json")

S = SG.Sugar(NB, NB)
SF = SG.Sugar(NBF, NBF)
print(f"ideal {NB}x{NB} blocks, forward {NBF}x{NBF}, blocks of {S.B}, rectangles {S.R}, {S.D} object values; "
      f"mu {MU} (z {np.exp(-MU):.3f}), presence bonus {BPRES}")


def held_out(obj):
    """Fraction of present objects with at least one present 4-neighbour block (not a fit statistic)."""
    g = obj.grid > 0
    nb = np.zeros_like(g)
    nb[1:] |= g[:-1]; nb[:-1] |= g[1:]; nb[:, 1:] |= g[:, :-1]; nb[:, :-1] |= g[:, 1:]
    return float((g & nb).sum() / max(g.sum(), 1))


# ---- 1. ideal
t0 = time.time()
ideal = SG.Ideal(S, MU, BPRES, seed=SEED)
ideal.sweep(1)
print(f"ideal: compile + 1 sweep {time.time() - t0:.1f}s")
t0 = time.time()
ideal.sweep(BURN)
acc = None
ho_i = []
for k in range(SWEEPS_I):
    ideal.sweep(1)
    st = S.stats(ideal.obj, ideal.tile)
    ho_i.append(held_out(ideal.obj))
    acc = st if acc is None else {kk: acc[kk] + st[kk] for kk in st}
target = {k: v / SWEEPS_I for k, v in acc.items()}
print(f"ideal: {BURN + SWEEPS_I} sweeps {time.time() - t0:.0f}s  present {target['present']:.3f}  sugar {target['sugar']:.3f}  "
      f"neighbour {np.mean(ho_i):.3f}")
ci = S.contact(target)
print("ideal contact  h:", {k: round(v, 3) for k, v in ci['h'].items()}, " v:", {k: round(v, 3) for k, v in ci['v'].items()})

# ---- 2. forward, untrained
fw = SG.Forward(SF, MU, BPRES, seed=SEED + 1)
def forward_stats(fw, n=10):
    acc, ho = None, []
    for _ in range(n):
        st = fw.run()
        ho.append(held_out(fw.obj))
        acc = st if acc is None else {kk: acc[kk] + st[kk] for kk in st}
    return {k: v / n for k, v in acc.items()}, float(np.mean(ho))
st0, ho0 = forward_stats(fw)
c0 = S.contact(st0)
print(f"forward untrained: present {st0['present']:.3f}  sugar {st0['sugar']:.3f}  neighbour {ho0:.3f}")
print("  contact  h:", {k: round(v, 3) for k, v in c0['h'].items()})

# ---- 3. fit
t0 = time.time()
mu, b = MU, BPRES
for step in range(STEPS):
    fw.mu, fw.b = mu, b
    st, _ = forward_stats(fw, RUNS)
    fw.theta_h = SG.moment_step(fw.theta_h, st["h"], target["h"], ETA)
    fw.theta_v = SG.moment_step(fw.theta_v, st["v"], target["v"], ETA)
    b += 2.0 * (target["present"] - st["present"])                      # too few present: raise the bonus
    mu += 2.0 * (st["sugar"] - target["sugar"])                          # too much sugar: raise mu
    if (step + 1) % 50 == 0:
        c = S.contact(st)
        print(f"step {step + 1:4d}  present {st['present']:.3f}  sugar {st['sugar']:.3f}  b {b:.2f}  mu {mu:.2f}  "
              f"h full {c['h']['full']:.3f} partial {c['h']['partial']:.3f} corner {c['h']['corner']:.3f}")
print(f"fit: {STEPS} steps {time.time() - t0:.0f}s")

# ---- 4. report
fw.mu, fw.b = mu, b
st1, ho1 = forward_stats(fw)
c1 = S.contact(st1)
print("\nsummary (horizontal pairs of present neighbours):")
print(f"{'':18s} {'full side':>10s} {'partial':>10s} {'corner':>10s} {'apart':>10s} {'present':>9s} {'sugar':>7s} {'neighbour':>10s}")
for name, c, st, ho in (("ideal", ci, target, np.mean(ho_i)), ("forward untrained", c0, st0, ho0), ("forward trained", c1, st1, ho1)):
    print(f"{name:18s} {c['h']['full']:10.3f} {c['h']['partial']:10.3f} {c['h']['corner']:10.3f} {c['h']['apart']:10.3f} "
          f"{st['present']:9.3f} {st['sugar']:7.3f} {ho:10.3f}")
# the learned table along one line: both rectangles at the shared edge, as a function of row misalignment
print("\nlearned theta_h for rectangles at the shared edge, by row misalignment (0 = full side contact):")
for d in range(S.noff):
    vals = [fw.theta_h[1 + oy * S.noff + (S.noff - 1), 1 + (oy + d) * S.noff + 0] for oy in range(S.noff - d)]
    print(f"  misalignment {d}: {np.mean(vals):+.3f}")
vals = [fw.theta_h[1 + oy * S.noff + (S.noff - 2), 1 + oy2 * S.noff + 0] for oy in range(S.noff) for oy2 in range(S.noff)]
print(f"  one cell apart (any rows): {np.mean(vals):+.3f}")
# picture: ideal state, untrained forward, trained forward (water blue, sugar white, stone grey)
from PIL import Image
def render(tile, px=4):
    col = np.array(SG.KINDS.colours, np.uint8)[tile.grid]
    return np.repeat(np.repeat(col, px, 0), px, 1)
fw0 = SG.Forward(SF, MU, BPRES, seed=SEED + 7); fw0.run()
fw.run()
panels = [render(ideal.tile), render(fw0.tile), render(fw.tile)]
hmax = max(p_.shape[0] for p_ in panels)
panels = [np.pad(p_, ((0, hmax - p_.shape[0]), (0, 8), (0, 0)), constant_values=255) for p_ in panels]
Image.fromarray(np.concatenate(panels, 1)).save(OUT.replace(".json", ".png"))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(dict(ideal=ci, untrained=c0, trained=c1, theta_h=fw.theta_h.tolist(), theta_v=fw.theta_v.tolist(), mu=mu, b=b),
          open(OUT, "w"))
print("wrote", OUT)
