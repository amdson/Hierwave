### Stage 8: migration

roots, ground and coord run through one protocol,
`sampler.ChannelSampler` (name, init, sweep(n, T, seed), energy,
candidates; relax, sweep_tempered and unary_logZ are optional).
`sampler.generate(levels, S, rng, painters)` runs the forward chain with
no per-model code.  `notes/experiments/migrate_combined.py`
(chan_combined3 as level 8 = [Sampler(u8)], level 1 =
[CoordSampler(joint)], with painters for the surface and the refinement)
reproduces `images/chan_combined3_sheet.png` pixel for pixel.

Interface changes: `S[l]` may be a list of (n, seed) chunks so that
scripts which reseed can be replayed; `Sampler` keeps an optional soft
model for `relax`; under a certificate, dormancy fixes only a mass-0
value, with d = INF, so it is a dead end to its neighbours, and a single
admitted value with mass stays active so its d is drawn.  The
certificate kernel already supported dormancy and inactive blocks at
K = None and needed no change.  Ground is a plain table set (no parent
rows, so init is a no-op).  For coordinate channels, `CoordSampler`
wraps `CoordKernel.sweep` / `sweep_joint` bit-identically and adds a
total `energy()`; they have no hard parent rows (the parent's refinement
is soft honour `mu` on `uref`), so `init` is a no-op.

What did not fit: the joint (u, t, d) kernel is one sampler owning three
channels, so "each channel owns its sampler" becomes "each sampled group
owns one"; the coordinate kernel has no colouring, no temperature in its
own mode, and no tempered kernel for AIS.

Remaining: the bit tile sampler (`bitgibbs.c`) behind the same protocol;
certificates with a candidate cap K (asserted off: the certificate draw
enumerates the whole domain); a tempered kernel and `unary_logZ` for
coordinate channels so AIS can run over them; dormancy for coordinate
channels if the parents' writes ever become hard.

Tests: `tests/test_migrate.py` (7).  The full suite has 8 pre-existing
failures unrelated to the channel code (`pipeline.py:217` NameError in
test_flow / test_promise / test_support; data files missing in
test_blockconn / test_blockfield).
