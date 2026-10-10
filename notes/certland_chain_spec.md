# Certland chain training: module contracts

The plan is the Claude Doc "Certland training: implementation plan"
(https://claude.ai/code/artifact/82e3c948-4a2b-4a50-9b48-00b594884f11);
this file fixes the interfaces between the modules so they can be built in
parallel. Change a signature here only by agreement (tell the integrator).

House rules: single core (`NUMBA_NUM_THREADS=1`, no `prange`), numba kernels
for anything per-cell, JAX (x64) only for fitting, hand-rolled Adam (no
optax). `certland.py` is **not edited** in this round: use it with
`world.params = [None] * 4` (learned = 0, designed terms only). Sign
convention everywhere: an energy; a distribution is `softmax(-E)`.

Designed p\* is certland's as it stands (count prior, smoothness, top noise,
hard support and row honour). The certificate tilt decision is open: do not
add terms on certificates.

## Notation

- Level index `i = 0..4` for `h = 16, 8, 4, 2, 1` (`cl.HS`, `cl.NL`).
- Root `(i, ry, rx)`, `i <= 3`. Its subtree: at level `l >= i`, rows
  `ry*2^(l-i) .. (ry+1)*2^(l-i) - 1`, the same for columns (no wrap needed:
  a block never straddles the seam).
- A **state** of a subtree: `list` over `l = i..4` of `(rho_block,
  cert_block)` int64 arrays (at `l = 4`, `cert_block is rho_block`).
- Channels `RHO = 0`, `CERT = 1`; at tiles one tied channel.

## `castlegen/channels/certland_pred.py` (agent: pred)

Per (level `i <= 3`, channel) a small MLP over the cell's context giving a
full-domain bias `b(ctx)` (length `D = h*h+1` for rho, `len(Aof[i])` for
cert). The predictor conditional is `softmax_t(-E_designed(t) - b_t)`.

Context of cell `(y, x)` at level `i`, channel `ch`: the parent's (rho,
cert) (none at `i = 0`: indicator), the 3x3 same-level neighbourhood in both
channels with the centre's own channel blanked, and the centre's other
channel; off-grid rows by indicator (below the map counts as full). Values
as ordinal features: value / range, a few bumps along the range, end
indicators. Fixed length `NF` (module constant, the same for all levels and
channels).

**Layouts (twist round).** The context is selected by the env
`CERTLAND_PRED_FEATS`, read when `certland_pred` is imported (one layout per
process; non-default layouts compile without numba's disk cache):

- `wide` (default, `NF = 651`): the parent (rho, cert) and quadrant as
  above; the same-level 4x4 window covering the cell's parent block and one
  ring (rows `y0-1..y0+2`, columns `x0-1..x0+2`, `(y0, x0)` the block's
  top-left; it contains the old 3x3; the cell's own channel blanked at its
  slot); the parent's 3x3 neighbourhood at the parent level. Same 25-feature
  encoding per slot for every level and channel (generic, symmetric).
- `widek` (`NF = 667`): `wide` plus one known flag per same-level slot (16).
  `cell_features_k` / `cell_bias_k` take the flags `kn (16,)`; a slot with
  `kn = 0` has its values zeroed. `cell_features` / `cell_bias` (spec
  signatures unchanged) set every flag to 1: the shipped sampler and the
  Gibbs recording. The subtree pass feeds its real known mask (outside the
  subtree / drawn already = 1; undrawn placeholders = 0; at the cell's own
  slot the flag of its other channel). The parent level is always known.
- `v0` (`NF = 251`): the old 3x3 layout, for comparisons.

**Proposal-only predictor (stage 2).** `cp.QPreds(nh)` = `Preds(nh, nf=NFQ)`,
`NFQ = NF + NBND` (`NBND = 72`): the pass's context (`cell_features_k`, known
flags) plus `boundary_features(i, ry, rx, l, y, x, rhos, certs, Aofs, Bofs,
out, o)`: the fixed cells adjacent to the subtree on all four sides at levels
`l+1..l+3`, under the cell's parent block (clipped to the subtree; the cell
itself at the root level), per (side, depth) `1{in map}`, mean rho, mean A,
mean B, min A, max B (normalised). Used only inside the pass
(`pass_subtree` / `subtree_move` / `move_level` take a QPreds like a Preds;
`_bias_args` flags it in `ons[8]`); the shipped sampler keeps `Preds`.
`init_params(..., nf=None)`, `unpack(..., nf=None)` and `Preds(nh, nf=None)`
take the width. Training: `Cfg.proposal = True` puts a QPreds at `preds.q`,
moves with it, records the Gibbs examples for `preds` and
`record_pass(proposal=True)` examples for `preds.q` (store keys
`(l + QOFF, ch)`); checkpoints carry `thq_i_ch`.

```python
@njit def cell_features_k(h, ch, y, x, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, has_parent, kn, out)
@njit def cell_bias_k(h, ch, y, x, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, has_parent, kn,
                      th, nh, on, out)
```

```python
NF: int
@njit def cell_features(h, ch, y, x, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, has_parent, out)  # out (NF,)
@njit def cell_bias(h, ch, y, x, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, has_parent,
                    th, nh, on, out)          # out[:D] = b(ctx); zeros when on == 0
def level_args(world, i)  -> (h, rho, cert, Aof, Bof, prho, pcert, pAof, pBof, has_parent)
                                              # i = 0: parent arrays = level 0's own, has_parent = 0
class Preds:                                  # .th[i][ch] flat float64 or None, .nh
    def __init__(self, nh=64)
    def args(self, i, ch) -> (th, nh, on)     # on = 0 and th = np.zeros(1) when None
    def set(self, i, ch, params)              # from a JAX/np param dict (packs it)
    def params(self, i, ch) -> dict | None    # unpacked
def init_params(i, ch, nh, seed) -> dict      # numpy arrays
def pack(p) -> np.ndarray ; def unpack(flat, i, ch, nh) -> dict
def jax_bias(p, feats)  -> (N, D)             # JAX, equal to cell_bias given the same features
def features_np(world, i, ch, y, x) -> (NF,)  # python wrapper
def generate_pred(world, preds, sweeps=20, seed=0, upto=cl.NL - 1) -> list[int]
    # the shipped sampler: cl.generate's shape (init_top, cl.refine, sweeps per level), Gibbs with
    # E = designed soft (+inf on violation) + bias; tiles designed only; returns no-candidate counts
```

## `castlegen/channels/certland_chain.py` (agent: chain)

```python
def subtree_slices(i, ry, rx) -> list[(l, y0, y1, x0, x1)]          # l = i..4
def get_subtree(world, i, ry, rx) -> state                           # copies
def set_subtree(world, i, ry, rx, state)
def refine_block(world, l, i, ry, rx)
    # deterministic placeholders for level l (> i) of the subtree from its (final) level l-1:
    # cl.refine's packing with side = left iff the parent column px is even, rho split evenly
    # with the remainder to the bottom row first. Must be deterministic: q depends on it.
def subtree_energy(world, i, ry, rx) -> (soft, nviol)
    # every designed term touching a subtree cell, each once (including terms with cells
    # outside: root's count prior against its parent, smoothness and support across the edge,
    # the root row's honour with its sibling and parent, the top noise when i = 0)
def cell_conditional(world, preds, l, ch, y, x, L=np.inf) -> e (D,)
    # designed soft of every value + bias (preds None: no bias) + L * violations against
    # all current cells (L = inf: excluded). The Gibbs conditional the shipped sampler uses.
def pass_subtree(world, preds, i, ry, rx, rng, forced=False, L0=np.inf, order="raster") -> (logq, dead)
    # in place. One ordered top-down pass: root rho then cert, then per level l = i+1..4:
    # refine_block(l) placeholders, then each cell (raster; rho then cert; tiles per 2x2 block
    # from the exact 16-state conditional) drawn from softmax(-(designed soft against current
    # values incl. placeholders + bias + hard rows against KNOWN cells only)), KNOWN = outside
    # the subtree or already drawn in this pass. Hard rows priced L0 per violation (inf:
    # excluded). Cheap necessary single-cell bounds (B <= B_P, A >= A_P - h) may also be used.
    # forced=True: outcomes are the current values (read before overwriting), logq their
    # probability; dead = no admissible value (sampling) / forced value excluded (forced).
    # order="bottomup": certificate rows of a level drawn bottom-up (optional).
    # The bias is cp.cell_bias_k with the pass's known mask (ignored unless layout widek).
def record_pass(world, i, ry, rx, state, L0=np.inf, order="raster") -> {(l, ch): (feats, offs, vals)}
    # the proposal's own training examples for `state`: a forced pass (bias off) recording, per
    # non-tile cell and channel in draw order, cell_features_k with the pass's known mask, the
    # pass's bias-free energies (designed soft against placeholders, +inf on hard exclusions
    # against KNOWN cells and the single-cell bounds) and the value. The world ends in `state`.
def subtree_move(world, preds, i, ry, rx, K, rng, collect=None, L0=np.inf) -> dict
    # i-SIR: particle 0 = current state (forced pass, assert logq finite), K proposals;
    # logw_j = -soft_j - logq_j (-inf on violation or dead); select j ~ w; if collect:
    # collect(world, i, ry, rx, states, wbar) then set_subtree(chosen).
    # returns dict(logw=(K+1,), chosen=int, dead=int, ess=float, changed=int cells)
def move_level(world, preds, i, K, rng, collect=None, move=None) -> dict
    # all roots of level i, colours (y%2, x%2) in a fixed order; move defaults to subtree_move
    # (same signature; certland_anneal provides another). returns aggregate stats:
    # dict(ess=mean ESS of (K+1) weights, changed=fraction of subtree cells changed,
    #      dead=dead / proposals, n=roots)
def tile_sweep(world, rng)   # one exact designed Gibbs sweep over the tiles (cl.sweep_level at i=4)
```

## `castlegen/channels/certland_anneal.py` (agent: anneal)

Annealed proposals of the doc's "Hard constraints" section, on top of the
chain API only (`pass_subtree(L0=...)`, `subtree_energy`,
`cell_conditional`, get/set).

```python
def anneal_path(M, L0) -> (betas (M+1,), Ls (M+1,))   # beta 0 -> 1 linear, L L0 -> inf geometric then inf
def annealed_forward(world, preds, i, ry, rx, M, L0, rng) -> (logw, dead)
    # soft pass, M MH sweeps on pi_m ~ q^(1-b) exp(-b E_soft - L viol), AIS weight; leaves the
    # final state in the world; dead (logw = -inf) if it still violates at the end
def annealed_backward(world, preds, i, ry, rx, M, L0, rng) -> logw
    # particle 0: from the current (valid) state down the path, the AIS weight of the reversed run
def annealed_move(world, preds, i, ry, rx, K, rng, collect=None, M=8, L0=3.0) -> dict
    # subtree_move's contract with annealed weights
```

## `castlegen/channels/certland_train.py` (agent: train)

```python
class Archive:        # per (i, ch): feats (N, NF), offs (N, D) designed energy (+inf excluded),
                      # val (N,), w (N,), upd (N,) update index; capacity cap; drop burn-in
    def add(self, i, ch, feats, offs, val, w, upd)
    def sample(self, i, ch, n, rng) -> dict
def collector(archive, upd) -> collect(world, i, ry, rx, states, wbar)
    # for each state j with wbar_j > 0: install it, for every non-tile cell of the subtree and
    # both channels record (features, cell_conditional(world, None, ...), value, wbar_j);
    # leave the world as found. Layout widek: also the proposal's own examples
    # (certland_chain.record_pass) with the same value and weight (two examples per cell).
def weighted_ce(p, feats, offs, val, w) -> scalar        # JAX; -sum w log softmax(-(offs + b))[val] / sum w
def fit_steps(p, batch, steps, lr, adam_state=None) -> (p, adam_state, loss)
def init_chain(world, preds, noise_seed, burn, rng)      # forward generation + burn-in moves
def update(world, preds, archive, cfg, upd, rng) -> log dict
    # the doc's "One update": root levels top-down (cfg.levels), colours, move_level with
    # cfg.K[i] (and cfg.anneal[i] -> certland_anneal.annealed_move), record, tile_sweep,
    # cfg.steps Adam steps per (i, ch) on this update's examples + archive sample,
    # Preds.set; logs ESS, changed, dead, loss per level
def train(world, preds, cfg, n_updates, log=print, eval_every=..., eval_seeds=...) -> history
```

## Tests and run script (agent: tests)

- `tests/channels/test_certland_chain.py`: numba `cell_bias` vs `jax_bias`;
  `pass_subtree` forced logq finite on valid states; subtree_energy vs a brute
  sum of `cl.window_terms`/site terms; the move leaves p\* invariant on a tiny
  case (one h = 2 root: 30 x 16 states, enumerable) by a long-run histogram vs
  exact p\*; annealed move likewise.
- `notes/experiments/certland_chain_run.py {ref,ess,capacity,h2,curriculum,e2e}`:
  the doc's tests 1-6 in order; logs and figures to
  `/Users/amdson/dev/Hierwave/images/certland_chain_*`.
