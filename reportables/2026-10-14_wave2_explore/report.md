# wave2-explore: own-draw-only kernels on ground's coarse levels

2026-10-14, agent wave2-explore. Problem: ground, LN+SiLU 6x6 (net="ln", win=(-2,3)), one seed per arm. Start: sweep2a arm-B params, so levels 0-2 are fit and levels 3-4 are at init (as in sweep2b). The RTX 3090 was shared with 3-4 other agents, so setup and iteration times are contended. Cores 40-47.

**Objective.** Weighted ML on states produced by p*-invariant kernels applied to the sampler's OWN τ = 1 draws, on all levels and draw-wise (amendment 2026-10-14). There is no reverse-KL term. **No probe lifts were used for training**; they were used only in the clamp-gate evaluation. Generation is unchanged (cfast's coloured pass).

## Result: no arm leaves the all-dirt state

Each arm trained for 300 s after setup (setup was 175-430 s of compiling). There were 32 own contexts per iteration and 8 ML updates per iteration (B = 8, lr 1e-3). The first 60 s were the curriculum phase: L1-L2 MTM (K = 8) plus fine Gibbs, with ML on levels 0-2 only. The kernels used λ = 20.

- Dirt and line_match are the mean of the last 5 iterations, raw own draws.
- Gate = kernel-a's clamp gate: fine cost per fine cell in the fill-clamped context vs the own context.
- Violations: own draws, per fine cell; nv: in the gate's L3 fill context.
- Accept: per move at L3 / L4.
- pure→part: accepted moves that turn a level-L block whose tiles are all dirt into a partial block, divided by moves proposed on pure-dirt blocks.

| arm | updates | dirt | line_match | gate L3 fill / own | gate L4 fill / own | viol/cell (nv gate) | accept L3 / L4 | pure→part L3 / L4 |
|---|---|---|---|---|---|---|---|---|
| start (sweep2a B) | 0 | 0.84 | 0.50 | -0.106 / 0.346 | 0.683 / 0.725 | 0.018 (2e-4) | | |
| sweep2b ck300_own (ref) | 280 | 0.90 | 0.50 | | | 0.005 | .54 / .38 flip | not logged |
| A1 curriculum + flips (K = 16, q τ_c = 2) | 336 | 0.87 | 0.50 | -0.102 / 0.107 | 0.084 / 0.133 | 0.005 (4e-4) | .59 / .54 | 22/1366 (1.6 %) / 56/1112 (5.0 %) |
| A2 = A1 + annealed L4 candidates, AIS-weighted | 272 | 0.89 | 0.50 | -0.100 / 0.144 | 0.132 / 0.204 | 0.006 (3e-4) | .66 / .69 flip | flips 17/747 / 38/624; anneal: 1.5 % valid, ESS 0.8 of 8 |
| A3a tempered transitions τ_hi 2, n 16 (no flips) | 384 | 0.88 | 0.50 | -0.100 / 0.102 | 0.090 / 0.120 | 0.004 (6e-4) | .67 / .48 | 41/1934 (2.1 %) / 74/1630 (4.5 %) |
| A3b TT τ_hi 6, n 32 | 608 | 0.90 | 0.50 | -0.095 / 0.116 | 0.103 / 0.133 | 0.005 (6e-4) | .33 / .21 | 19/1187 (1.6 %) / 5/973 (0.5 %) |
| A3c TT τ_hi 6, n 32, λ 20→1 along the ladder | 992 | 0.84 | 0.50 | -0.105 / 0.051 | 0.029 / 0.057 | 0.004 (5e-4) | .008 / .000 | 1/4532 / 0/3682 |

- **Own-draw line_match stays at 0.50 in every arm.** Dirt stays at 0.84-0.90.
- The clamp-gate gap closes in all arms (own-context cost goes from 0.35 to 0.05-0.14). As in sweep2a/2b, that is the fine levels fitting own contexts; it is not exploration.

![dirt and line_match vs update](images/loop_dirt_line.png)
![acceptance per level](images/loop_accept.png)
![own draws](images/own_draws.png)

In the own-draws image, training makes the top air band smoother but no more fill-dependent than at the start. The yellow line is the fill-implied ground line.

## Why: the collapsed variable is the certificate, not rho
Raw: runs/inspect_own_start.json, own draws at the start params.

**Rho already follows fill.** By L4 block row, top to bottom:
- rho4 is 0.017 / 0.12 / 0.87 / 0.99;
- the fill is 0.00 / 0.11 / 0.88 / 1.00.

So own draws are not "rho = full": no L4 cell has rho4 = 256.

**The tiles follow the certificates instead.**
- The tile count per L4 row is 0.60 / 0.86 / 0.95 / 0.99.
- B4 (occupied columns) is 0.66 / 0.89 / 0.97 / 0.99.
- The mean |rho - tile count| per cell is 0.18 at L1, 0.30 at L2, 0.40 at L3 and 0.37 at L4. That is a large soft count-prior debt that q and the refined states both carry.

**So the collapse is in cert B.**
- Support (hard) means an occupied column is dirt to the bottom.
- Lowering B_L needs every child row's B sum to drop first (honour, hard).
- Freeing a column of a top block is therefore a bottom-up chain through 4 levels of hard constraints.

**What each move does to that chain.**
- **Flips** propose a new cell value. Their q subtree redraw puts the certificates back high, so flips are accepted at about 55 % but almost only within the mode.
- **Tempered transitions** at τ ≤ 6 with λ = 20 do not cross the hard chain.
- **Lowering λ at the hot end** lets the chain cross, but the cooled states keep violations, so they are rejected: median log a is -756 at L4. This matches speed-2's finding that a λ ramp destroys valid states.
- Where pure→partial moves are accepted (A1 and A3a L4, about 5 %), they remove a few tiles. ML does not accumulate them, because each iteration re-draws fresh contexts from q.

## What was built (all generic: factors + layout + q)
`castlegen/kernels/explore.py`:
1. **`make_flip_chain_tracked`.** coarse_flip's flip chain, plus old and new values of the cell's first channel and the fine-tile sum of the block.
2. **`make_annealed_candidates`.** x0 ~ q(subtree | outside) with the least-violating mask dropped, then speed-2's `anneal_block_valid` (τ 8→1, λ 20 const, hard final). The weight is log w = -β0(S0 + λ0 N0) - log q(x0) + log w_AIS.
   - These are exact AIS weights; the test checks log Z on both toys.
   - With q's mask kept (`full_support=False`), the toy log Z is 5-6 nats low, so the weights are biased.
   - In A2 the weights entered ML by importance resampling: 4 candidates per update drawn ∝ SNIS weight, alongside 8 refined states.
3. **`make_tempered_transition`.** Neal's tempered transitions on a level-L subtree.
   - Heating uses forward masked colour-Gibbs sweeps; cooling uses the exactly reversed sweeps; the move is MH-accepted.
   - An optional λ ladder is included.
   - It is p*-invariant, and it is the admissible form of "annealed block from the current own block". speed-2's unweighted anneal from the own block is not invariant.

`castlegen/variants/ckernel_own.py`: subclasses ckernel. It uses own contexts only, the two-phase curriculum, the tracked flips, optional annealed candidates and optional tempered transitions, and it logs all of the above every iteration.

**Tests.** tests/kernels/test_explore.py, 7 pass:
- the tracked-flip records;
- AIS log Z and the SNIS rho2 mean on toy_support and toy_modes against exact enumeration;
- invariance of tempered transitions, with and without the λ ladder, from exact toy samples (|z| < 4.5, acceptance not trivial).

**Delayed acceptance (candidate 3): not built.**
- A valid stage-1 screen must depend only on (x, x') symmetrically. The coarse-level factors are allowed only with the outside (smoothness, support, parent honour, fill), not with the current children, because the reverse move sees the new children.
- On GPU under vmap the screen saves no compute, because lax.cond becomes select.
- With flips already accepted at 55 %, a screen cannot help: the bottleneck is the content of the subtree redraw, not the cost of rejections.

## Caveats
- One seed per arm.
- Setup and compile took 175-430 s per arm on top of the 300 s, so update counts differ: TT arms are cheaper per iteration than K = 16 flips.
- The GPU was shared.
- The A2 anneal used full-support q starts (exact weights). Those starts are only 1.5 % valid at 32 steps, against speed-2's 98 % from own blocks, which have heuristic weights.

## Next
1. **Price the certificate debt in the move rather than in the start.** A joint L1 block move that lowers cert B and clears a column at once (exact_block over (cert, tiles) of a column strip is not generic). Generic version: a multiple-try subtree move whose proposal is q at the coarse cell and **exact Gibbs, not q, for the certificates of levels ≤ 2** (kernel-a's (2,0)/(1,1) blocks used as the proposal inside MTM).
2. **Persistent chains.** Keep the refined own states across iterations (a replay buffer of own-started chains), so accepted pure→partial moves (5 % at L4) accumulate instead of being re-drawn from q every iteration. Each chain is still started from q, so KL(qK^n ‖ p*) ≤ KL(q ‖ p*) holds. This is the cheapest next arm.
3. Tempered transitions at L4 with more steps (n = 128) and λ kept at 20, run on persistent chains.

## Files
- `castlegen/kernels/explore.py` (new)
- `castlegen/variants/ckernel_own.py` (new)
- `tests/kernels/test_explore.py` (new)
- `reportables/2026-10-14_wave2_explore/`:
  - report.md
  - images/: loop_dirt_line.png, loop_accept.png, own_draws.png
  - scripts/: run_own.py, diag_accept.py, inspect_own.py, plot_runs.py, render_own.py, env.sh, eval_ground.py / clamp_tools.py / gate.py (sweep2b copies)
  - runs/: A1_curr_flip, A2_anneal_L4, A3a_tt_t2, A3b_tt_t6, A3c_tt_t6_lam1 (loop.jsonl, params_ground.pkl, ground_summary.json), *.log, diag_accept.log, inspect_own_start.json, test_explore.log
