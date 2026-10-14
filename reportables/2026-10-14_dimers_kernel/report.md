# dimers-kernel: fine-block Gibbs + footprint MTM + ML on dimers (2026-10-14)

Agent dimers-kernel, cores 64-71, shared GPU (update counts depend on contention). One seed per arm, 300 s training each, bench region (32 x 64, periodic x). Objective: weighted draw-wise ML on kernel-refined own draws (amendment 2026-10-14); no reverse-KL term.

## Answer
- **k = 2 blocks cannot change the diagonal count; k = 3 can; both mix slowly.** Pure kernel chains (no learning) from cfast's collapsed draws (h/v/d = 0.467/0.533/0.000): k=2 stays at 0.509/0.491/**0.000** after 600 cycles; k=3 reaches 0.38/0.39/0.23 (150 cycles, unchanged at 600). From h/v-only squares tilings k=2 also stays at d = 0; from h/d-only rhombi tilings k=2 keeps d = 0.505 exactly (only h <-> v square flips happen). Reason: a flippable {E,SE} or {S,SE} rhombus spans 2 x 3 or 3 x 2 cells and never fits an aligned 2 x 2 footprint; the only k=2 flips are 2 x 2 squares (E,E <-> S,S), which conserve the number of diagonal dimers. k=3 (3 x 3) contains every rhombus, so d can change.
- **k=3 chains from opposite starts did not meet in 600 cycles** (squares start 0.383/0.383/0.235; rhombi start 0.400/0.207/0.393). So the stationary orientation mix of the strip is not measured here, and block moves of size 3 mix the h/v-rich and h/d-rich sectors slowly. I did not test k=4 on the bench region (internal-pruned set 2.9e6, too large for the batched GPU path); k=4 is exact on a 4 x 8 region in numpy (test passes) but too slow to train with.
- **The ML loop did not give a usable answer on the third orientation**: with kernels and ML only, q's own violation rate rose from 2e-4 to ~0.05-0.1 per cell within ~50 updates, in every arm. Orientation fractions of those draws (about 0.35/0.33/0.32) are fractions of violating states and say nothing about the target. It needs a violation term (ckernel_rkl style) that I did not build.

## Tests (tests/kernels/test_fine_block.py, 7 passed, CPU, 4 min)
Dimers on H=4 rows x W=8 periodic (26 817 perfect matchings enumerated). iid exact samples stay uniform after one sweep (state-histogram chi-square within 5 sigma; orientation z < 4.5) for k=2 offsets (0,0), (1,1), k=3, and k=4 (per-tile boundary enumeration, numpy); a 4-offset k=4 mixture started from one matching reproduces the exact orientation means; the GPU path (internal-pruned completions, batched) also preserves uniformity for k=3. Caveat: the chi-square has 26 817 bins and 6000 samples, so it is weak; the orientation z-scores are the sharper test.

## Kernel (castlegen/kernels/fine_block.py)
`FineBlockGibbs(problem, region, k, offset, lam)` subclasses exact_block's structure: k x k tiles of level-0 cells (all sampled channels) at stride k from an offset, x wrapping, rows kept inside [0, H); conflict-graph colouring as in exact_block. Completions of a tile = product of domains pruned by the hard instances lying wholly inside the tile (amortised.internal_pruned; a superset of every feasible set), then every touching instance prices them exactly, so the conditional is exact. Dimers: k=2 -> 210 completions, k=3 -> 15 480, k=4 -> 2.9e6 (too large for GPU; numpy `internal="boundary"` path works). `JaxFineBlockGibbs` reuses amortised.JaxBlockGibbs's fast sweep; `FineBlockKernel` mixes 4 offsets (random per sweep, p*-invariant). Sweep cost on 32 worlds: k=2 8 ms, k=3 0.13 s. Generic: factors and layout only.

## Loop (scripts/dk_loop.py)
Contexts = 32 own tau=1 draws per iteration; kernel cycle = k=2 x3 sweeps, k=3 x3 sweeps, then 8 multiple-try 4 x 4 footprint redraws from q (K=8, lam 20; block_mh.block_pass with a footprint mask); 8 ML updates (batch 8, lr 1e-3) per iteration. Starts: cfast trained 60 s (906 updates, h/v/d 0.47/0.53/0.00) and untrained init.

Arm a1: fine-block lam = infeasible (None), ML weights 1. Arm a2: fine-block lam = 20, ML weight exp(-20 V) per minibatch. Orientation fractions h/v/d; "first" = mean of first 3 iterations, "last" = mean of last 10; viol = violated hard instances per cell.

| arm | start | updates | raw first | raw last | refined first | refined last | viol raw last | viol refined last |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a1 | cfast 60 s | 2040 | .551/.449/.000 | .341/.338/.321 | .462/.457/.080 | .348/.329/.323 | 0.051 | 0.031 |
| a1 | init | 2040 | .386/.314/.300 | .351/.334/.315 | .371/.326/.304 | .352/.328/.320 | 0.052 | 0.031 |
| a2 | cfast 60 s | 1800 | .550/.450/.000 | .358/.323/.319 | .452/.446/.102 | .353/.325/.322 | 0.056 | 0.016 |
| a2 | init | 1784 | .376/.313/.311 | .348/.327/.325 | .354/.324/.322 | .351/.324/.326 | 0.057 | 0.016 |

Figures: images/orientation_vs_update_a1.png, images/orientation_vs_update_a2.png (raw own draws above, refined states below).

Observations. (1) One kernel cycle on valid collapsed draws already puts 8-10 % of cells on diagonals (refined first row), from d = 0.000 in the raw draws; by ~50 updates the raw draws carry a diagonal too. (2) By then raw violations are 0.1 per cell: the ML update on refined states destroys the validity cfast had (the init start is invalid from the start: untrained lam-20 draws violate at 0.15 per cell). The refined states keep 1.6-3 % violations (kernel cycles with finite lam do not repair them), so ML trains on invalid states. (3) The weight exp(-20 V) is nearly zero on every state in the batch after update 50 (valid fraction of refined states 0), so it only normalises. (4) cbound_lam (three orientations, 0.47/0.30/0.23, viol 6e-5) and cfast (0.51/0.49/0.00, viol 5e-5) remain the only valid-state references; none of my arms reaches their violation level, so no orientation comparison with them is meaningful.

## Pure-kernel chains (scripts/chain_relax.py; runs/chain_relax.json 150 cycles, chain_relax600.json 600 cycles)
Orientation h/v/d after 600 cycles, 32 worlds, violations per cell at the end in brackets.

| start | k=2 only | k=3 only | k=2+3 |
| --- | --- | --- | --- |
| cfast60 draws (.467/.533/0) | .509/.491/.000 (1.5e-4) | .382/.390/.228 (3e-5) | .392/.371/.237 (3e-5) |
| probe squares (.495/.505/0) | .506/.494/.000 | .383/.383/.235 | .389/.372/.240 |
| probe rhombi (.495/0/.505) | .420/.075/.505 | .400/.207/.393 | .395/.212/.393 |
| untrained draws (invalid) | .369/.324/.308 (0.054) | .363/.315/.322 (0.032) | .357/.321/.322 (0.029) |

Local block moves can leave the h/v-only state (k=3), but k=3 does not equilibrate the two sectors with each other within 600 cycles. A k >= 3 footprint is necessary; whether it is sufficient for the full mix I cannot say. The bulk target is 1/3 each by the triangular symmetry (strip edges and the 4 x 4 exact 0.505/0.248/0.248 differ), so the k=3 plateaus 0.38/0.38/0.23 and 0.40/0.21/0.39 are both off it.

## What failed / not done
- No valid-state training run (violations stayed 1-5 %); no violation-controlled variant (reverse-KL mix, ckernel_rkl style) was tried; no second seed; k=4 was only tested in numpy on 4 x 8.
- The MT footprint redraw accepted 30-40 % of 4 x 4 moves (changed state in 29-41 %).

## Files
castlegen/kernels/fine_block.py; tests/kernels/test_fine_block.py; scripts/{dk_loop.py, chain_relax.py, train60.py, plot.py, run.sh}; runs/{cfast60.pkl, a1_*.jsonl, a2_*.jsonl, summary_a1/a2.json, chain_relax.json, chain_relax600.json}; images/orientation_vs_update_{a1,a2}.png.

# Part 2: kernel loop + lambda-controlled reverse-KL steps (wave2-bench recipe)

Same loop as above (k=2 and k=3 fine-block Gibbs with lam 20, 4x4 multiple-try redraws from q, draw-wise ML with unit weights, 8 updates per iteration), plus N cfast fused reverse-KL steps (tau 1, own draws, adaptive lambda <= 20, lr 1e-3) per iteration. Start: cfast 60 s (collapsed, h/v/d .467/.533/0). 300 s, one seed, shared GPU. Script: scripts/dk_loop.py `rkl_n=N`; runs/p2_rkl16.jsonl, p2_rkl32.jsonl. Numbers are means over the last 10 iterations.

| arm | updates (ML) | own draws h/v/d | refined h/v/d | viol/cell own draws | viol/cell refined | lambda end |
| --- | --- | --- | --- | --- | --- | --- |
| N = 16 | 1816 | .521/.469/.011 | .466/.449/.085 | 2.2e-3 | 1.4e-3 | 20 |
| N = 32 | 1256 | .496/.503/.001 | .472/.459/.069 | 3.4e-4 | 1.9e-4 | 20 |
| ML only, no rkl (a2) | 1800 | .358/.323/.319 | .353/.325/.322 | 5.6e-2 | 1.6e-2 | - |
| cbound_lam | | .47/.30/.23 | | 6e-5 | | |
| cfast | | .51/.49/.00 | | 5e-5 | | |

- **Validity:** N = 32 keeps own draws near budget (3.4e-4 per cell, target < 2e-4; individual iterations reach 6e-5 to 2e-4). N = 16 does not (2.2e-3, rising over the run from 2e-4 at the start to 2.5e-3).
- **Diagonal:** it does not appear in own draws. N = 32 stays at d = 0.000-0.001 for the whole run; N = 16 creeps to 0.01 as violations rise (the diagonal appears together with violations, as in cdim's curve). The mix stays two orientations, as cfast.
- **Refined states do carry a diagonal** (d = 0.07-0.09 after one kernel cycle, as in Part 1), so the kernel does move mass to the third orientation and ML is trained on it, but 8 ML updates per iteration at lr 1e-3 do not move q's draws there: the reverse-KL steps pull them back. This is the collapse pressure: with enough reverse-KL steps to keep violations near budget, the own-draw mix is the two-orientation one; with too few, the validity goes and the diagonal returns only as violations. At these settings I found no regime with valid draws and three orientations.
- Caveats: one seed per arm; refined-state violations (1.9e-4 at N = 32) are those of the repaired-by-kernel states, comparable with the raw level; the refined diagonal fraction (0.07) is far below the cbound_lam 0.23, so even the kernel's output is not near the three-orientation mix after one cycle. Untested: more ML updates per iteration or higher ML lr relative to the reverse-KL steps, longer runs.
