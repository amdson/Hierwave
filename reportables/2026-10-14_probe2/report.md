# probe2: bench v2 block check (per-level rho and starvation), calibration

2026-10-14. GPU shared with other jobs. Cores 56-63. No git. Not an acceptance gate; the v1 probe flag stays on the board.

## What changed (bench version 1 -> 2, see notes/bench.md "Version 2")
- `Variant.score_sites(x, region)`: per-draw log q by cell (sums to `score`; checked on the first world, n/a otherwise). Implemented for the cpass family (cpass/cfast/cbig/...) via `_cpass_core.coloured_pass(record=True)` and for greedy. Every other variant: n/a.
- `castlegen/bench/shares.py` (new), `run.py` (`block_check`, `BENCH_VERSION`), `board.py` (two columns), `variant.py` (hook).
- Block = 2 x 2 patch of level-l sites with one parent; at the top level the same rule (2 x 2 top cells, as if level T+1 existed). Weight = -S - 10 * violations (lam = 10, cpass lam0; the bench has no lambda of its own), factor energy to the finer level, same-level pairs across a block edge half-half; count factors to the parent block's children; cross-level pairs to the finer cell's block. Reconciled: the shares sum to -S - 10 viol exactly on ground, dimers, columns, toy_modes (valid and random states).
- Pool: 128 own raw worlds + 128 probe worlds (32 probe seeds x 4). Starvation: probe blocks with log q share > 5 nats below the own-draw Theil-Sen line at the same weight; the same fraction on own blocks is the null.
- **Extra, not in the spec: "subtree" weight** (a block's direct share plus all blocks below it). Direct weight is exactly constant at toy_modes levels 1 and 2 (their factors are hard equalities, 0 on valid states), so direct rho is undefined there; subtree gives the coarse draws something to track. Both are reported; brackets/"sub" = subtree.

## Calibration, toy_modes (128 own + 128 probes; rho pool / probes / own; starved % probe / own)
(a) exact = exact samples scored by the exact conditional of each draw in the sampler's order (block-local enumeration, F = 0.2513 = F*). (b) cbig256, 60 s, 955 updates, F = 0.2577 (this run). (c) greedy.

| sampler | level | rho direct (pool / probe / own) | rho subtree (pool / probe / own) | starved % direct (probe / own) |
| --- | --- | --- | --- | --- |
| exact | 0 | 0.87 / 0.87 / 0.88 | same | 0.0 / 0.0 |
| exact | 1 | n/a | 0.97 / 0.97 / 0.98 | 1.4 / 1.1 |
| exact | 2 | n/a (log q share is 0: rho2 is forced) | n/a | 0 / 0 |
| cbig256 | 0 | 0.87 / 0.87 / 0.86 | same | 0.0 / 0.0 |
| cbig256 | 1 | n/a | 0.87 / 0.89 / 0.86 | 3.2 / 1.7 |
| cbig256 | 2 | n/a | n/a | 0 / 0 |
| greedy | 0 | 0.37 / 0.47 / 0.56 | same | 0.1 / 0.0 |
| greedy | 1 | 1.00 / n/a / 1.00 (violation penalties on own draws) | 0.53 / n/a / 0.45 | 0 / 0 |
| greedy | 2 | n/a | n/a | 0 / 0 |

Reading. The ideal is rho 0.87 at level 0 (not 1: the half-split of boundary energies and the per-draw decomposition cost that much) and 0.97 at level 1 (subtree). cbig256 reaches 0.87 and 0.87, starvation at the null level (0 %, 3 % vs 1 % ideal): the block check passes a trained sampler that is 0.006 nats/cell from F* while the v1 flag fires on it. Greedy is clearly separated (rho 0.37 at level 0, but its own draws violate hard constraints; probes alone 0.47).

## Ground, cfast 60 s (427 updates, F = 0.91, i.e. collapsed; same shapes as the earlier ground runs)
Own draws are the collapsed state; probes are designed ground states.

| level | rho direct pool / probe / own | rho subtree pool | starved % probe (direct) | starved own (null) | median deficit, probes (nats) |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.09 / 0.78 / 0.01 | 0.09 | 0 | 0 | 0.0 |
| 1 | -0.61 / 0.10 / 0.54 | -0.81 | 100 | 1.4 | 24 |
| 2 | -0.75 / 0.44 / -0.13 | -0.72 | 100 | 2.6 | 46 |
| 3 | -0.68 / 0.55 / 0.14 | -0.65 | 100 | 0.3 | 57 |
| 4 | -0.64 / 0.21 / 0.32 | -0.78 | 100 | 3.0 | 77 |

Worst offenders (probe world, block [row, col] in level-l blocks, deficit nats): L4 [0,2] 108, [0,2] 107, [0,2] 106, [0,1] 105; L3 [1,6] 117, [1,4] 113; L2 [4,3] 82, [4,11] 81; L1 [6,26] 49.5, [8,24] 49.5; L0 [14,47] 5.1 (barely over threshold). So **the fine level does not starve; levels 1-4 starve completely**: every designed-ground block at levels 1-4 has a log q share 24-77 nats below what the sampler gives its own blocks, and the deficit grows with level (coarse blocks sum more draws, but the median deficit per level shows the sampler assigns the target's states essentially zero mass from level 1 up). This agrees with the collapse story (coarse levels avoid the fill-following contexts) and says it is the whole coarse stack, not one level; level 0 is fine given its context.

Caveats. (1) Own and probe weights barely overlap on ground (own blocks sit at -20 to -250, probes near 0), so "at the same energy" is partly an extrapolation of the flat own line; the near-zero-weight own blocks at L1-L2 do have log q share 0 to -10, against -25 to -50 for probes, so the conclusion survives, but starvation % saturating at 100 means the count has no resolution there. The own null is 0.3-3 %. (2) Pooled rho on ground is negative at levels 1-4 because own and probe clusters sit on opposite sides (Simpson), so read rho_probe and rho_own, not the pooled one, for a collapsed sampler. (3) One seed, 60 s, 427 updates under a shared GPU; no repeat.

## Cost and status
Block check: 4 s (exact toy), 12-16 s (toy, 256 worlds), 43 s on ground (256 worlds, 128 own generated one at a time). Integration test: `python -m castlegen.bench run greedy toy_modes --seeds 2 --train 1 --tag probe2v2` (row is on the regenerated board; it replaces greedy's earlier toy_modes row, deterministic variant). Other board rows are "(old harness)" with n/a in the new columns until rerun. `bench.jsonl` backup before my row: /tmp/probe2_bench_backup.jsonl.

Files: `calib.py`, `analyse.py`, `*.json`/`*.npz` (pooled arrays), `images/block_scatter.png` (log q share vs weight share per level, own vs probes; rows: toy cbig256 direct/subtree, ground cfast direct/subtree).

## Not done / proposals
- The ideal (a) is computed for toy_modes only (needs the exact conditional per draw).
- Threshold c = 5 nats and top-block size 2 are untuned choices. A gate would need own-vs-own null rates (reported) and a range-overlap rule for starvation (e.g. only count probe blocks whose weight is inside the own 5-95 % range; on ground that leaves 0-8 % of blocks at L1-L4).
