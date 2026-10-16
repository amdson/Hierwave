# Overnight 2026-10-16 (coordinator log)

User: integrate recursion (Amendment 2026-10-16 (b)) and the coarse-planning strategies into a small set of waves; 1 Opus + 4 Sonnet max; ~8 hours; longer experiments and programmatic sweeps; conserve tokens. Rules: `notes/overnight_brief_2026-10-16.md`.

Starting point: `cfitpen_ml2_stage` (fitted own-draws pass + stage R 30): 0 violations on 4/4 seeds, F -0.09, line_match 0.65, roughness 1.43 (chain 1.40); starvation 0/4/9/20/35 % at levels 0-4 (coarse levels wrong in distribution). Fails only the old probe flag.

## Plan
| wave | hours | agents |
| --- | --- | --- |
| A | 0-3 | Opus `recur` (recursion: exact check, clamp gate, first ground runs); Sonnet `starve` (coarse starvation diagnosis: population vs pass vs probes); Sonnet `cdata` (coarse data-starvation sweep: population size x coarse ML weight); Sonnet `certfree` (certificate-free pass + derived certificates + tight-refill stage: build and first runs); Sonnet `longfit` (brute-force long ML fit, no penalty, wider window + LN/SiLU, 60 min, then stage). |
| B | 3-6 | Opus `recur` continues (K sweep on ground, longer runs, bench); Sonnets: follow-ups chosen from A (best cdata config long bench; certfree training + bench; longfit stage/bench; dimers sector kernel if a slot is free). |
| C | 6-8 | consolidation: 4-seed benches of the best 1-2 rows at 300 s and at a long budget, board, synthesis, handoff. |

## Wave A (launched hour 0)
Results: pending.
