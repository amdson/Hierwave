# csat: does the fine-level bias saturate first, and do lr / τ levers stop it? (2026-10-14)

Agent: csat. Acts on cresid's open question (`../2026-10-13_cresid/report.md`, "Next" 1). Ground only, one training seed per arm, 150 s each, on a GPU shared with other agents' jobs. Diagnostics only: training and generation code are cfast's, unchanged.

## Answer

**No. The saturation hypothesis is wrong about the order of events, and neither lever fixes the gate.**

1. **The all-dirt collapse starts at the top level, not the fine levels.** With cfast's defaults, cert4's entropy and the dirt fraction move together: cert4 entropy falls below 0.3 nats at update 200, and dirt reaches 0.95 at update 200. rho3 and rho2 go deterministic later, at about update 350; rho1 goes earlier, at about 175. With lr 1e-3, dirt reaches 0.95 at update 225 and cert4 collapses by update 275. At that point rho3, rho2 and rho1 still have 2–3 nats of entropy, and they never go below 0.3 nats. So the dirt collapse happens while the fine levels are still stochastic. Note also that the untrained model at τ = 2 is already 86 % dirt (the fill line implies about 50 %).
2. **A lower lr keeps the fine levels stochastic but does not make them read context.** At the end of training, csat_lr1's rho3, rho2 and rho1 entropies are 0.37, 1.37 and 0.70 nats; the default's are 0.03, 0.00 and 0.17. Yet with the same noise, its rho3, rho2 and rho1 draws are still **95.2 %, 99.4 % and 99.9 % identical** whether the coarse levels come from the model or from a fill-following probe. So the fine levels are stochastic and context-blind: their q does not depend on the parent count.
3. **Gate (fine cost per fine cell; cfast reference: 0.72 vs 1.97 at L = 4, 0.34 vs 2.23 at L = 3; the probe's own cost is 0.014):**

| run | updates | L = 4: own | L = 4: fill | L = 3: own | L = 3: fill | own dirt | own line_match | clamped line_match L4 / L3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| untrained (τ = 1, b = 0) | 0 | 2.47 | 2.32 | 2.07 | 1.98 | 0.88 | 0.503 | 0.561 / 0.606 |
| csat (cfast defaults) | 2018 | 1.22 | 1.92 | 0.93 | 2.00 | 1.000 | 0.500 | 0.534 / 0.583 |
| **csat_lr1** (actor lr 1e-3) | 2805 | **0.57** | **1.58** | **0.37** | 2.04 | 1.000 | 0.500 | 0.547 / 0.592 |
| csat_tauall (τ 2 → 1 over all 150 s) | 2311 | 0.71 | 1.88 | 0.44 | 2.11 | 1.000 | 0.500 | 0.553 / 0.601 |

The lower lr reduces the fill-context cost at L = 4 from 1.92 to 1.58, but the gap to the run's own contexts stays at 1.0 nat per cell, and nothing changes at L = 3. The slower anneal does nothing useful. The ground draws do **not** follow the fill line in any arm: every own draw is 100 % dirt (line_match 0.500). Fine levels clamped under a probe's coarse levels match the line no better than untrained ones do (0.53–0.60 against 0.56–0.61); they produce the same block staircases (images below).
4. **Why τ is a weak lever here:** the logits are −s/τ − b, so τ tempers only the designed soft term, not the learned bias. The bias's spread over allowed values reaches 15–37 nats at levels 1–4 in the default run, and 60 at level 4 in csat_tauall, which dwarfs any τ in [1, 4]. A higher τ0 (lever b') would behave the same way, so I did not run it. Lever (c), clipping, was not run either: Adam normalises the step size, so a tighter per-level clip only reweights noisy batches. The diagnostics show the step count, not gradient spikes, driving the drift (the bias spread grows roughly linearly with updates).

## Figures

- `images/sat_all.png`: per level, against update, for each run: the non-forced entropy of q; the same with b = 0; the spread of b over the allowed set A; the forced fraction; and the dirt fraction with τ − 1.
- `images/collapse_timing.png`: per-channel entropy (cert4, rho4, rho3, rho2, rho1, cert3) against the dirt fraction, for the first 1000 updates.
- `images/clamped_examples_<run>.png`: tiles drawn with the model's own coarse levels and with probe-clamped coarse levels, from the same noise.

![](images/sat_all.png)
![](images/collapse_timing.png)
![](images/clamped_examples_csat_lr1.png)

## What else the curves show

- **The forced fraction at level 0 rises to 0.99–1.00 within about 300 updates in every arm.** Once cert4 is near full, t is forced to dirt, and from then on training only visits that context.
- **λ.** csat ended at λ = [2.0, 6.7, 7.8, 4.4, 0]; both lever arms ended at λ = 0 everywhere, because their raw violations stayed under ε. The bench was not run, so these runs have no repaired-violation or free-energy numbers. Their own-context cost (0.57 and 0.71) is lower than csat's (1.22), partly because they ended with a smaller λ.
- **The context blindness is not only off-distribution.** Even in its own contexts, csat_lr1 keeps a stochastic rho3 whose mean depends only on the row (`runs/csat_lr1/same_check.txt`). In the conditional, the count prior reaches only the last of the four children, so the other three get the parent only through b and noisy cone returns. 2800 updates at B = 8 are not enough for b to learn to read the parent, even when it is not saturated.

## Method

- **Variant** `castlegen/variants/csat.py`: CSat(CFast) wraps the train log. At update 1 and every 25 updates it runs `train_pass` with the current parameters, τ and λ on 8 fixed diagnostic worlds (seeds 900000+). Per level and per channel it logs:
  - H: the mean entropy of q over valid draws with |A| > 1;
  - H0: the same with b = 0;
  - the forced fraction (|A| = 1);
  - the mean |b| over A;
  - the mean spread (max − min) of b over A at non-forced draws;
  - the dirt fraction.

  The overhead was 4.5 s per 150 s run. Lever arms: `csat_lr1` (lr = 1e-3), `csat_tauall` (anneal = 1.0), and also registered but not run, `csat_tau4` (τ0 = 4) and `csat_clip1` (clip = 1).
- **Scripts** (here):
  - `train_save.py`: it warms the update and the diagnostic compiles in the same process, then trains seed 0 for 150 s and saves the parameters.
  - `eval_ground.py`, `same_check.py`, `clamp_tools.py`: the clamp gate, reused from cresid with `resid` off. It uses 32 seeds × 4 probes.
  - `gate.py`, `line_match.py` (the bench's `ground.stats` on the saved t), `plot_sat.py`, `plot_timing.py`.
  - Per-run outputs are in `runs/<run>/`.
- **Conditions.** Cores 24–31, with the GPU shared with other agents. csat_lr1 trained while my csat gate eval was running, and csat_tauall trained while the csat_lr1 eval was running, so update counts are only roughly comparable. All arms were 150 s, the brief's budget, not the new 300 s bench budget.

## Caveats

- One seed per arm, at 150 s.
- The diagnostic batch is only 8 worlds.
- "Collapse" timings are read off a 25-update grid.

## Proposal (outside the levers tested; nothing built)

Engineering levers can't fix this: the collapse starts at the coarse level, and the fine levels don't learn to read the parent count even while stochastic. Two pieces would be needed:

1. **Per-candidate gradient at the fine count draws.** This is an allowed freedom: "branching at any level, enumerating small candidate sets". For a subsample of non-last rho children, enumerate their allowed values with coupled replays scored by the cone, as creplay and cfast_k4 do at the top. That gives every candidate an exact cost difference, so b gets a direct signal to read the parent in the contexts that are visited. It stays unbiased, so the objective is unchanged.
2. **Visiting fill contexts.** (1) does not reach contexts that training never visits, and those are set by cert4 going full by about update 250. ideas #12 (draw at τ_b > 1, importance-weighted back to τ = 1; the user says it is in scope) covers this, with one catch from this report: with logits −s/τ − b, raising τ does not temper b. Exploration must temper the whole logit, e.g. draw from q^(1/τ_b), and use the ratio q / q^(1/τ_b) as the importance weight; otherwise saturated levels stay deterministic. Training fine levels directly on probe-clamped coarse contexts would change the context distribution of the objective (constraints 3 and 4). That needs the user's approval and is not proposed as an experiment.
