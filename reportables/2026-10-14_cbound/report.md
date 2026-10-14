# cbound: curricula that keep the policy soft while the penalty dominates (2026-10-14)

Agent: cbound. Cores 16-23. Exact runs ran on the CPU (`JAX_PLATFORMS=cpu`). Sampled runs ran on a GPU shared with other agents, so update counts differ between runs; compare runs by update, not by time. One seed per arm.
Code:
- `castlegen/variants/_exactopt.py`: new `--cbound key=val,...` flag (`tau0, tau_frac, lam0, lam_frac, c0, cmax, c_frac, cap=tanh|clip|proj, whole=0|1, post`), with `train_cbound`, `sched_at`, `project_out` and `Enum.fns_sched`. Existing kinds and defaults are unchanged.
- `castlegen/variants/cbound.py`: a cfast subclass. Every option defaults to off, so plain `cbound` is cfast.
- Scripts here: `run_exact.sh`, `dimers.py` (copied from cdim), `train_save.py` (copied from csat), `plot.py`. The gate is csat's `eval_ground.py`.
- Run outputs are in `runs/`.

## Answer
- **Exact gradients (dimers 4x4, λ 20):** two schedules reach and keep the diverse optimum.
  - **λ ramp + whole-logit τ 8→1** reaches KL **4e-5**, entropy 0.376, h/v/d 0.505/0.248/0.248. That is the target; the table reached KL 0.018.
  - **Whole-logit τ 8→1 alone** reaches KL 0.031, entropy 0.374, h/v/d 0.507/0.252/0.241.
  - Both hold for the 400 post steps at τ = 1, full λ.
  - The λ ramp alone stays diverse but stops at KL 0.24.
  - Bounding b (tanh cap or a last-layer norm projection) and whole-logit τ 4→1 all go to stripes, the same as the baseline.
- **Sampled training (cfast, dimers 60 s):** the exact result does **not** carry over cleanly.
  - Every cbound arm keeps all three orientations alive while violations are still about 1-2 %. Each loses the diagonal when violations fall (the same entropy-violation curve cdim found), but only down to 2-5 %. cfast loses it at update 10 and goes to exactly 0.
  - `cbound_lam` (λ ramp only) then **recovered** the diagonal at zero violations. It ended at h/v/d 0.47/0.30/0.23, entropy 0.271, raw violations 6e-5 (budget 2e-4). cfast ended at 0.51/0.49/0.000 and 0.224. This is one seed.
  - The exact winner (`cbound_wholelam`) collapsed to diagonal 0.000 in sampled training.
- **Ground:** the λ ramp does not stop the all-dirt collapse, and the gate does not move. The policy is all dirt by update 500, while λ is still 3.1. Gate: 0.75 vs 1.62 at L = 4 and 0.56 vs 2.08 at L = 3 (cfast: 1.97 vs 0.72 and 2.23 vs 0.34). I did not run the 300 s bench.

## 1. Exact gradients (dimers 4x4 periodic, cdim2 region)
Setup: λ 20, Adam lr 3e-3, clip 10, 600 schedule steps, then 400 post steps at τ = 1, full λ and the final bound. Metrics are exact and evaluated at τ = 1 and λ = 20. Target: KL 0, entropy 0.377, h/v/d 0.505/0.248/0.248. The table_feat reference is KL 0.018 / 0.377 / 0.51/0.25/0.25.

| arm | schedule (over the 600 steps) | KL @600 | KL @1000 | entropy/cell @1000 | P(viol) | h / v / d @1000 | diverse after the schedule? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline (cdim2/3) | τ 2→1 over 20 % | 3.249 | | 0.173 | 0 | 1 / 0 / 0 | no |
| capA | b = c tanh(b/c), c 1→30 geometric over 50 %; τ 2→1 / 20 % | 3.249 | 3.249 | 0.173 | 6e-6 | 1 / 0 / 0 | no |
| projA | last layer projected to Σ\|W3\|+\|b3\| ≤ c, c 1→30 / 50 %; τ 2→1 / 20 % | 3.251 | 3.249 | 0.173 | 0 | 1 / 0 / 0 | no |
| lamB | λ 1→20 linear over 50 %; τ 2→1 / 20 % | 0.234 | 0.236 | 0.362 | 6e-5 | 0.552 / 0.240 / 0.208 | yes, KL stalls |
| wholeC4 | q ∝ exp((−s−b)/τ), τ 4→1 over 100 % | 3.248 | 3.248 | 0.173 | 3e-6 | 1 / 0 / 0 | no |
| **wholeC8** | same, τ 8→1 over 100 % | 0.030 | 0.031 | 0.374 | 8e-7 | 0.507 / 0.252 / 0.241 | **yes** |
| **lamB + wholeC8** | λ 1→20 / 50 % and whole τ 8→1 / 100 % | 0.069 | **0.00004** | **0.376** | 1e-6 | **0.505 / 0.248 / 0.248** | **yes** |

![exact](images/exact_schedules.png)

- **A bound on b does nothing.** The tanh cap keeps the *in-use* policy soft (its KL at c = 1.25 is 15.6, with many violations). But the raw MLP output goes horizontal in every context within 20 steps, the same as without the cap, so the direction is unchanged. Once the bound is lifted, the policy is stripes. The projection version (a bounded last layer, with the gradient kept linear) also reaches stripes by step 100. The margin cap limits how far the network moves, not where it goes.
- **Whole-logit τ** lowers the effective step of the last layer by a factor τ and keeps the in-use policy soft. With τ0 = 4 that is not enough; with τ0 = 8 it is. The τ = 1 curves in the figure show stripe-like states mid-run (h 0.94 at step 300 in the combination). Those are the τ = 1 readout; the policy actually sampled at τ = 4.5 is soft. Diversity returns as τ falls.
- **λ ramp:** at small λ the gradient is entropy-dominated, and the network picks up some context structure before the penalty arrives. It never saturates, but it then converges slowly (KL 0.24, still falling at step 1000).

## 2. Sampled training: dimers, 60 s (cfast, bench region, 4 held-out worlds every 5 updates, final row on 16 worlds)
| run | updates | diag < 0.05 at | min diag after | final h / v / d (raw) | entropy | raw viol | repaired viol |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cfast (control, rerun now) | 3806 | update 10 | 0.000 | 0.513 / 0.487 / 0.000 | 0.225 | 4.7e-5 | 4.7e-5 |
| cbound_whole8 (whole τ 8→1 over 60 %) | 3411 | ~1600 (τ 2.3) | 0.02 | 0.506 / 0.448 / 0.047 | 0.236 | 1.6e-5 | 1.6e-5 |
| **cbound_lam** (λ 1→lam0 over 50 %, control frozen during the ramp) | 3960 | ~1200 (λ 6.5) | 0.03 | **0.472 / 0.298 / 0.230** | **0.272** | 6.2e-5 | 6.2e-5 |
| cbound_wholelam (both) | 2770 | ~1450 (τ 1.1) | 0.000 | 0.492 / 0.508 / 0.000 | 0.222 | 0 | 0 |

For reference, cdim's cfast defaults run (1050 updates) had diag 0.000 and entropy 0.221.

![dimers](images/dimers_sampled.png)

- All three cbound arms hold about a third per orientation while raw violations stay above about 1 %. Each drops the diagonal in a sharp step when violations fall from about 1 % to 1e-4, which is cdim's curve again. A slower schedule moves the collapse later; it does not remove it.
- In cbound_lam the diagonal stayed at 3-5 % after the drop instead of going to 0. From update 3000 to 3500 it grew back to 0.25 with zero violations, and entropy rose from 0.237 to 0.272. cfast's diagonal stays at exactly 0 for all 3800 updates. This is the only sampled run in this project so far that ends with three orientations within budget. It is one seed, and the recovery happened in the last 15 % of the run, so it may be seed luck. It needs a second seed and a longer run before it counts.
- The combination that is best under exact gradients collapsed fully. With sampled gradients, whole-logit τ did not protect the diagonal through the end of the anneal (τ ≈ 1-2.3).
- How `whole` is implemented in cfast: the actor stores W_eff = W_raw/τ for the last layer. Each Adam step on it is divided by τ, and W_eff is rescaled when τ changes. This is what Adam on W_raw would do, up to eps and the moment scale. While τ > 1, the dimers measurements generate with W_eff at τ = 1, which is the in-use b at the generation temperature.

## 3. Ground (cbound_lam_diag, 120 s, 3981 updates, λ ramp 1→10 over 50 %)
| update | λ | τ | dirt | line_match | H L0 / L1 / L2 / L3 / L4 | b spread L1-L4 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.0 | 2.0 | 0.857 | 0.505 | 0.66 / 1.43 / 2.28 / 3.04 / 3.20 | 0.1 |
| 200 | 1.8 | 1.77 | 0.939 | 0.502 | 0.68 / 0.37 / 1.06 / 1.56 / 1.71 | 5.5 / 7.0 / 7.7 / 11.7 |
| 500 | 3.1 | 1.42 | 1.000 | 0.500 | 0.00 / 0.26 / 0.55 / 1.31 / 0.39 | 8.3 / 13.7 / 9.1 / 17.7 |
| 3500 | 0 (control) | 1.0 | 1.000 | 0.500 | 0.00 / 0.01 / 0.04 / 0.02 / 0.01 | 15.8 / 29.2 / 30.4 / 32.1 |

Clamp gate (csat's `eval_ground.py`, 32 seeds × 4 probes; fine cost per fine cell, own coarse levels vs fill-following probe):

| run | updates | L = 4 own | L = 4 fill | L = 3 own | L = 3 fill | own dirt |
| --- | --- | --- | --- | --- | --- | --- |
| untrained | 0 | 2.47 | 2.32 | 2.07 | 1.98 | 0.88 |
| cfast (reference) | | 0.72 | 1.97 | 0.34 | 2.23 | 1.00 |
| csat_lr1 (csat report) | 2805 | 0.57 | 1.58 | 0.37 | 2.04 | 1.00 |
| cbound_lam | 3981 | 0.75 | 1.62 | 0.56 | 2.08 | 1.00 |

- On ground the collapse is not driven by the penalty. The run goes to all dirt at λ ≈ 1-3, while the b spread is still growing (it reaches 30 nats). This matches csat: the untrained model is already 86 % dirt, and the top level locks first. Keeping λ small does not keep cert4 soft.
- The gate gap is 0.9 nats at L = 4 and 1.5 at L = 3, which is within the range of every other arm. The gate did not move, so I did not run the 300 s bench (task 3 skipped as instructed).
- Not run on ground: `cbound_whole8_diag` / `cbound_wholelam_diag` (registered). Whole-logit τ is the lever csat's report said was missing (it tempers b), so it is the next ground run. The dimers sampled result makes it unlikely to be enough alone.

## Variants registered (cbound.py)
`cbound` (= cfast), `cbound_lam`, `cbound_lam_diag`, `cbound_proj`, `cbound_projlam`, `cbound_whole8`, `cbound_whole8_diag`, `cbound_tau8` (control: τ 8→1 on −s only), `cbound_wholelam`, `cbound_wholelam_diag`, `cbound_off_diag`. All the changes are engineering freedoms (schedules, optimiser projection, λ-control defaults). λ stays ≤ 20 throughout, and the objective is unchanged.

## Caveats
- One seed for every arm.
- The exact runs use one 4x4 region.
- The sampled dimers runs use 4-world readouts, which are noisy (±0.05).
- The GPU was shared: 2770-3960 updates in 60 s.
- The ground run's λ was 0 at the end (raw violations under ε), as in csat's lever arms, so its own-context cost is not comparable with a λ > 0 run.
