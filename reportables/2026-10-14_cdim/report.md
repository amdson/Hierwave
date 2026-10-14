# cdim: dimers entropy collapse vs lambda and tau (2026-10-14)

Question: with T = 0 (no hierarchy), is the collapse to stripes caused by lambda*violations outweighing tau*log q?
Setup: dimers, cfast, 60 s training per run, one seed (0), compile warm, GPU shared with other agents (so updates per run differ: 462 to 1050). Logged every 5 updates on 4 held-out raw worlds: orientation fractions (h, v, diag), entropy = -score/cells (nats per cell, generation at tau = 1), raw violation fraction per hard factor. Final row on 16 worlds, raw and repaired. Scripts: `run.py`, `an.py`; data `run_*.json`, `log_*.txt`.

## Configs (appended to `castlegen/variants/cfast.py`, cfg overrides only)
| variant | overrides | meaning |
| --- | --- | --- |
| cfast | none | lam0 = 10, eta = 0.01 control, tau 2 -> 1 over 20 %, lr 3e-3 |
| cfast_lam1 / lam4 / lam20 | lam0 = 1 / 4 / 20, eta = 0 | lambda fixed (control off) |
| cfast_tau2_lam4 | lam0 = 4, eta = 0, tau1 = 2 | tau = 2 for the whole run (tau0 = tau1 = 2), lambda fixed 4 |
| cfast_lr1e3 | lr = 1e-3 | actor lr 1e-3, defaults otherwise (same cfg as cfast_s64) |

## Results
Budget = 2e-4 on repaired violations per hard factor. "Diag < 0.05 at" is the first logged update (resolution 5, first log at update 1).

| run | diag < 0.05 at | updates | final h / v / diag (raw) | entropy | raw viol | repaired viol | meets 2e-4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| defaults | update 10 | 1050 | 0.519 / 0.481 / 0.000 | 0.221 | 0 | 0 | yes |
| lambda 1 | survives | 934 | 0.333 / 0.336 / 0.332 | 0.800 | 4.35e-2 | 3.64e-2 | no |
| lambda 4 | survives | 821 | 0.440 / 0.288 / 0.272 | 0.596 | 1.89e-2 | 1.75e-2 | no |
| lambda 20 | update 10 | 487 | 0.076 / 0.924 / 0.000 | 0.078 | 3.9e-4 | 3.9e-4 | no (3.9e-4) |
| tau 2, lambda 4 | survives | 462 | 0.365 / 0.338 / 0.297 | 0.756 | 3.58e-2 | 3.08e-2 | no |
| lr 1e-3 | update 65 | 594 | 0.891 / 0.109 / 0.000 | 0.104 | 7.0e-4 | 6.8e-4 | no (7e-4, still falling) |

Reference: untrained is 0.36/0.32/0.32, 0.78 nats, 4.0 % violations. Max entropy of this model class is about 0.78 at the untrained policy, so 0.80 for lambda 1 is the untrained value.

Figure: `images/cdim_dimers.png` (diagonal fraction, entropy, raw violation vs update, symlog x).

## Answer
Every run that cuts violations does so by trading entropy, and no setting gets both. Entropy and violation move together along one curve: 0.80 nats / 4.4 % (lambda 1), 0.76 / 3.6 % (tau 2, lambda 4), 0.60 / 1.9 % (lambda 4), 0.22 / 0 (defaults), 0.08 / 0.04 % (lambda 20). Orientations survive only where violations have barely fallen.

- The penalty does drive the collapse. Fixed lambda 1 and tau = 2 with lambda 4 keep all three orientations at about a third and entropy at the untrained 0.78 for the whole run, but the policy learns essentially nothing (violations 4.4 % and 3.6 %, no better than untrained 4.0 %). Lambda 4 is in between: violations fall slowly (3.9 % to 1.9 %), diagonal 0.31 to 0.27, entropy 0.78 to 0.60, still sliding at update 820 (not collapsed, not converged).
- Lambda 20 and the defaults (lambda 10 rising to 11.8) kill the diagonal within 10 updates; lambda 20 collapses further (92 % vertical) and still misses the budget at 3.9e-4.
- Lower lr (1e-3) only delays the loss of the diagonal (update 65 instead of 10) and ends in horizontal stripes (0.89) with violations 7e-4 and falling.
- So the stripes do not win at every lambda: at lambda 1 nothing wins because nothing is learned. Lambda is the knob that decides when stripes arrive; it does not reveal a lambda where learning the true local structure works. Within lambda in [0, 20] and tau in {1..2}, I found no setting where violations fall below ~2 % with the diagonal alive.
- The next suspect (one-step lookahead plus colour order making stripes the only locally learnable zero-violation strategy) is consistent with this but not tested here: the evidence is that once violations fall appreciably, the policy that does it is a stripe policy. A test would be an exact-gradient or larger-capacity fit at fixed lambda 4 (does the mixed-orientation policy reach 0 violations at all?), or the lambda 4 run extended; both are outside this 60 s box.

## Caveats
One seed, 60 s each, GPU shared (update counts differ by 2x, so compare against update, not time). Lambda 4 and tau 2 may collapse later. The entropy is the sampler's own -log q at tau = 1, not the true entropy. Training-time tau in the lambda runs follows the default 2 -> 1 anneal; only cfast_tau2_lam4 holds it at 2 (its eval is still at tau = 1). No constraint-7 issue: all lambda <= 20; the 2e-4 misses are reported as misses.
