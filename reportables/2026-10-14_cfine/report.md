# cfine: coupled enumeration at the fine levels (2026-10-14)

Agent: cfine (chat / csat agent's proposal 1). Code: `castlegen/variants/cfine.py`, `_cfine_core.py`, `tests/variants/test_cfine.py`. Scripts and run outputs are in this folder. Cores 16-23. **The GPU was shared** with 3-5 other agents' jobs, so every timing is contended.

## Answer

**The estimator works, but training on ground does not change.** Branching the fine levels raises the per-world gradient SNR at levels 0-2 by ×400 to ×10 000, and the gradient means agree with cfast's. But at equal updates (300) the fine levels **do not become parent-aware** in any useful way, and **the fill-context cost does not drop** (cfine lr 1e-3: 1.85 / 2.03 nats per fine cell at L = 4 / 3; the cfast estimator at the same lr: 1.87 / 1.74). The draws do not follow the fill in any run (line_match 0.50). One cfine update costs 1.1-1.2 s, against 0.17 s for the cfast estimator. The dimers pre-check also failed: one orientation still dies, as with cfast. So I did not run the bench (gate 4).

## 1. What was built

- **Branch levels.** cfine branches any tuple of levels (`blevs`, default all of 4..0) with R_l draws per world and level. Each branched draw gets K = 4 actions: all of A when |A| ≤ K, otherwise the drawn action plus K − 1 draws from q with the leave-one-out baseline. This is creplay's estimator: t1 is kept for every draw, every t2 at a branched level is dropped, and the coupled term c_i is added.
- **Non-forced selection.** R_l draws are chosen uniformly among the N_nf valid draws with |A| > 1, each weighted N_nf / min(R_l, N_nf).
  - A forced draw has ∇q ≡ 0, so its replay term is identically zero, and the subset sum over non-forced draws equals the full sum pointwise.
  - The selection depends on the trajectory but not on the replay randomness, so the estimator stays unbiased (hard constraint 3).
  - This matters because ground's level 0 is 96 % forced at init and 99 % after training. Uniform selection would waste almost every replay.
- **Windowed replay** (levels ≤ `wlev`). The replay starts from the original pass's final state, with every cell drawn at a step ≥ base[l_b] reset to unknown. At each later step it recomputes only the sites inside the branched draw's cone box, padded to a static size per (branch level, level, channel). Every other site of the step gets its original value back: it lies outside the cone, so a full replay would reproduce it bit for bit. G^a is the sum of the recomputed sites' costs. Noise is keyed by coordinate, so each recomputed draw uses the same Gumbels as a full replay.
- creplay's defaults are unchanged. A second registered variant, `cfine_fineonly`, branches levels 2-0 only.

## 2. Gate 1: tests (CPU, all pass at the first run; no tolerance changed)

- `test_window_faithful`, **2/2** (toy_support and ground, 4 min).
  - For selected draws at the windowed levels, the windowed replay's final state equals the full override replay's final state **bit for bit** in every cell of every sampled channel.
  - G^a equals the full replay's cone-box sum (rel 1e-4).
  - The batched windowed returns equal the batched full-world returns for the same draws and actions.
- `test_cfine_unbiased`, **7/7** on the tiny enumerable problem, 300 × 16 worlds, 4 min. The configurations:
  - the fine level alone, every draw enumerated, windowed;
  - a non-forced subset of the fine level (R = 3);
  - both levels as non-forced subsets with LOO at the top, windowed and full-world;
  - creplay's loo_top;
  - creplay's uniform selection, windowed;
  - a wrong fixed critic.

## 3. Cost of a replay: windowed vs full (ground, B = 8, R = 16, K = 4; `timing_levels.json`)

| branched level | full-world replay | windowed | cone size (later draws, mean / of all later) |
| --- | --- | --- | --- |
| 2 | 1.49 ms | 1.53 ms | 8206 / 13312 |
| 1 | 1.35 ms | **0.84 ms** | 2520 / 12288 |
| 0 | 0.96 ms | **0.079 ms (×12)** | 131 / 8192 |

![](images/replay_timing.png)

- The window pays off at level 0 and a little at level 1. At level 2 the padded box is nearly the whole world, so the gate runs use `wlev = 1`: levels 2-4 get full-world replays.
- Whole update (`timing.py`), all levels with R = (2, 2, 8, 16, 32) and 1920 replays: **1.55 s windowed vs 3.33 s full**. creplay's default measured 0.29 s in the same session; it reported 0.18 s earlier with less contention.
- The gate configuration, R = (2, 2, 4, 8, 32) with 1600 replays, ran at **1.1-1.2 s/update** in training; the cfast estimator ran at 0.17 s.
- Remaining costs: full replays at levels 2-4 (320 × ~1-1.5 ms) and windowed level 1. The padding uses the maximum box over all draws and colours. Per-colour padding would trim it further; I did not build that.

## 4. Gradient SNR per world (`snr.py`, 24 worlds, B = 1, critic zeroed)

Same worlds and parameters; cfast estimator vs cfine (R = (2, 2, 4, 8, 32)).

| level | SNR cfast, init | SNR cfine, init | SNR cfast, trained* | SNR cfine, trained* |
| --- | --- | --- | --- | --- |
| 4 | 0.010 | 42 | 0.013 | 245 |
| 3 | 0.014 | 23 | 0.043 | 436 |
| 2 | 0.006 | 25 | 0.004 | 1.8 |
| 1 | 0.011 | 85 | 0.056 | 49 |
| 0 | 0.012 | 65 | 0.003 | 28 |

\* At the cfast estimator's parameters after 300 updates (`runs/cfast300`).

- Means agree, with diff²/se² of 0.5-1.3 at init. At the trained parameters the values are 0.1-1.2, except 5.3 (level 1) and 4.1 (level 3); the se there is dominated by cfast's heavy-tailed variance over only 24 worlds.
- At the trained parameters, only 19 % of the selected level-1 draws and 3 % of the level-2 draws are enumerable (mean |A| 4.6 and 16 with K = 4). The rest use LOO samples from q. **When q is nearly deterministic, those samples are almost all the same action, so the unvisited actions still get almost no signal at levels 1-2.** Only level 0 (|A| = 2) is fully enumerated.

## 5. Gate 2: the fast loop

### Dimers pre-check (60 s, 778 updates; `dimers_cfine.json`)
Dimers has one level, so every branch is a fine-level enumeration.

| update | h | v | diag | −log q per cell |
| --- | --- | --- | --- | --- |
| 1 | 0.370 | 0.304 | 0.326 | 0.78 |
| 53 | 0.480 | 0.071 | 0.449 | 0.52 |
| 126 | 0.244 | 0.0005 | 0.756 | 0.17 |
| 778 | 0.450 | **0.000** | 0.550 | 0.22 |

The diagonal survives, but **vertical dies instead**, by about update 126, and the entropy falls to 0.22 nats per cell, the same as cfast (0.225). The collapse to two orientations is not prevented, only relabelled. (cfast kills the diagonal within about 20 updates.)

### Ground: 120 s at equal time, then 300 updates at equal count
At equal time, cfine reached 124 updates in 120 s and cfast reached 1508. That comparison says nothing about the estimator (see `images/gate2_entropy_120s.png`). The comparison that does is at equal updates: cfine against the cfast estimator in the same code (`creplay_b0_cap300`), both stopped at 300 updates, at the default lr 3e-3 and at lr 1e-3.

| run (300 updates) | s/update | dirt | H non-forced L4 / L3 / L2 / L1 / L0 (nats) | raw viol L0 / L1 / L2 / L3 | L = 4: own / fill | L = 3: own / fill | rho3 / rho2 / rho1 changed by probe clamp | line_match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cfine, lr 3e-3 | 1.21 | 0.913 | 1.55 / 1.55 / 1.14 / 0.81 / 0.02 | 0.004 / **0.060 / 0.042 / 0.060** | 2.77 / 2.40 | 2.27 / 1.90 | 2.4 % / 0.4 % / **18.5 %** | 0.501 |
| cfast est., lr 3e-3 | 0.17 | 0.966 | 0.28 / 0.17 / 0.12 / 0.10 / 0.05 | 0.000 / 0.002 / 0.003 / 0.001 | 1.12 / 1.99 | 0.66 / 2.13 | 0.6 % / 0.1 % / 0.1 % | 0.500 |
| cfine, lr 1e-3 | 1.08 | 0.919 | 2.78 / 2.50 / 1.52 / 1.02 / 0.11 | 0.001 / 0.006 / 0.009 / 0.019 | 1.24 / **1.85** | 1.11 / **2.03** | 6.2 % / 1.1 % / 1.4 % | 0.503 |
| cfast est., lr 1e-3 | 0.17 | 0.957 | 2.29 / 2.46 / 2.17 / 1.20 / 0.13 | 0.001 / 0.006 / 0.006 / 0.010 | 1.75 / 1.87 | 1.36 / 1.74 | 3.3 % / 2.7 % / 5.5 % | 0.505 |

Reference: cfast (150 s, about 1400 updates) 1.97 vs 0.72 and 2.23 vs 0.34; the probe's own cost is 0.014.

![](images/gate2_entropy_eq300.png)
![](images/gate2_entropy_eq300_lr1.png)
![](images/clamp_gate_eq300.png)

- **Stochastic? Partly, at lr 3e-3 only, and not cleanly.** At the default lr, cfine keeps 0.8-1.6 nats at levels 1-4, where the cfast estimator has fallen to 0.1-0.3. But that run is unstable:
  - after a good start (cost per world 8.1k at update 31), raw violations rise to 4-6 % at levels 1-3 and λ climbs;
  - its own-context fine cost (2.77) ends **above the untrained 2.47**.

  At lr 1e-3, both estimators stay stochastic at 300 updates, and cfast's levels 1-2 are even a little more stochastic than cfine's. The extra entropy at lr 3e-3 looks like a by-product of the instability, not a cure for saturation. This matches creplay's level-3 oscillation: a much cleaner gradient under Adam at the same lr takes larger effective steps.
- **Parent-aware? No.** The only large effect is rho1 changing in 18.5 % of draws under the probe clamp, in the unstable lr 3e-3 run. At lr 1e-3 cfine changes 1-6 %, against 3-6 % for cfast at the same lr. Neither is meaningfully above the untrained 0-2 %.
- **Fill-context cost? No drop.** cfine's fill cost is 1.85 / 2.03 against cfast's 1.87 / 1.74 at lr 1e-3. In the lr 3e-3 run fill sits below own (2.40 vs 2.77), but only because the own contexts got worse, not because fill contexts got better.
- Every own draw is 91-97 % dirt, and line_match is 0.50 in every run.

## 6. Gate 4 (bench): not run
Gate 2 shows no movement on the gate metrics. At 1.1-1.2 s/update, a 300 s bench gives cfine about 250 updates against cfast's 2000 or more.

## Failures and caveats
- One training seed per arm. Training stopped at 300 updates, where collapse is only beginning for cfast (its dirt reaches 1.0 by update ~300 in the 120 s run).
- The SNR comparison was measured on only 24 worlds.
- The dimers check was one seed at 60 s.
- Contended GPU: the s/update figures move by ±50 % between runs.

## What I'd do next
1. **Cover unvisited actions when |A| > K** (an allowed freedom: enumerating or sampling small candidate sets). Draw the K actions from a proposal that does not collapse with q, e.g. uniform over A, and weight by q / proposal: Σ_a ∇q(a)(G^a − Q(a)) ≈ (|A| / K) Σ_k ∇q(a_k)(G^{a_k} − Q(a_k)), which is unbiased. Or raise K to 5 at level 1, since rho1 has D = 5 and every draw would then be enumerable. Today only level 0 is fully enumerated in trained states, so the core claim, a gradient for unvisited actions, holds only there.
2. **Use a lower lr with clean gradients.** Pair cfine with lr 1e-3 or lower at the branched levels; the lr 3e-3 run diverged in violations.
3. **Cheaper replays.** Use per-colour box padding, and fold the levels 2-4 full replays into the windowed path with tighter padding.
4. Even with all of this, ideas #12 / #14 (visiting fill contexts) is probably needed: the gradient only acts in contexts that training visits, and those are all dirt by update ~200.

## Files
- `timing.py`, `timing_levels.py`, `cone_sizes.py`, `snr.py` (with `snr_ground_init.json` and `snr_ground_cfast300.json`), `train_gate.py` (any variant plus csat's diagnostics), `dimers_check.py`.
- Clamp-gate scripts copied from csat: `eval_ground.py`, `clamp_tools.py`, `same_check.py`, `gate.py`, `line_match.py`.
- `plot_gate.py`, `plot_clamp.py`, `plot_timing.py`.
- Per-run outputs in `runs/<run>/`: train logs, params, `ground_summary.json`, `same_check.txt`, and clamped example images in `runs/<run>/images/`.
