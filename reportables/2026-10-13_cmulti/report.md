# cmulti (ideas #3): stopped early

2026-10-13 · `castlegen/variants/cmulti.py` (`cmulti` with P = 2; also `cmulti_p1`, `cmulti_p3`), `_cmulti_core.py`, `tests/variants/test_cmulti.py` (11 pass on the box). The agent's text, condensed by the orchestrator.

**Needs your approval before cmulti can be a candidate (constitution, hard constraint 4):** it optimises an upper bound, not the stated objective.

## Generation

- At each level, coarse to fine, pass 1 is cfast's 16-colour pass with lookahead.
- Passes 2..P sweep the colours again and redraw every site given all its current neighbours. Each (level, channel, pass) has its own actor.
- Then the repair round runs.
- Finite support holds, with the margin scaled by P. Same-colour sites sharing no factor is checked generically.

## Objective (an upper bound)

log q of the final draw is intractable once sites are redrawn. cmulti minimises

J = E_q[S(x^P) + λc(x^P) + τ(log q(x^1..x^P) − log r(x^{<P} | x^P))] = stated objective + τ·E KL(q(x^{<P} | x^P) ‖ r).

- r is a learned per-site backward kernel, fit by maximum likelihood.
- Per-draw costs add up exactly to J's integrand (tested), and the returns use generalised cones.
- The gradient is unbiased for J, but not for the true reverse KL.
- Because r is per-site, the bound is loose wherever a later pass mixes, which discourages later passes from changing things.

## Results

| problem | variant | worst final viol | notes |
| --- | --- | --- | --- |
| toy_support | cfast (GPU) | 0 / 0 | F −0.432 / −0.441 |
| toy_support | cmulti | 0 (seed 0) | bench F missing (no score) |
| dimers | cmulti (smoke, 60 s) | 0 | 100 % vertical, fully collapsed |

**No evidence that multi-pass breaks the collapse.**
- In a valid perfect matching every site is fixed by its neighbours, so single-site redraws are frozen; pass 2 can at best act as a learned repair.
- Columns should behave the same way (untested).

The dimers and columns benches died at start-up because of the box's thread limit.

## Box state

- **Running:** `scripts/chain.sh` (setsid + nohup) was on cmulti toy_support seed 1, with logs in `/tmp/cmulti_*.log`.
- **Not started:** `scripts/chain2.sh` (cfast then cmulti on dimers and columns, `taskset -c 0-15`, with a retry).

## Next

1. Run chain2.sh.
2. Add generic block moves to the later passes, redrawing all members of a factor instance together.
3. Parameterise the colour lattice, to try fewer colours.
