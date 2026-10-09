# Notes index

Reading order for the current design, then the test notes with their
results, then reading and history.  Dates are when a note was last
brought up to date.

## Current design

| note | what |
|---|---|
| `constitution.md` | The goals behind every design decision.  Start here. |
| `handoff.md` | Where things stand, what is not yet in git and how to bring it in, what to do first. |
| `todo.md` | Every future addition under consideration, grouped, with sources. |
| `dsl_updates.md` | The committed design (C1 samplers, C2 learned potentials and training, C3 documentation, C4 constraints among latents, C5 scaling the library), what the build measured, the deferred bias field, open questions. |
| `dsl_interface.md` | The programmatic shape: Channel, Local, Bias, Features, Targets, Trainer, the generate loop and the training algorithm in pseudocode. |
| `reference_math.tex` (`.pdf`) | The mathematics of the current DSL: objects, the two models and the free-energy identity, the sampler, AIS and paint-potential training, convpot, self-play, toy closed forms. |
| `reference.md` | Compact reference of the current DSL with a code map. |
| `channels_system.tex` | Research questions around the channel system; partly current. |

## Test notes (model, interfaces, experiment, results)

In the order they were run; each builds on the previous.

| note | what it established |
|---|---|
| `channels_demo.md` | The first channel sets: ground and roots. |
| `potts_test.md` | Self-play training of coarse tables on a three-level Potts hierarchy; the S0 to S3 schemes and the "leak". |
| `circles_test.md` | Induced repulsion between objects with footprints; exact free energies. |
| `induce_test.md` | The window free-energy estimator with AIS. |
| `convpot_test.md` | Learned conv potentials over value embeddings. |
| `paintpot_test.md` | Paint potentials: the induced free energy as a local energy over the painted map, exact on held-out pairs. |
| `circles_biome_test.md` | The next-version DSL built and validated: per-channel samplers, support from stamps, the bootstrap trainer with collapsed targets, stamp versus learned features, scaling, migration of the real channel sets, the boundary, the Potts leak explained. |

## Reading

`reading.md`: literature notes (renormalisation multigrid, inverse Monte
Carlo RG, coarse-graining), with consequences for the design.

## History

`history/`: superseded designs, each with a status line naming what
replaced it.  The generations, oldest first: JAX castles (the original
README), promises and texture synthesis (`promises.md`,
`promise_estimation.md`, `texture_sampling.md`, `coarse_to_fine.tex`),
the generic joint sampler (`generic.md`, `joint_sampler.md`), macro
objects (`macro_objects.md`, `dsl.md`), block connectivity and the bit
sampler (`port_gibbs.md`), the first channels design (`channels.tex`).
The code for these is `castlegen/legacy/`.

## Experiments

`experiments/`: the scripts behind the current test notes, named after
them; `experiments/legacy/` for the earlier generations.  Every script
writes to `images/`, which is gitignored.
