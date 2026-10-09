"""Legacy: the earlier generations of the project, superseded by castlegen/channels.

Kept runnable (tests/legacy, notes/experiments/legacy) but not developed.

G0   JAX castle sampler: annealed checkerboard Gibbs on tile signatures with a
     connectivity plan (core, tileset, base, schedule, e2e, render, preview,
     conn, refsampler, hier_probe).  Described in the original README.md (git
     history).
G1   Promises and texture synthesis: exemplar coordinates by parallel texture
     synthesis, promise variables with exact bottom-up summaries (connmetrics,
     exemplar, texsyn, hier, promise, corpus, plfit, pipeline, quantities/).
     notes/history/promises.md, promise_estimation.md, texture_sampling.md.
G1b  Coarse-to-fine promises: support envelopes and average heights with
     reference samplers (envelope_var, average_var, envpredict, heights,
     refheights, exchain).  notes/history/coarse_to_fine.tex.
G1c  Generic joint sampler over exemplar coordinates and promise values
     (generic, generic_nb, ports).  notes/history/generic.md, joint_sampler.md.
G1d  Macro objects: rigid multi-cell components with ports (objects,
     macroobj, macrocontact).  notes/history/macro_objects.md, dsl.md.
G2   Block connectivity precursors to the bit sampler (blockconn,
     blockfield).  notes/history/port_gibbs.md, coarse_to_fine.tex.
"""
