# Reading: forward chains fitted to a bidirectional energy

Literature for the scheme in channels_system.tex ("Research questions,
and the learnable components"): a forward procedure f_theta fitted to the
undirected joint q_beta = argmin KL(q || p_base) subject to requested
moments.  Citations were given from memory (2026-10-06); check years and
venues before citing.

Start with Shell (2008), then Pitera & Chodera (2012) for the beta side,
then Nijkamp et al. (2019) for the procedure-as-model caveat.

## Coarse tables learned from the fine level

Coarse-grained molecular simulation fits tabulated coarse pair potentials
so a coarse model reproduces a fine model's statistics; the learned
potential is the fine level's free energy (potential of mean force).  The
closest analogue of the sugar test.

- Lyubartsev & Laaksonen (1995), "Calculation of effective interaction
  potentials from radial distribution functions: a reverse Monte Carlo
  approach", Phys. Rev. E.  Inverse Monte Carlo: moment matching of a
  tabulated pair potential against target pair distributions; essentially
  `sugar.moment_step`.
- Reith, Pütz & Müller-Plathe (2003), "Deriving effective mesoscale
  potentials from atomistic simulations", J. Comput. Chem.  Iterative
  Boltzmann inversion, a cheaper heuristic version.
- Shell (2008), "The relative entropy is fundamental to multiscale and
  inverse thermodynamic problems", J. Chem. Phys.  Coarse-graining as
  minimising KL(fine || coarse); why the learned table is the induced free
  energy, and why knobs and tables come from one objective.
- Noid et al. (2008), "The multiscale coarse-graining method", J. Chem.
  Phys.; and Noid (2013), "Perspective: coarse-grained models for
  biomolecular systems", J. Chem. Phys.  What a coarse model can and
  cannot represent.

## Biasing a base distribution toward requested moments (beta)

- Pitera & Chodera (2012), "On the use of experimental observations to
  bias simulated ensembles", J. Chem. Theory Comput.  The minimum-KL
  correction matching given averages is one linear bias per observable:
  Signal B, one knob per request.
- White & Voth (2014), "Efficient and minimal method to bias molecular
  simulations with experimental data", J. Chem. Theory Comput.  Fits those
  multipliers online by stochastic updates during sampling.
- Ganchev, Graça, Gillenwater & Taskar (2010), "Posterior regularization
  for structured latent variable models", JMLR.  The same KL-plus-
  expectation-constraint projection in machine learning.
- Csiszár (1975), "I-divergence geometry of probability distributions and
  minimization problems", Annals of Probability.  I-projections.

## Maximum-entropy random fields for textures

- Zhu, Wu & Mumford (1997/1998), FRAME: "Minimax entropy principle and its
  application to texture modeling", Neural Computation; "Filters, random
  fields and maximum entropy", IJCV.  Matches feature histograms by Gibbs
  sampling with a multiplier per bin (tables, not knobs); minimax entropy
  also chooses the features.
- Della Pietra, Della Pietra & Lafferty (1997), "Inducing features of
  random fields", IEEE PAMI.  Greedy feature induction where the model's
  moments are worst: a principled form of "fit the knob, check the
  per-entry residuals, unlock the table".

## The procedure is the model

- Nijkamp, Hill, Zhu & Wu (2019), "Learning non-convergent non-persistent
  short-run MCMC toward energy-based model", NeurIPS.  Maximum-likelihood
  updates under a fixed short MCMC budget learn parameters right for that
  procedure, not for the equilibrium: the coefficients belong to a
  schedule.
- Xie, Lu, Gao, Zhu & Wu (2018), "Cooperative training of descriptor and
  generator networks", IEEE PAMI.  An energy model and a forward generator
  trained together; the closest ML analogue of fitting beta and theta at
  once.
- Tieleman (2008), "Training restricted Boltzmann machines using
  approximations to the likelihood gradient", ICML.  Persistent
  contrastive divergence, the update used in learn_sugar.py.

## What a coarse level of a random field is

- Gidas (1989), "A renormalization group approach to image processing
  problems", IEEE PAMI.
- Pérez & Heitz (1996), "Restriction of a Markov random field on a graph
  and multiresolution statistical image modeling", IEEE Trans. Inf.
  Theory.  The restriction of an MRF to a coarse level has induced,
  generally non-local interactions: the theory behind "coarse tables are
  implied by the fine level".
- Mehta & Schwab (2014), "An exact mapping between the variational
  renormalization group and deep learning", arXiv.
