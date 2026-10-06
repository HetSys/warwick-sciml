"""Support modules for the Warwick PX914/ES98E module on Predictive Modelling,
Uncertainty Quantification and Scientific Machine Learning.

Submodules (import them explicitly, e.g. ``from sciml.blr import design_matrix``):

- ``sciml.blr``           Bayesian linear regression: basis functions, design matrices, posteriors
- ``sciml.sensitivity``   finite-difference sensitivities and Taylor expansions
- ``sciml.fluidsolvers``  Poiseuille flow solvers
- ``sciml.femsolvers``    2D plane-stress finite element solver
- ``sciml.gputils``       Gaussian process fitting and plotting with JAX/tinygp   [extra: gp]
- ``sciml.atomistic``     Tersoff calculators and vacancy formation energies       [extra: atomistic]
- ``sciml.tersoff_jax``   differentiable Tersoff potential in JAX                  [extra: atomistic]
- ``sciml.progress``      tqdm wrappers suited to lecture notebooks
"""

__version__ = "2026.1.1"
