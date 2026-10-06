# warwick-sciml

Support modules for the University of Warwick module on Predictive Modelling,
Uncertainty Quantification and Scientific Machine Learning (PX914 / ES98E).

```bash
pip install warwick-sciml              # core: BLR, sensitivity analysis, fluid/FE solvers
pip install "warwick-sciml[gp]"        # + Gaussian process utilities (JAX, tinygp)
pip install "warwick-sciml[atomistic]" # + atomistic examples (ASE, atomistica, matscipy, JAX)
```

In a marimo notebook, add it to the PEP 723 header instead, e.g.

```python
# /// script
# dependencies = ["warwick-sciml[atomistic]>=2026.1", "marimo>=0.25"]
# ///
```

then `from sciml.blr import design_matrix`, `from sciml import tersoff_jax`, etc.

| Module | Contents |
|---|---|
| `sciml.blr` | Bayesian linear regression: basis functions, design matrices, posteriors |
| `sciml.sensitivity` | finite-difference sensitivities and Taylor expansions |
| `sciml.fluidsolvers` | Poiseuille flow solvers |
| `sciml.femsolvers` | 2D plane-stress finite element solver |
| `sciml.gputils` | Gaussian process fitting and plotting (extra `gp`) |
| `sciml.atomistic` | Tersoff calculators, vacancy formation energies (extra `atomistic`) |
| `sciml.tersoff_jax` | differentiable Tersoff potential in JAX (extra `atomistic`) |
| `sciml.progress` | tqdm wrappers for lecture notebooks |

Versions follow the academic year (`2026.x` for 2026/27).

## Development

```bash
uv run --extra test pytest
```
