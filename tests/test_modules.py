"""Smoke tests for the support modules, with known answers where available."""

import numpy as np
import pytest


def test_blr_polynomial_least_squares_recovers_coefficients():
    from sciml.blr import PolynomialBasis, design_matrix, least_squares_MLE

    X = np.linspace(-1, 1, 20)[:, None]
    y = 0.5 - 2.0 * X[:, 0] + 3.0 * X[:, 0] ** 2
    Phi = design_matrix(X, PolynomialBasis(2))
    assert Phi.shape == (20, 3)
    w = least_squares_MLE(Phi, y)
    w = w[0] if isinstance(w, tuple) else w
    np.testing.assert_allclose(np.ravel(w), [0.5, -2.0, 3.0], atol=1e-10)


def test_sensitivity_first_order_of_quadratic():
    import scipy.stats as st
    from sciml.sensitivity import first_order_sensitivities

    Q = lambda x: x[0] ** 2 + 3 * x[1]
    variables = [st.norm(2.0, 0.1), st.norm(1.0, 0.1)]
    g = first_order_sensitivities(Q, variables, Q([2.0, 1.0]))
    np.testing.assert_allclose(g, [4.0, 3.0], rtol=1e-5)


def test_poiseuille_flow_rate_matches_analytic():
    from sciml.fluidsolvers import Q_analytic, poiseuille_solver_1

    F, W = 1.0, 2.0
    assert poiseuille_solver_1([F, W], N=2000) == pytest.approx(Q_analytic(F, W), rel=1e-3)


def test_quad_solver_returns_three_stress_components():
    from sciml.femsolvers import quad_solver

    sigma = quad_solver([200e9, 0.3, 0.01, -1e3])
    assert np.shape(sigma) == (3,) and np.all(np.isfinite(sigma))


def test_progress_tqdm_iterates():
    from sciml.progress import trange

    assert sum(trange(5, disable=True)) == 10


def test_gp_fit_runs():
    pytest.importorskip("tinygp")
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    from tinygp import GaussianProcess, kernels

    from sciml.gputils import gp_fit

    jax.config.update("jax_enable_x64", True)
    X = jnp.linspace(0, 5, 20)
    y = jnp.sin(X)

    def build_gp(params, X):
        k = params["variance"] * kernels.ExpSquared(params["lengthscale"])
        return GaussianProcess(k, X, diag=params["noise"])

    params = {"variance": 1.0, "lengthscale": 1.0, "noise": 1e-2}
    negative_mll, learned = gp_fit(jr.PRNGKey(0), build_gp, X, y, params)
    assert float(negative_mll(learned)) < float(negative_mll(params))


def test_atomistic_vacancy_formation_energy():
    pytest.importorskip("atomistica")
    from sciml.atomistic import Tersoff_parameter_values, vacancy_formation_energy

    assert vacancy_formation_energy(Tersoff_parameter_values) == pytest.approx(4.1025, abs=1e-3)
