import numpy as np
import matplotlib.pyplot as plt


def design_matrix(X, phi):
    """
    Arguments:

    X   -  The observed inputs
    phi -  The basis functions
    """
    num_observations = X.shape[0]
    num_basis = phi.num_basis
    Phi = np.zeros((num_observations, num_basis))
    for i in range(num_observations):
        Phi[i, :] = phi(X[i, :])
    return Phi


class LinearBasis:
    """
    Represents a 1D linear basis.
    """

    def __init__(self):
        self.num_basis = 2  # The number of basis functions

    def __call__(self, x):
        """
        ``x`` should be a 1D array of inputs
        """
        return [1.0, x[0]]


class PolynomialBasis:
    """
    A set of polynomial basis functions.

    Arguments:
    degree  -  The degree of the polynomial.
    """

    def __init__(self, degree):
        self.degree = degree
        self.num_basis = degree + 1

    def __call__(self, x):
        return np.array([x[0] ** i for i in range(self.degree + 1)])


class RadialBasisFunctions:
    """
    A set of linear basis functions.

    Arguments:
    X   -  The centers of the radial basis functions.
    ell -  The assumed lengthscale.
    """

    def __init__(self, X, ell):
        self.X = X
        self.ell = ell
        self.num_basis = X.shape[0]

    def __call__(self, x):
        return np.exp(-0.5 * (x - self.X) ** 2 / self.ell**2).flatten()


def least_squares_MLE(X, y):
    """Compute maximum likelihood estimate of mean and standard deviation of weights"""
    w_MLE, res_MLE, _, _ = np.linalg.lstsq(X, y, rcond=None)
    sigma_MLE = np.sqrt(res_MLE / X.shape[0])
    return w_MLE, sigma_MLE


def prior(alpha, N):
    """Compute mean and covariance matrices of the weight prior"""
    m0 = np.zeros(N)
    S0 = 1.0 / alpha * np.eye(N)
    return m0, S0


def posterior(Phi, y, alpha, beta, return_inverse=False):
    """Computes mean and covariance matrix of the posterior distribution."""
    S_N_inv = alpha * np.eye(Phi.shape[1]) + beta * Phi.T.dot(Phi)
    S_N = np.linalg.inv(S_N_inv)
    m_N = beta * S_N.dot(Phi.T).dot(y)
    m_N = m_N
    if return_inverse:
        return m_N, S_N, S_N_inv
    else:
        return m_N, S_N


def posterior_predictive(Phi_test, m_N, S_N, beta):
    """Computes mean and variances of the posterior predictive distribution."""
    y = Phi_test.dot(m_N).ravel()
    # Only compute variances (diagonal elements of covariance matrix)
    y_epi = np.sum(Phi_test.dot(S_N) * Phi_test, axis=1)
    y_var = 1 / beta + y_epi
    return y, y_epi, y_var


def plot_data(X, y):
    plt.plot(X[:, 0], y[:, 0], "kx", ms=10)


def plot_truth(X, y, label="Truth"):
    plt.plot(X[:, 0], y[:, 0], "k--", label=label)


def plot_posterior_samples(X, y):
    plt.plot(X, y, "r-")
    plt.axis("equal")


def plot_predictive(X, y, y_epi, y_var):
    sigma_epi = np.sqrt(y_epi)  # epistemitic uncertainty
    sigma_tot = np.sqrt(y_var)  # total uncertainty

    y_el = y - 2 * sigma_epi
    y_tl = y - 2 * sigma_tot
    y_eu = y + 2 * sigma_epi
    y_tu = y + 2 * sigma_tot

    plt.plot(X[:, 0], y, "b-", label="Prediction")
    plt.fill_between(
        X[:, 0], y_el, y_eu, color="C2", label="Epistemic uncertainty", alpha=0.3
    )
    plt.fill_between(
        X[:, 0], y_tl, y_el, color="C1", label="Total uncertainty", alpha=0.3
    )
    plt.fill_between(X[:, 0], y_eu, y_tu, color="C1", alpha=0.3)
