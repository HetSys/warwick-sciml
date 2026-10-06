import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jax import grad, jit
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize


def _show(obj):
    """Display a DataFrame or HTML string in marimo, Jupyter, or a terminal."""
    try:
        import marimo as mo

        if mo.running_in_notebook():
            mo.output.append(mo.Html(obj) if isinstance(obj, str) else obj)
            return
    except ImportError:
        pass
    try:
        from IPython.display import HTML, display

        display(HTML(obj) if isinstance(obj, str) else obj)
    except ImportError:
        print(obj)


def gp_print(params, negative_mll, grad_negative_mll=None):
    """
    Print the tingp Gaussian Process with parameter set `params`.

    Arguments:

    params: dict
        Dictionary of parameters
    negative_mll: func
        Negative marginal log likelihood function
    grad_negative_mll: func
        Gradient of negative_mll. If not present, computed with `jax.grad()`.
    """
    if grad_negative_mll is None:
        grad_negative_mll = grad(negative_mll)
    _show(f"<b>Negative log marginal likelihood:</b> {negative_mll(params):.3f}")

    def params_to_dict(params):
        D = {}
        for k, v in params.items():
            if isinstance(v, jnp.ndarray):
                if v.shape == ():
                    v = float(v)
                    D[k] = v
                elif v.shape == (1,):
                    D[k] = v[0]
                else:
                    for j, v2 in enumerate(v):
                        D[f"{k}[{j}]"] = v2
            else:
                D[k] = v
        return D

    df = pd.DataFrame.from_dict(
        {
            "parameter": params_to_dict(params),
            "gradient": params_to_dict(grad_negative_mll(params)),
        }
    )
    _show(f"<b>Number of parameters:</b> {len(df)}")
    _show(df)


def gp_plot(
    cond_gp,
    X,
    y,
    X_p,
    legend=True,
    ax=None,
    true_func=None,
    epistemic_uncertainty=True,
    full_uncertainty=True,
):
    """
    Visualise a 1D GPJax Gaussian Process

    Arguments:

    cond_gp: tinygp.GaussianProcesss
        The conditioned GP
    X: array
        Input values at training points
    y: array
        target values
    X_p: array
        Input values at prediction points
    """

    return_ax = False
    if ax is None:
        return_ax = True
        fig, ax = plt.subplots()

    Y_p = cond_gp.mean.squeeze()
    sigma_tot = jnp.sqrt(cond_gp.variance)
    sigma_epi = jnp.sqrt(cond_gp.variance - cond_gp.noise.diag)
    ax.plot(X.squeeze(), y.squeeze(), "bx", label="Data")
    ax.plot(X_p, Y_p, label="Predictive mean", color="C0")
    if epistemic_uncertainty:
        ax.fill_between(
            X_p.squeeze(),
            Y_p - 2 * sigma_epi,
            Y_p + 2 * sigma_epi,
            color="C0",
            alpha=0.2,
            label="Epistemic uncertainty",
        )
    if full_uncertainty:
        ax.fill_between(
            X_p.squeeze(),
            Y_p - 2 * sigma_tot,
            Y_p - 2 * sigma_epi,
            color="C1",
            alpha=0.2,
            label="Total uncertainty",
        )
        ax.fill_between(
            X_p.squeeze(),
            Y_p + 2 * sigma_epi,
            Y_p + 2 * sigma_tot,
            color="C1",
            alpha=0.2,
        )
    if true_func:
        ax.plot(
            X_p,
            true_func(X_p),
            label="True function",
            color="black",
            linestyle="--",
            linewidth=1,
        )
    if legend:
        ax.legend()
    if return_ax:
        return fig, ax


def gp_fit(
    key,
    build_gp,
    X,
    y,
    params,
    randomise=False,
    verbose=False,
    num_iters=100,
    gtol=1e-5,
):
    # we just-in-time compile the objective function for speed

    if randomise:
        flat, unravel = ravel_pytree(params)
        # split the pseudo-random number generator key. NB: this is mandatory in JAX
        # or you'll keep getting the same random numbers.
        # See https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#jax-prng
        key, subkey = jr.split(key)
        params = unravel(jnp.exp(jr.normal(subkey, (len(flat),))))

    def negative_mll(params):
        gp = build_gp(params, X)
        return -gp.log_probability(y)

    if verbose:
        _show("<h3>Initial parameters</h3>")
        gp_print(params, negative_mll)

    f, df = negative_mll, grad(negative_mll)
    x0, unravel = ravel_pytree(params)
    x0 = jnp.log(x0)  # user-supplied params are not yet log transformed

    @jit
    def obj_and_grad(x):
        x_trans = jnp.exp(x)  # transform to ensure x_trans is always positive
        dx, _ = ravel_pytree(df(unravel(x_trans)))  # compute gradients
        return (
            f(unravel(x_trans)),
            dx * x_trans,
        )  # don't forget Jacobian of transformation

    res = minimize(
        obj_and_grad,
        x0,
        jac=True,
        method="BFGS",
        options={"disp": verbose, "maxiter": num_iters, "gtol": gtol},
    )
    learned_params = unravel(jnp.exp(res.x))

    if verbose:
        _show("<h3>Final parameters</h3>")
        gp_print(learned_params, negative_mll)

    return negative_mll, learned_params


def gp_fit_restarts(
    key, build_gp, X, y, params, num_restarts=5, return_all=False, **fitkwargs
):
    best_value = np.inf
    best_params = None
    results = []
    for i in range(num_restarts):
        key, subkey = jr.split(key)
        negative_mll, params = gp_fit(
            subkey, build_gp, X, y, params, randomise=True, **fitkwargs
        )
        value = negative_mll(params)
        global_min = ""
        if value < best_value:
            best_value = value
            best_params = params
            global_min = "(new global minimum)"
        print(f"Fit {i+1}/{num_restarts}: {value:.4f} {global_min} {params}")
        results.append((negative_mll, params))
    if return_all:
        return results
    else:
        return negative_mll, best_params


def repredict(gp, cond_gp, X_test, return_var=False, return_cov=False):
    assert gp.X is cond_gp.mean_function.X
    assert gp.kernel is cond_gp.mean_function.kernel
    k_star = gp.kernel(gp.X, X_test)
    mean = k_star.T @ cond_gp.mean_function.alpha
    if return_var or return_cov:
        v = gp.solver.solve_triangular(k_star)
        cov = gp.kernel(X_test, X_test) - v.T @ v
    if return_var:
        var = jnp.diag(cov)
        return mean, var
    if return_cov:
        return mean, cov
    return mean
