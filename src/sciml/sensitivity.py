from functools import wraps

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def count_calls(f):
    @wraps(f)
    def wrap_f(*args, **kwargs):
        wrap_f.n_calls += 1
        return f(*args, **kwargs)
    wrap_f.n_calls = 0
    return wrap_f

def first_order_sensitivities(Q, variables, Q0, eps=1e-6):
    x0 = np.array([v.mean() for v in variables])
    dQ_dx = np.zeros(len(x0))
    for i, xi in enumerate(x0):
        xp = x0.copy()
        # Use absolute step size when xi is zero to avoid division by zero
        h = eps * abs(xi) if xi != 0 else eps
        xp[i] = xi + h
        dQ_dx[i] = (Q(xp) - Q0)/h
    return dQ_dx

def first_order_variance(dQ_dx, Sigma):
    var_Q = dQ_dx.T @ Sigma @ dQ_dx
    sigma_Q = np.sqrt(var_Q)
    return var_Q, sigma_Q

def second_order_sensitivities(Q, variables, Q0, eps=1e-4):
    x0 = np.array([v.mean() for v in variables])
    dQ2_dx2 = np.zeros((len(x0), len(x0)))

    for i, xi in enumerate(x0):
        for j, xj in enumerate(x0):
            if i == j:
                # diagonal matrix entries
                # Use absolute step size when xi is zero to avoid division by zero
                hi = eps * abs(xi) if xi != 0 else eps

                xp = x0.copy()
                xp[i] = xi + hi

                xm = x0.copy()
                xm[i] = xi - hi

                dQ2_dx2[i,i] = (Q(xp) - 2*Q0 + Q(xm))/hi**2

            elif i > j:
                # off-diagonal matrix entries, noting symmetry on swapping of i and j
                # Use absolute step size when xi/xj is zero to avoid division by zero
                hi = eps * abs(xi) if xi != 0 else eps
                hj = eps * abs(xj) if xj != 0 else eps

                x1 = x0.copy()
                x1[i] += hi
                x1[j] += hj

                x2 = x0.copy()
                x2[i] += hi
                x2[j] -= hj

                x3 = x0.copy()
                x3[i] -= hi
                x3[j] += hj

                x4 = x0.copy()
                x4[i] -= hi
                x4[j] -= hj

                dQ2_dx2[i,j] = (Q(x1) - Q(x2) - Q(x3) + Q(x4))/(4*hi*hj)

    # fill upper triangle
    i_upper = np.triu_indices(len(x0))
    dQ2_dx2[i_upper] = dQ2_dx2.T[i_upper]

    return dQ2_dx2

def taylor_expansion(x, variables, Q0, dQ_dx, d2Q_dx2=None):
    """
    Taylor expansion to linear or quadratic order
    """
    x0 = np.array([v.mean() for v in variables])
    delta = x - x0
    Q = Q0 + dQ_dx.T @ delta
    if d2Q_dx2 is not None:
        Q += 0.5*(delta.T @ d2Q_dx2 @ delta)
    return Q

def plot_sensitivities(f, variables, variable_names,
                       Q0=None, dQ_dx=None, d2Q_dx2=None,
                       second_order=False, logscale=False):
    x0 = np.array([v.mean() for v in variables])
    sigma0 = np.array([v.std() for v in variables])
    if Q0 is None:
        Q0 = f(x0)
    if dQ_dx is None:
        dQ_dx = first_order_sensitivities(f, x0, Q0)
    df = pd.DataFrame.from_dict(dict(parameter=variable_names,
                                     mean=x0,
                                     sigma=sigma0,
                                     sensitivity=dQ_dx,
                                     scaled_sensitivity=x0*dQ_dx,
                                     sensitivity_index=sigma0*dQ_dx))

    if second_order:
        if d2Q_dx2 is None:
            d2Q_dx2 = second_order_sensitivities(f, x0, Q0)
        for i, p1 in enumerate(variable_names):
            for j, p2 in enumerate(variable_names):
                if i < j:
                    continue
                df.loc[len(df)] = {'parameter': f'{p1}{p2}',
                                   'mean': x0[i]*x0[j],
                                   'sigma': sigma0[i]*sigma0[j],
                                   'sensitivity': d2Q_dx2[i,j],
                                   'scaled_sensitivity': x0[i]*x0[j]*d2Q_dx2[i,j],
                                   'sensitivity_index': sigma0[i]*sigma0[j]*d2Q_dx2[i,j]}

    ax = df['scaled_sensitivity'].abs().plot(xticks=df.index, logy=logscale)
    ax.set_xticklabels(df.parameter, rotation=45)
    ax.set_ylabel('Absolute scaled sensitivity')
    ax.set_xlabel('Parameter')
    return df, ax

def plot_response(f, variables, variable_names,
                  Q0=None, dQ_dx=None, d2Q_dx2=None):
    x0 = np.array([v.mean() for v in variables])
    sigma0 = np.array([v.std() for v in variables])
    if Q0 is None:
        Q0 = f(x0)
    if dQ_dx is None:
        dQ_dx = first_order_sensitivities(f, x0, Q0)
    if d2Q_dx2 is None:
        d2Q_dx2 = second_order_sensitivities(f, x0, Q0)

    for i, parameter in enumerate(variable_names):
        mu = x0[i]
        sigma = sigma0[i]

        # evaluate 10 points in range (mean - 2*stdev, mean + 2*stdev)
        X_exact = np.linspace(mu-2*sigma, mu+2*sigma, 10)
        Q_exact = []
        for value in X_exact:
            x = x0.copy()
            x[i] = value
            Q_exact.append(f(x))

        # now do Taylor expansion - use more points to get smooth curves
        X = np.linspace(mu-2*sigma, mu+2*sigma, 100)
        Q_taylor_1 = []
        Q_taylor_2 = []
        for value in X:
            x = x0.copy()
            x[i] = value
            Q_taylor_1.append(taylor_expansion(x, variables, Q0, dQ_dx))
            Q_taylor_2.append(taylor_expansion(x, variables, Q0, dQ_dx, d2Q_dx2))

        plt.subplot(4, 3, i+1)
        plt.plot(X_exact, Q_exact, 'kx', label='Exact')
        plt.plot(X, Q_taylor_1, 'r-', label='First order')
        plt.plot(X, Q_taylor_2, 'b--', label='Second order')
        plt.xlabel(parameter)

    plt.subplots_adjust(hspace=0.5)

def build_data_matrix(X, y, x0, second_order=False):
    I, J = X.shape # I = number of samples, J = number of variable_names
    N = J # default is first-order, number of columns = number of variable_names
    if second_order:
        # add columns for the second derivatives
        N += J + J*(J-1)//2

        # construct a map from parameter index (j1, j2) to order used by plot_sensitivities()
        index_map = {}
        n = 0
        for j1 in range(J):
            for j2 in range(J):
                if j1 < j2:
                    continue
                index_map[(j1,j2)] = n
                n += 1

    Xt = np.zeros((I, N))

    # first J entries in Xt correspond to the first derivatives
    for j in range(J):
        Xt[:, j] = (X[:, j] - x0[j]) / x0[j]

    if second_order:
        # next we have J single-variable second derivatives, note factor of 1/2
        for j in range(J):
            idx = J + index_map[(j,j)]
            #print(idx, variable_names[j], variable_names[j])
            Xt[:, idx] = 0.5*((X[:, j] - x0[j])/x0[j])**2

        # finally, J*(J-1)/2 mixed-variable second derivatives
        for j in range(J):
            for jp in range(j):
                idx = J + index_map[(j, jp)]
                #print(idx, variable_names[j], variable_names[jp])
                Xt[:, idx] = ((X[:, j] - x0[j])*(X[:, jp] - x0[jp]) /
                                     (x0[j] * x0[jp]))

    return Xt
