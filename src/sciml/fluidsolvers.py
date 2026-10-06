import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg

def poiseuille_solver_1(x, N=1000, return_all=False):
    F, W = x # unpack parameters
    F = F*np.ones(N) # vector for RHS
    F[0] = F[-1] = 0  # enforce boundary conditions on RHS
    h = 2*W/(N-1)
    A = sparse.dia_matrix((N, N)) # assemble matrix A

    # Build diagonals with boundary condition modifications
    main_dia = np.ones(N) * 2/h**2
    main_dia[0] = main_dia[-1] = 1  # enforce u[0] = u[-1] = 0
    upper_dia = np.ones(N-1) * -1/h**2
    upper_dia[0] = 0  # no coupling at left boundary
    lower_dia = np.ones(N-1) * -1/h**2
    lower_dia[-1] = 0  # no coupling at right boundary

    A.setdiag(main_dia)
    A.setdiag(upper_dia,  1)
    A.setdiag(lower_dia, -1)
    A = A.tocsc() # convert to CSC storage for efficient solve
    u = linalg.spsolve(A, F)
    Q = np.sum(u)*h
    if return_all:
        return u, F, Q
    else:
        return Q

def poiseuille_solver_2(x, N=1000, return_all=False):
    F_l, F_h, W = x # unpack parameters

    # build vector for F used for RHS of linear system
    y = np.linspace(-W, W, N)
    F = np.where((y > 0) & (y < W/2), F_l, F_h)
    F[0] = F[-1] = 0  # enforce boundary conditions on RHS

    h = 2*W/(N-1)
    A = sparse.dia_matrix((N, N)) # assemble matrix A

    # Build diagonals with boundary condition modifications
    main_dia = np.ones(N) * 2/h**2
    main_dia[0] = main_dia[-1] = 1  # enforce u[0] = u[-1] = 0
    upper_dia = np.ones(N-1) * -1/h**2
    upper_dia[0] = 0  # no coupling at left boundary
    lower_dia = np.ones(N-1) * -1/h**2
    lower_dia[-1] = 0  # no coupling at right boundary

    A.setdiag(main_dia)
    A.setdiag(upper_dia,  1)
    A.setdiag(lower_dia, -1)
    A = A.tocsc() # convert to CSC storage for efficient solve
    u = linalg.spsolve(A, F)
    Q = np.sum(u)*h

    if return_all:
        return u, F, Q
    else:
        return Q

def poiseuille_solver_3(F, W=2, return_all=False):
    N = len(F)
    F = F.copy()  # avoid modifying input
    F[0] = F[-1] = 0  # enforce boundary conditions on RHS
    h = 2*W/(N-1)
    A = sparse.dia_matrix((N, N)) # assemble matrix A

    # Build diagonals with boundary condition modifications
    main_dia = np.ones(N) * 2/h**2
    main_dia[0] = main_dia[-1] = 1  # enforce u[0] = u[-1] = 0
    upper_dia = np.ones(N-1) * -1/h**2
    upper_dia[0] = 0  # no coupling at left boundary
    lower_dia = np.ones(N-1) * -1/h**2
    lower_dia[-1] = 0  # no coupling at right boundary

    A.setdiag(main_dia)
    A.setdiag(upper_dia,  1)
    A.setdiag(lower_dia, -1)
    A = A.tocsc() # convert to CSC storage for efficient solve
    u = linalg.spsolve(A, F) # parameter vector specifies RHS
    Q = np.sum(u)*h
    if return_all:
        return u, F, Q
    else:
        return Q


# +
def u_analytic(y, F, W):
    return F / 2 * (W ** 2 - y ** 2)

def Q_analytic(F, W):
    return 2 * W ** 3 * F / 3
