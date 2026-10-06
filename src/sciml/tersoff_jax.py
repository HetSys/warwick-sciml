"""A differentiable Tersoff potential for silicon, written in JAX.

Only the energy is written down. Forces, stresses, Hessians and derivatives with
respect to the potential *parameters* all come from automatic differentiation.
Compare with matscipy's `TersoffBrenner`, which hand-codes first and second
derivatives of every term.

Functional form (Tersoff, Phys. Rev. B 39, 5566 (1989)), with pairs p = (i, j):

    E      = 1/2 sum_p f_c(r_p) [ f_R(r_p) + b_p f_A(r_p) ]
    f_R(r) = A exp(-lambda r),   f_A(r) = -B exp(-mu r)
    b_p    = xi (1 + (beta zeta_p)^n)^(-1/(2n))
    zeta_p = sum_{q = (i, k), k != j} f_c(r_q) g(cos theta_pq)
    g      = 1 + c^2/d^2 - c^2 / (d^2 + (h - cos theta)^2)

The angular exponential factor is omitted since it is identically 1 for silicon
(atomistica's `mubo = 0`). Parameters follow atomistica's naming and order, so a
parameter vector can be passed to either implementation.

Neighbour (pair and triplet) lists are built once with matscipy, outside JAX, and
treated as fixed connectivity. This is fine for the strains and relaxations used
here, but not for molecular dynamics where neighbours change.
"""

import jax
import jax.numpy as jnp
import numpy as np
from matscipy.neighbours import neighbour_list

jax.config.update("jax_enable_x64", True)

PARAMETER_NAMES = ["A", "B", "xi", "lambda", "mu", "beta", "n", "c", "d", "h", "r1", "r2"]
# Tersoff, Phys. Rev. B 39, 5566 (1989), Table I, silicon column
SI_PARAMETERS = np.array([
    1830.8, 471.18, 1.0, 2.4799, 1.7322, 1.1e-6, 0.78734, 1.0039e5, 16.217, -0.59825, 2.7, 3.0,
])
EV_PER_A3_TO_GPA = 160.21766208


def cutoff(r, r1, r2):
    """Smooth cutoff: 1 below r1, 0 above r2, cosine switch in between."""
    # Clip the argument so the unused branch of `where` stays finite and its
    # (masked) gradient cannot contaminate the result with NaNs.
    x = jnp.clip((r - r1) / (r2 - r1), 0.0, 1.0)
    return jnp.where(r < r1, 1.0, jnp.where(r > r2, 0.0, 0.5 + 0.5 * jnp.cos(jnp.pi * x)))


def tersoff_energy(params, D_p, triplet_p, triplet_q):
    """Energy from pair vectors D_p (n_pairs x 3) and triplet index arrays."""
    A, B, xi, lam, mu, beta, n, c, d, h, r1, r2 = params
    r = jnp.linalg.norm(D_p, axis=1)
    fc = cutoff(r, r1, r2)

    cos_theta = jnp.sum(D_p[triplet_p] * D_p[triplet_q], axis=1) / (r[triplet_p] * r[triplet_q])
    g = 1.0 + c**2 / d**2 - c**2 / (d**2 + (h - cos_theta) ** 2)
    zeta = jax.ops.segment_sum(fc[triplet_q] * g, triplet_p, num_segments=D_p.shape[0])

    b = xi * (1.0 + (beta * zeta) ** n) ** (-1.0 / (2.0 * n))
    f_R = A * jnp.exp(-lam * r)
    f_A = -B * jnp.exp(-mu * r)
    return 0.5 * jnp.sum(fc * (f_R + b * f_A))


class TersoffModel:
    """Tersoff energy for a fixed set of atoms and neighbour lists.

    `energy(params, positions)` is a pure JAX function of the parameters and
    positions, so `jax.grad`, `jax.hessian` etc. can be applied to either.
    """

    def __init__(self, atoms, skin=0.5):
        self.atoms = atoms.copy()
        self.cell = np.array(atoms.cell)
        rcut = SI_PARAMETERS[PARAMETER_NAMES.index("r2")] + skin
        i, j, S = neighbour_list("ijS", atoms, rcut)
        self.i, self.j, self.shift = i, j, S @ self.cell

        # Triplets (p, q): pairs p = (i, j) and q = (i, k) sharing atom i, with k != j
        triplet_p, triplet_q = [], []
        for atom in range(len(atoms)):
            pairs = np.nonzero(i == atom)[0]
            for p in pairs:
                for q in pairs:
                    if p != q:
                        triplet_p.append(p)
                        triplet_q.append(q)
        self.triplet_p = np.array(triplet_p, dtype=int)
        self.triplet_q = np.array(triplet_q, dtype=int)
        self.volume = atoms.get_volume()

    def pair_vectors(self, positions, strain=0.0):
        positions = jnp.asarray(positions)
        D = positions[self.j] - positions[self.i] + self.shift
        return (1.0 + strain) * D

    def energy(self, params, positions):
        return tersoff_energy(params, self.pair_vectors(positions), self.triplet_p, self.triplet_q)

    def forces(self, params, positions):
        return -jax.grad(self.energy, argnums=1)(params, positions)

    def strained_energy(self, params, strain):
        """Energy under uniform isotropic strain (atoms scale with the cell)."""
        D = self.pair_vectors(self.atoms.positions, strain)
        return tersoff_energy(params, D, self.triplet_p, self.triplet_q)

    def bulk_modulus(self, params, newton_steps=8):
        """B = V d2E/dV2 at the equilibrium volume, in GPa.

        The equilibrium strain solves dE/de = 0. We find it with a few Newton
        steps; AD differentiates through them, so dB/dparams includes the shift
        of the equilibrium volume with the parameters.
        """
        dE = jax.grad(self.strained_energy, argnums=1)
        d2E = jax.grad(dE, argnums=1)

        def newton(e, _):
            return e - dE(params, e) / d2E(params, e), None

        e_star, _ = jax.lax.scan(newton, jnp.zeros(()), None, length=newton_steps)
        V = self.volume * (1.0 + e_star) ** 3
        # V(e) = V0 (1+e)^3, so d2E/dV2 = (E'' (1+e) - 2 E') / (9 V0^2 (1+e)^5)
        E1, E2 = dE(params, e_star), d2E(params, e_star)
        d2E_dV2 = (E2 * (1.0 + e_star) - 2.0 * E1) / (9.0 * self.volume**2 * (1.0 + e_star) ** 5)
        return V * d2E_dV2 * EV_PER_A3_TO_GPA


def relax(model, params, positions, newton_steps=5, hessian=None):
    """Minimise the energy with respect to positions at fixed parameters.

    BFGS gets close; a few Newton steps with the AD Hessian then converge to
    machine precision. Returns the relaxed positions and the Hessian there
    (3N x 3N), which the adjoint solve reuses. Pass a fixed `hessian` (e.g. from
    a nearby relaxation) to skip recomputing it at every Newton step; with
    `newton_steps=0` this is plain BFGS at its default tolerance.
    """
    import scipy.optimize

    shape = np.shape(positions)
    if not hasattr(model, "_compiled"):  # compile once per model, not per call
        model._compiled = (
            jax.jit(model.energy),
            jax.jit(jax.grad(model.energy, argnums=1)),
            jax.jit(jax.hessian(model.energy, argnums=1)),
        )
    energy, grad, hessian_fn = model._compiled

    res = scipy.optimize.minimize(
        lambda x: float(energy(params, x.reshape(shape))),
        np.ravel(positions),
        jac=lambda x: np.asarray(grad(params, x.reshape(shape))).ravel(),
        method="BFGS",
    )
    R, H = res.x, hessian
    for _ in range(newton_steps):
        if hessian is None:
            H = np.asarray(hessian_fn(params, R.reshape(shape))).reshape(R.size, R.size)
        # H is singular (rigid translations), so use a least-squares solve
        R = R - np.linalg.lstsq(H, np.asarray(grad(params, R.reshape(shape))).ravel(), rcond=1e-10)[0]
    return R.reshape(shape), H


def adjoint_sensitivity(model, params, R, H, qoi):
    """d qoi(R*(params)) / d params by the adjoint method.

    The relaxed positions satisfy g = dE/dR = 0. Solve H lambda = d qoi/dR once,
    then d qoi/d params = -lambda^T d2E/(dR dparams), for all parameters at once.
    """
    lam = np.linalg.lstsq(H, np.asarray(jax.grad(qoi)(R)).ravel(), rcond=1e-10)[0]
    d2E_dR_dp = jax.jacfwd(lambda p: jax.grad(model.energy, argnums=1)(p, R))(params)
    return -lam @ np.asarray(d2E_dR_dp).reshape(lam.size, -1)
