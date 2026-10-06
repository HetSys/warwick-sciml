"""Tests for tersoff_jax against the atomistica reference implementation.

Run with:
    uv run --extra test pytest
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("atomistica")

import jax.numpy as jnp  # noqa: E402
from ase.build import bulk  # noqa: E402
from atomistica import Tersoff, Tersoff_PRB_39_5566_Si_C  # noqa: E402

from sciml import tersoff_jax as tj  # noqa: E402

jax.config.update("jax_enable_x64", True)


def atomistica_calculator(params):
    d = {"el": "Si", "m": 1}
    d.update(dict(zip(tj.PARAMETER_NAMES, params)))
    return Tersoff(**d)


def rattled_bulk():
    at = bulk("Si", cubic=True) * 2
    at.rattle(0.05, seed=1)
    return at


def vacancy_cell():
    at = bulk("Si", cubic=True) * 3
    del at[0]
    at.rattle(0.02, seed=2)
    return at


def test_nominal_parameters_match_atomistica():
    ref = np.array([Tersoff_PRB_39_5566_Si_C[k][-1] for k in tj.PARAMETER_NAMES])
    np.testing.assert_allclose(tj.SI_PARAMETERS, ref, rtol=1e-12)


@pytest.mark.parametrize("make_atoms", [rattled_bulk, vacancy_cell])
def test_energy_and_forces_match_atomistica(make_atoms):
    at = make_atoms()
    at.calc = atomistica_calculator(tj.SI_PARAMETERS)
    e_ref, f_ref = at.get_potential_energy(), at.get_forces()

    model = tj.TersoffModel(at)
    p = jnp.asarray(tj.SI_PARAMETERS)
    assert float(model.energy(p, at.positions)) == pytest.approx(e_ref, rel=1e-10)
    np.testing.assert_allclose(np.asarray(model.forces(p, at.positions)), f_ref, atol=1e-8)


def test_parameter_gradient_matches_finite_differences():
    at = vacancy_cell()
    model = tj.TersoffModel(at)
    p0 = np.array(tj.SI_PARAMETERS)
    grad = np.asarray(jax.grad(model.energy)(jnp.asarray(p0), at.positions))

    def e_ref(p):
        a = at.copy()
        a.calc = atomistica_calculator(p)
        return a.get_potential_energy()

    for k in range(len(p0)):
        h = 1e-6 * abs(p0[k])
        pp, pm = p0.copy(), p0.copy()
        pp[k] += h
        pm[k] -= h
        fd = (e_ref(pp) - e_ref(pm)) / (2 * h)
        assert grad[k] == pytest.approx(fd, rel=1e-5, abs=1e-8), tj.PARAMETER_NAMES[k]


def test_relax_and_adjoint_bond_length_sensitivity():
    """Adjoint dl/dtheta for a relaxed bond length (lambda != 0) matches finite
    differences through tight relaxations."""
    at = bulk("Si", cubic=True) * 2
    del at[0]
    at.rattle(0.01, seed=3)
    model = tj.TersoffModel(at)
    p0 = jnp.asarray(tj.SI_PARAMETERS)
    R, H = tj.relax(model, p0, at.positions)
    assert np.abs(np.asarray(model.forces(p0, R))).max() < 1e-10

    a, b = 1, 2
    bond = lambda R: jnp.linalg.norm(R[a] - R[b])
    dl_adj = tj.adjoint_sensitivity(model, p0, R, H, bond)

    for k in range(len(p0) - 2):  # r1, r2 do not affect this structure
        h = 1e-4 * abs(float(p0[k]))
        lp = float(bond(tj.relax(model, p0.at[k].add(h), R)[0]))
        lm = float(bond(tj.relax(model, p0.at[k].add(-h), R)[0]))
        assert dl_adj[k] == pytest.approx((lp - lm) / (2 * h), rel=1e-4, abs=1e-10), tj.PARAMETER_NAMES[k]


def test_bulk_modulus_matches_equation_of_state_fit():
    """B from AD at the equilibrium volume agrees with L02's EOS fit to atomistica energies."""
    from ase.eos import EquationOfState
    import ase.units as units

    at = bulk("Si")
    v, E = [], []
    for e in np.linspace(-0.05, 0.05, 13):
        a = at.copy()
        a.set_cell((1.0 + e) * a.cell, scale_atoms=True)
        a.calc = atomistica_calculator(tj.SI_PARAMETERS)
        v.append(a.get_volume())
        E.append(a.get_potential_energy())
    B_eos = EquationOfState(v, E).fit()[2] / units.GPa

    model = tj.TersoffModel(at)
    B_ad = float(model.bulk_modulus(jnp.asarray(tj.SI_PARAMETERS)))
    assert B_ad == pytest.approx(B_eos, rel=5e-3)


def test_bulk_modulus_parameter_gradient_matches_finite_differences():
    """dB/dtheta by AD (through the equilibrium-volume solve) vs central differences of B."""
    model = tj.TersoffModel(bulk("Si"))
    p0 = np.array(tj.SI_PARAMETERS)
    grad = np.asarray(jax.grad(model.bulk_modulus)(jnp.asarray(p0)))
    for k in range(len(p0)):
        h = 1e-5 * abs(p0[k])
        pp, pm = p0.copy(), p0.copy()
        pp[k] += h
        pm[k] -= h
        fd = (float(model.bulk_modulus(jnp.asarray(pp))) - float(model.bulk_modulus(jnp.asarray(pm)))) / (2 * h)
        assert grad[k] == pytest.approx(fd, rel=1e-4, abs=1e-6), tj.PARAMETER_NAMES[k]
