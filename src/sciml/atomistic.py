import sys

import numpy as np

from ase.build import bulk
from ase.optimize.precon import PreconLBFGS
from atomistica import Tersoff, Tersoff_PRB_39_5566_Si_C

Tersoff_parameter_names = list(Tersoff_PRB_39_5566_Si_C.keys())[2:]
del Tersoff_parameter_names[Tersoff_parameter_names.index('m')]
del Tersoff_parameter_names[Tersoff_parameter_names.index('mubo')]

Tersoff_parameter_values = np.array([Tersoff_PRB_39_5566_Si_C[key][-1] for key in Tersoff_parameter_names])

def tersoff_calculator(parameters):
    # build a dictionary of parameters using values passed in
    param_dict = { 'el': 'Si', 'm': 1 } # fixed parameters
    param_dict.update(dict(zip(Tersoff_parameter_names, parameters)))
    calc = Tersoff(**param_dict)
    return calc

def vacancy_cell(Ncell=3, rattle=0.0):
    """
    Build an Ncell x Ncell x Ncell cubic Si supercell and a copy with a vacancy.

    The atom nearest the cell midpoint is removed. Positions in the vacancy cell
    are optionally rattled (standard deviation `rattle`, in Angstrom) before the
    atom is deleted.

    Returns
    -------
    at_sup:   perfect supercell
    at_vac:   supercell with the vacancy
    vac_pos:  position of the removed atom
    """
    at_sup = bulk('Si', cubic=True) * Ncell
    at_vac = at_sup.copy()
    if rattle:
        at_vac.rattle(rattle)
    cell_midpoint = np.diag(at_vac.cell) * 0.5
    vac_atom = ((at_vac.positions - cell_midpoint)**2).sum(axis=1).argmin()
    vac_pos = at_vac.positions[vac_atom].copy()
    del at_vac[vac_atom]
    return at_sup, at_vac, vac_pos


def vacancy_formation_energy(parameters, Ncell=3, relax=False, fmax=1e-4, logfile=None):
    """
    Compute vacancy formation energy for a Ncell x Ncell x Ncell Si system

    Modified to use given values of the Tersoff parameters

    Arguments
    ---------

    parameters: vector of the Tersoff parameters to be used
    Ncell:      number of unit cells (default 3)
    relax:      if true, rattle the atoms slightly and perform a geometry
                   optimisation before computing the vacancy formation energy
    fmax:       the maximum force tolerance to use when `relax=True`

    Returns
    -------

    e_vac       Vacancy formation energy, in eV

    """
    at_sup, at_vac, _ = vacancy_cell(Ncell, rattle=0.01 if relax else 0.0)

    calc = tersoff_calculator(parameters)
    at_sup.calc = calc
    e_bulk_per_atom = at_sup.get_potential_energy() / len(at_sup)

    at_vac.calc = calc
    if relax:
        opt = PreconLBFGS(at_vac, logfile=logfile)
        opt.run(fmax=fmax)
    return at_vac.get_potential_energy() - e_bulk_per_atom * len(at_vac)


def vacancy_formation_energy_first5(x):
    X = Tersoff_parameter_values.copy()
    X[:5] = x
    _stdout = sys.stdout
    sys.stdout = None # suppress output
    Ef = vacancy_formation_energy(X, relax=True)
    sys.stdout = _stdout
    return Ef
