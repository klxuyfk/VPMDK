"""Relaxation execution flow."""

from __future__ import annotations

import csv
import sys
from contextlib import contextmanager
from typing import Callable, List

from ..compat.vasp import VaspCompatConfig, VaspRelaxConfig
from ..models import RelaxConfig
from ..observers import PrintProgressObserver, VaspCompatObserver


def _root():
    return sys.modules["vpmdk_core"]


class _EnergyConvergenceMonitor:
    """Track ionic step energies and test for convergence."""

    def __init__(self, atoms, tolerance: float):
        self._atoms = atoms
        self._tolerance = tolerance
        self._previous: float | None = None

    def update(self) -> bool:
        """Return True when the total energy change falls below the tolerance."""

        energy = self._atoms.get_potential_energy()
        if self._previous is None:
            self._previous = energy
            return False
        delta = abs(energy - self._previous)
        self._previous = energy
        return delta <= self._tolerance


def _make_relaxation_builder(
    isif: int,
    scalar_pressure: float | None,
    scalar_pressure_kwarg: float,
) -> tuple[Callable[[object], object], bool]:
    """Return a factory for the relaxation object and freeze requirement."""

    root = _root()

    def build_identity(atoms):
        return atoms

    if isif == 3:
        if scalar_pressure is None:
            return root.UnitCellFilter, False

        def build_ucf(atoms):
            return root.UnitCellFilter(atoms, scalar_pressure=scalar_pressure)

        return build_ucf, False

    if isif == 4:

        def build_constant_volume(atoms):
            return root.UnitCellFilter(
                atoms,
                constant_volume=True,
                scalar_pressure=scalar_pressure_kwarg,
            )

        return build_constant_volume, False

    if isif == 5:

        def build_constant_volume_frozen(atoms):
            return root.UnitCellFilter(
                atoms,
                constant_volume=True,
                scalar_pressure=scalar_pressure_kwarg,
            )

        return build_constant_volume_frozen, True

    if isif == 6:
        return root.StrainFilter, False

    if isif == 7:

        def build_hydrostatic_frozen(atoms):
            return root.UnitCellFilter(
                atoms,
                mask=[1, 1, 1, 0, 0, 0],
                hydrostatic_strain=True,
                scalar_pressure=scalar_pressure_kwarg,
            )

        return build_hydrostatic_frozen, True

    if isif == 8:

        def build_hydrostatic(atoms):
            return root.UnitCellFilter(
                atoms,
                mask=[1, 1, 1, 0, 0, 0],
                hydrostatic_strain=True,
                scalar_pressure=scalar_pressure_kwarg,
            )

        return build_hydrostatic, False

    return build_identity, False


@contextmanager
def _temporarily_freeze_atoms(atoms, freeze_required: bool):
    """Temporarily constrain ionic positions when required by ISIF."""

    if not freeze_required:
        yield
        return

    current_constraints = getattr(atoms, "constraints", None)
    if current_constraints is None:
        original_constraints = None
        base_constraints: list[object] = []
    else:
        try:
            base_constraints = list(current_constraints)
        except TypeError:
            base_constraints = [current_constraints]
        original_constraints = base_constraints

    frozen = _root().FixAtoms(indices=list(range(len(atoms))))
    atoms.set_constraint(base_constraints + [frozen])
    try:
        yield
    finally:
        if original_constraints is None:
            atoms.set_constraint()
        else:
            atoms.set_constraint(original_constraints)


def run_relaxation(
    atoms,
    calculator,
    steps: int,
    fmax: float,
    write_energy_csv: bool = False,
    isif: int = 2,
    pstress: float | None = None,
    energy_tolerance: float | None = None,
    ibrion: int = 2,
    stress_isif: int | None = None,
    neb_mode: bool = False,
    neb_prev_positions=None,
    neb_next_positions=None,
    oszicar_pseudo_scf: bool = False,
):
    result = _root().relax(
        atoms,
        calculator=calculator,
        config=RelaxConfig(
            steps=steps,
            fmax=fmax,
            relax_cell=isif >= 3,
            pressure_kbar=pstress,
            energy_tolerance=energy_tolerance,
            compat=VaspRelaxConfig(
                isif=isif,
                stress_isif=stress_isif,
                ibrion=ibrion,
            ),
        ),
        observer=[VaspCompatObserver(), PrintProgressObserver()],
        compatibility=VaspCompatConfig(
            enabled=True,
            write_pseudo_scf=oszicar_pseudo_scf,
            write_contcar=True,
            neb_mode=neb_mode,
            neb_prev_positions=neb_prev_positions,
            neb_next_positions=neb_next_positions,
        ),
    )
    if write_energy_csv:
        with open("energy.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for step in result.steps:
                writer.writerow([float(step.potential_energy)])
    return result.potential_energy
