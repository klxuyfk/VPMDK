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


def _warn_pressure_is_inert_at_constant_volume(
    scalar_pressure: float | None, isif: int
) -> None:
    """Say so when PSTRESS cannot act, instead of dropping it silently."""

    if not scalar_pressure:
        return
    print(
        f"Warning: PSTRESS is ignored for ISIF={isif}, which relaxes at constant "
        "volume; an external hydrostatic pressure does no work there. Use ISIF=3 "
        "(or 6/7/8) to relax against a pressure."
    )


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

    if isif in (4, 5):
        # NO scalar_pressure at constant volume. An external hydrostatic pressure
        # can do no work when the volume is fixed, so it must be inert -- but ASE
        # adds -V*P*I to the virial and only removes the TRACE after solving
        # against the deformation gradient, so once F != I a traceless remainder
        # survives as a spurious driving force. Measured (EMT/Cu, NSW=100): ISIF=5
        # lost 18.5% of its volume at PSTRESS=500 and 41.7% at 2000, ISIF=4 lost
        # 42.2% at 2000, with the energy diverging rather than relaxing -- and both
        # modes are documented as constant volume.
        _warn_pressure_is_inert_at_constant_volume(scalar_pressure, isif)

        def build_constant_volume(atoms):
            return root.UnitCellFilter(atoms, constant_volume=True)

        return build_constant_volume, isif == 5

    if isif == 6:
        if not scalar_pressure:
            # No pressure to apply (unset, or an explicit PSTRESS=0, which is also
            # VASP's default): keep the documented StrainFilter mapping so runs that
            # were already correct stay bit-for-bit unchanged.
            return root.StrainFilter, False

        # ase.filters.StrainFilter has NO scalar_pressure argument, so a PSTRESS
        # given with ISIF=6 -- a mode where cell degrees of freedom ARE active, so
        # docs/reference/incar-tags.md says PSTRESS applies -- was silently dropped
        # and the cell relaxed to zero stress instead: a high-pressure ISIF=6 sweep
        # produced the same structure at every PSTRESS. UnitCellFilter with the ions
        # frozen is cell-only in the same way (this is exactly the shape ISIF=5 uses
        # above) and it does accept the pressure.
        def build_strain_with_pressure(atoms):
            return root.UnitCellFilter(atoms, scalar_pressure=scalar_pressure)

        return build_strain_with_pressure, True

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


_INTERNAL_ISIF_FREEZE_MARKER = "_vpmdk_internal_isif_freeze"


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
    # Mark it as VPMDK's OWN device for holding the ions while only the cell
    # relaxes (ISIF 5/6/7). The VASP-compat recorder reads forces with
    # apply_constraint=True, so this constraint zeroed the OUTCAR TOTAL-FORCE
    # table, "FORCES: max atom, RMS", the total drift and the vasprun.xml forces
    # varray -- a run with real residual forces looked perfectly converged. Real
    # VASP reports the physical forces for these ISIF modes; the marker lets the
    # recorder drop THIS constraint (and only this one, so a user's selective
    # dynamics still applies) while reading them.
    setattr(frozen, _INTERNAL_ISIF_FREEZE_MARKER, True)
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

    if write_energy_csv:
        # This run WILL write energy.csv at the end; fail on an unwritable
        # node now, before the relaxation is paid for (the recorder's
        # unconditional preflight of flag-gated artifacts was scoped down).
        _root()._require_writable_artifact_path("energy.csv")
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
            pstress_kbar=pstress,
            nsw_requested=steps,
        ),
    )
    if write_energy_csv:
        with open("energy.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for step in result.steps:
                writer.writerow([float(step.potential_energy)])
    return result.potential_energy
