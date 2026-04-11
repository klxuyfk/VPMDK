"""Relaxation execution flow."""

from __future__ import annotations

import csv
import sys
from contextlib import contextmanager
from typing import Callable, List


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
    root = _root()
    atoms.calc = root._resolve_calculator(calculator)
    recorder = root._initialize_vasp_compat_outputs(
        atoms,
        ibrion=ibrion,
        isif=isif if stress_isif is None else stress_isif,
        neb_mode=neb_mode,
        write_oszicar_pseudo_scf=oszicar_pseudo_scf,
        neb_prev_positions=neb_prev_positions,
        neb_next_positions=neb_next_positions,
    )
    energies: List[float] = []
    previous_energy: float | None = None
    step_counter = 0
    scalar_pressure = pstress * root.KBAR_TO_EV_PER_A3 if pstress is not None else None
    scalar_pressure_kwarg = scalar_pressure if scalar_pressure is not None else 0.0

    builder, freeze_required = _make_relaxation_builder(
        isif, scalar_pressure, scalar_pressure_kwarg
    )

    with _temporarily_freeze_atoms(atoms, freeze_required):
        relax_object = builder(atoms)
        dyn = root.BFGS(relax_object, logfile="OUTCAR")

        def _log_relaxation_energy() -> None:
            nonlocal previous_energy, step_counter
            target = getattr(relax_object, "atoms", atoms)
            energy = target.get_potential_energy()
            delta = 0.0 if previous_energy is None else energy - previous_energy
            previous_energy = energy
            step_counter += 1
            root._record_vasp_compat_step(
                recorder,
                target,
                step_index=step_counter,
                potential_energy=energy,
                total_energy=energy,
            )
            print(
                f"{step_counter:4d} F= {root._format_energy_value(energy)} "
                f"E0= {root._format_energy_value(energy)}  d E ={root._format_energy_value(delta)}"
            )

        if write_energy_csv:
            dyn.attach(lambda: energies.append(atoms.get_potential_energy()))
        dyn.attach(_log_relaxation_energy)
        if energy_tolerance is None:
            dyn.run(fmax=fmax, steps=steps)
        else:
            monitor = _EnergyConvergenceMonitor(atoms, energy_tolerance)
            dyn.fmax = fmax
            for force_converged in dyn.irun(steps=steps):
                energy_converged = monitor.update()
                if energy_converged or force_converged:
                    break

    target_atoms = getattr(relax_object, "atoms", atoms)
    target_atoms.wrap()
    if not recorder.steps:
        fallback_energy = target_atoms.get_potential_energy()
        root._record_vasp_compat_step(
            recorder,
            target_atoms,
            step_index=1,
            potential_energy=fallback_energy,
            total_energy=fallback_energy,
        )
    root._write_vasprun_xml(recorder, target_atoms)
    root._append_outcar_footer(recorder)
    root.write("CONTCAR", target_atoms, direct=True)
    if write_energy_csv:
        with open("energy.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for energy in energies:
                writer.writerow([energy])
    return target_atoms.get_potential_energy()
