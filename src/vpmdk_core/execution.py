"""Filesystem-independent execution helpers for the public API."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from .models import (
    CalculationResult,
    MDConfig,
    MDResult,
    RelaxConfig,
    RelaxResult,
    RunContext,
    RunStep,
    SinglePointConfig,
    SinglePointResult,
)
from .observers import coerce_observer


def _root():
    return sys.modules["vpmdk_core"]


def _build_result(atoms, calculator, potential_energy: float) -> CalculationResult:
    """Return common result fields for the final structure."""

    root = _root()
    forces = root._safe_get_forces(atoms)
    stress = root._safe_get_stress_matrix(atoms, mode="full")
    resolved_calculator = getattr(atoms, "calc", None)
    return CalculationResult(
        atoms=atoms,
        calculator=resolved_calculator if resolved_calculator is not None else calculator,
        potential_energy=float(potential_energy),
        forces=forces,
        stress=stress,
    )


def execute_single_point(
    atoms,
    calculator,
    *,
    config: SinglePointConfig | None = None,
    observer=None,
    context: RunContext | None = None,
) -> SinglePointResult:
    """Run one energy/forces evaluation, plus stress when exposed, without writing files."""

    root = _root()
    config = config or SinglePointConfig()
    observer = coerce_observer(observer)
    context = context or RunContext(
        mode="single_point",
        ibrion=config.effective_ibrion,
        isif=config.effective_isif,
    )
    atoms.calc = root._resolve_calculator(calculator)

    if observer is not None:
        observer.on_start(atoms, context)

    energy = float(atoms.get_potential_energy())
    kinetic_energy = root._extract_numeric_attribute(atoms, ("get_kinetic_energy",))
    temperature = root._extract_numeric_attribute(atoms, ("get_temperature",))
    step = RunStep(
        index=1,
        potential_energy=energy,
        total_energy=energy + kinetic_energy,
        kinetic_energy=kinetic_energy,
        temperature=temperature,
    )
    if observer is not None:
        observer.on_step(atoms, step, context)

    common = _build_result(atoms, calculator, energy)
    result = SinglePointResult(
        atoms=common.atoms,
        calculator=common.calculator,
        potential_energy=common.potential_energy,
        forces=common.forces,
        stress=common.stress,
    )
    if observer is not None:
        observer.on_finish(atoms, result, context)
    return result


def execute_relaxation(
    atoms,
    calculator,
    *,
    config: RelaxConfig,
    observer=None,
    context: RunContext | None = None,
) -> RelaxResult:
    """Run a geometry optimization without implicit filesystem side effects."""

    root = _root()
    observer = coerce_observer(observer)
    context = context or RunContext(
        mode="relax",
        ibrion=config.effective_ibrion,
        isif=config.effective_stress_isif,
    )
    atoms.calc = root._resolve_calculator(calculator)
    if observer is not None:
        observer.on_start(atoms, context)

    if config.steps == 0:
        energy = float(atoms.get_potential_energy())
        fallback_step = RunStep(index=1, potential_energy=energy, total_energy=energy)
        if observer is not None:
            observer.on_step(atoms, fallback_step, context)
        common = _build_result(atoms, calculator, energy)
        result = RelaxResult(
            atoms=common.atoms,
            calculator=common.calculator,
            potential_energy=common.potential_energy,
            forces=common.forces,
            stress=common.stress,
            steps=[fallback_step],
            converged=False,
        )
        if observer is not None:
            observer.on_finish(atoms, result, context)
        return result

    recorded_steps: list[RunStep] = []
    scalar_pressure = (
        config.pressure_kbar * root.KBAR_TO_EV_PER_A3
        if config.pressure_kbar is not None
        else None
    )
    scalar_pressure_kwarg = scalar_pressure if scalar_pressure is not None else 0.0
    builder, freeze_required = root._make_relaxation_builder(
        config.effective_isif,
        scalar_pressure,
        scalar_pressure_kwarg,
    )

    previous_energy: float | None = None
    relax_object = None
    dyn = None
    converged: bool | None = None
    with root._temporarily_freeze_atoms(atoms, freeze_required):
        relax_object = builder(atoms)
        dyn = root.BFGS(relax_object, logfile=None)

        def _record_step() -> None:
            nonlocal previous_energy
            target = getattr(relax_object, "atoms", atoms)
            energy = float(target.get_potential_energy())
            previous_energy = energy
            step = RunStep(
                index=len(recorded_steps) + 1,
                potential_energy=energy,
                total_energy=energy,
            )
            recorded_steps.append(step)
            if observer is not None:
                observer.on_step(target, step, context)

        dyn.attach(_record_step)
        if config.energy_tolerance is None:
            converged = bool(dyn.run(fmax=config.fmax, steps=config.steps))
        else:
            monitor = root._EnergyConvergenceMonitor(atoms, config.energy_tolerance)
            # ASE's Optimizer.irun signature is `irun(self, fmax=0.05, steps=...)`
            # and its FIRST statement is `self.fmax = fmax`, so assigning dyn.fmax
            # beforehand is dead: every yielded force_converged flag was evaluated
            # against 0.05 eV/A. EDIFFG > 0 deliberately sets a NEGATIVE force limit
            # so that only the energy criterion can stop the run (see
            # docs/reference/incar-tags.md), so the leak silently terminated
            # relaxations at a force threshold the INCAR never asked for -- measured
            # eight orders of magnitude above the requested EDIFFG, writing an
            # under-relaxed CONTCAR and exiting 0.
            # The assignment is kept for an optimizer stand-in whose irun takes only
            # `steps` and reads self.fmax itself.
            dyn.fmax = config.fmax
            if root._callable_supports_parameter(dyn.irun, "fmax"):
                iterator = dyn.irun(fmax=config.fmax, steps=config.steps)
            else:
                iterator = dyn.irun(steps=config.steps)
            converged = False
            for force_converged in iterator:
                energy_converged = monitor.update()
                if energy_converged or force_converged:
                    converged = True
                    break

    target_atoms = getattr(relax_object, "atoms", atoms)
    target_atoms.wrap()
    if not recorded_steps:
        energy = float(target_atoms.get_potential_energy())
        fallback_step = RunStep(index=1, potential_energy=energy, total_energy=energy)
        recorded_steps.append(fallback_step)
        if observer is not None:
            observer.on_step(target_atoms, fallback_step, context)

    common = _build_result(target_atoms, calculator, recorded_steps[-1].potential_energy)
    result = RelaxResult(
        atoms=common.atoms,
        calculator=common.calculator,
        potential_energy=common.potential_energy,
        forces=common.forces,
        stress=common.stress,
        steps=recorded_steps,
        converged=converged,
    )
    if observer is not None:
        observer.on_finish(target_atoms, result, context)
    return result


class _MDDivergenceGuardCalculator:
    """Refuse force calls on a diverged MD configuration.

    The run-time twin of the input-time _MAX_PERIODIC_CELL_VOLUME cap: the
    backend's neighbour search bins over the UNWRAPPED coordinate span, so a
    trajectory thrown out of the cell by an oversized POTIM or thermostat
    parameter turns the next force call into an OOM-grade allocation (a
    measured 152 GB request) or an uninterruptible native spin that ignores
    SIGINT and wedges a resident server. No input-time cap can catch this --
    the divergence step depends on the system (POTIM=1e2 completes at NSW=3
    and diverges at NSW=30) -- so the bound has to sit in front of the force
    call itself. Positions must be checked BEFORE delegating: the blowing-up
    call happens inside dyn.run(1), after the integrator moved the atoms and
    before execute_md regains control.
    """

    def __init__(self, calculator):
        self._vpmdk_inner_calculator = calculator

    def _vpmdk_check_positions(self, atoms) -> None:
        if atoms is None:
            return
        from .io.inputs import (
            _MAX_PERIODIC_CELL_VOLUME,
            _MAX_UNWRAPPED_AXIS_SPAN,
        )

        positions = np.asarray(atoms.get_positions(), dtype=float)
        if positions.size == 0:
            return
        if not np.isfinite(positions).all():
            raise RuntimeError(
                "MD trajectory diverged: atomic positions became non-finite. "
                "This usually means POTIM (or a thermostat parameter such as "
                "LANGEVIN_GAMMA) is too large for this system; reduce it and "
                "rerun."
            )
        # Each axis is floored at 1 A: for collinear or constrained motion the
        # other spans are ~zero, and a raw product stayed zero no matter how
        # far the atoms flew apart, letting a one-axis divergence reach the
        # neighbour search anyway (its bin count keeps at least one bin per
        # axis, so its cost tracks this floored product, not the raw one).
        span = positions.max(axis=0) - positions.min(axis=0)
        floored_span = np.maximum(span, 1.0)
        span_volume = float(floored_span[0] * floored_span[1] * floored_span[2])
        # The bound is per-cell, not the global constant alone: the input-time
        # cap admits cells whose BOUNDING BOX exceeds 1e9 A^3 (e.g. one very
        # long axis with a sub-Angstrom cross-section), and wrapped positions
        # legitimately span that box today, so the run-time guard must not
        # reject what the cell itself allows.
        cell_widths = np.abs(np.asarray(atoms.get_cell())).sum(axis=0)
        cell_volume_bound = float(np.prod(np.maximum(cell_widths, 1.0)))
        limit = max(_MAX_PERIODIC_CELL_VOLUME, cell_volume_bound)
        # A SINGLE-axis divergence escapes the product (the other floored
        # factors stay 1) while the neighbour search's per-axis image
        # replication keeps growing linearly -- a 3.9e7 A single-axis span
        # is a measured MemoryError. Bound each axis on its own too; a cell
        # whose own width exceeds the cap keeps its width as the limit
        # (mirrors the NEB image-read guard).
        axis_limits = np.maximum(
            np.maximum(cell_widths, 1.0), _MAX_UNWRAPPED_AXIS_SPAN
        )
        if span_volume > limit or bool(np.any(span > axis_limits)):
            # Name BOTH bounds, like the NEB twin: when the per-axis rule is
            # what fires, a volume-only message reported a span-volume far
            # BELOW the printed maximum -- a self-contradictory diagnostic
            # with no actionable number.
            raise RuntimeError(
                "MD trajectory diverged: the unwrapped atomic positions span "
                f"a {span[0]:g} x {span[1]:g} x {span[2]:g} A bounding box "
                f"(supported maximum {limit:g} A^3 with per-axis spans "
                f"floored at 1 A, and {_MAX_UNWRAPPED_AXIS_SPAN:g} A per "
                "axis), so the next force evaluation would exhaust memory in "
                "the backend's neighbour search. This usually means POTIM "
                "(or a thermostat parameter such as LANGEVIN_GAMMA) is too "
                "large for this system; reduce it and rerun."
            )

    def get_forces(self, atoms=None):
        self._vpmdk_check_positions(atoms)
        return self._vpmdk_inner_calculator.get_forces(atoms)

    def get_potential_energy(self, atoms=None, force_consistent=False):
        self._vpmdk_check_positions(atoms)
        try:
            return self._vpmdk_inner_calculator.get_potential_energy(
                atoms, force_consistent=force_consistent
            )
        except TypeError:
            return self._vpmdk_inner_calculator.get_potential_energy(atoms)

    def get_stress(self, atoms=None):
        self._vpmdk_check_positions(atoms)
        return self._vpmdk_inner_calculator.get_stress(atoms)

    def calculate(self, atoms=None, *args, **kwargs):
        self._vpmdk_check_positions(atoms)
        return self._vpmdk_inner_calculator.calculate(atoms, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._vpmdk_inner_calculator, name)


def execute_md(
    atoms,
    calculator,
    *,
    config: MDConfig,
    observer=None,
    context: RunContext | None = None,
) -> MDResult:
    """Run molecular dynamics without implicit VASP-style file output."""

    root = _root()
    observer = coerce_observer(observer)
    mdalgo = config.effective_mdalgo
    context = context or RunContext(
        mode="md",
        ibrion=0,
        isif=config.effective_isif,
        potim=config.timestep_fs,
        mdalgo=mdalgo,
    )
    atoms.calc = root._resolve_calculator(calculator)
    if observer is not None:
        observer.on_start(atoms, context)

    if config.steps == 0:
        potential_energy = float(atoms.get_potential_energy())
        fallback_step = RunStep(
            index=1,
            potential_energy=potential_energy,
            total_energy=potential_energy,
            kinetic_energy=0.0,
            temperature=0.0,
            advanced=False,
        )
        if observer is not None:
            observer.on_step(atoms, fallback_step, context)
        atoms.wrap()
        common = _build_result(atoms, calculator, potential_energy)
        result = MDResult(
            atoms=common.atoms,
            calculator=common.calculator,
            potential_energy=common.potential_energy,
            forces=common.forces,
            stress=common.stress,
            steps=[fallback_step],
        )
        if observer is not None:
            observer.on_finish(atoms, result, context)
        return result

    target_end = config.temperature if config.temperature_end is None else config.temperature_end
    # Measured per thermostat in ase 3.29. A NEGATIVE absolute temperature makes ASE
    # take the square root of a negative number with no exception at all, so the
    # trajectory silently becomes nan and the nan surfaces far away as a raw
    # ValueError from the energy formatter -- reported as calculation_error, i.e.
    # exit 2, which the server-mode exit-code contract documents as RETRYABLE for a permanently
    # invalid INCAR. Caught up front it is exit 1 instead.
    #   MDALGO 2/4 (Nose-Hoover chain): T <= 0 breaks (Q = 3NkT*tdamp^2 = 0).
    #   MDALGO 5   (CSVR/Bussi):        T <= 0 breaks ("Initial kinetic energy is
    #                                  zero" from ase.md.bussi).
    #   MDALGO 1/3 (Andersen/Langevin): T = 0 is FINE -- it is the legal 0 K limit
    #                                  and completes today -- only T < 0 breaks.
    #   MDALGO 0   (plain NVE):         unaffected; it already completes at 0 K with
    #                                  exit 0, and legacy one-shot behavior must not
    #                                  change (one-shot compatibility contract).
    if mdalgo in (1, 3) and (config.temperature < 0 or target_end < 0):
        raise _root().WorkdirInputError(
            "TEBEG and TEEND must not be negative; an absolute temperature below "
            "zero makes the thermostat produce nan velocities. Use TEBEG=0 for "
            "zero-temperature dynamics."
        )
    if mdalgo == 5 and (config.temperature <= 0 or target_end <= 0):
        raise _root().WorkdirInputError(
            "CSVR (Bussi) dynamics require positive TEBEG and TEEND temperatures; "
            "the thermostat cannot rescale a zero initial kinetic energy. Use "
            "MDALGO=0 for zero-temperature NVE-style dynamics."
        )
    if mdalgo in (2, 4) and (config.temperature <= 0 or target_end <= 0):
        # TEBEG/TEEND <= 0 is bad INCAR input caught up front, before any dynamics
        # run: exit 1 (input_error), not a retryable calculation failure (exit 2).
        raise _root().WorkdirInputError(
            "Nose-Hoover chain dynamics require positive TEBEG and TEEND "
            "temperatures. Use MDALGO=0 for zero-temperature NVE-style dynamics "
            "or choose a positive temperature ramp."
        )

    if config.temperature <= 0:
        velocities = atoms.get_velocities()
        if velocities is None:
            atoms.set_velocities([[0.0, 0.0, 0.0] for _ in range(len(atoms))])
        else:
            atoms.set_velocities(velocities * 0.0)
    else:
        root.velocitydistribution.MaxwellBoltzmannDistribution(
            atoms,
            temperature_K=config.temperature,
        )

    # Wrapped only for the dynamics run and restored before the result is
    # built: _build_result publishes atoms.calc as result.calculator, and a
    # resident server reuses the calculator across jobs.
    resolved_md_calculator = atoms.calc
    atoms.calc = _MDDivergenceGuardCalculator(resolved_md_calculator)
    try:
        dyn, update_temperature = root._select_md_dynamics(
            atoms,
            mdalgo,
            config.timestep_fs,
            config.temperature,
            config.smass,
            config.thermostat_kwargs,
        )
        recorded_steps: list[RunStep] = []

        for step_index in range(1, config.steps + 1):
            dyn.run(1)
            atoms.wrap()
            potential_energy = float(atoms.get_potential_energy())
            kinetic_energy = root._extract_numeric_attribute(atoms, ("get_kinetic_energy",))
            thermostat_potential, thermostat_kinetic = root._thermostat_energy_terms(dyn)
            temperature = root._extract_numeric_attribute(atoms, ("get_temperature",))
            step = RunStep(
                index=step_index,
                potential_energy=potential_energy,
                total_energy=potential_energy + kinetic_energy + thermostat_potential + thermostat_kinetic,
                kinetic_energy=kinetic_energy,
                thermostat_potential=thermostat_potential,
                thermostat_kinetic=thermostat_kinetic,
                temperature=temperature,
            )
            recorded_steps.append(step)
            if observer is not None:
                observer.on_step(atoms, step, context)
            if config.steps > 1 and step_index < config.steps and target_end != config.temperature:
                next_temp = config.temperature + (
                    (target_end - config.temperature) * step_index / (config.steps - 1)
                )
                update_temperature(next_temp)
    finally:
        atoms.calc = resolved_md_calculator

    if not recorded_steps:
        potential_energy = float(atoms.get_potential_energy())
        kinetic_energy = root._extract_numeric_attribute(atoms, ("get_kinetic_energy",))
        fallback_step = RunStep(
            index=1,
            potential_energy=potential_energy,
            total_energy=potential_energy + kinetic_energy,
            kinetic_energy=kinetic_energy,
            temperature=float(config.temperature),
        )
        recorded_steps.append(fallback_step)
        if observer is not None:
            observer.on_step(atoms, fallback_step, context)

    atoms.wrap()
    common = _build_result(atoms, calculator, recorded_steps[-1].potential_energy)
    result = MDResult(
        atoms=common.atoms,
        calculator=common.calculator,
        potential_energy=common.potential_energy,
        forces=common.forces,
        stress=common.stress,
        steps=recorded_steps,
    )
    if observer is not None:
        observer.on_finish(atoms, result, context)
    return result
