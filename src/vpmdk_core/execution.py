"""Filesystem-independent execution helpers for the public API."""

from __future__ import annotations

import sys
from typing import Any

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
    return CalculationResult(
        atoms=atoms,
        calculator=calculator,
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
    """Run one energy/force/stress evaluation without writing files by default."""

    root = _root()
    config = config or SinglePointConfig()
    observer = coerce_observer(observer)
    context = context or RunContext(mode="single_point", ibrion=-1, isif=config.isif)
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
        ibrion=config.ibrion,
        isif=config.stress_isif if config.stress_isif is not None else config.isif,
    )
    atoms.calc = root._resolve_calculator(calculator)
    if observer is not None:
        observer.on_start(atoms, context)

    recorded_steps: list[RunStep] = []
    scalar_pressure = (
        config.pressure_kbar * root.KBAR_TO_EV_PER_A3
        if config.pressure_kbar is not None
        else None
    )
    scalar_pressure_kwarg = scalar_pressure if scalar_pressure is not None else 0.0
    builder, freeze_required = root._make_relaxation_builder(
        config.isif,
        scalar_pressure,
        scalar_pressure_kwarg,
    )

    previous_energy: float | None = None
    relax_object = None
    dyn = None
    with root._temporarily_freeze_atoms(atoms, freeze_required):
        relax_object = builder(atoms)
        dyn = root.BFGS(relax_object, logfile="OUTCAR")

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
            dyn.run(fmax=config.fmax, steps=config.steps)
        else:
            monitor = root._EnergyConvergenceMonitor(atoms, config.energy_tolerance)
            dyn.fmax = config.fmax
            for force_converged in dyn.irun(steps=config.steps):
                energy_converged = monitor.update()
                if energy_converged or force_converged:
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
    converged = getattr(dyn, "converged", None)
    result = RelaxResult(
        atoms=common.atoms,
        calculator=common.calculator,
        potential_energy=common.potential_energy,
        forces=common.forces,
        stress=common.stress,
        steps=recorded_steps,
        converged=bool(converged) if converged is not None else None,
    )
    if observer is not None:
        observer.on_finish(target_atoms, result, context)
    return result


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
        isif=config.isif,
        potim=config.timestep_fs,
        mdalgo=mdalgo,
    )
    atoms.calc = root._resolve_calculator(calculator)
    if observer is not None:
        observer.on_start(atoms, context)

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

    dyn, update_temperature = root._select_md_dynamics(
        atoms,
        mdalgo,
        config.timestep_fs,
        config.temperature,
        config.smass,
        config.thermostat_kwargs,
    )
    target_end = config.temperature if config.temperature_end is None else config.temperature_end
    recorded_steps: list[RunStep] = []

    for step_index in range(1, config.steps + 1):
        dyn.run(1)
        atoms.wrap()
        potential_energy = float(atoms.get_potential_energy())
        kinetic_energy = root._extract_numeric_attribute(atoms, ("get_kinetic_energy",))
        thermostat_potential = root._extract_numeric_attribute(
            dyn,
            (
                "thermostat_potential_energy",
                "thermostat_potential",
                "nose_potential_energy",
                "nhc_potential_energy",
            ),
        )
        thermostat_kinetic = root._extract_numeric_attribute(
            dyn,
            (
                "thermostat_kinetic_energy",
                "thermostat_kinetic",
                "nose_kinetic_energy",
                "nhc_kinetic_energy",
            ),
        )
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
