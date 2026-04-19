"""Execution observers for progress reporting and compatibility output."""

from __future__ import annotations

import sys
from typing import Iterable

from .models import CalculationResult, RunContext, RunStep


def _root():
    return sys.modules["vpmdk_core"]


class RunObserver:
    """Base observer interface used by the pure execution layer."""

    def on_start(self, atoms, context: RunContext) -> None:  # pragma: no cover - interface
        return None

    def on_step(self, atoms, step: RunStep, context: RunContext) -> None:  # pragma: no cover - interface
        return None

    def on_finish(self, atoms, result: CalculationResult, context: RunContext) -> None:  # pragma: no cover - interface
        return None


class CompositeObserver(RunObserver):
    """Dispatch events to multiple observers."""

    def __init__(self, observers: Iterable[RunObserver]):
        self._observers = [observer for observer in observers if observer is not None]

    def on_start(self, atoms, context: RunContext) -> None:
        for observer in self._observers:
            observer.on_start(atoms, context)

    def on_step(self, atoms, step: RunStep, context: RunContext) -> None:
        for observer in self._observers:
            observer.on_step(atoms, step, context)

    def on_finish(self, atoms, result: CalculationResult, context: RunContext) -> None:
        for observer in self._observers:
            observer.on_finish(atoms, result, context)


def coerce_observer(
    observer: RunObserver | Iterable[RunObserver] | None,
) -> RunObserver | None:
    """Normalize observer input to either ``None`` or one composite observer."""

    if observer is None:
        return None
    if isinstance(observer, RunObserver):
        return observer
    return CompositeObserver(observer)


class PrintProgressObserver(RunObserver):
    """Emit the legacy stdout progress lines used by the CLI wrappers."""

    def __init__(self):
        self._previous_energy: float | None = None

    def on_start(self, atoms, context: RunContext) -> None:
        self._previous_energy = None

    def on_step(self, atoms, step: RunStep, context: RunContext) -> None:
        root = _root()
        if context.mode == "md":
            print(
                f"{step.index:7d} T={step.temperature:7.1f} "
                f"E= {root._format_energy_value(step.total_energy)} "
                f"F= {root._format_energy_value(step.potential_energy)} "
                f"E0= {root._format_energy_value(step.potential_energy)}  "
                f"EK= {root._format_energy_value(step.kinetic_energy)} "
                f"SP= {root._format_energy_value(step.thermostat_potential)} "
                f"SK= {root._format_energy_value(step.thermostat_kinetic)}"
            )
            return

        delta = 0.0 if self._previous_energy is None else step.potential_energy - self._previous_energy
        self._previous_energy = step.potential_energy
        print(
            f"{step.index:4d} F= {root._format_energy_value(step.potential_energy)} "
            f"E0= {root._format_energy_value(step.potential_energy)}  "
            f"d E ={root._format_energy_value(delta)}"
        )


class VaspCompatObserver(RunObserver):
    """Bridge pure execution events to legacy VASP-style file outputs."""

    def __init__(self):
        self._recorder = None

    def on_start(self, atoms, context: RunContext) -> None:
        config = context.vasp_compat
        if config is None or not config.enabled:
            return
        self._recorder = _root()._initialize_vasp_compat_outputs(
            atoms,
            ibrion=context.ibrion,
            potim=context.potim,
            mdalgo=context.mdalgo,
            isif=context.isif,
            neb_mode=config.neb_mode,
            write_oszicar_pseudo_scf=config.write_pseudo_scf,
            neb_prev_positions=config.neb_prev_positions,
            neb_next_positions=config.neb_next_positions,
        )

    def on_step(self, atoms, step: RunStep, context: RunContext) -> None:
        config = context.vasp_compat
        if config is None or not config.enabled or self._recorder is None:
            return
        _root()._record_vasp_compat_step(
            self._recorder,
            atoms,
            step_index=step.index,
            potential_energy=step.potential_energy,
            total_energy=step.total_energy,
            kinetic_energy=step.kinetic_energy,
            thermostat_potential=step.thermostat_potential,
            thermostat_kinetic=step.thermostat_kinetic,
            temperature=step.temperature,
            sc_time=step.sc_time,
        )
        if context.mode == "md":
            if config.write_xdatcar and step.advanced:
                _root()._write_xdatcar_step("XDATCAR", atoms, step.index - 1)
            if (
                config.write_lammps_traj
                and step.advanced
                and (step.index - 1) % config.lammps_traj_interval == 0
            ):
                _root()._write_lammps_trajectory_step(
                    config.lammps_traj_path,
                    atoms,
                    step.index - 1,
                )

    def on_finish(self, atoms, result: CalculationResult, context: RunContext) -> None:
        config = context.vasp_compat
        if config is None or not config.enabled or self._recorder is None:
            return
        _root()._write_vasprun_xml(self._recorder, atoms)
        _root()._append_outcar_footer(self._recorder)
        if config.write_contcar:
            _root().write("CONTCAR", atoms, direct=True)
