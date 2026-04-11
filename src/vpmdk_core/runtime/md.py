"""Molecular-dynamics execution flow."""

from __future__ import annotations

import sys
from typing import Dict


def _root():
    return sys.modules["vpmdk_core"]


def _rescale_velocities(atoms, target_temperature: float) -> None:
    """Scale velocities so that kinetic temperature approaches target."""

    root = _root()
    if target_temperature <= 0:
        velocities = atoms.get_velocities()
        if velocities is None:
            zeros = [[0.0, 0.0, 0.0] for _ in range(len(atoms))]
            atoms.set_velocities(zeros)
        else:
            atoms.set_velocities(velocities * 0.0)
        return

    ndof = getattr(atoms, "get_number_of_degrees_of_freedom", lambda: 0)()
    if ndof <= 0:
        root.velocitydistribution.MaxwellBoltzmannDistribution(
            atoms, temperature_K=target_temperature
        )
        return

    kinetic_energy = atoms.get_kinetic_energy()
    if kinetic_energy <= 0:
        root.velocitydistribution.MaxwellBoltzmannDistribution(
            atoms, temperature_K=target_temperature
        )
        return

    current_temperature = 2.0 * kinetic_energy / (ndof * root.units.kB)
    if current_temperature <= 0:
        root.velocitydistribution.MaxwellBoltzmannDistribution(
            atoms, temperature_K=target_temperature
        )
        return

    scaling = (target_temperature / current_temperature) ** 0.5
    velocities = atoms.get_velocities()
    if velocities is None:
        root.velocitydistribution.MaxwellBoltzmannDistribution(
            atoms, temperature_K=target_temperature
        )
        return
    atoms.set_velocities(velocities * scaling)


def _estimate_tdamp(smass: float | None, timestep: float) -> float:
    """Return Nose-Hoover time constant (in fs)."""

    if smass is None or smass == 0:
        return max(100.0 * timestep, timestep)
    return abs(smass)


def _select_md_dynamics(
    atoms,
    mdalgo: int,
    timestep: float,
    initial_temperature: float,
    smass: float | None,
    thermostat_params: Dict[str, float],
):
    """Create ASE molecular dynamics driver and temperature updater."""

    root = _root()
    timestep_ase = timestep * root.units.fs

    def default_update(temp: float) -> None:
        root._rescale_velocities(atoms, temp)

    def make_update(dyn, *, allow_attribute_update: bool = False):
        def update(temp: float) -> None:
            try:
                dyn.set_temperature(temperature_K=temp)
            except TypeError:
                dyn.set_temperature(temp)
            except AttributeError:
                if not allow_attribute_update:
                    raise
                dyn.temp = temp * root.units.kB
                dyn.target_kinetic_energy = 0.5 * dyn.temp * dyn.ndof
            root._rescale_velocities(atoms, temp)

        return update

    if mdalgo == 1:
        if root.Andersen is None:
            raise RuntimeError(
                "Andersen thermostat requested but ase.md.andersen.Andersen "
                "is unavailable. Install the optional ASE thermostat "
                "dependencies or choose a supported MDALGO value."
            )
        andersen_prob = float(thermostat_params.get("ANDERSEN_PROB", 0.1))
        dyn = root.Andersen(
            atoms,
            timestep_ase,
            temperature_K=initial_temperature,
            andersen_prob=andersen_prob,
            logfile="OUTCAR",
        )

        return dyn, make_update(dyn)

    if mdalgo in (2, 4) and root.NoseHooverChainNVT is not None:
        tdamp_fs = _estimate_tdamp(smass, timestep)
        if mdalgo == 2:
            chain_length = int(thermostat_params.get("NHC_NCHAINS", 1))
        else:
            chain_length = int(thermostat_params.get("NHC_NCHAINS", 3))
        dyn = root.NoseHooverChainNVT(
            atoms,
            timestep=timestep_ase,
            temperature_K=initial_temperature,
            tdamp=tdamp_fs * root.units.fs,
            tchain=chain_length,
            logfile="OUTCAR",
        )

        return dyn, make_update(dyn)
    if mdalgo in (2, 4) and root.NoseHooverChainNVT is None and mdalgo != 0:
        raise RuntimeError(
            "Nose-Hoover thermostat requested but ase.md.nose_hoover_chain.NoseHooverChainNVT "
            "is unavailable. Install the optional ASE thermostat dependencies or choose "
            "a supported MDALGO value."
        )

    if mdalgo == 3:
        if root.Langevin is None:
            raise RuntimeError(
                "Langevin thermostat requested but ase.md.langevin.Langevin "
                "is unavailable. Install the optional ASE thermostat dependencies or "
                "choose a supported MDALGO value."
            )
        gamma = thermostat_params.get("LANGEVIN_GAMMA")
        if gamma is None and smass is not None and smass < 0:
            gamma = abs(smass)
        if gamma is None:
            gamma = 1.0
        friction = (float(gamma) / 1000.0) / root.units.fs
        dyn = root.Langevin(
            atoms,
            timestep_ase,
            temperature_K=initial_temperature,
            friction=friction,
            logfile="OUTCAR",
        )

        return dyn, make_update(dyn)

    if mdalgo == 5:
        if root.Bussi is None:
            raise RuntimeError(
                "CSVR thermostat requested but ase.md.bussi.Bussi is unavailable. "
                "Install the optional ASE thermostat dependencies or choose a supported "
                "MDALGO value."
            )
        taut = thermostat_params.get("CSVR_PERIOD")
        if taut is None:
            taut = max(100.0 * timestep, timestep)
        dyn = root.Bussi(
            atoms,
            timestep_ase,
            temperature_K=initial_temperature,
            taut=float(taut) * root.units.fs,
            logfile="OUTCAR",
        )

        return dyn, make_update(dyn, allow_attribute_update=True)

    dyn = root.VelocityVerlet(atoms, timestep_ase, logfile="OUTCAR")
    return dyn, default_update


def run_md(
    atoms,
    calculator,
    steps: int,
    temperature: float,
    timestep: float,
    *,
    mdalgo: int = 0,
    teend: float | None = None,
    smass: float | None = None,
    thermostat_params: Dict[str, float] | None = None,
    isif: int | None = 0,
    oszicar_pseudo_scf: bool = False,
    neb_mode: bool = False,
    neb_prev_positions=None,
    neb_next_positions=None,
    write_lammps_traj: bool = False,
    lammps_traj_interval: int = 1,
    lammps_traj_path: str = "lammps.lammpstrj",
):
    root = _root()
    atoms.calc = root._resolve_calculator(calculator)
    recorder = root._initialize_vasp_compat_outputs(
        atoms,
        ibrion=0,
        isif=isif,
        potim=timestep,
        mdalgo=mdalgo,
        write_oszicar_pseudo_scf=oszicar_pseudo_scf,
        neb_mode=neb_mode,
        neb_prev_positions=neb_prev_positions,
        neb_next_positions=neb_next_positions,
    )
    if temperature <= 0:
        velocities = atoms.get_velocities()
        if velocities is None:
            zeros = [[0.0, 0.0, 0.0] for _ in range(len(atoms))]
            atoms.set_velocities(zeros)
        else:
            atoms.set_velocities(velocities * 0.0)
    else:
        root.velocitydistribution.MaxwellBoltzmannDistribution(
            atoms, temperature_K=temperature
        )
    params = thermostat_params or {}
    dyn, update_temperature = root._select_md_dynamics(
        atoms,
        mdalgo,
        timestep,
        temperature,
        smass,
        params,
    )
    target_end = temperature if teend is None else teend
    md_step = 0

    def _log_md_state() -> None:
        nonlocal md_step
        md_step += 1
        potential_energy = atoms.get_potential_energy()
        try:
            kinetic_energy = atoms.get_kinetic_energy()
        except Exception:
            kinetic_energy = 0.0
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
        total_energy = potential_energy + kinetic_energy + thermostat_potential + thermostat_kinetic
        try:
            temperature_inst = atoms.get_temperature()
        except Exception:
            temperature_inst = 0.0
        print(
            f"{md_step:7d} T={temperature_inst:7.1f} "
            f"E= {root._format_energy_value(total_energy)} "
            f"F= {root._format_energy_value(potential_energy)} "
            f"E0= {root._format_energy_value(potential_energy)}  "
            f"EK= {root._format_energy_value(kinetic_energy)} "
            f"SP= {root._format_energy_value(thermostat_potential)} "
            f"SK= {root._format_energy_value(thermostat_kinetic)}"
        )
        root._record_vasp_compat_step(
            recorder,
            atoms,
            step_index=md_step,
            potential_energy=potential_energy,
            total_energy=total_energy,
            kinetic_energy=kinetic_energy,
            thermostat_potential=thermostat_potential,
            thermostat_kinetic=thermostat_kinetic,
            temperature=temperature_inst,
        )

    for i in range(steps):
        dyn.run(1)
        atoms.wrap()
        _log_md_state()
        root._write_xdatcar_step("XDATCAR", atoms, i)
        if write_lammps_traj and i % lammps_traj_interval == 0:
            root._write_lammps_trajectory_step(lammps_traj_path, atoms, i)
        if steps > 1 and i + 1 < steps and target_end != temperature:
            next_temp = temperature + (target_end - temperature) * (i + 1) / (steps - 1)
            update_temperature(next_temp)
    if not recorder.steps:
        potential_energy = atoms.get_potential_energy()
        kinetic_energy = 0.0
        try:
            kinetic_energy = atoms.get_kinetic_energy()
        except Exception:
            kinetic_energy = 0.0
        root._record_vasp_compat_step(
            recorder,
            atoms,
            step_index=1,
            potential_energy=potential_energy,
            total_energy=potential_energy + kinetic_energy,
            kinetic_energy=kinetic_energy,
            temperature=float(temperature),
        )
    atoms.wrap()
    root._write_vasprun_xml(recorder, atoms)
    root._append_outcar_footer(recorder)
    root.write("CONTCAR", atoms, direct=True)
    return atoms.get_potential_energy()
