"""Molecular-dynamics execution flow."""

from __future__ import annotations

import sys
from typing import Dict

from ..compat.vasp import VaspCompatConfig, VaspMDConfig
from ..models import MDConfig
from ..observers import PrintProgressObserver, VaspCompatObserver


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


def _estimate_tdamp(
    smass: float | None,
    timestep: float,
    thermostat_params: Dict[str, float] | None = None,
) -> float:
    """Return Nose-Hoover time constant (in fs)."""

    if thermostat_params is not None and "NHC_PERIOD" in thermostat_params:
        nhc_period = float(thermostat_params["NHC_PERIOD"])
        if nhc_period <= 0:
            raise RuntimeError(
                "NHC_PERIOD must be positive for VPMDK Nose-Hoover chain runs. "
                "VASP uses NHC_PERIOD=0 to switch to NVE; use MDALGO=0 in VPMDK "
                "for that mode."
            )
        return nhc_period * timestep
    if smass is None or smass == 0:
        return max(100.0 * timestep, timestep)
    return abs(smass)


def _raise_nose_hoover_chain_temperature_error(reason: str) -> None:
    """Raise a clear error for unsupported ASE Nose-Hoover chain updates."""

    raise RuntimeError(
        "Temperature ramping with Nose-Hoover chain requires ASE "
        "NoseHooverChainNVT to provide set_temperature() or the expected "
        "3.27-style internal thermostat state. "
        f"{reason}"
    )


def _set_nose_hoover_chain_temperature(dyn, atoms, target_temperature: float) -> None:
    """Update ASE NoseHooverChainNVT target temperature for a ramp step."""

    root = _root()
    target_temperature = float(target_temperature)
    if target_temperature <= 0:
        _raise_nose_hoover_chain_temperature_error(
            "The requested target temperature is not positive; use a positive "
            "TEBEG/TEEND range or a non-NHC integrator for zero-temperature runs."
        )

    thermostat = getattr(dyn, "_thermostat", None)
    missing = []
    if thermostat is None:
        missing.append("_thermostat")
    else:
        for attr in ("_Q", "_kT", "_num_atoms_global", "_tdamp"):
            if not hasattr(thermostat, attr):
                missing.append(f"_thermostat.{attr}")
    if not hasattr(dyn, "_p"):
        missing.append("_p")
    if missing:
        _raise_nose_hoover_chain_temperature_error(
            "Missing attributes: " + ", ".join(missing) + "."
        )

    q = thermostat._Q
    try:
        chain_length = len(q)
    except TypeError:
        _raise_nose_hoover_chain_temperature_error(
            "_thermostat._Q is not an indexable thermostat-mass array."
        )
    if chain_length < 1:
        _raise_nose_hoover_chain_temperature_error(
            "_thermostat._Q is empty; at least one thermostat mass is required."
        )

    try:
        num_atoms_global = float(thermostat._num_atoms_global)
        tdamp = float(thermostat._tdamp)
    except (TypeError, ValueError):
        _raise_nose_hoover_chain_temperature_error(
            "Could not interpret Nose-Hoover chain atom count or damping time."
        )
    if num_atoms_global <= 0 or tdamp <= 0:
        _raise_nose_hoover_chain_temperature_error(
            "Nose-Hoover chain atom count and damping time must be positive."
        )

    kT = root.units.kB * target_temperature
    thermostat._kT = kT
    first_mass = 3.0 * num_atoms_global * kT * tdamp**2
    remaining_mass = kT * tdamp**2
    try:
        q[0] = first_mass
        if chain_length > 1:
            try:
                q[1:] = remaining_mass
            except TypeError:
                for index in range(1, chain_length):
                    q[index] = remaining_mass
    except (TypeError, ValueError, IndexError):
        _raise_nose_hoover_chain_temperature_error(
            "_thermostat._Q is not a mutable thermostat-mass array."
        )

    root._rescale_velocities(atoms, target_temperature)
    dyn._p = atoms.get_momenta()


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

    def make_nose_hoover_chain_update(dyn):
        def update(temp: float) -> None:
            setter = getattr(dyn, "set_temperature", None)
            if setter is None:
                root._set_nose_hoover_chain_temperature(dyn, atoms, temp)
                return
            try:
                setter(temperature_K=temp)
            except TypeError:
                setter(temp)
            root._rescale_velocities(atoms, temp)
            if hasattr(dyn, "_p"):
                dyn._p = atoms.get_momenta()

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
        if initial_temperature <= 0:
            raise RuntimeError(
                "Nose-Hoover chain dynamics require a positive initial "
                "temperature. Use a positive TEBEG/temperature value or "
                "MDALGO=0 for zero-temperature velocity-Verlet dynamics."
            )
        tdamp_fs = _estimate_tdamp(smass, timestep, thermostat_params)
        if mdalgo == 2:
            chain_length = int(thermostat_params.get("NHC_NCHAINS", 1))
        else:
            chain_length = int(thermostat_params.get("NHC_NCHAINS", 3))
        if chain_length < 1:
            raise RuntimeError(
                "NHC_NCHAINS must be at least 1 for VPMDK Nose-Hoover chain "
                "runs. VASP uses NHC_NCHAINS=0 to switch off the thermostat; "
                "use MDALGO=0 in VPMDK for NVE dynamics."
            )
        dyn = root.NoseHooverChainNVT(
            atoms,
            timestep=timestep_ase,
            temperature_K=initial_temperature,
            tdamp=tdamp_fs * root.units.fs,
            tchain=chain_length,
            logfile="OUTCAR",
        )

        return dyn, make_nose_hoover_chain_update(dyn)
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
    result = _root().md(
        atoms,
        calculator=calculator,
        config=MDConfig(
            steps=steps,
            temperature=temperature,
            timestep_fs=timestep,
            thermostat="nve",
            temperature_end=teend,
            thermostat_kwargs=dict(thermostat_params or {}),
            smass=smass,
            compat=VaspMDConfig(isif=isif, mdalgo=mdalgo),
        ),
        observer=[VaspCompatObserver(), PrintProgressObserver()],
        compatibility=VaspCompatConfig(
            enabled=True,
            write_pseudo_scf=oszicar_pseudo_scf,
            write_contcar=True,
            write_xdatcar=True,
            write_lammps_traj=write_lammps_traj,
            lammps_traj_interval=lammps_traj_interval,
            lammps_traj_path=lammps_traj_path,
            neb_mode=neb_mode,
            neb_prev_positions=neb_prev_positions,
            neb_next_positions=neb_next_positions,
        ),
    )
    return result.potential_energy
