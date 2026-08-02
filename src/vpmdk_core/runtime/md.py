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


"""Below this multiple of POTIM the Nose-Hoover chain is reported as stiff.

VPMDK's own default damping time is ``100 * POTIM``, and the ASE chain
integrator diverges to NaN once the damping time approaches the timestep
(measured: with ``tdamp <= 1.5 * dt`` it produces NaN for some cells/seeds and
survives for others, so it is not a hard rejection -- see
_warn_if_thermostat_coupling_is_stiff).
"""
_STIFF_TDAMP_POTIM_RATIO = 10.0

# Ceiling for the Nose-Hoover chain length. Literature and VASP practice use
# 3-10 links; 100 is >10x beyond that while keeping ASE's O(tchain) Python
# integration loop negligible (~ms per substep).
_MAX_NHC_NCHAINS = 100

# Floor for the Nose-Hoover damping time in fs. A tiny-positive SMASS or
# NHC_PERIOD (1e-300) passes the positive-value guards and the 1e9 magnitude
# bound, but the chain mass Q = 3N*kT*tdamp^2 UNDERFLOWS to exactly 0.0 and
# the first MD step is nan -- classified retryable exit 2 for a permanently
# broken INCAR. 1e-6 fs is >100000x below any damping time measured to
# complete (~POTIM scale), so nothing that runs today is rejected.
_MIN_TDAMP_FS = 1e-6


def _warn_if_thermostat_coupling_is_stiff(
    tdamp_fs: float, timestep: float, *, source: str
) -> None:
    """Say out loud when the resolved NHC damping time is near the timestep.

    ``SMASS`` is read here as an ABSOLUTE damping time in femtoseconds, while
    VASP's SMASS is a Nose MASS -- the two are not interchangeable, and no
    document stated the unit (its siblings NHC_PERIOD and CSVR_PERIOD do). An
    ordinary VASP value such as ``SMASS=1.0`` therefore becomes a 1 fs damping
    time: with ``POTIM=2`` that is a thermostat ~200x stiffer than VPMDK's own
    omitted-SMASS default, which pins the temperature instead of sampling the
    canonical ensemble and can make the ASE chain integrator blow up to NaN.

    This WARNS rather than rejecting: whether the integration survives depends on
    the cell, the temperature and the drawn velocities (measured: tdamp = POTIM
    completes for some systems and NaNs for others), so a hard rule would reject
    runs that currently complete. Converting SMASS from VASP's Nose mass to a
    damping time is deliberately NOT done here -- that needs VASP's own Q
    definition, which cannot be verified offline, and silently redefining the tag
    would change results for anyone already using it.
    """

    if timestep <= 0 or tdamp_fs <= 0:
        return
    if tdamp_fs >= _STIFF_TDAMP_POTIM_RATIO * timestep:
        return
    print(
        f"Warning: {source} sets the Nose-Hoover damping time to {tdamp_fs:g} fs, "
        f"only {tdamp_fs / timestep:g}x POTIM={timestep:g} fs (VPMDK's default is "
        "100x POTIM). Such strong coupling pins the temperature instead of "
        "sampling the canonical ensemble and can diverge. Note SMASS is read as "
        "a damping time in fs, not as VASP's Nose mass; NHC_PERIOD sets it in MD "
        "steps."
    )


def _warn_md_is_fixed_cell(*, isif: int | None, pstress: float | None) -> None:
    """Say so when an MD INCAR asks for cell dynamics VPMDK does not do.

    In real VASP, ``IBRION=0`` with ``ISIF>=3`` (plus ``MDALGO``/``PSTRESS``)
    is the documented NPT / Parrinello-Rahman mode: the cell responds to the
    pressure. VPMDK's MD integrates IONS ONLY -- no barostat exists and the
    cell never moves -- yet every artifact actively claimed the pressure
    ensemble: ISIF/MDALGO/PSTRESS echoed, per-step 'Pullay stress' lines, the
    enthalpy E+PV in vasprun. An EOS/density consumer read a fixed-volume NVT
    trajectory as equilibrated NPT with exit 0 and no warning. Same
    warn-don't-reject remedy as the R132 MDALGO normalization: the run is
    still a valid fixed-cell trajectory, so completing inputs keep completing.
    """

    wants_cell = isif is not None and isif >= 3
    wants_pressure = pstress is not None and pstress != 0.0
    if not (wants_cell or wants_pressure):
        return
    requested = []
    if wants_cell:
        requested.append(f"ISIF={isif}")
    if wants_pressure:
        requested.append(f"PSTRESS={pstress:g}")
    print(
        f"Warning: {' and '.join(requested)} requests cell dynamics, but VPMDK "
        "MD is FIXED-CELL (NVE/NVT): no barostat is applied and the cell never "
        "moves. ISIF/PSTRESS only affect the stress and enthalpy OUTPUT "
        "conventions. Real VASP would run NPT here; use a relaxation (IBRION=2, "
        "ISIF=3) to equilibrate the cell instead."
    )


def _reject_underflowing_tdamp(tdamp_fs: float, *, source: str) -> None:
    if 0 < tdamp_fs < _MIN_TDAMP_FS:
        raise _root().WorkdirInputError(
            f"{source} resolves to a Nose-Hoover damping time of {tdamp_fs:g} "
            f"fs, below the supported minimum of {_MIN_TDAMP_FS:g} fs: the "
            "thermostat mass underflows to zero and the first MD step is nan."
        )


def _estimate_tdamp(
    smass: float | None,
    timestep: float,
    thermostat_params: Dict[str, float] | None = None,
) -> float:
    """Return Nose-Hoover time constant (in fs)."""

    if thermostat_params is not None and "NHC_PERIOD" in thermostat_params:
        nhc_period = float(thermostat_params["NHC_PERIOD"])
        if nhc_period <= 0:
            # A user-supplied INCAR value that is out of range: a fix-your-input
            # condition (exit 1), not a retryable calculation failure (exit 2).
            raise _root().WorkdirInputError(
                "NHC_PERIOD must be positive for VPMDK Nose-Hoover chain runs. "
                "VASP uses NHC_PERIOD=0 to switch to NVE; use MDALGO=0 in VPMDK "
                "for that mode."
            )
        tdamp = nhc_period * timestep
        _reject_underflowing_tdamp(tdamp, source="NHC_PERIOD")
        _warn_if_thermostat_coupling_is_stiff(tdamp, timestep, source="NHC_PERIOD")
        return tdamp
    if smass is None or smass == 0:
        return max(100.0 * timestep, timestep)
    tdamp = abs(smass)
    _reject_underflowing_tdamp(tdamp, source="SMASS")
    _warn_if_thermostat_coupling_is_stiff(tdamp, timestep, source="SMASS")
    return tdamp


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
        # A non-positive ramp target comes from TEBEG/TEEND (user INCAR input), so
        # classify it as input_error (exit 1) rather than routing it through the
        # ASE-compatibility helper (which reports a calculation_error, exit 2).
        raise root.WorkdirInputError(
            "Nose-Hoover chain temperature ramp reached a non-positive target "
            "temperature; use a positive TEBEG/TEEND range or a non-NHC "
            "integrator for zero-temperature runs."
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

    # Deliberately NO velocity rescaling here. Retargeting the chain (_kT and the
    # thermostat masses) is the whole ramp: the chain then drives the system to
    # the new temperature over its own damping time, which is what a Nose-Hoover
    # ramp means. Rescaling the velocities on top of it -- once per ionic step,
    # since execute_md calls this after EVERY step of a TEBEG->TEEND run --
    # replaced the canonical ensemble with an ISOKINETIC trajectory: measured
    # temperature spread collapsed from 85.3 K to 12.3 K and it pumped in energy
    # that neither the Nose SP/SK terms nor Bussi's transferred_energy account
    # for, so the total energy the previous rounds made conserved drifted 2.37 eV
    # over 150 steps instead of 0.0013 eV. Every property read off canonical
    # fluctuations (heat capacity, RDF broadening, velocity autocorrelation) was
    # wrong, with exit 0 and MDALGO=2 still recorded in OUTCAR/vasprun.


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
    # Measured per integrator in ase 3.29 with real velocities (POTIM reaches the
    # thermostats both directly as dt and through _estimate_tdamp = 100*POTIM):
    #   POTIM = 0: MDALGO 1/2/3/4 produce nan positions with NO exception, and 5
    #              raises ZeroDivisionError -- every thermostat mass Q collapses to
    #              zero. MDALGO 0 (plain NVE) simply does not move and completes.
    #   POTIM < 0: only MDALGO 3 breaks (UFuncTypeError from a complex sqrt); the
    #              others integrate backwards in time and complete.
    # Reject exactly what breaks: anything that merely completes today keeps doing
    # so (SPEC 1.1). Left unguarded, the nan surfaced far downstream as a bare
    # ValueError, i.e. calculation_error = exit 2 = documented RETRYABLE for a
    # permanently invalid INCAR -- the same class as the POTIM guard IBRION=5/6
    # already has.
    if mdalgo != 0 and timestep == 0.0:
        raise root.WorkdirInputError(
            "POTIM must be non-zero for thermostatted molecular dynamics "
            f"(MDALGO={mdalgo}); a zero time step leaves the thermostat undefined."
        )
    if mdalgo == 3 and timestep < 0.0:
        raise root.WorkdirInputError(
            "POTIM must be positive for Langevin dynamics (MDALGO=3); "
            f"got {timestep:g}."
        )
    timestep_ase = timestep * root.units.fs

    # A TEBEG->TEEND ramp retargets the THERMOSTAT; it does not rescale the
    # velocities. execute_md calls the updater after every ionic step of a ramp,
    # so a rescale there pins the instantaneous kinetic temperature to the ramp
    # line and turns a requested Nose-Hoover / Langevin / Andersen / CSVR run
    # into an isokinetic one (measured: temperature spread 85.3 K -> 12.3 K,
    # conserved-energy drift 0.0013 eV -> 2.37 eV over 150 steps) while
    # OUTCAR/vasprun still report the requested MDALGO. Only MDALGO=0 keeps
    # rescaling: for plain NVE there is no thermostat to retarget, so rescaling
    # IS the ramp.
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
                # ase.md.bussi.Bussi has no setter and reads these two directly.
                dyn.temp = temp * root.units.kB
                dyn.target_kinetic_energy = 0.5 * dyn.temp * dyn.ndof

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

        return update

    if mdalgo == 1:
        if root.Andersen is None:
            raise RuntimeError(
                "Andersen thermostat requested but ase.md.andersen.Andersen "
                "is unavailable. Install the optional ASE thermostat "
                "dependencies or choose a supported MDALGO value."
            )
        if "ANDERSEN_PROB" not in (thermostat_params or {}):
            # Real VASP's documented default is ANDERSEN_PROB=0, i.e. a
            # collision-free (microcanonical) trajectory; VPMDK's legacy
            # default of 0.1 samples a strongly-coupled Andersen ensemble
            # instead (measured: conserved-energy range 0.21 eV vs 0.0007 eV
            # over 25 steps). Changing the default would alter existing runs
            # (SPEC 1.1), so it is DISCLOSED instead, like the MDALGO and
            # fixed-cell warnings.
            print(
                "Warning: MDALGO=1 without ANDERSEN_PROB uses VPMDK's default "
                "collision probability 0.1 per atom per step. Real VASP "
                "defaults to ANDERSEN_PROB=0 (no collisions, NVE); write "
                "ANDERSEN_PROB = 0.0 for that behavior."
            )
        andersen_prob = float(thermostat_params.get("ANDERSEN_PROB", 0.1))
        # ASE's Andersen keeps fixcm=True: the total momentum is zeroed every
        # step, so only 3N-3 kinetic degrees of freedom are populated -- but
        # the OSZICAR/stdout temperature divides by all 3N (no FixCom
        # constraint is registered), so the reported T reads (3N-3)/3N of
        # TEBEG (-25% for 4 atoms; measured 225 K for TEBEG=300). The sampled
        # ensemble itself is at TEBEG and real VASP reports over its DOF =
        # 3N-3, so the NUMBER is a disclosed convention divergence
        # (warn-don't-change, the POMASS/LCLIMB precedent): rescaling the
        # reported value would silently change every existing OSZICAR.
        atom_count = len(atoms)
        if atom_count > 0:
            print(
                "Warning: MDALGO=1 (Andersen) freezes the center of mass, but "
                "the reported temperature divides by all 3N degrees of "
                "freedom, so OSZICAR/stdout read about "
                f"{100.0 * (3 * atom_count - 3) / (3 * atom_count):.0f}% of "
                "TEBEG for this cell. The sampled ensemble is at TEBEG; VASP "
                "reports over 3N-3."
            )
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
            # TEBEG=0 (or any non-positive initial temperature) is bad INCAR input,
            # caught at setup before any dynamics run: exit 1 (input), not exit 2.
            raise root.WorkdirInputError(
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
            # NHC_NCHAINS=0 is an out-of-range user INCAR value: exit 1 (input),
            # not a retryable calculation failure.
            raise root.WorkdirInputError(
                "NHC_NCHAINS must be at least 1 for VPMDK Nose-Hoover chain "
                "runs. VASP uses NHC_NCHAINS=0 to switch off the thermostat; "
                "use MDALGO=0 in VPMDK for NVE dynamics."
            )
        if chain_length > _MAX_NHC_NCHAINS:
            # ASE integrates the chain in a Python loop whose per-substep cost
            # and state arrays are O(tchain), so a finite-huge NHC_NCHAINS
            # (1e8: ~25 min PER IONIC STEP; 1e12: a 7 TiB allocation) either
            # wedges a resident worker for weeks or dies mid-run as a
            # "retryable" MemoryError after writing partial outputs. Chains
            # beyond ~10 links add nothing physically; judge the value at
            # input time.
            raise root.WorkdirInputError(
                f"NHC_NCHAINS={chain_length} exceeds the supported maximum of "
                f"{_MAX_NHC_NCHAINS}. Nose-Hoover chains longer than ~10 "
                "links have no physical effect; this magnitude only makes the "
                "integrator arbitrarily slow."
            )
        if getattr(atoms, "constraints", None):
            # ASE's NoseHooverChainNVT integrates its OWN _q/_p arrays and
            # never re-applies constraints to them: get_forces(md=True) skips
            # adjust_forces for FixAtoms/FixCartesian/FixScaled, and set_momenta
            # constrains only the atoms' copy, not the integrator's. Frozen
            # atoms therefore accumulate phantom momenta that the thermostat
            # counts against its target -- which is itself hard-coded to
            # 3*N_global*kT with no constrained-DOF reduction -- so a POSCAR
            # with selective dynamics sampled 25-85 K where TEBEG said 300 K,
            # with exit 0 (measured: 16/32 frozen -> ~25 K; even 1/32 frozen
            # -> ~82 K). Langevin/Andersen/CSVR write momenta back through
            # set_momenta every step and are unaffected. Until the integrator
            # honors constraints, this combination must be an explicit input
            # error, not a silently wrong trajectory.
            raise root.UnsupportedInputError(
                "MDALGO=2/4 (Nose-Hoover chain) does not support constrained "
                "atoms (POSCAR selective dynamics): ASE's integrator keeps "
                "internal momenta that ignore the constraints, which distorts "
                "the sampled temperature far below TEBEG. Remove the "
                "selective-dynamics flags or use MDALGO=1 (Andersen), "
                "MDALGO=3 (Langevin), or MDALGO=5 (CSVR), which handle "
                "constraints correctly."
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
        elif float(gamma) < 0.0:
            # ase.md.langevin computes sigma = sqrt(2 * T * friction / masses), so a
            # negative friction yields nan velocities on the very first step with NO
            # exception -- the nan only surfaces later as a raw ValueError from the
            # energy formatter, i.e. calculation_error = exit 2 (RETRYABLE) for a
            # permanently invalid INCAR. GAMMA=0 is a legal "no damping" limit and
            # stays accepted.
            raise _root().WorkdirInputError(
                "LANGEVIN_GAMMA must not be negative; a negative friction makes the "
                f"Langevin thermostat produce nan velocities. Got {float(gamma):g}."
            )
        if len(atoms) < 2:
            # ase.md.langevin's fixcm=True default computes
            # sqrt(natoms/(natoms - 1)): for a single atom that is a
            # ZeroDivisionError on the first step, classified retryable
            # exit 2 for a fixed property of the input (one-shot exits 1 on
            # the identical tree). Removing the COM of one atom would freeze
            # it anyway, so no 1-atom Langevin run is meaningful; MDALGO
            # 0/1/2/4/5 all handle a single atom fine (measured).
            raise root.WorkdirInputError(
                "MDALGO=3 (Langevin) requires at least two atoms: ASE's "
                "integrator removes the center-of-mass motion and divides "
                "by N-1, which is a division by zero for a single atom. "
                "Use another MDALGO or a supercell."
            )
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
        elif float(taut) <= 0.0:
            # Same class as the NHC_PERIOD guard above: ase.md.bussi.Bussi is
            # mathematically undefined here (taut=0 raises ZeroDivisionError in
            # exp(-dt/taut); taut<0 raises "math domain error" from the sqrt in
            # calculate_alpha), so the run cannot proceed at all. Without this the
            # failure surfaced MID-RUN as calculation_error, i.e. exit 2, which
            # SERVER_MODE_SPEC 2.5 documents as RETRYABLE -- so a retry driver
            # resubmits a permanently invalid INCAR forever. This is not the
            # "invalid values are ignored with warnings" case that applies to tags
            # ASE tolerates (ANDERSEN_PROB, LANGEVIN_GAMMA): there is no value to
            # fall back to that the user asked for.
            raise _root().WorkdirInputError(
                "CSVR_PERIOD must be positive for VPMDK CSVR (Bussi) runs; "
                f"got {float(taut):g}. Use MDALGO=0 for plain NVE dynamics."
            )
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
    pstress: float | None = None,
):
    _warn_md_is_fixed_cell(isif=isif, pstress=pstress)
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
            # See run_single_point: the PSTRESS output correction applies in
            # every mode that prints stress, not just relaxations.
            pstress_kbar=pstress,
            nsw_requested=steps,
        ),
    )
    return result.potential_energy
