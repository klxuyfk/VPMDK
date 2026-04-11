"""VASP-compatible output writers used across execution modes."""

from __future__ import annotations

import os
import sys
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, List

import numpy as np


def _root():
    return sys.modules["vpmdk_core"]


@dataclass
class _VasprunStep:
    """Container for one ionic step written to ``vasprun.xml``."""

    cell: list[list[float]]
    scaled_positions: list[list[float]]
    forces: list[list[float]]
    stress: list[list[float]] | None
    potential_energy: float
    total_energy: float
    kinetic_energy: float
    thermostat_potential: float
    thermostat_kinetic: float
    temperature: float
    sc_time: float = 0.0


@dataclass(frozen=True)
class _PseudoScfSettings:
    """Pseudo electronic-step settings used for VASP-compatibility output."""

    enabled: bool = False
    nelm: int = 60
    nelmin: int = 2
    nelmdl: int = 0
    ediff: float = 1.0e-4


_PSEUDO_SCF_INCAR_TAGS = frozenset({"NELM", "NELMIN", "NELMDL", "EDIFF"})
_ACTIVE_PSEUDO_SCF_SETTINGS: _PseudoScfSettings | None = None


@dataclass(frozen=True)
class _VaspInputPaths:
    """Selected run input paths reused by compatibility writers."""

    incar_path: str | None = None
    potcar_path: str | None = None
    kpoints_path: str | None = None


_ACTIVE_VASP_INPUT_PATHS: _VaspInputPaths | None = None


@dataclass
class _VaspCompatRecorder:
    """State tracker for VASP-like ``OUTCAR``/``OSZICAR``/``vasprun.xml`` output."""

    symbols: List[str]
    initial_cell: list[list[float]]
    initial_scaled_positions: list[list[float]]
    ibrion: int
    potim: float | None
    mdalgo: int | None
    isif: int | None = None
    stress_mode: str = "none"
    neb_mode: bool = False
    pseudo_scf: _PseudoScfSettings = field(default_factory=_PseudoScfSettings)
    oszicar_scf_header_written: bool = False
    neb_prev_positions: np.ndarray | None = None
    neb_next_positions: np.ndarray | None = None
    started_at: float = field(default_factory=time.perf_counter)
    previous_energy: float | None = None
    steps: List[_VasprunStep] = field(default_factory=list)


@dataclass
class _NebImageResult:
    """Final-step summary extracted from one NEB image directory."""

    image_name: str
    atoms: Any
    potential_energy: float
    forces: np.ndarray
    stress: np.ndarray | None


@dataclass
class _NebChainApproximation:
    """Approximate NEB chain components derived from neighboring images."""

    tangential_force: float
    tangent_vectors: np.ndarray
    chain_force_vectors: np.ndarray
    chain_plus_total: np.ndarray


def _coerce_neb_reference_positions(values) -> np.ndarray | None:
    """Return neighbor image positions as ``(n_atoms, 3)`` array when valid."""

    if values is None:
        return None
    try:
        array = np.asarray(values, dtype=float)
    except Exception:
        return None
    if array.ndim != 2 or array.shape[1] != 3:
        return None
    return np.array(array, dtype=float, copy=True)


def _matrix_to_nested_list(values) -> list[list[float]]:
    """Return ``values`` as nested Python ``float`` lists."""

    return np.asarray(values, dtype=float).tolist()


def _safe_get_forces(atoms) -> np.ndarray:
    """Return per-atom forces or zeros when unavailable."""

    try:
        return np.asarray(atoms.get_forces(), dtype=float)
    except Exception:
        return np.zeros((len(atoms), 3), dtype=float)


def _stress_mode_from_isif(isif: int | None) -> str:
    """Return stress output mode from VASP ``ISIF`` semantics."""

    if isif is None:
        return "none"
    if isif <= 0:
        return "none"
    if isif == 1:
        return "trace"
    return "full"


def _voigt_to_full_stress(stress_voigt: np.ndarray) -> np.ndarray:
    """Convert ASE Voigt stress ``[xx, yy, zz, yz, xz, xy]`` to 3x3 matrix."""

    xx, yy, zz, yz, xz, xy = [float(v) for v in stress_voigt]
    return np.array(
        [
            [xx, xy, xz],
            [xy, yy, yz],
            [xz, yz, zz],
        ],
        dtype=float,
    )


def _full_to_voigt_stress(stress_matrix: np.ndarray) -> np.ndarray:
    """Convert full 3x3 stress matrix to ASE Voigt convention."""

    return np.array(
        [
            float(stress_matrix[0, 0]),
            float(stress_matrix[1, 1]),
            float(stress_matrix[2, 2]),
            float(stress_matrix[1, 2]),
            float(stress_matrix[0, 2]),
            float(stress_matrix[0, 1]),
        ],
        dtype=float,
    )


def _safe_get_stress_matrix(atoms, *, mode: str) -> np.ndarray | None:
    """Return stress matrix in eV/A^3 based on output mode."""

    if mode == "none":
        return None

    try:
        raw = np.asarray(atoms.get_stress(voigt=True), dtype=float)
    except Exception:
        return None

    if raw.shape == (6,):
        stress_voigt = raw
    elif raw.shape == (3, 3):
        stress_voigt = _full_to_voigt_stress(raw)
    else:
        return None

    if mode == "trace":
        mean_pressure = float(np.mean(stress_voigt[:3]))
        stress_voigt = np.array([mean_pressure, mean_pressure, mean_pressure, 0.0, 0.0, 0.0])

    return _voigt_to_full_stress(stress_voigt)


def _estimate_neb_chain_approximation(
    *,
    positions: np.ndarray,
    forces: np.ndarray,
    prev_positions: np.ndarray | None,
    next_positions: np.ndarray | None,
) -> _NebChainApproximation | None:
    """Estimate NEB chain vectors from neighboring image displacements."""

    if positions.shape != forces.shape or positions.ndim != 2 or positions.shape[1] != 3:
        return None

    prev = prev_positions if prev_positions is not None and prev_positions.shape == positions.shape else None
    nxt = next_positions if next_positions is not None and next_positions.shape == positions.shape else None

    if prev is not None and nxt is not None:
        tangent_raw = nxt - prev
    elif nxt is not None:
        tangent_raw = nxt - positions
    elif prev is not None:
        tangent_raw = positions - prev
    else:
        tangent_raw = np.zeros_like(forces)

    tangent_norm = float(np.linalg.norm(tangent_raw.ravel()))
    if tangent_norm <= 1e-14:
        tangent_vectors = np.zeros_like(forces)
        tangential_force = 0.0
        chain_force_vectors = np.zeros_like(forces)
    else:
        tangent_vectors = tangent_raw / tangent_norm
        tangential_force = float(np.dot(forces.ravel(), tangent_vectors.ravel()))
        chain_force_vectors = tangent_vectors * tangential_force

    if forces.size:
        chain_plus_total = np.sum(chain_force_vectors + forces, axis=0)
    else:
        chain_plus_total = np.zeros(3, dtype=float)

    return _NebChainApproximation(
        tangential_force=tangential_force,
        tangent_vectors=tangent_vectors,
        chain_force_vectors=chain_force_vectors,
        chain_plus_total=chain_plus_total,
    )


def _read_non_comment_lines(path: str) -> list[str]:
    """Return stripped non-empty lines with ``#``/``!`` comments removed."""

    if not os.path.exists(path):
        return []
    lines: list[str] = []
    with open(path, encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            for marker in ("#", "!"):
                if marker in line:
                    line = line.split(marker, 1)[0]
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
    return lines


def _extract_potcar_titles(path: str) -> list[str]:
    """Return POTCAR TITEL strings from ``path`` when available."""

    if not os.path.exists(path):
        return []
    titles: list[str] = []
    with open(path, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "TITEL" not in line or "=" not in line:
                continue
            title = line.split("=", 1)[1].strip()
            if title:
                titles.append(title)
    return titles


def _append_outcar_metadata_header(handle, atoms) -> None:
    """Append VASP-like metadata/header blocks to ``OUTCAR``."""

    paths = _ACTIVE_VASP_INPUT_PATHS or _VaspInputPaths("INCAR", "POTCAR", "KPOINTS")
    incar_lines = _read_non_comment_lines(paths.incar_path) if paths.incar_path else []
    if incar_lines:
        handle.write(" INCAR:\n")
        for line in incar_lines:
            handle.write(f"   {line}\n")

    potcar_titles = _extract_potcar_titles(paths.potcar_path) if paths.potcar_path else []
    if potcar_titles:
        for title in potcar_titles:
            handle.write(f" POTCAR:    {title}\n")
    elif len(atoms):
        seen: OrderedDict[str, None] = OrderedDict()
        for symbol in atoms.get_chemical_symbols():
            seen.setdefault(symbol, None)
        for symbol in seen:
            handle.write(f" POTCAR:    PAW_PBE {symbol}\n")

    cell = np.asarray(atoms.get_cell().array, dtype=float)
    if abs(np.linalg.det(cell)) > 1e-14:
        reciprocal = np.linalg.inv(cell).T
    else:
        reciprocal = np.zeros((3, 3), dtype=float)
    handle.write("      direct lattice vectors                 reciprocal lattice vectors\n")
    for direct, recip in zip(cell, reciprocal):
        handle.write(
            f"  {direct[0]:12.9f} {direct[1]:12.9f} {direct[2]:12.9f}"
            f"   {recip[0]:12.9f} {recip[1]:12.9f} {recip[2]:12.9f}\n"
        )

    kp_lines = _read_non_comment_lines(paths.kpoints_path) if paths.kpoints_path else []
    kpoint_label = "Gamma"
    if len(kp_lines) >= 3:
        kpoint_label = kp_lines[2]
    handle.write(
        f" k-points in reciprocal lattice and weights: {kpoint_label:<40}\n"
    )
    handle.write("   0.00000000   0.00000000   0.00000000      1.00000000\n\n")


def _append_kpoints_xml(parent) -> None:
    """Append a minimal Gamma-only ``kpoints`` section."""

    kpoints = ET.SubElement(parent, "kpoints")
    ET.SubElement(kpoints, "generation", {"param": "Gamma"})
    kpointlist = ET.SubElement(kpoints, "varray", {"name": "kpointlist"})
    ET.SubElement(kpointlist, "v").text = "       0.00000000       0.00000000       0.00000000 "
    weights = ET.SubElement(kpoints, "varray", {"name": "weights"})
    ET.SubElement(weights, "v").text = "       1.00000000 "


def _pseudo_scf_settings_from_incar(incar, *, enabled: bool) -> _PseudoScfSettings:
    """Return pseudo-SCF settings derived from the selected run ``INCAR``."""

    if not enabled:
        return _PseudoScfSettings(enabled=False)

    if not hasattr(incar, "get"):
        return _PseudoScfSettings(enabled=enabled)

    def _parse_int_tag(key: str, default: int) -> int:
        raw = incar.get(key, default)
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            return default

    def _parse_float_tag(key: str, default: float) -> float:
        raw = incar.get(key, default)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    nelm = max(1, _parse_int_tag("NELM", 60))
    nelmin = min(max(1, _parse_int_tag("NELMIN", 2)), nelm)
    nelmdl = _parse_int_tag("NELMDL", 0)
    ediff = max(_parse_float_tag("EDIFF", 1.0e-4), 0.0)
    return _PseudoScfSettings(
        enabled=enabled,
        nelm=nelm,
        nelmin=nelmin,
        nelmdl=nelmdl,
        ediff=ediff,
    )


@contextmanager
def _active_pseudo_scf_settings(settings: _PseudoScfSettings):
    """Temporarily expose pseudo-SCF settings to nested output writers."""

    global _ACTIVE_PSEUDO_SCF_SETTINGS
    previous = _ACTIVE_PSEUDO_SCF_SETTINGS
    _ACTIVE_PSEUDO_SCF_SETTINGS = settings
    try:
        yield
    finally:
        _ACTIVE_PSEUDO_SCF_SETTINGS = previous


@contextmanager
def _active_vasp_input_paths(paths: _VaspInputPaths):
    """Temporarily expose selected run input paths to compatibility writers."""

    global _ACTIVE_VASP_INPUT_PATHS
    previous = _ACTIVE_VASP_INPUT_PATHS
    _ACTIVE_VASP_INPUT_PATHS = paths
    try:
        yield
    finally:
        _ACTIVE_VASP_INPUT_PATHS = previous


def _selected_incar_path() -> str:
    """Return the active run ``INCAR`` path or the caller's local ``INCAR``."""

    paths = _ACTIVE_VASP_INPUT_PATHS or _VaspInputPaths()
    return paths.incar_path or "INCAR"


def _resolve_pseudo_scf_settings(*, enabled: bool) -> _PseudoScfSettings:
    """Return pseudo-SCF settings from the active run or selected ``INCAR``."""

    if _ACTIVE_PSEUDO_SCF_SETTINGS is not None:
        active = _ACTIVE_PSEUDO_SCF_SETTINGS
        return _PseudoScfSettings(
            enabled=enabled,
            nelm=active.nelm,
            nelmin=active.nelmin,
            nelmdl=active.nelmdl,
            ediff=active.ediff,
        )
    if not enabled:
        return _PseudoScfSettings(enabled=False)
    return _pseudo_scf_settings_from_incar(_root()._load_incar(_selected_incar_path()), enabled=True)


def _format_outcar_ediff(value: float) -> str:
    """Return VASP-like scientific notation for ``EDIFF`` lines in ``OUTCAR``."""

    if value == 0.0:
        return "0.0E+00"
    mantissa_text, exponent_text = f"{value:.8E}".split("E")
    digits = mantissa_text.replace(".", "").rstrip("0")
    exponent = int(exponent_text) + 1
    return f"0.{digits or '0'}E{exponent:+03d}"


def _initialize_vasp_compat_outputs(
    atoms,
    *,
    ibrion: int,
    potim: float | None = None,
    mdalgo: int | None = None,
    isif: int | None = None,
    neb_mode: bool = False,
    write_oszicar_pseudo_scf: bool = False,
    neb_prev_positions: np.ndarray | None = None,
    neb_next_positions: np.ndarray | None = None,
) -> _VaspCompatRecorder:
    """Initialize compatibility outputs and return recorder state."""

    pseudo_scf = _resolve_pseudo_scf_settings(enabled=write_oszicar_pseudo_scf)
    initial_cell = _matrix_to_nested_list(atoms.get_cell().array)
    initial_scaled_positions = _matrix_to_nested_list(atoms.get_scaled_positions())
    current_positions = np.asarray(atoms.get_positions(), dtype=float)
    prev_positions = _coerce_neb_reference_positions(neb_prev_positions)
    if prev_positions is not None and prev_positions.shape != current_positions.shape:
        prev_positions = None
    next_positions = _coerce_neb_reference_positions(neb_next_positions)
    if next_positions is not None and next_positions.shape != current_positions.shape:
        next_positions = None
    recorder = _VaspCompatRecorder(
        symbols=list(atoms.get_chemical_symbols()),
        initial_cell=initial_cell,
        initial_scaled_positions=initial_scaled_positions,
        ibrion=ibrion,
        potim=potim,
        mdalgo=mdalgo,
        isif=isif,
        stress_mode=_stress_mode_from_isif(isif),
        neb_mode=neb_mode,
        pseudo_scf=pseudo_scf,
        neb_prev_positions=prev_positions,
        neb_next_positions=next_positions,
    )

    with open("OUTCAR", "w", encoding="utf-8") as handle:
        handle.write(" vasp.6.x compatible output generated by VPMDK\n")
        handle.write(f"   IBRION = {ibrion:6d}\n")
        if isif is not None:
            handle.write(f"   ISIF   = {isif:6d}\n")
        if mdalgo is not None:
            handle.write(f"   MDALGO = {mdalgo:6d}\n")
        if potim is not None:
            handle.write(f"   POTIM  = {potim:6.4f}    time-step for ionic-motion\n")
        if pseudo_scf.enabled:
            handle.write(" Electronic Relaxation 1\n")
            handle.write(
                f"   NELM   = {pseudo_scf.nelm:6d};   NELMIN={pseudo_scf.nelmin:3d};"
                f" NELMDL={pseudo_scf.nelmdl:3d}     # of ELM steps \n"
            )
            handle.write(
                f"   EDIFF  = {_format_outcar_ediff(pseudo_scf.ediff):>10s}"
                "   stopping-criterion for ELM\n"
            )
        handle.write(f"   ICHAIN = {0:6d}\n")
        handle.write(
            f"   number of dos      NEDOS =    301   number of ions     NIONS = {len(atoms):6d}\n"
        )
        handle.write("\n")
        _append_outcar_metadata_header(handle, atoms)

    with open("OSZICAR", "w", encoding="utf-8"):
        pass

    return recorder


def _append_outcar_compat_step(
    step_index: int,
    atoms,
    forces: np.ndarray,
    stress_matrix: np.ndarray | None,
    pseudo_scf: _PseudoScfSettings,
    potential_energy: float,
    total_energy: float,
    kinetic_energy: float,
    thermostat_potential: float,
    thermostat_kinetic: float,
    neb_mode: bool = False,
    neb_chain: _NebChainApproximation | None = None,
) -> None:
    """Append a VTST-friendly ionic-step block to ``OUTCAR``."""

    positions = np.asarray(atoms.get_positions(), dtype=float)
    if forces.size:
        norms = np.linalg.norm(forces, axis=1)
        force_max = float(np.max(norms))
        force_rms = float(np.sqrt(np.mean(norms * norms)))
    else:
        force_max = 0.0
        force_rms = 0.0

    with open("OUTCAR", "a", encoding="utf-8") as handle:
        handle.write(
            f"\n--------------------------------------- Ionic step {step_index:8d}  -------------------------------------------\n\n"
        )
        handle.write(
            f"\n--------------------------------------- Iteration {step_index:6d}(   1)  ---------------------------------------\n"
        )
        handle.write(" POSITION                                       TOTAL-FORCE (eV/Angst)\n")
        handle.write(" -----------------------------------------------------------------------------------\n")
        for position, force in zip(positions, forces):
            handle.write(
                f" {position[0]:16.8f} {position[1]:16.8f} {position[2]:16.8f}"
                f" {force[0]:16.8f} {force[1]:16.8f} {force[2]:16.8f}\n"
            )
        handle.write(" -----------------------------------------------------------------------------------\n")
        drift = np.sum(forces, axis=0) if forces.size else np.zeros(3, dtype=float)
        handle.write(
            "    total drift:                               "
            f"{drift[0]:12.6f} {drift[1]:12.6f} {drift[2]:12.6f}\n\n"
        )
        handle.write(f" FORCES: max atom, RMS {force_max:16.8f} {force_rms:16.8f}\n\n")
        if stress_matrix is not None:
            xx = float(stress_matrix[0, 0])
            yy = float(stress_matrix[1, 1])
            zz = float(stress_matrix[2, 2])
            xy = float(stress_matrix[0, 1])
            yz = float(stress_matrix[1, 2])
            zx = float(stress_matrix[2, 0])
            to_kbar = 1.0 / _root().KBAR_TO_EV_PER_A3
            ext_pressure = (xx + yy + zz) / 3.0 * to_kbar
            handle.write("  FORCE on cell =-STRESS in cart. coord.  units (eV):\n")
            handle.write("  Direction    XX          YY          ZZ          XY          YZ          ZX\n")
            handle.write("  -------------------------------------------------------------------------------------\n")
            handle.write(
                f"  Total   {xx:11.5f} {yy:11.5f} {zz:11.5f}"
                f" {xy:11.5f} {yz:11.5f} {zx:11.5f}\n"
            )
            handle.write(
                f"  in kB   {xx * to_kbar:11.2f} {yy * to_kbar:11.2f} {zz * to_kbar:11.2f}"
                f" {xy * to_kbar:11.2f} {yz * to_kbar:11.2f} {zx * to_kbar:11.2f}\n"
            )
            handle.write(
                f"  external pressure = {ext_pressure:11.2f} kB  Pullay stress =        0.00 kB\n\n"
            )
        if neb_mode:
            if neb_chain is None:
                tangential_force = 0.0
                tangent_vectors = np.zeros_like(forces)
                chain_force_vectors = np.zeros_like(forces)
                chain_plus_total = np.zeros(3, dtype=float)
            else:
                tangential_force = float(neb_chain.tangential_force)
                tangent_vectors = np.asarray(neb_chain.tangent_vectors, dtype=float)
                if tangent_vectors.shape != forces.shape:
                    tangent_vectors = np.zeros_like(forces)
                chain_force_vectors = np.asarray(neb_chain.chain_force_vectors, dtype=float)
                if chain_force_vectors.shape != forces.shape:
                    chain_force_vectors = np.zeros_like(forces)
                chain_plus_total = np.asarray(neb_chain.chain_plus_total, dtype=float)
                if chain_plus_total.shape != (3,):
                    chain_plus_total = np.zeros(3, dtype=float)

            perpendicular_forces = forces - chain_force_vectors
            if perpendicular_forces.size:
                chain_force_max = float(np.max(np.linalg.norm(perpendicular_forces, axis=1)))
            else:
                chain_force_max = 0.0

            handle.write(
                " NEB: projections on to tangent (spring, REAL) "
                f"{0.0:12.6f} {tangential_force:12.6f} {chain_force_max:12.6f}\n\n"
            )
        handle.write("  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)\n")
        handle.write("  ---------------------------------------------------\n")
        handle.write(f"  free  energy   TOTEN  = {potential_energy:16.8f} eV\n")
        handle.write(
            f"  energy  without entropy= {potential_energy:16.8f}  "
            f"energy(sigma->0) = {potential_energy:16.8f}\n"
        )
        handle.write(f"  kinetic energy EKIN    = {kinetic_energy:16.8f} eV\n")
        handle.write(f"  nose potential         = {thermostat_potential:16.8f} eV\n")
        handle.write(f"  nose kinetic           = {thermostat_kinetic:16.8f} eV\n")
        handle.write(f"  total energy ETOTAL    = {total_energy:16.8f} eV\n\n")
        if neb_mode:
            handle.write(f"  tangential force (eV/A) {tangential_force:16.6f}\n")
            handle.write(" TANGENT                                        CHAIN-FORCE (eV/Angst)\n")
            handle.write(" -------------------------------------------------------------------------------\n")
            for tangent, chain_force in zip(tangent_vectors, chain_force_vectors):
                handle.write(
                    f" {tangent[0]:12.6f} {tangent[1]:12.6f} {tangent[2]:12.6f}"
                    f" {chain_force[0]:16.6f} {chain_force[1]:12.6f} {chain_force[2]:12.6f}\n"
                )
            handle.write(" -------------------------------------------------------------------------------\n\n")
            handle.write(" CHAIN + TOTAL  (eV/Angst)\n")
            handle.write(" ----------------------------------------------\n")
            handle.write(
                f" {chain_plus_total[0]:12.5f} {chain_plus_total[1]:12.5f} {chain_plus_total[2]:12.5f}\n"
            )
            handle.write(" ----------------------------------------------\n\n")


def _append_oszicar_compat_step(
    recorder: _VaspCompatRecorder,
    step_index: int,
    *,
    potential_energy: float,
    total_energy: float,
    kinetic_energy: float,
    thermostat_potential: float,
    thermostat_kinetic: float,
    temperature: float,
    forces: np.ndarray | None = None,
) -> None:
    """Append one ionic step to ``OSZICAR``."""

    delta = 0.0 if recorder.previous_energy is None else potential_energy - recorder.previous_energy
    recorder.previous_energy = potential_energy

    with open("OSZICAR", "a", encoding="utf-8") as handle:
        if recorder.pseudo_scf.enabled:
            if not recorder.oszicar_scf_header_written:
                handle.write("       N       E                     dE             d eps       ncg     rms          rms(c)\n")
                recorder.oszicar_scf_header_written = True
            if forces is not None and forces.size:
                force_rms = float(np.sqrt(np.mean(np.sum(forces * forces, axis=1))))
                ncg = max(1, int(12 * len(forces)))
            else:
                force_rms = 0.0
                ncg = 1
            handle.write(
                f"DAV: {1:3d} {potential_energy:21.12E} "
                f"{_root()._format_oszicar_residual(0.0):>14s} {_root()._format_oszicar_residual(0.0):>14s} "
                f"{ncg:7d} {_root()._format_oszicar_residual(force_rms):>12s}\n"
            )

        if recorder.ibrion == 0:
            handle.write(
                f"{step_index:7d} T={temperature:8.1f} "
                f"E= {_root()._format_oszicar_energy(total_energy)} "
                f"F= {_root()._format_oszicar_energy(potential_energy)} "
                f"E0= {_root()._format_oszicar_energy(potential_energy)}  "
                f"EK= {_root()._format_oszicar_energy(kinetic_energy)} "
                f"SP= {_root()._format_oszicar_energy(thermostat_potential)} "
                f"SK= {_root()._format_oszicar_energy(thermostat_kinetic)}\n"
            )
        else:
            handle.write(
                f"{step_index:4d} F= {_root()._format_oszicar_energy(potential_energy)} "
                f"E0= {_root()._format_oszicar_energy(potential_energy)}  "
                f"d E = {_root()._format_oszicar_energy(delta)}\n"
            )


def _record_vasp_compat_step(
    recorder: _VaspCompatRecorder,
    atoms,
    *,
    step_index: int,
    potential_energy: float,
    total_energy: float,
    kinetic_energy: float = 0.0,
    thermostat_potential: float = 0.0,
    thermostat_kinetic: float = 0.0,
    temperature: float = 0.0,
    sc_time: float = 0.0,
    neb_chain: _NebChainApproximation | None = None,
) -> None:
    """Capture step data and append compatibility records."""

    forces = _safe_get_forces(atoms)
    stress_matrix = _safe_get_stress_matrix(atoms, mode=recorder.stress_mode)
    if recorder.neb_mode and neb_chain is None:
        neb_chain = _estimate_neb_chain_approximation(
            positions=np.asarray(atoms.get_positions(), dtype=float),
            forces=forces,
            prev_positions=recorder.neb_prev_positions,
            next_positions=recorder.neb_next_positions,
        )
    _append_outcar_compat_step(
        step_index,
        atoms,
        forces,
        stress_matrix,
        recorder.pseudo_scf,
        potential_energy,
        total_energy,
        kinetic_energy,
        thermostat_potential,
        thermostat_kinetic,
        recorder.neb_mode,
        neb_chain=neb_chain,
    )
    _append_oszicar_compat_step(
        recorder,
        step_index,
        potential_energy=potential_energy,
        total_energy=total_energy,
        kinetic_energy=kinetic_energy,
        thermostat_potential=thermostat_potential,
        thermostat_kinetic=thermostat_kinetic,
        temperature=temperature,
        forces=forces,
    )
    recorder.steps.append(
        _VasprunStep(
            cell=_matrix_to_nested_list(atoms.get_cell().array),
            scaled_positions=_matrix_to_nested_list(atoms.get_scaled_positions()),
            forces=forces.tolist(),
            stress=None if stress_matrix is None else stress_matrix.tolist(),
            potential_energy=float(potential_energy),
            total_energy=float(total_energy),
            kinetic_energy=float(kinetic_energy),
            thermostat_potential=float(thermostat_potential),
            thermostat_kinetic=float(thermostat_kinetic),
            temperature=float(temperature),
            sc_time=float(sc_time),
        )
    )


def _append_structure_xml(
    parent,
    *,
    cell: list[list[float]],
    scaled_positions: list[list[float]],
    name: str | None = None,
):
    """Append a minimal VASP-like ``structure`` node."""

    attrs = {"name": name} if name is not None else {}
    structure = ET.SubElement(parent, "structure", attrs)
    crystal = ET.SubElement(structure, "crystal")

    basis = ET.SubElement(crystal, "varray", {"name": "basis"})
    cell_array = np.asarray(cell, dtype=float)
    for vector in cell_array:
        ET.SubElement(basis, "v").text = (
            f"{vector[0]:16.8f} {vector[1]:16.8f} {vector[2]:16.8f}"
        )

    volume = float(abs(np.linalg.det(cell_array)))
    ET.SubElement(crystal, "i", {"name": "volume"}).text = f"{volume:16.8f}"

    rec_basis = ET.SubElement(crystal, "varray", {"name": "rec_basis"})
    if abs(np.linalg.det(cell_array)) > 1e-14:
        reciprocal = np.linalg.inv(cell_array).T
    else:
        reciprocal = np.zeros((3, 3), dtype=float)
    for vector in reciprocal:
        ET.SubElement(rec_basis, "v").text = (
            f"{vector[0]:16.8f} {vector[1]:16.8f} {vector[2]:16.8f}"
        )

    positions = ET.SubElement(structure, "varray", {"name": "positions"})
    for row in scaled_positions:
        ET.SubElement(positions, "v").text = f"{row[0]:16.8f} {row[1]:16.8f} {row[2]:16.8f}"

    return structure


def _build_atominfo_xml(parent, symbols: List[str]) -> None:
    """Append a compact ``atominfo`` section."""

    atominfo = ET.SubElement(parent, "atominfo")
    ET.SubElement(atominfo, "atoms").text = str(len(symbols))

    counts: OrderedDict[str, int] = OrderedDict()
    for symbol in symbols:
        counts[symbol] = counts.get(symbol, 0) + 1
    ET.SubElement(atominfo, "types").text = str(len(counts))

    atom_array = ET.SubElement(atominfo, "array", {"name": "atoms"})
    ET.SubElement(atom_array, "field", {"type": "string"}).text = "element"
    atom_set = ET.SubElement(atom_array, "set")
    for symbol in symbols:
        row = ET.SubElement(atom_set, "rc")
        ET.SubElement(row, "c").text = symbol

    type_array = ET.SubElement(atominfo, "array", {"name": "atomtypes"})
    ET.SubElement(type_array, "field", {"type": "int"}).text = "atomspertype"
    ET.SubElement(type_array, "field", {"type": "string"}).text = "element"
    ET.SubElement(type_array, "field", {"type": "float"}).text = "mass"
    ET.SubElement(type_array, "field", {"type": "float"}).text = "valence"
    ET.SubElement(type_array, "field", {"type": "string"}).text = "pseudopotential"
    type_set = ET.SubElement(type_array, "set")
    for symbol, count in counts.items():
        row = ET.SubElement(type_set, "rc")
        ET.SubElement(row, "c").text = str(count)
        ET.SubElement(row, "c").text = symbol
        ET.SubElement(row, "c").text = f"{1.0:8.4f}"
        ET.SubElement(row, "c").text = f"{0.0:8.4f}"
        ET.SubElement(row, "c").text = f"PAW_PBE {symbol}"


def _append_pseudo_scf_xml_step(parent, step: _VasprunStep) -> None:
    """Append one minimal ``scstep`` block for VASP XML reader compatibility."""

    scstep = ET.SubElement(parent, "scstep")
    ET.SubElement(scstep, "time", {"name": "dav"}).text = f"{step.sc_time:8.2f} {step.sc_time:8.2f}"
    ET.SubElement(scstep, "time", {"name": "total"}).text = (
        f"{step.sc_time:8.2f} {step.sc_time:8.2f}"
    )
    energy = ET.SubElement(scstep, "energy")
    ET.SubElement(energy, "i", {"name": "e_fr_energy"}).text = f"{step.potential_energy:16.8f}"
    ET.SubElement(energy, "i", {"name": "e_wo_entrp"}).text = f"{step.potential_energy:16.8f}"
    ET.SubElement(energy, "i", {"name": "e_0_energy"}).text = f"{step.potential_energy:16.8f}"


def _write_vasprun_xml(recorder: _VaspCompatRecorder, final_atoms) -> None:
    """Write a minimal ``vasprun.xml`` with ionic-step data."""

    root = ET.Element("modeling")
    generator = ET.SubElement(root, "generator")
    ET.SubElement(generator, "i", {"name": "program", "type": "string"}).text = "VPMDK"
    ET.SubElement(generator, "i", {"name": "version", "type": "string"}).text = "0"

    incar = ET.SubElement(root, "incar")
    ET.SubElement(incar, "i", {"name": "IBRION", "type": "int"}).text = str(recorder.ibrion)
    if recorder.isif is not None:
        ET.SubElement(incar, "i", {"name": "ISIF", "type": "int"}).text = str(recorder.isif)
    ET.SubElement(incar, "i", {"name": "NSW", "type": "int"}).text = str(len(recorder.steps))
    if recorder.pseudo_scf.enabled:
        ET.SubElement(incar, "i", {"name": "NELM", "type": "int"}).text = str(recorder.pseudo_scf.nelm)
        ET.SubElement(incar, "i", {"name": "NELMIN", "type": "int"}).text = str(recorder.pseudo_scf.nelmin)
        ET.SubElement(incar, "i", {"name": "NELMDL", "type": "int"}).text = str(recorder.pseudo_scf.nelmdl)
        ET.SubElement(incar, "i", {"name": "EDIFF", "type": "float"}).text = (
            f"{recorder.pseudo_scf.ediff:.8E}"
        )
    if recorder.potim is not None:
        ET.SubElement(incar, "i", {"name": "POTIM", "type": "float"}).text = f"{recorder.potim:.6f}"
    if recorder.mdalgo is not None:
        ET.SubElement(incar, "i", {"name": "MDALGO", "type": "int"}).text = str(recorder.mdalgo)

    _append_structure_xml(
        root,
        name="primitive_cell",
        cell=recorder.initial_cell,
        scaled_positions=recorder.initial_scaled_positions,
    )
    primitive_index = ET.SubElement(root, "varray", {"name": "primitive_index"})
    for index in range(1, len(recorder.symbols) + 1):
        ET.SubElement(primitive_index, "v").text = f"{index:9d} "

    _append_kpoints_xml(root)

    parameters = ET.SubElement(root, "parameters")
    electronic = ET.SubElement(parameters, "separator", {"name": "electronic"})
    ET.SubElement(electronic, "i", {"name": "NELM", "type": "int"}).text = str(
        recorder.pseudo_scf.nelm
    )
    if recorder.pseudo_scf.enabled:
        ET.SubElement(electronic, "i", {"name": "NELMDL", "type": "int"}).text = str(
            recorder.pseudo_scf.nelmdl
        )
        ET.SubElement(electronic, "i", {"name": "NELMIN", "type": "int"}).text = str(
            recorder.pseudo_scf.nelmin
        )
        ET.SubElement(electronic, "i", {"name": "EDIFF", "type": "float"}).text = (
            f"{recorder.pseudo_scf.ediff:.8E}"
        )
        ET.SubElement(electronic, "i", {"name": "NBANDS", "type": "int"}).text = str(
            max(1, 4 * len(recorder.symbols))
        )
    ionic = ET.SubElement(parameters, "separator", {"name": "ionic"})
    ET.SubElement(ionic, "i", {"name": "IBRION", "type": "int"}).text = str(recorder.ibrion)
    if recorder.isif is not None:
        ET.SubElement(ionic, "i", {"name": "ISIF", "type": "int"}).text = str(recorder.isif)
    ET.SubElement(ionic, "i", {"name": "NSW", "type": "int"}).text = str(len(recorder.steps))
    if recorder.potim is not None:
        ET.SubElement(ionic, "i", {"name": "POTIM", "type": "float"}).text = f"{recorder.potim:.6f}"

    _build_atominfo_xml(root, recorder.symbols)
    _append_structure_xml(
        root,
        name="initialpos",
        cell=recorder.initial_cell,
        scaled_positions=recorder.initial_scaled_positions,
    )

    for step in recorder.steps:
        calculation = ET.SubElement(root, "calculation")
        if recorder.pseudo_scf.enabled:
            _append_pseudo_scf_xml_step(calculation, step)
        _append_structure_xml(
            calculation,
            cell=step.cell,
            scaled_positions=step.scaled_positions,
        )

        forces = ET.SubElement(calculation, "varray", {"name": "forces"})
        for row in step.forces:
            ET.SubElement(forces, "v").text = f"{row[0]:16.8f} {row[1]:16.8f} {row[2]:16.8f}"
        if step.stress is not None:
            stress = ET.SubElement(calculation, "varray", {"name": "stress"})
            for row in step.stress:
                ET.SubElement(stress, "v").text = f"{row[0]:16.8f} {row[1]:16.8f} {row[2]:16.8f}"

        energy = ET.SubElement(calculation, "energy")
        ET.SubElement(energy, "i", {"name": "e_fr_energy"}).text = f"{step.potential_energy:16.8f}"
        ET.SubElement(energy, "i", {"name": "e_wo_entrp"}).text = f"{step.potential_energy:16.8f}"
        ET.SubElement(energy, "i", {"name": "e_0_energy"}).text = f"{step.potential_energy:16.8f}"
        ET.SubElement(energy, "i", {"name": "kinetic"}).text = f"{step.kinetic_energy:16.8f}"
        ET.SubElement(energy, "i", {"name": "nosepot"}).text = f"{step.thermostat_potential:16.8f}"
        ET.SubElement(energy, "i", {"name": "nosekinetic"}).text = f"{step.thermostat_kinetic:16.8f}"
        ET.SubElement(energy, "i", {"name": "total"}).text = f"{step.total_energy:16.8f}"
        if recorder.pseudo_scf.enabled:
            ET.SubElement(calculation, "time", {"name": "totalsc"}).text = (
                f"{step.sc_time:8.2f} {step.sc_time:8.2f}"
            )

    _append_structure_xml(
        root,
        name="finalpos",
        cell=_matrix_to_nested_list(final_atoms.get_cell().array),
        scaled_positions=_matrix_to_nested_list(final_atoms.get_scaled_positions()),
    )

    tree = ET.ElementTree(root)
    ET.indent(tree, space=" ")
    tree.write("vasprun.xml", encoding="utf-8", xml_declaration=True)


def _append_outcar_footer(recorder: _VaspCompatRecorder) -> None:
    """Append simplified VASP-like timing/memory footer to ``OUTCAR``."""

    elapsed = max(time.perf_counter() - recorder.started_at, 0.0)
    peak_memory_kb = 0.0
    minor_page_faults = 0
    major_page_faults = 0
    voluntary_context_switches = 0
    involuntary_context_switches = 0
    if _root().resource is not None:
        try:
            usage = _root().resource.getrusage(_root().resource.RUSAGE_SELF)
            peak_memory_kb = float(usage.ru_maxrss)
            if sys.platform.startswith("darwin"):
                peak_memory_kb /= 1024.0
            minor_page_faults = int(getattr(usage, "ru_minflt", 0))
            major_page_faults = int(getattr(usage, "ru_majflt", 0))
            voluntary_context_switches = int(getattr(usage, "ru_nvcsw", 0))
            involuntary_context_switches = int(getattr(usage, "ru_nivcsw", 0))
        except Exception:
            peak_memory_kb = 0.0
            minor_page_faults = 0
            major_page_faults = 0
            voluntary_context_switches = 0
            involuntary_context_switches = 0

    with open("OUTCAR", "a", encoding="utf-8") as handle:
        handle.write(" General timing and accounting informations for this job:\n")
        handle.write(" ========================================================\n")
        handle.write(
            f"   Total CPU time used (sec):{elapsed:16.3f}\n"
            f"   User time (sec):{elapsed:25.3f}\n"
            f"   System time (sec):{0.0:23.3f}\n"
            f"   Elapsed time (sec):{elapsed:22.3f}\n"
            f"   Maximum memory used (kb):{peak_memory_kb:15.1f}\n"
            f"   Average memory used (kb):{peak_memory_kb:15.1f}\n"
            f"   Number of ionic steps:{len(recorder.steps):21d}\n\n"
            f"   Minor page faults:{minor_page_faults:25d}\n"
            f"   Major page faults:{major_page_faults:25d}\n"
            f"   Voluntary context switches:{voluntary_context_switches:15d}\n"
            f"   Involuntary context switches:{involuntary_context_switches:13d}\n\n"
        )
