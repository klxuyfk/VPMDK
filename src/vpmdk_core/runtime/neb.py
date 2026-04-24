"""NEB execution helpers."""

from __future__ import annotations

import os
import re
import sys
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

import numpy as np


_NEB_IMAGE_DIR_RE = re.compile(r"^\d+$")


def _root():
    return sys.modules["vpmdk_core"]


def _discover_neb_image_directories(workdir: str) -> List[str]:
    """Return numbered NEB image directories sorted by numeric index."""

    try:
        entries = os.listdir(workdir)
    except OSError:
        return []

    indexed_dirs: list[tuple[int, str]] = []
    for entry in entries:
        if _NEB_IMAGE_DIR_RE.fullmatch(entry) is None:
            continue
        path = os.path.join(workdir, entry)
        if os.path.isdir(path):
            indexed_dirs.append((int(entry), path))
    indexed_dirs.sort(key=lambda item: item[0])
    return [path for _, path in indexed_dirs]


def _resolve_neb_image_structure_path(image_dir: str, *, prefer_contcar: bool = False) -> str:
    """Return structure path for one NEB image (POSCAR/CONTCAR)."""

    poscar_path = os.path.join(image_dir, "POSCAR")
    contcar_path = os.path.join(image_dir, "CONTCAR")
    candidates = (
        (contcar_path, poscar_path) if prefer_contcar else (poscar_path, contcar_path)
    )
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Neither POSCAR nor CONTCAR found in NEB image directory: {image_dir}"
    )


def _parse_vasprun_varray_rows(varray) -> np.ndarray:
    """Return numeric rows from a ``vasprun.xml`` ``varray`` element."""

    rows: list[list[float]] = []
    for vector in varray.findall("v"):
        parts = str(vector.text or "").split()
        if not parts:
            continue
        rows.append([float(value) for value in parts])
    return np.asarray(rows, dtype=float)


def _read_last_vasprun_step(path: str) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    """Return ``(energy, forces, stress)`` from last ``calculation`` in ``vasprun.xml``."""

    root = ET.parse(path).getroot()
    calculations = root.findall("calculation")
    if not calculations:
        raise ValueError("vasprun.xml has no <calculation> blocks")
    calculation = calculations[-1]

    energy_value: float | None = None
    energy = calculation.find("energy")
    if energy is not None:
        for name in ("e_wo_entrp", "e_fr_energy", "e_0_energy", "total"):
            node = energy.find(f"./i[@name='{name}']")
            if node is None or node.text is None:
                continue
            try:
                energy_value = float(node.text)
                break
            except ValueError:
                continue
    if energy_value is None:
        raise ValueError("Unable to parse energy from vasprun.xml")

    forces_varray = calculation.find("./varray[@name='forces']")
    forces = _parse_vasprun_varray_rows(forces_varray) if forces_varray is not None else None

    stress_varray = calculation.find("./varray[@name='stress']")
    stress = _parse_vasprun_varray_rows(stress_varray) if stress_varray is not None else None
    if stress is not None and stress.shape != (3, 3):
        stress = None

    return energy_value, forces, stress


def _collect_neb_image_results(
    image_dirs: list[str], *, potcar_path: str | None
):
    """Collect final structures/energies/forces for each NEB image directory."""

    root = _root()
    results: list[root._NebImageResult] = []
    for image_dir in image_dirs:
        image_name = os.path.basename(image_dir)
        structure_path = _resolve_neb_image_structure_path(image_dir, prefer_contcar=True)
        structure = root.read_structure(structure_path, potcar_path)
        atoms = root.AseAtomsAdaptor.get_atoms(structure)
        atoms.wrap()

        potential_energy = 0.0
        forces = np.zeros((len(atoms), 3), dtype=float)
        stress: np.ndarray | None = None
        vasprun_path = os.path.join(image_dir, "vasprun.xml")
        if os.path.exists(vasprun_path):
            try:
                potential_energy, parsed_forces, parsed_stress = _read_last_vasprun_step(vasprun_path)
                if parsed_forces is None or parsed_forces.shape != (len(atoms), 3):
                    raise ValueError(
                        f"Unexpected forces shape in {vasprun_path}: "
                        f"{None if parsed_forces is None else parsed_forces.shape}"
                    )
                forces = parsed_forces
                if parsed_stress is not None:
                    stress = parsed_stress
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to parse NEB image vasprun.xml for {image_name}: {vasprun_path}"
                ) from exc

        results.append(
            root._NebImageResult(
                image_name=image_name,
                atoms=atoms,
                potential_energy=float(potential_energy),
                forces=forces,
                stress=stress,
            )
        )
    return results


def _write_neb_parent_aggregate_outputs(
    *,
    workdir: str,
    settings,
    image_results,
    oszicar_pseudo_scf: bool = False,
) -> None:
    """Write parent-level NEB ``OUTCAR``/``OSZICAR``/``vasprun.xml`` summaries."""

    root = _root()
    if not image_results:
        return

    first_atoms = image_results[0].atoms.copy()
    recorder = root._initialize_vasp_compat_outputs(
        first_atoms,
        ibrion=settings.ibrion,
        isif=settings.stress_isif,
        potim=settings.potim,
        mdalgo=settings.mdalgo if settings.ibrion == 0 else None,
        neb_mode=True,
        write_oszicar_pseudo_scf=oszicar_pseudo_scf,
    )
    image_positions = [np.asarray(image.atoms.get_positions(), dtype=float) for image in image_results]
    for image_index, image in enumerate(image_results):
        step_index = image_index + 1
        atoms_step = image.atoms.copy()
        prev_positions = image_positions[image_index - 1] if image_index > 0 else None
        next_positions = (
            image_positions[image_index + 1] if image_index + 1 < len(image_positions) else None
        )
        neb_chain = root._estimate_neb_chain_approximation(
            positions=np.asarray(atoms_step.get_positions(), dtype=float),
            forces=np.asarray(image.forces, dtype=float),
            prev_positions=prev_positions,
            next_positions=next_positions,
        )
        calculator_kwargs: Dict[str, Any] = {
            "energy": image.potential_energy,
            "forces": image.forces,
        }
        if image.stress is not None:
            calculator_kwargs["stress"] = root._full_to_voigt_stress(np.asarray(image.stress, dtype=float))
        atoms_step.calc = root.SinglePointCalculator(atoms_step, **calculator_kwargs)
        root._record_vasp_compat_step(
            recorder,
            atoms_step,
            step_index=step_index,
            potential_energy=image.potential_energy,
            total_energy=image.potential_energy,
            sc_time=0.0,
            neb_chain=neb_chain,
        )

    final_atoms = image_results[-1].atoms.copy()
    root._write_vasprun_xml(recorder, final_atoms)
    root._append_outcar_footer(recorder)


def run_neb_images(
    *,
    workdir: str,
    incar,
    settings,
    bcar: Dict[str, str],
    potcar_path: str | None,
    write_energy_csv: bool,
    write_lammps_traj: bool,
    lammps_traj_interval: int,
    oszicar_pseudo_scf: bool,
) -> None:
    """Run independent per-image calculations for a NEB-like directory layout."""

    root = _root()
    workdir_abs = os.path.abspath(workdir)
    potcar_path_abs = os.path.abspath(potcar_path) if potcar_path else None
    pseudo_scf_settings = root._pseudo_scf_settings_from_incar(incar, enabled=oszicar_pseudo_scf)
    input_paths = root._VaspInputPaths(
        incar_path=os.path.join(workdir_abs, "INCAR"),
        potcar_path=potcar_path_abs or os.path.join(workdir_abs, "POTCAR"),
        kpoints_path=os.path.join(workdir_abs, "KPOINTS"),
    )
    image_dirs = root._discover_neb_image_directories(workdir_abs)
    if len(image_dirs) < 2:
        raise RuntimeError(
            "NEB mode requires numbered image directories (for example 00, 01, 02)."
        )

    images_hint = root._parse_neb_image_count(incar)
    if images_hint is not None:
        expected_dirs = images_hint + 2
        if expected_dirs != len(image_dirs):
            print(
                f"Warning: IMAGES={images_hint} implies {expected_dirs} image directories, "
                f"but found {len(image_dirs)} under {workdir_abs}. Proceeding with discovered directories."
            )

    with root._active_pseudo_scf_settings(pseudo_scf_settings), root._active_vasp_input_paths(input_paths):
        total_images = len(image_dirs)
        image_reference_positions: list[np.ndarray] = []
        for image_dir in image_dirs:
            structure_path = root._resolve_neb_image_structure_path(image_dir)
            structure = root.read_structure(structure_path, potcar_path_abs)
            image_atoms = root.AseAtomsAdaptor.get_atoms(structure)
            image_atoms.wrap()
            image_reference_positions.append(np.asarray(image_atoms.get_positions(), dtype=float))

        for image_index, image_dir in enumerate(image_dirs, start=1):
            image_name = os.path.basename(image_dir)
            structure_path = root._resolve_neb_image_structure_path(image_dir)
            structure = root.read_structure(structure_path, potcar_path_abs)
            atoms = root.AseAtomsAdaptor.get_atoms(structure)
            atoms.wrap()
            root._apply_initial_magnetization(atoms, incar)
            with root._working_directory(workdir_abs):
                calculator = root._build_calculator_from_tags(bcar, structure=structure)
            neb_prev_positions = image_reference_positions[image_index - 2] if image_index > 1 else None
            neb_next_positions = image_reference_positions[image_index] if image_index < total_images else None

            print(f"Running NEB image {image_name} ({image_index}/{total_images})")
            with root._working_directory(image_dir):
                if settings.nsw <= 0 or settings.ibrion < 0:
                    root.run_single_point(
                        atoms,
                        calculator,
                        isif=settings.stress_isif,
                        oszicar_pseudo_scf=oszicar_pseudo_scf,
                        neb_mode=True,
                        neb_prev_positions=neb_prev_positions,
                        neb_next_positions=neb_next_positions,
                    )
                elif settings.ibrion == 0:
                    root.run_md(
                        atoms,
                        calculator,
                        settings.nsw,
                        settings.tebeg,
                        settings.potim,
                        mdalgo=settings.mdalgo,
                        teend=settings.teend,
                        smass=settings.smass,
                        thermostat_params=settings.thermostat_params,
                        isif=settings.stress_isif,
                        oszicar_pseudo_scf=oszicar_pseudo_scf,
                        neb_mode=True,
                        neb_prev_positions=neb_prev_positions,
                        neb_next_positions=neb_next_positions,
                        write_lammps_traj=write_lammps_traj,
                        lammps_traj_interval=lammps_traj_interval,
                    )
                else:
                    root.run_relaxation(
                        atoms,
                        calculator,
                        settings.nsw,
                        settings.force_limit,
                        write_energy_csv,
                        isif=settings.isif,
                        pstress=settings.pstress,
                        energy_tolerance=settings.energy_tolerance,
                        ibrion=settings.ibrion,
                        stress_isif=settings.stress_isif,
                        neb_mode=True,
                        neb_prev_positions=neb_prev_positions,
                        neb_next_positions=neb_next_positions,
                        oszicar_pseudo_scf=oszicar_pseudo_scf,
                    )
        with root._working_directory(workdir_abs):
            image_results = root._collect_neb_image_results(image_dirs, potcar_path=potcar_path_abs)
            root._write_neb_parent_aggregate_outputs(
                workdir=workdir_abs,
                settings=settings,
                image_results=image_results,
                oszicar_pseudo_scf=oszicar_pseudo_scf,
            )
