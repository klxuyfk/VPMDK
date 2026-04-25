"""NEB execution helpers."""

from __future__ import annotations

import csv
import os
import re
import sys
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

import numpy as np


_NEB_IMAGE_DIR_RE = re.compile(r"^\d+$")
_DEFAULT_VTST_SPRING = -5.0


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


def _parse_neb_ichain(incar) -> int:
    """Return VTST ``ICHAIN`` with the NEB default."""

    root = _root()
    raw_value = getattr(incar, "get", lambda *_: 0)("ICHAIN", 0)
    parsed = root._parse_optional_float(raw_value, key="ICHAIN")
    if parsed is None:
        return 0
    return int(parsed)


def _parse_neb_iopt(incar) -> int:
    """Return VTST ``IOPT`` with the VASP optimizer default."""

    root = _root()
    raw_value = getattr(incar, "get", lambda *_: 0)("IOPT", 0)
    parsed = root._parse_optional_float(raw_value, key="IOPT")
    if parsed is None:
        return 0
    return int(parsed)


def _parse_neb_spring_constant(incar) -> float:
    """Return ASE NEB spring magnitude from VASP/VTST ``SPRING``."""

    root = _root()
    raw_value = getattr(incar, "get", lambda *_: _DEFAULT_VTST_SPRING)(
        "SPRING", _DEFAULT_VTST_SPRING
    )
    parsed = root._parse_optional_float(raw_value, key="SPRING")
    if parsed is None:
        parsed = _DEFAULT_VTST_SPRING
    return abs(float(parsed))


def _select_neb_optimizer(incar, ibrion: int):
    """Return an ASE optimizer class approximating VTST ``IOPT``/``IBRION``."""

    root = _root()
    iopt = _parse_neb_iopt(incar)
    if iopt == 1:
        return root.LBFGS, "LBFGS"
    if iopt == 3:
        return root.MDMin, "Quick-Min"
    if iopt == 5:
        return root.BFGS, "BFGS"
    if iopt == 7:
        return root.FIRE, "FIRE"
    if iopt in {2, 4, 6, 8}:
        print(
            f"Warning: VTST IOPT={iopt} has no exact ASE optimizer mapping in VPMDK; "
            "using BFGS."
        )
        return root.BFGS, "BFGS"

    if iopt != 0:
        print(f"Warning: Unsupported VTST IOPT={iopt}; using BFGS.")
        return root.BFGS, "BFGS"

    if ibrion == 3:
        return root.MDMin, "Quick-Min"
    return root.BFGS, "BFGS"


def _neb_force_limit(settings) -> float:
    """Return an ASE ``fmax`` value for NEB optimization."""

    ediffg = getattr(settings, "ediffg", None)
    if ediffg is not None:
        try:
            ediffg_float = float(ediffg)
        except (TypeError, ValueError):
            ediffg_float = 0.0
        if ediffg_float < 0.0:
            return abs(ediffg_float)
        if ediffg_float > 0.0:
            print(
                "Warning: NEB optimization uses force convergence; "
                "EDIFFG should be negative. Using EDIFFG magnitude as fmax."
            )
            return abs(ediffg_float)
    force_limit = float(getattr(settings, "force_limit", 0.05))
    return force_limit if force_limit > 0.0 else 0.05


def _build_neb_images(
    *,
    image_dirs: list[str],
    workdir_abs: str,
    incar,
    bcar: Dict[str, str],
    potcar_path_abs: str | None,
):
    """Read image structures and attach one calculator per image."""

    root = _root()
    images = []
    for image_dir in image_dirs:
        structure_path = root._resolve_neb_image_structure_path(image_dir)
        structure = root.read_structure(structure_path, potcar_path_abs)
        atoms = root.AseAtomsAdaptor.get_atoms(structure)
        atoms.wrap()
        root._apply_initial_magnetization(atoms, incar)
        with root._working_directory(workdir_abs):
            atoms.calc = root._build_calculator_from_tags(bcar, structure=structure)
        images.append(atoms)
    return images


def _validate_neb_path(images) -> None:
    """Raise a clear error when adjacent images cannot define a NEB tangent."""

    for image_index, (left, right) in enumerate(zip(images, images[1:])):
        displacement = np.asarray(right.get_positions(), dtype=float) - np.asarray(
            left.get_positions(), dtype=float
        )
        if float(np.linalg.norm(displacement.ravel())) <= 1e-12:
            raise RuntimeError(
                "NEB path contains duplicate adjacent image geometries at "
                f"indices {image_index} and {image_index + 1}; "
                "provide distinct 00, intermediate, and final POSCAR/CONTCAR files."
            )


def _select_neb_method(images) -> str:
    """Return the ASE NEB tangent method for the current band."""

    energies: list[float] = []
    for image in images:
        try:
            energies.append(float(image.get_potential_energy()))
        except Exception:
            return "improvedtangent"
    if energies and max(energies) - min(energies) <= 1e-12:
        print(
            "Warning: initial NEB image energies are degenerate; "
            "using ASE standard tangent to avoid undefined improved tangents."
        )
        return "aseneb"
    return "improvedtangent"


def _initialize_neb_image_recorders(
    *,
    image_dirs: list[str],
    images,
    settings,
    oszicar_pseudo_scf: bool,
) -> dict[str, Any]:
    """Create VASP-compatible output recorders for every NEB image."""

    root = _root()
    recorders: dict[str, Any] = {}
    image_positions = [np.asarray(image.get_positions(), dtype=float) for image in images]
    total_images = len(images)
    for image_index, (image_dir, atoms) in enumerate(zip(image_dirs, images)):
        prev_positions = image_positions[image_index - 1] if image_index > 0 else None
        next_positions = (
            image_positions[image_index + 1]
            if image_index + 1 < total_images
            else None
        )
        with root._working_directory(image_dir):
            recorders[image_dir] = root._initialize_vasp_compat_outputs(
                atoms,
                ibrion=settings.ibrion,
                isif=settings.stress_isif,
                neb_mode=True,
                write_oszicar_pseudo_scf=oszicar_pseudo_scf,
                neb_prev_positions=prev_positions,
                neb_next_positions=next_positions,
            )
    return recorders


def _evaluate_neb_image_for_output(atoms, *, stress_isif: int | None):
    """Return real image energy, forces, and optional stress for output."""

    root = _root()
    potential_energy = float(atoms.get_potential_energy())
    forces = root._safe_get_forces(atoms)
    stress_matrix = root._safe_get_stress_matrix(
        atoms, mode=root._stress_mode_from_isif(stress_isif)
    )
    return potential_energy, forces, stress_matrix


def _record_neb_band_step(
    *,
    step_index: int,
    image_dirs: list[str],
    images,
    recorders: dict[str, Any],
    energy_history: dict[str, list[float]],
    stress_isif: int | None,
) -> None:
    """Append one VTST-style ionic step for all images in the band."""

    root = _root()
    image_positions = [np.asarray(image.get_positions(), dtype=float) for image in images]
    total_images = len(images)
    for image_index, (image_dir, atoms) in enumerate(zip(image_dirs, images)):
        potential_energy, forces, stress_matrix = _evaluate_neb_image_for_output(
            atoms, stress_isif=stress_isif
        )
        output_atoms = atoms.copy()
        calculator_kwargs: Dict[str, Any] = {
            "energy": potential_energy,
            "forces": forces,
        }
        if stress_matrix is not None:
            calculator_kwargs["stress"] = root._full_to_voigt_stress(stress_matrix)
        output_atoms.calc = root.SinglePointCalculator(
            output_atoms, **calculator_kwargs
        )
        prev_positions = image_positions[image_index - 1] if image_index > 0 else None
        next_positions = (
            image_positions[image_index + 1]
            if image_index + 1 < total_images
            else None
        )
        neb_chain = root._estimate_neb_chain_approximation(
            positions=np.asarray(output_atoms.get_positions(), dtype=float),
            forces=np.asarray(forces, dtype=float),
            prev_positions=prev_positions,
            next_positions=next_positions,
        )
        with root._working_directory(image_dir):
            root._record_vasp_compat_step(
                recorders[image_dir],
                output_atoms,
                step_index=step_index,
                potential_energy=potential_energy,
                total_energy=potential_energy,
                sc_time=0.0,
                neb_chain=neb_chain,
            )
        energy_history[image_dir].append(potential_energy)


def _finalize_neb_image_outputs(
    *,
    image_dirs: list[str],
    images,
    recorders: dict[str, Any],
    energy_history: dict[str, list[float]],
    write_energy_csv: bool,
) -> None:
    """Write final image ``vasprun.xml``, ``CONTCAR``, and optional CSV logs."""

    root = _root()
    for image_dir, atoms in zip(image_dirs, images):
        atoms.wrap()
        with root._working_directory(image_dir):
            root._write_vasprun_xml(recorders[image_dir], atoms)
            root._append_outcar_footer(recorders[image_dir])
            root.write("CONTCAR", atoms, direct=True)
            if write_energy_csv:
                with open("energy.csv", "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    for potential_energy in energy_history[image_dir]:
                        writer.writerow([float(potential_energy)])


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


def _run_ase_neb_relaxation(
    *,
    image_dirs: list[str],
    workdir_abs: str,
    incar,
    settings,
    bcar: Dict[str, str],
    potcar_path_abs: str | None,
    write_energy_csv: bool,
    oszicar_pseudo_scf: bool,
) -> None:
    """Run a spring-coupled ASE NEB optimization for VTST-style inputs."""

    root = _root()
    ichain = root._parse_neb_ichain(incar)
    if ichain != 0:
        raise NotImplementedError(
            "VPMDK currently implements VTST-style NEB for ICHAIN=0 only. "
            f"ICHAIN={ichain} TS methods such as dimer/lanczos are not implemented."
        )
    if root._is_truthy_flag(getattr(incar, "get", lambda *_: None)("LNEBCELL")):
        print(
            "Warning: LNEBCELL is not implemented in ASE NEB mode; "
            "the NEB band will use fixed cells."
        )
    elif settings.isif >= 3:
        print(
            "Warning: ASE NEB optimizes image positions only; "
            f"ISIF={settings.stress_isif} cell relaxation is ignored for the band."
        )

    images = _build_neb_images(
        image_dirs=image_dirs,
        workdir_abs=workdir_abs,
        incar=incar,
        bcar=bcar,
        potcar_path_abs=potcar_path_abs,
    )
    _validate_neb_path(images)
    neb_method = _select_neb_method(images)
    spring_constant = root._parse_neb_spring_constant(incar)
    climb = root._is_truthy_flag(getattr(incar, "get", lambda *_: None)("LCLIMB"))
    optimizer_cls, optimizer_name = root._select_neb_optimizer(incar, settings.ibrion)
    fmax = _neb_force_limit(settings)

    neb = root.NEB(
        images,
        k=spring_constant,
        climb=climb,
        method=neb_method,
    )
    recorders = _initialize_neb_image_recorders(
        image_dirs=image_dirs,
        images=images,
        settings=settings,
        oszicar_pseudo_scf=oszicar_pseudo_scf,
    )
    energy_history = {image_dir: [] for image_dir in image_dirs}
    step_count = 0

    def record_step() -> None:
        nonlocal step_count
        step_count += 1
        _record_neb_band_step(
            step_index=step_count,
            image_dirs=image_dirs,
            images=images,
            recorders=recorders,
            energy_history=energy_history,
            stress_isif=settings.stress_isif,
        )

    print(
        "Running VTST-style NEB "
        f"({len(images) - 2} moving images, spring={spring_constant:g}, "
        f"climb={climb}, method={neb_method}, optimizer={optimizer_name})"
    )
    dyn = optimizer_cls(neb, logfile=None)
    dyn.attach(record_step)
    converged = bool(dyn.run(fmax=fmax, steps=settings.nsw))

    if step_count == 0:
        record_step()

    _finalize_neb_image_outputs(
        image_dirs=image_dirs,
        images=images,
        recorders=recorders,
        energy_history=energy_history,
        write_energy_csv=write_energy_csv,
    )
    if converged:
        print(f"NEB converged in {step_count} ionic steps (fmax <= {fmax:g}).")
    else:
        print(f"NEB stopped after {step_count} ionic steps (NSW={settings.nsw}).")


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
    """Run NEB-style numbered image directories."""

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
        if settings.nsw > 0 and settings.ibrion > 0:
            _run_ase_neb_relaxation(
                image_dirs=image_dirs,
                workdir_abs=workdir_abs,
                incar=incar,
                settings=settings,
                bcar=bcar,
                potcar_path_abs=potcar_path_abs,
                write_energy_csv=write_energy_csv,
                oszicar_pseudo_scf=oszicar_pseudo_scf,
            )
        else:
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
        with root._working_directory(workdir_abs):
            image_results = root._collect_neb_image_results(image_dirs, potcar_path=potcar_path_abs)
            root._write_neb_parent_aggregate_outputs(
                workdir=workdir_abs,
                settings=settings,
                image_results=image_results,
                oszicar_pseudo_scf=oszicar_pseudo_scf,
            )
