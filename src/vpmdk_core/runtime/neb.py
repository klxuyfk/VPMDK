"""NEB execution helpers."""

from __future__ import annotations

import csv
import functools
import inspect
import os
import re
import sys
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

import numpy as np
from ase.calculators.calculator import Calculator


_NEB_IMAGE_DIR_RE = re.compile(r"^\d+$")
_DEFAULT_VTST_SPRING = -5.0


class _PerImageResultCache(Calculator):
    """Per-image ASE result cache in front of the ONE resident calculator.

    One-shot NEB builds a separate calculator per image, so ASE's built-in
    check_state caching evaluates each geometry once (frozen endpoints once
    per RUN). Server mode attaches the single resident calculator to every
    image, and each image's evaluation evicted the previous image's cached
    results: a 5-image / 21-step band measured 420 forward passes where the
    byte-identical one-shot run measured 35 -- 92% of the resident's work
    recomputed geometries the model had already computed, growing with
    nimages x ionic steps while the model-load saving server mode exists for
    is a one-time constant. This subclass gives each image its own results
    dict; every actual evaluation still runs on the one loaded model, and
    results are unchanged -- only redundant evaluations disappear.

    The delegate is evaluated through a CONSTRAINT-FREE copy of the image:
    calculators cache raw results and the constraint adjustment belongs to
    the Atoms layer (applied by the caller either way), so evaluating through
    a constrained copy would cache adjusted forces where a per-image real
    calculator caches raw ones.
    """

    def __init__(self, calculator: Any) -> None:
        Calculator.__init__(self)
        self._calculator = calculator
        declared = list(getattr(calculator, "implemented_properties", []) or [])
        self.implemented_properties = declared or [
            "energy",
            "free_energy",
            "forces",
            "stress",
            "energies",
            "stresses",
            "magmom",
            "magmoms",
            "dipole",
            "charges",
        ]

    def calculate(self, atoms=None, properties=("energy",), system_changes=None):
        if atoms is not None:
            self.atoms = atoms.copy()
        scratch = self.atoms.copy()
        scratch.set_constraint()
        scratch.calc = self._calculator
        results: Dict[str, Any] = {"energy": scratch.get_potential_energy()}
        for name in properties or ():
            if name in results:
                continue
            if name == "forces":
                results[name] = scratch.get_forces()
            elif name == "free_energy":
                results[name] = scratch.get_potential_energy(force_consistent=True)
            elif name == "stress":
                results[name] = scratch.get_stress()
            else:
                results[name] = self._calculator.get_property(name, scratch)
        if "forces" not in results:
            results["forces"] = scratch.get_forces()
        # Whatever else the delegate produced in the same pass (free_energy,
        # stress, magmoms for backends that compute everything at once) is
        # cached too, exactly as it would sit in a per-image calculator.
        for name, value in dict(getattr(self._calculator, "results", {}) or {}).items():
            results.setdefault(name, value)
        self.results = results


class _ResidentCalculatorProxy:
    """Give one serial resident calculator a distinct identity per NEB image."""

    def __init__(self, calculator: Any) -> None:
        self._calculator = calculator

    def __getattr__(self, name: str) -> Any:
        return getattr(self._calculator, name)


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

    # The image file is self-describing about PSTRESS (echoed in <incar> per
    # the R139 fix), and its calculation-level energies/stress carry the VASP
    # PSTRESS transformations: energy = E + PSTRESS*V (enthalpy) and the
    # stress diagonal has PSTRESS subtracted. This reader feeds the PARENT
    # aggregate writers, which re-apply those transformations from the parent
    # recorder -- so both must be UNDONE here, or a PSTRESS NEB band reported
    # E+2PV in the parent vasprun (pymatgen), E+PV in the parent OSZICAR/
    # OUTCAR, and a doubly-shifted parent pressure ('external pressure =
    # -202.59 kB' where the images correctly said ~-68 kB), while the
    # per-image files were right -- so nothing looked wrong.
    pstress_kbar = 0.0
    pstress_node = root.find("./incar/i[@name='PSTRESS']")
    if pstress_node is not None and pstress_node.text is not None:
        try:
            pstress_kbar = float(pstress_node.text)
        except ValueError:
            pstress_kbar = 0.0
    volume = 0.0
    basis = calculation.find("./structure/crystal/varray[@name='basis']")
    if basis is not None:
        rows = _parse_vasprun_varray_rows(basis)
        if rows is not None and rows.shape == (3, 3):
            volume = abs(float(np.linalg.det(rows)))

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
    if pstress_kbar and volume:
        energy_value -= pstress_kbar * _root().KBAR_TO_EV_PER_A3 * volume

    forces_varray = calculation.find("./varray[@name='forces']")
    forces = _parse_vasprun_varray_rows(forces_varray) if forces_varray is not None else None

    stress_varray = calculation.find("./varray[@name='stress']")
    stress = _parse_vasprun_varray_rows(stress_varray) if stress_varray is not None else None
    if stress is not None and stress.shape != (3, 3):
        stress = None
    if stress is not None:
        # The file holds VASP's convention (kBar, sign opposite to ASE, with
        # PSTRESS subtracted from the diagonal); every consumer of this
        # function -- the parent OUTCAR/vasprun writers -- works in RAW
        # ASE-signed eV/A^3, so reverse both transformations here rather than
        # round-tripping a VPMDK-only convention.
        # NOTE: `root` here is the parsed XML element, not the package root.
        if pstress_kbar:
            stress = stress + np.eye(3) * pstress_kbar
        stress = -stress * _root().KBAR_TO_EV_PER_A3

    return energy_value, forces, stress


def _parse_neb_ichain(incar) -> int:
    """Return VTST ``ICHAIN`` with the NEB default."""

    root = _root()
    return root._parse_vtst_ichain(incar)


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
    if abs(float(parsed)) > 1.0e9:
        # The one MD/NEB scalar the absurd-finite sweep missed: SPRING=-1e300
        # froze the band silently (CONTCAR identical to POSCAR, exit 0). Same
        # 1e9 ceiling as every sibling scalar.
        raise root.WorkdirInputError(
            f"SPRING = {parsed:g} exceeds the supported magnitude of 1e9; "
            "check the exponent."
        )
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


def _read_neb_image_structure(image_dir: str, potcar_path: str | None):
    """Read a NEB image's input structure, tagging failures as input errors.

    Mirrors ``run_workdir``'s top-level POSCAR handling: a missing or malformed
    image structure is user input, so classify it as a ``WorkdirInputError``
    (one-shot exit 1 / server ``input_error``) instead of letting the raw parse
    exception propagate and be misreported as a calculation failure (exit 2).
    This keeps NEB and non-NEB input handling consistent. Result collection
    (``_collect_neb_image_results`` reading CONTCAR) is deliberately not routed
    here: those reads happen after the calculation and are output, not input.
    """

    root = _root()
    try:
        structure_path = root._resolve_neb_image_structure_path(image_dir)
        return root.read_structure(structure_path, potcar_path)
    except root.WorkdirInputError:
        raise
    except Exception as exc:
        raise root.WorkdirInputError(
            f"Failed to read NEB image structure in {image_dir}: {exc}"
        ) from exc


def _read_neb_image_atoms(
    image_dir: str,
    potcar_path: str | None,
    *,
    incar=None,
    wrap: bool = False,
):
    """Read a NEB image AND convert it to ASE Atoms, all as input handling.

    ``_read_neb_image_structure`` classified only the READ. The steps that follow
    -- ``AseAtomsAdaptor.get_atoms``, ``wrap()`` and ``_apply_initial_magnetization``
    -- consume the same user input, and a degenerate image lattice raises a raw
    ``numpy.linalg.LinAlgError`` from the cell inversion inside them. Left
    outside, that escaped as ``calculation_error`` (exit 2, which
    SERVER_MODE_SPEC 2.5 documents as RETRYABLE, so a retry driver resubmits a
    permanently broken NEB directory forever) while the byte-identical POSCAR in
    a flat workdir is ``input_error`` (exit 1) -- run_workdir wraps exactly these
    steps. Same input, two classifications, purely because of the directory
    layout.

    A degenerate cell is rejected HERE, in both branches, by the same
    ``_validate_finite_geometry`` the flat path uses -- which also rejects
    non-finite lattices and non-finite POSITIONS, the case this branch missed while
    it checked only the determinant. Note that ``get_scaled_positions()`` is NOT a
    sufficient probe: ASE's ``Cell.complete()`` silently substitutes unit vectors
    for all-zero lattice rows, so a POSCAR whose third vector is ``0 0 0`` sails
    through it (only ``wrap()`` raises). The explicit determinant check catches both
    that case and a collinear cell, so the optimization branch no longer runs an
    entire NEB relaxation before dying on a raw AssertionError deep in the recorder
    setup. A cell that is ENTIRELY zero means "no cell given" (a legitimate
    molecular NEB) and is deliberately allowed.
    """

    root = _root()
    structure = _read_neb_image_structure(image_dir, potcar_path)
    try:
        atoms = root.AseAtomsAdaptor.get_atoms(structure)
        root._apply_vasp_comment_from_structure(atoms, structure)
        # One shared rule for the lattice AND the positions: this branch used to
        # check only the cell determinant, so non-finite POSITIONS passed here as
        # well as in the flat path.
        root._validate_finite_geometry(atoms)
        if wrap:
            atoms.wrap()
        if incar is not None:
            root._apply_initial_magnetization(atoms, incar)
    except root.WorkdirInputError:
        raise
    except Exception as exc:
        raise root.WorkdirInputError(
            f"Invalid NEB image geometry in {image_dir}: {exc}"
        ) from exc
    return structure, atoms


def _build_neb_images(
    *,
    image_dirs: list[str],
    workdir_abs: str,
    incar,
    bcar: Dict[str, str],
    potcar_path_abs: str | None,
    calculator=None,
    backend_tags=None,
):
    """Read image structures and attach calculators to the images."""

    root = _root()
    images = []
    for image_dir in image_dirs:
        structure, atoms = _read_neb_image_atoms(
            image_dir, potcar_path_abs, incar=incar
        )
        root._check_backend_species_coverage(
            atoms, bcar, backend_tags=backend_tags, calculator=calculator
        )
        if calculator is None:
            image_calculator = root._build_workdir_calculator(
                bcar, structure=structure, workdir_abs=workdir_abs
            )
            # Second half of the species gate, mirroring the flat path
            # (cli.py, after _build_workdir_calculator): the check above ran
            # with calculator=None, so a model that DECLARES its coverage
            # (matgl's element_types) was never consulted -- a Po band with
            # MLP=MATGL escaped as a raw KeyError traceback after writing
            # partial per-image artifacts, while the byte-identical flat
            # workdir and the resident-server submission both got the clean
            # input-error diagnostic.
            root._check_model_declared_species_coverage(atoms, image_calculator)
            atoms.calc = root._resolve_calculator(image_calculator)
        else:
            # Server mode: one resident calculator for the whole band. Attach
            # a per-image result cache so ASE's check_state caching works per
            # image, as it does with one-shot's per-image calculators (see
            # _PerImageResultCache).
            atoms.calc = _PerImageResultCache(root._resolve_calculator(calculator))
        images.append(atoms)
    return images


def _validate_neb_band_consistency(images, *, require_common_cell: bool) -> None:
    """Reject a band ASE itself would refuse, as INPUT rather than mid-run.

    Mirrors ``ase.mep.neb.BaseNEB.__init__``'s checks (atom count, boundary
    conditions, species order, and the periodic cell directions) so both branches
    of ``run_neb_images`` agree:

    * the OPTIMIZATION branch used to reach ``root.NEB(...)`` and let ASE's raw
      ValueError escape, which server mode reported as calculation_error (exit 2,
      documented RETRYABLE) for a permanently broken directory while one-shot
      exits 1;
    * the SINGLE-POINT/MD branch never checked at all, so the byte-identical
      directory ran to completion and wrote a tangent/chain-force summary
      computed between images describing different atoms or cells -- a
      meaningless result reported as success.

    Deliberately NOT stricter than ASE: the rules are copied from it so a band ASE
    accepts is still accepted here.

    ``require_common_cell`` splits the ONE rule whose severity legitimately
    differs between the branches. The optimizer must subtract images, so ASE
    refuses a varying periodic cell outright and so do we. The single-point/MD
    branch evaluates every image in ISOLATION -- differing cells affect only the
    aggregate TANGENT/CHAIN-FORCE summary columns, while each image's energy and
    forces stay exact -- and its 1e-8 A tolerance trips on nothing more than
    independently ISIF>=3-relaxed endpoints or POSCAR text precision. Making it a
    hard error there aborted previously-working, documented "independent image
    single points" runs with ZERO outputs, so that branch warns instead. The
    atom-count/species/pbc rules stay hard errors on both: they make the band
    meaningless either way.
    """

    if not images:
        return
    root = _root()
    reference = images[0]
    reference_numbers = np.asarray(reference.get_atomic_numbers())
    reference_pbc = np.asarray(reference.pbc)
    reference_cell = np.asarray(reference.get_cell(), dtype=float)
    for image_index, image in enumerate(images[1:], start=1):
        if len(image) != len(reference):
            raise root.WorkdirInputError(
                "NEB images have inconsistent atom counts at indices "
                f"0 ({len(reference)} atoms) and {image_index} ({len(image)} atoms); "
                "every image must describe the same atoms in the same order."
            )
        if np.any(np.asarray(image.pbc) != reference_pbc):
            raise root.WorkdirInputError(
                f"NEB image {image_index} has different boundary conditions than "
                "image 0; every image must use the same periodicity."
            )
        if np.any(np.asarray(image.get_atomic_numbers()) != reference_numbers):
            raise root.WorkdirInputError(
                f"NEB image {image_index} lists atoms in a different order (or of "
                "different species) than image 0; every image must describe the "
                "same atoms in the same order."
            )
        image_cell = np.asarray(image.get_cell(), dtype=float)
        for axis, periodic in enumerate(reference_pbc):
            if not periodic or not np.any(
                np.abs(image_cell[axis] - reference_cell[axis]) > 1e-8
            ):
                continue
            if require_common_cell:
                raise root.WorkdirInputError(
                    f"NEB image {image_index} has a different periodic cell than "
                    "image 0 (lattice vector "
                    f"{axis}); relax the endpoints with a common cell before "
                    "running the band."
                )
            print(
                f"Warning: NEB image {image_index} has a different periodic cell "
                f"than image 0 (lattice vector {axis}); each image is still "
                "evaluated independently, but the band tangent/chain-force "
                "summary is approximate."
            )
            break


def _validate_neb_image_shapes(positions_by_image) -> None:
    """Reject a band whose adjacent images describe different atoms.

    Shared by BOTH branches of ``run_neb_images``. Mismatched atom counts make
    the band meaningless either way: the optimizer cannot subtract the images,
    and the single-point/MD path silently emits an all-zero TANGENT/CHAIN-FORCE
    summary (io/vasp_compat drops neighbours whose shape differs) that looks like
    a successful run. Before this was shared, the byte-identical directory was
    rejected as invalid input under NSW>0/IBRION>0 and quietly mis-computed under
    NSW=0 -- the same input classified two different ways.

    Only the SHAPE rule is shared. The duplicate-adjacent-geometry rule below
    stays exclusive to the optimization path, where a zero tangent breaks the
    NEB math; identical adjacent images are a legitimate single-point/MD input
    (they simply yield zero projections).
    """

    for image_index, (left_positions, right_positions) in enumerate(
        zip(positions_by_image, positions_by_image[1:])
    ):
        left_positions = np.asarray(left_positions, dtype=float)
        right_positions = np.asarray(right_positions, dtype=float)
        if left_positions.shape != right_positions.shape:
            # Adjacent images with different atom counts come from inconsistent
            # user image POSCAR/CONTCAR files. Subtracting them would raise a raw
            # numpy shape-mismatch ValueError; classify it as invalid input (exit
            # 1 / input_error), consistent with the duplicate-geometry check and
            # _read_neb_image_structure, instead of leaking a calculation_error.
            raise _root().WorkdirInputError(
                "NEB images have inconsistent atom counts at indices "
                f"{image_index} ({left_positions.shape[0]} atoms) and "
                f"{image_index + 1} ({right_positions.shape[0]} atoms); "
                "every image must describe the same atoms in the same order."
            )


def _validate_neb_path(images) -> None:
    """Raise a clear error when adjacent images cannot define a NEB tangent.

    The optimization path's full check: the shared shape rule plus the
    duplicate-adjacent-geometry rule, which only the optimizer needs (a zero
    tangent has no direction to project onto).
    """

    _validate_neb_band_consistency(images, require_common_cell=True)
    positions_by_image = [
        np.asarray(image.get_positions(), dtype=float) for image in images
    ]
    _validate_neb_image_shapes(positions_by_image)
    for image_index, (left_positions, right_positions) in enumerate(
        zip(positions_by_image, positions_by_image[1:])
    ):
        displacement = right_positions - left_positions
        if float(np.linalg.norm(displacement.ravel())) <= 1e-12:
            # Duplicate adjacent geometries come from the user's image POSCAR/
            # CONTCAR files, so this is invalid input (exit 1 / input_error), not
            # a calculation failure -- consistent with _read_neb_image_structure.
            raise _root().WorkdirInputError(
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
                pstress_kbar=settings.pstress,
                nsw_requested=settings.nsw,
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
            cell=np.asarray(output_atoms.get_cell(), dtype=float),
            pbc=np.asarray(output_atoms.get_pbc(), dtype=bool),
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
            root._write_vasp_structure("CONTCAR", atoms, direct=True)
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
        root._apply_vasp_comment_from_structure(atoms, structure)
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
        pstress_kbar=settings.pstress,
        nsw_requested=settings.nsw,
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
            cell=np.asarray(atoms_step.get_cell(), dtype=float),
            pbc=np.asarray(atoms_step.get_pbc(), dtype=bool),
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


@functools.lru_cache(maxsize=None)
def _neb_supports_shared_calculator(neb_cls: Any) -> bool:
    """Whether the ASE ``NEB`` class accepts ``allow_shared_calculator``.

    The class is stable for a process, so the signature reflection is cached per
    class object rather than recomputed on every NEB relaxation. Keying on the
    class keeps a monkeypatched ``root.NEB`` (tests) correctly re-detected.
    """

    try:
        parameters = inspect.signature(neb_cls).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(parameter.name == "allow_shared_calculator" for parameter in parameters)


def _construct_ase_neb(
    *,
    images: list[Any],
    spring_constant: float,
    climb: bool,
    method: str,
    calculator: Any | None = None,
) -> Any:
    """Construct ASE NEB across releases with optional calculator sharing."""

    root = _root()
    supports_shared_calculator = _neb_supports_shared_calculator(root.NEB)

    kwargs = {
        "k": spring_constant,
        "climb": climb,
        "method": method,
    }
    if calculator is not None:
        if supports_shared_calculator:
            kwargs["allow_shared_calculator"] = True
        else:
            # Older ASE releases reject repeated calculator object identities.
            # VPMDK's resident worker is serial, so distinct delegating proxies
            # satisfy that guard while every evaluation still uses the one
            # loaded model and its cache.
            for image in images:
                image_calculator = getattr(image, "calc", None)
                if image_calculator is not None and not isinstance(
                    image_calculator, _PerImageResultCache
                ):
                    # A _PerImageResultCache already gives each image a
                    # distinct identity; re-wrapping would only hide its
                    # cache behind another delegation layer.
                    image.calc = _ResidentCalculatorProxy(image_calculator)
    return root.NEB(images, **kwargs)


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
    calculator=None,
    backend_tags=None,
) -> None:
    """Run a spring-coupled ASE NEB optimization for VTST-style inputs."""

    root = _root()
    ichain = root._parse_neb_ichain(incar)
    if ichain != 0:
        raise root.UnsupportedInputError(
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
        calculator=calculator,
        backend_tags=backend_tags,
    )
    _validate_neb_path(images)
    with root._working_directory(workdir_abs):
        neb_method = _select_neb_method(images)
    spring_constant = root._parse_neb_spring_constant(incar)
    raw_lclimb = getattr(incar, "get", lambda *_: None)("LCLIMB")
    climb = root._is_truthy_flag(raw_lclimb)
    if raw_lclimb is None:
        # VTST's documented default is LCLIMB=.TRUE. (climbing-image NEB);
        # VPMDK's absent-tag default is plain NEB, which UNDERESTIMATES the
        # barrier (measured with EMT: 0.2848 vs 0.3745 eV, 24% low, for an
        # Au/Al(001) hop with 2 moving images) -- silently, with exit 0.
        # Changing the default would alter existing runs (SPEC 1.1), so the
        # divergence is disclosed instead, like ANDERSEN_PROB's.
        print(
            "Warning: LCLIMB is not set; VPMDK runs PLAIN NEB (climb=False), "
            "while VTST's documented default is LCLIMB=.TRUE. (climbing "
            "image). A plain band underestimates the barrier; write "
            "LCLIMB = .TRUE. for VTST's default behavior."
        )
    optimizer_cls, optimizer_name = root._select_neb_optimizer(incar, settings.ibrion)
    fmax = _neb_force_limit(settings)

    neb = _construct_ase_neb(
        images=images,
        spring_constant=spring_constant,
        climb=climb,
        method=neb_method,
        calculator=calculator,
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

    with root._working_directory(workdir_abs):
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
    calculator=None,
    backend_tags=None,
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
        # The numbered-directory layout is user input, so a missing/insufficient
        # layout is invalid input (exit 1 / input_error), not a calc failure.
        raise root.WorkdirInputError(
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

    root._reject_unsupported_vtst_modes(incar)

    with root._active_pseudo_scf_settings(pseudo_scf_settings), root._active_vasp_input_paths(input_paths):
        if settings.nsw > 0 and settings.ibrion > 0:
            if len(image_dirs) < 3:
                # Too few image directories for an NEB optimization is a user
                # layout problem -> invalid input (exit 1), not a calc failure.
                raise root.WorkdirInputError(
                    "ASE NEB optimization requires at least three numbered image "
                    "directories: initial, one moving image, and final "
                    "(for example 00, 01, 02)."
                )
            _run_ase_neb_relaxation(
                image_dirs=image_dirs,
                workdir_abs=workdir_abs,
                incar=incar,
                settings=settings,
                bcar=bcar,
                potcar_path_abs=potcar_path_abs,
                write_energy_csv=write_energy_csv,
                oszicar_pseudo_scf=oszicar_pseudo_scf,
                calculator=calculator,
                backend_tags=backend_tags,
            )
        else:
            total_images = len(image_dirs)
            reference_images: list[Any] = []
            image_reference_positions: list[np.ndarray] = []
            for image_dir in image_dirs:
                _, image_atoms = _read_neb_image_atoms(
                    image_dir, potcar_path_abs, wrap=True
                )
                reference_images.append(image_atoms)
                image_reference_positions.append(np.asarray(image_atoms.get_positions(), dtype=float))

            # Reject an inconsistent band BEFORE running any image, with exactly
            # the rules the ASE-optimization branch enforces. Otherwise mismatched
            # atom counts, species order, pbc or cells run to completion and write
            # a tangent/chain-force summary computed between images describing
            # different systems -- a meaningless result reported as success, while
            # the byte-identical directory with NSW>0/IBRION>0 fails fast.
            # Duplicate adjacent geometries are NOT rejected here: they are a
            # legitimate single-point/MD input that simply yields zero projections.
            _validate_neb_band_consistency(
                reference_images, require_common_cell=False
            )
            _validate_neb_image_shapes(image_reference_positions)

            for image_index, image_dir in enumerate(image_dirs, start=1):
                image_name = os.path.basename(image_dir)
                structure, atoms = _read_neb_image_atoms(
                    image_dir, potcar_path_abs, incar=incar, wrap=True
                )
                root._check_backend_species_coverage(
                    atoms, bcar, backend_tags=backend_tags, calculator=calculator
                )
                if calculator is None:
                    image_calculator = root._build_workdir_calculator(
                        bcar, structure=structure, workdir_abs=workdir_abs
                    )
                    # Second half of the species gate, mirroring the flat path
                    # and _build_neb_images (see the comment there).
                    root._check_model_declared_species_coverage(
                        atoms, image_calculator
                    )
                else:
                    image_calculator = calculator
                neb_prev_positions = image_reference_positions[image_index - 2] if image_index > 1 else None
                neb_next_positions = image_reference_positions[image_index] if image_index < total_images else None

                print(f"Running NEB image {image_name} ({image_index}/{total_images})")
                with root._working_directory(image_dir):
                    if settings.nsw <= 0 or settings.ibrion < 0:
                        root.run_single_point(
                            atoms,
                            image_calculator,
                            isif=settings.stress_isif,
                            oszicar_pseudo_scf=oszicar_pseudo_scf,
                            neb_mode=True,
                            neb_prev_positions=neb_prev_positions,
                            neb_next_positions=neb_next_positions,
                            # Without this the per-image artifacts used the
                            # RAW stress convention while the parent (whose
                            # recorder carries settings.pstress) used the
                            # corrected one: the same image differed by the
                            # full PSTRESS/PV between its own files and the
                            # parent's.
                            pstress=settings.pstress,
                            # Without this the per-image vasprun echoed NSW=1
                            # (the len(steps) fallback) while the parent
                            # aggregate of the SAME run and a flat workdir
                            # with the same INCAR both echo the requested
                            # value (the R142 wiring's missing half).
                            nsw=settings.nsw,
                        )
                    elif settings.ibrion == 0:
                        root.run_md(
                            atoms,
                            image_calculator,
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
                            pstress=settings.pstress,
                        )
        with root._working_directory(workdir_abs):
            image_results = root._collect_neb_image_results(image_dirs, potcar_path=potcar_path_abs)
            root._write_neb_parent_aggregate_outputs(
                workdir=workdir_abs,
                settings=settings,
                image_results=image_results,
                oszicar_pseudo_scf=oszicar_pseudo_scf,
            )
