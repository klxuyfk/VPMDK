"""Single-point execution flow."""

from __future__ import annotations

import sys

import numpy as np

from ..compat.vasp import VaspCompatConfig, VaspSinglePointConfig
from ..models import SinglePointConfig
from ..observers import PrintProgressObserver, VaspCompatObserver


def _root():
    return sys.modules["vpmdk_core"]


# Floor for a finite-difference displacement (POTIM for IBRION=5/6,
# FORCE_CONSTANTS_DISPLACEMENT for IBRION=7/8), in Angstrom. Positions are
# O(1 Angstrom) doubles (ulp ~4e-16), so `positions += direction * 1e-30` is a
# bit-for-bit no-op: F(+d) == F(-d) exactly and every force constant becomes
# 0/(2d) = 0.0 -- an ALL-ZERO Hessian written to vasprun.xml as a SUCCESSFUL
# run (the underflow mirror of the non-finite-POTIM rejection below; absurd
# values have BOTH ends). Displacements above the underflow line but below
# ~1e-6 are still garbage: backend force noise divided by 2d dominates the
# quotient. Real inputs use 1e-3..3e-2, so 1e-6 rejects nothing legitimate.
_MIN_FINITE_DIFFERENCE_DISPLACEMENT = 1e-6


def run_single_point(
    atoms,
    calculator,
    *,
    isif: int | None = None,
    oszicar_pseudo_scf: bool = False,
    neb_mode: bool = False,
    neb_prev_positions=None,
    neb_next_positions=None,
    strict_forces: bool = False,
    pstress: float | None = None,
    nsw: int | None = None,
):
    result = _root().single_point(
        atoms,
        calculator=calculator,
        config=SinglePointConfig(compat=VaspSinglePointConfig(isif=isif)),
        observer=[VaspCompatObserver(), PrintProgressObserver()],
        compatibility=VaspCompatConfig(
            enabled=True,
            write_pseudo_scf=oszicar_pseudo_scf,
            write_contcar=True,
            neb_mode=neb_mode,
            neb_prev_positions=neb_prev_positions,
            neb_next_positions=neb_next_positions,
            strict_forces=strict_forces,
            # VASP applies the PSTRESS output correction in EVERY mode that
            # prints stress; wiring it only into the relaxation path made the
            # same INCAR flip pressure convention when IBRION/NSW changed
            # (measured: identical geometry read -2.59 kB static vs -502.59 kB
            # relax at PSTRESS=500).
            pstress_kbar=pstress,
            nsw_requested=nsw,
        ),
    )
    return result.potential_energy


def _force_constants_displacement_from_bcar(bcar_tags) -> float:
    """Return finite-difference displacement for VASP-like force constants."""

    raw = bcar_tags.get(
        "FORCE_CONSTANTS_DISPLACEMENT",
        bcar_tags.get("PHONON_DISPLACEMENT", "0.01"),
    )
    if isinstance(raw, str):
        try:
            float(raw.strip())
        except (TypeError, ValueError):
            # Mirror of the INCAR-side corrupted-token rule (POTIM et al.):
            # _parse_optional_float extracts the LEADING DIGITS, so a Fortran
            # D exponent ('1D-2') or a letter O for zero ('.O1') silently
            # became a 1.0 Angstrom step where 0.01 was meant -- a 100x-wrong
            # finite-difference displacement with exit 0. The same physical
            # quantity spelled in INCAR (POTIM) is a clean input error.
            raise ValueError(
                f"FORCE_CONSTANTS_DISPLACEMENT value {raw!r} is not a plain "
                "number; refusing to read its leading digits as the "
                "displacement. Write it like 0.01 (Angstrom)."
            ) from None
    value = _root()._parse_optional_float(raw, key="FORCE_CONSTANTS_DISPLACEMENT")
    # Mirrors the numbered rejection list in _finite_difference_force_response:
    # (1) non-finite (None from the parser), (2) not positive, (3) below the
    # underflow/noise floor.
    if (
        value is None
        or not np.isfinite(value)
        or value <= 0.0
        or value < _MIN_FINITE_DIFFERENCE_DISPLACEMENT
    ):
        raise ValueError(
            "FORCE_CONSTANTS_DISPLACEMENT must be a positive length of at least "
            f"{_MIN_FINITE_DIFFERENCE_DISPLACEMENT:g} Angstrom; smaller values "
            "underflow the double-precision positions and produce an all-zero "
            "or noise-dominated Hessian."
        )
    return float(value)


def _validate_nfree(nfree: int) -> None:
    """Validate supported VASP finite-difference stencils."""

    if nfree not in {1, 2, 4}:
        raise _root().UnsupportedInputError(
            "VPMDK currently implements VASP finite-difference phonons for "
            "NFREE=1, NFREE=2, and NFREE=4 only."
        )


def _forces_for_displacement(reference, displaced_atom_index: int, direction, distance: float):
    """Return forces for one displaced geometry."""

    displaced = reference.copy()
    displaced.positions[displaced_atom_index] += np.asarray(direction, dtype=float) * distance
    displaced.calc = reference.calc
    return _root()._safe_get_forces(
        displaced,
        strict=True,
        apply_constraint=False,
    )


def _finite_difference_force_response(
    reference,
    displaced_atom_index: int,
    direction,
    *,
    displacement: float,
    nfree: int,
    reference_forces=None,
) -> np.ndarray:
    """Return ``-dF/dx`` for one displaced atom and direction."""

    _validate_nfree(nfree)
    # Rejection conditions (numbered; every finite-difference entry point and
    # _force_constants_displacement_from_bcar must mirror the full list):
    # (1) non-finite, (2) not positive, (3) below the underflow/noise floor.
    if (
        not np.isfinite(displacement)
        or displacement <= 0.0
        or displacement < _MIN_FINITE_DIFFERENCE_DISPLACEMENT
    ):
        raise _root().WorkdirInputError(
            "Finite-difference displacement (POTIM / FORCE_CONSTANTS_DISPLACEMENT) "
            f"must be a positive length of at least "
            f"{_MIN_FINITE_DIFFERENCE_DISPLACEMENT:g} Angstrom; smaller values "
            "underflow the double-precision positions and produce an all-zero "
            "or noise-dominated Hessian."
        )

    if nfree == 1:
        forces_reference = (
            _root()._safe_get_forces(
                reference,
                strict=True,
                apply_constraint=False,
            )
            if reference_forces is None
            else np.asarray(reference_forces, dtype=float)
        )
        forces_plus = _forces_for_displacement(
            reference,
            displaced_atom_index,
            direction,
            displacement,
        )
        force_derivative = (forces_plus - forces_reference) / displacement
    elif nfree == 2:
        forces_plus = _forces_for_displacement(
            reference,
            displaced_atom_index,
            direction,
            displacement,
        )
        forces_minus = _forces_for_displacement(
            reference,
            displaced_atom_index,
            direction,
            -displacement,
        )
        force_derivative = (forces_plus - forces_minus) / (2.0 * displacement)
    else:
        forces_plus = _forces_for_displacement(
            reference,
            displaced_atom_index,
            direction,
            displacement,
        )
        forces_minus = _forces_for_displacement(
            reference,
            displaced_atom_index,
            direction,
            -displacement,
        )
        forces_plus2 = _forces_for_displacement(
            reference,
            displaced_atom_index,
            direction,
            2.0 * displacement,
        )
        forces_minus2 = _forces_for_displacement(
            reference,
            displaced_atom_index,
            direction,
            -2.0 * displacement,
        )
        force_derivative = (
            -forces_plus2 + 8.0 * forces_plus - 8.0 * forces_minus + forces_minus2
        ) / (12.0 * displacement)

    return -force_derivative


def _finite_difference_force_constants(atoms, calculator, *, displacement: float, nfree: int = 2):
    """Return force constants from finite differences of MLP forces."""

    _validate_nfree(nfree)
    # Rejection conditions (numbered; every finite-difference entry point and
    # _force_constants_displacement_from_bcar must mirror the full list):
    # (1) non-finite, (2) not positive, (3) below the underflow/noise floor.
    if (
        not np.isfinite(displacement)
        or displacement <= 0.0
        or displacement < _MIN_FINITE_DIFFERENCE_DISPLACEMENT
    ):
        raise _root().WorkdirInputError(
            "Finite-difference displacement (POTIM / FORCE_CONSTANTS_DISPLACEMENT) "
            f"must be a positive length of at least "
            f"{_MIN_FINITE_DIFFERENCE_DISPLACEMENT:g} Angstrom; smaller values "
            "underflow the double-precision positions and produce an all-zero "
            "or noise-dominated Hessian."
        )

    root = _root()
    reference = atoms.copy()
    reference.calc = root._resolve_calculator(calculator)
    num_atoms = len(reference)
    force_constants = np.zeros((num_atoms, num_atoms, 3, 3), dtype=float)
    reference_forces = (
        root._safe_get_forces(
            reference,
            strict=True,
            apply_constraint=False,
        )
        if nfree == 1
        else None
    )

    for displaced_atom_index in range(num_atoms):
        for cart_axis in range(3):
            axis_vector = np.eye(3)[cart_axis]
            force_constants[:, displaced_atom_index, :, cart_axis] = (
                _finite_difference_force_response(
                    reference,
                    displaced_atom_index,
                    axis_vector,
                    displacement=displacement,
                    nfree=nfree,
                    reference_forces=reference_forces,
                )
            )

    return force_constants


def _cartesian_rotation_from_spglib(rotation, cell) -> np.ndarray:
    """Return Cartesian rotation matrix for a spglib fractional rotation."""

    lattice = np.asarray(cell, dtype=float)
    return lattice.T @ np.asarray(rotation, dtype=float) @ np.linalg.inv(lattice.T)


def _symmetry_operations(atoms, *, symprec: float):
    """Return Cartesian rotations and atom mappings from ASE/spglib symmetry."""

    try:
        from ase.spacegroup.symmetrize import prep_symmetry
    except ImportError as exc:  # pragma: no cover - depends on optional ASE extras
        raise RuntimeError(
            "IBRION=6/8 symmetry reduction requires ASE's spglib-backed "
            "spacegroup.symmetrize module."
        ) from exc

    try:
        rotations, _translations, atom_maps = prep_symmetry(atoms, symprec=symprec)
    except Exception as exc:
        # SYMPREC is a DISTANCE tolerance, so whether a value is usable depends on
        # the structure: 1e-5 works, and the classic `1E5` typo (or any value near
        # the interatomic spacing) makes spglib raise "too close distance between
        # atoms". A static range check would therefore reject values that are fine
        # for some cells. Classify the failure instead: it is unusable INPUT for
        # this structure (exit 1), not a retryable calculation failure (exit 2),
        # which is what the server-mode exit-code contract says exit 2 means -- a retry driver
        # resubmitted the permanently broken INCAR forever.
        raise _root().WorkdirInputError(
            f"Symmetry analysis failed at SYMPREC={symprec:g}: {exc}. "
            "Adjust SYMPREC (VASP's default is 1E-5) or check the structure."
        ) from exc
    cell = atoms.get_cell().array
    operations = []
    for rotation, atom_map in zip(rotations, atom_maps):
        operations.append(
            (
                _cartesian_rotation_from_spglib(rotation, cell),
                np.asarray(atom_map, dtype=int),
            )
        )
    return operations


def _representative_atoms_from_mappings(num_atoms: int, atom_maps) -> list[int]:
    """Return one atom index for each symmetry orbit."""

    visited: set[int] = set()
    representatives: list[int] = []
    for atom_index in range(num_atoms):
        if atom_index in visited:
            continue
        orbit = {int(mapping[atom_index]) for mapping in atom_maps}
        visited.update(orbit)
        representatives.append(min(orbit))
    return representatives


def _matrix_rank(rows: list[np.ndarray], *, tolerance: float = 1e-8) -> int:
    """Return rank of row vectors with an empty-list guard."""

    if not rows:
        return 0
    return int(np.linalg.matrix_rank(np.vstack(rows), tol=tolerance))


def _site_symmetry_displacement_axes(
    site_rotations: list[np.ndarray],
    *,
    tolerance: float = 1e-8,
) -> list[int]:
    """Return Cartesian axes needed after site-symmetry direction reduction."""

    if not site_rotations:
        return [0, 1, 2]

    selected_axes: list[int] = []
    generated_directions: list[np.ndarray] = []
    for cart_axis in range(3):
        axis_vector = np.eye(3)[cart_axis]
        orbit_directions = [rotation @ axis_vector for rotation in site_rotations]
        previous_rank = _matrix_rank(generated_directions, tolerance=tolerance)
        trial_rank = _matrix_rank(
            generated_directions + orbit_directions,
            tolerance=tolerance,
        )
        if trial_rank > previous_rank:
            selected_axes.append(cart_axis)
            generated_directions.extend(orbit_directions)
        if trial_rank >= 3:
            break

    if _matrix_rank(generated_directions, tolerance=tolerance) < 3:
        return [0, 1, 2]
    return selected_axes


def _symmetry_reduced_finite_difference_force_constants(
    atoms,
    calculator,
    *,
    displacement: float,
    nfree: int = 2,
    symprec: float = 1e-5,
):
    """Return force constants using symmetry-equivalent atom displacements."""

    _validate_nfree(nfree)
    # Rejection conditions (numbered; every finite-difference entry point and
    # _force_constants_displacement_from_bcar must mirror the full list):
    # (1) non-finite, (2) not positive, (3) below the underflow/noise floor.
    if (
        not np.isfinite(displacement)
        or displacement <= 0.0
        or displacement < _MIN_FINITE_DIFFERENCE_DISPLACEMENT
    ):
        raise _root().WorkdirInputError(
            "Finite-difference displacement (POTIM / FORCE_CONSTANTS_DISPLACEMENT) "
            f"must be a positive length of at least "
            f"{_MIN_FINITE_DIFFERENCE_DISPLACEMENT:g} Angstrom; smaller values "
            "underflow the double-precision positions and produce an all-zero "
            "or noise-dominated Hessian."
        )

    root = _root()
    reference = atoms.copy()
    reference.calc = root._resolve_calculator(calculator)
    num_atoms = len(reference)
    operations = _symmetry_operations(reference, symprec=symprec)
    if not operations:
        return _finite_difference_force_constants(
            reference,
            reference.calc,
            displacement=displacement,
            nfree=nfree,
        )

    atom_maps = [mapping for _rotation, mapping in operations]
    representatives = _representative_atoms_from_mappings(num_atoms, atom_maps)
    reference_forces = (
        root._safe_get_forces(
            reference,
            strict=True,
            apply_constraint=False,
        )
        if nfree == 1
        else None
    )

    partial = np.zeros((num_atoms, num_atoms, 3, 3), dtype=float)
    for displaced_atom_index in representatives:
        site_operations = [
            (rotation, atom_map)
            for rotation, atom_map in operations
            if int(atom_map[displaced_atom_index]) == displaced_atom_index
        ]
        site_rotations = [rotation for rotation, _atom_map in site_operations]
        selected_axes = _site_symmetry_displacement_axes(site_rotations)
        direction_observations: list[np.ndarray] = []
        response_observations: list[np.ndarray] = []

        for cart_axis in selected_axes:
            axis_vector = np.eye(3)[cart_axis]
            direct_response = _finite_difference_force_response(
                reference,
                displaced_atom_index,
                axis_vector,
                displacement=displacement,
                nfree=nfree,
                reference_forces=reference_forces,
            )

            for rotation, atom_map in site_operations:
                direction_observations.append(rotation @ axis_vector)
                transformed_response = np.zeros((num_atoms, 3), dtype=float)
                for atom_index in range(num_atoms):
                    target_atom = int(atom_map[atom_index])
                    transformed_response[target_atom] = rotation @ direct_response[atom_index]
                response_observations.append(transformed_response)

        directions = np.vstack(direction_observations)
        if np.linalg.matrix_rank(directions, tol=1e-8) < 3:  # pragma: no cover - defensive
            raise RuntimeError(
                "Site-symmetry displacement directions did not span Cartesian space "
                f"for atom {displaced_atom_index}."
            )
        for atom_index in range(num_atoms):
            responses = np.vstack(
                [observation[atom_index] for observation in response_observations]
            )
            coefficients, *_ = np.linalg.lstsq(directions, responses, rcond=None)
            partial[atom_index, displaced_atom_index] = coefficients.T

    force_constants = np.zeros((num_atoms, num_atoms, 3, 3), dtype=float)
    counts = np.zeros((num_atoms, num_atoms), dtype=int)
    for representative in representatives:
        for rotation, atom_map in operations:
            target_displaced_atom = int(atom_map[representative])
            for atom_index in range(num_atoms):
                target_atom = int(atom_map[atom_index])
                force_constants[target_atom, target_displaced_atom] += (
                    rotation @ partial[atom_index, representative] @ rotation.T
                )
                counts[target_atom, target_displaced_atom] += 1

    if np.any(counts == 0):  # pragma: no cover - defensive guard
        missing = np.argwhere(counts == 0)
        raise RuntimeError(f"Symmetry reconstruction left missing force-constant blocks: {missing!r}")

    return force_constants / counts[:, :, None, None]


def run_force_constants(
    atoms,
    calculator,
    *,
    displacement: float = 0.01,
    nfree: int = 2,
    potim: float | None = None,
    isif: int | None = None,
    ibrion: int = 7,
    use_symmetry: bool = False,
    symprec: float = 1e-5,
    oszicar_pseudo_scf: bool = False,
    pstress: float | None = None,
    nsw: int | None = None,
):
    """Write a VASP-like ``dynmat`` block from MLP force finite differences."""

    root = _root()
    _validate_nfree(nfree)
    atoms.calc = root._resolve_calculator(calculator)
    recorder = root._initialize_vasp_compat_outputs(
        atoms,
        ibrion=ibrion,
        potim=potim,
        nfree=nfree if ibrion in {5, 6} else None,
        isif=isif,
        write_oszicar_pseudo_scf=oszicar_pseudo_scf,
        pstress_kbar=pstress,
        # Echo the INCAR's NSW like every other run mode; without it the
        # vasprun fell back to len(steps)=1 whatever the INCAR said.
        nsw_requested=nsw,
    )
    energy = float(atoms.get_potential_energy())
    root._record_vasp_compat_step(
        recorder,
        atoms,
        step_index=1,
        potential_energy=energy,
        total_energy=energy,
        strict_forces=True,
        apply_force_constraints=False,
    )
    if use_symmetry:
        force_constants = _symmetry_reduced_finite_difference_force_constants(
            atoms,
            atoms.calc,
            displacement=displacement,
            nfree=nfree,
            symprec=symprec,
        )
    else:
        force_constants = _finite_difference_force_constants(
            atoms,
            atoms.calc,
            displacement=displacement,
            nfree=nfree,
        )
    recorder.force_constants = force_constants.tolist()
    root._write_vasprun_xml(recorder, atoms)
    root._append_outcar_footer(recorder)
    root._write_vasp_structure("CONTCAR", atoms, direct=True)
    return np.asarray(recorder.force_constants, dtype=float)
