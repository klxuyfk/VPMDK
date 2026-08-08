"""CLI entrypoint for VPMDK."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Mapping, Sequence

from vpmdk_client import add_client_subcommands
from vpmdk_protocol import CLIENT_SUBCOMMANDS

from .compat import vasp as vasp_compat


def _root():
    return sys.modules["vpmdk_core"]


class WorkdirInputError(RuntimeError):
    """An invalid-input error raised from the workdir input phase.

    ``reported`` records whether the human-facing message was already written to
    the normal CLI output (stdout) before the exception was raised -- e.g.
    ``print("POSCAR not found.")`` immediately before ``raise``. The one-shot
    entrypoint uses it to avoid printing the diagnostic twice: wrapped parse
    failures (``reported=False``) carry their message only in the exception, so
    _legacy_main surfaces them, while already-reported errors are left alone to
    preserve exact ``vpmdk --dir`` output.
    """

    def __init__(self, *args, reported: bool = False):
        super().__init__(*args)
        self.reported = reported


class UnsupportedInputError(NotImplementedError):
    """VPMDK does not implement the mode the user's INCAR asked for.

    Raised only by VPMDK itself (VTST ICHAIN!=0, unsupported NFREE, ...) for a
    fix-your-input condition, so the server can classify it as input_error
    (exit 1) WITHOUT also capturing a NotImplementedError that a third-party
    backend raises mid-calculation -- e.g. torch's "Could not run 'aten::...'
    with arguments from the 'CUDA' backend" for an unregistered kernel. That one
    is an exception DURING the calculation, which the server-mode exit-code contract defines as
    exit 2, and misclassifying it as exit 1 also suppressed its traceback (only
    the calculation-failure branch prints one).

    Subclasses NotImplementedError so existing callers and tests that catch the
    builtin keep working unchanged.
    """


def _print_unused_input_notices(workdir: str) -> None:
    """Print the Note: lines for VASP inputs VPMDK detects but never uses.

    One home for the loop: run_workdir emits these as its FIRST output, and
    the server calls the same helper on its hoisted request-BCAR failure path
    so a failing request still shows the notices the byte-identical one-shot
    run prints first (the server protocol's log-event example is exactly the
    KPOINTS line).
    """

    for fname in ("KPOINTS", "WAVECAR", "CHGCAR"):
        if os.path.exists(os.path.join(workdir, fname)):
            print(f"Note: {fname} detected but not used in MLP calculations.")


def _read_workdir_input(description: str, thunk):
    """Run an input-parsing step, tagging its failures as input errors.

    A malformed POSCAR or INCAR (e.g. ``NSW = not-a-number``) raises an ordinary
    parsing exception; classify it as a WorkdirInputError so both one-shot mode
    (exit 1) and server mode (input_error) honor the documented invalid-input
    contract instead of reporting it as a calculation failure (exit 2).
    """

    try:
        return thunk()
    except WorkdirInputError:
        raise
    except Exception as exc:
        raise WorkdirInputError(f"{description}: {exc}") from exc


def _reset_calculator(calculator) -> None:
    """Clear request-local ASE results while retaining loaded model weights."""

    root = _root()
    for candidate in root._calculator_candidates(calculator):
        reset = getattr(candidate, "reset", None)
        if callable(reset):
            reset()
            continue
        results = getattr(candidate, "results", None)
        if hasattr(results, "clear"):
            results.clear()


def _build_workdir_calculator(bcar, *, structure, workdir_abs):
    """Build the configured backend, classifying BCAR selector errors as input.

    BCAR SELECTOR problems -- an unknown or empty MLP, a MODEL path that does not
    exist -- are user input, like every other BCAR parse in run_workdir, and must
    produce the same one-line diagnostic instead of a raw multi-frame traceback.

    Narrow on purpose: a RuntimeError from this call means the backend PACKAGE is
    missing ("... not available. Install ..."), which is an ENVIRONMENT failure,
    not bad input, and keeps its own message and classification.

    Shared by the flat path AND both NEB per-image build sites. Those sites were
    left behind when the flat path gained the guard, so the identical BCAR typo
    was a clean diagnostic in a flat workdir and a traceback in a NEB one --
    keeping the rule in one place is what stops that drift recurring.
    """

    root = _root()
    try:
        with root._working_directory(workdir_abs):
            return root._build_calculator_from_tags(bcar, structure=structure)
    except (ValueError, FileNotFoundError) as exc:
        raise WorkdirInputError(f"Invalid BCAR backend settings: {exc}") from exc


def _capability_backend_tags(bcar, backend_tags) -> dict:
    """Return the tags that describe the backend this run will ACTUALLY use.

    In server mode ``bcar`` holds only what the REQUEST spelled out, and
    the server-mode backend-compatibility contract deliberately lets a request omit backend tags to
    inherit the resident (the documented batch pattern -- see
    examples/server_batch/calculations/0001/BCAR). Resolving capabilities from
    the request alone therefore fell back to ``BackendConfig``'s CHGNET default,
    so a request that restated the resident's tags and a byte-identical one that
    inherited them were checked against different backends.

    ``backend_tags`` is the resident's effective configuration, and it is the
    authority for anything the request did not spell out. A present-but-BLANK
    request tag is not a value either: the builder reads blanks as omissions
    (``get("MATRIS_TASK") or "efs"``), and in server mode the calculator was
    built from the resident's tags regardless. One-shot passes no resident tags
    and is unaffected.
    """

    if not backend_tags:
        return dict(bcar)
    merged = {str(key).upper(): value for key, value in backend_tags.items()}
    for key, value in dict(bcar).items():
        normalized_key = str(key).upper()
        if normalized_key in merged and str(value).strip() == "":
            continue
        merged[normalized_key] = value
    return merged


def _check_backend_output_capabilities(
    bcar, settings, *, backend_tags=None, neb_mode: bool = False
) -> None:
    """Refuse a backend that cannot produce the forces every run reports.

    Every VASP-format output this tool writes carries a TOTAL-FORCE table, and
    ``MATRIS_TASK=e`` -- documented as "downgrades capabilities to energy-only"
    -- makes the calculator return no forces at all. _safe_get_forces used to
    turn that into an exact zero force field, i.e. a converged-looking run; it
    now fails, but only after the whole calculation has been paid for and as
    calculation_error (exit 2, documented RETRYABLE) for what is a permanent
    input problem. Deciding it here from the SAME tag that configures the
    backend makes it exit 1 before anything is computed.

    Stress is only warned about: ISIF defaults to 2 for every relaxation, so an
    ion-only relaxation with a force-capable but stress-less backend is a
    perfectly good run whose OUTCAR simply has no stress block -- rejecting it
    would break working setups. Saying so out loud is what stops the omission
    from being silent.
    """

    root = _root()
    try:
        config = root.BackendConfig.from_mapping(
            _capability_backend_tags(bcar, backend_tags)
        )
        capabilities = root.get_backend_capabilities(config)
    except Exception:
        # Selector problems (unknown/empty MLP) are reported with their own
        # message by _build_workdir_calculator; never pre-empt them here.
        return
    if not capabilities.forces:
        raise WorkdirInputError(
            f"Backend {config.mlp} is configured for energy only and cannot "
            "produce forces, which every VASP-format output of this run reports "
            "(OUTCAR TOTAL-FORCE, 'FORCES: max atom, RMS', vasprun.xml forces). "
            "Select a force-capable configuration (for MATRIS: MATRIS_TASK=ef "
            "or efs)."
        )
    if not capabilities.stress:
        # ISIF>=3 makes stress part of the DYNAMICS, not just an output
        # block: run_relaxation passes relax_cell=(isif>=3) and ASE's
        # UnitCellFilter.get_forces() calls atoms.get_stress(), so a
        # stress-less backend dies on the FIRST optimizer step -- measured
        # exit 1 one-shot and exit 2 (documented RETRYABLE) in server mode,
        # for a condition fixed by the BCAR tag x the INCAR alone. Decide it
        # here, like the sibling forces gate above. The warn-don't-reject
        # rationale still holds for stress as OUTPUT (ISIF<=2, single point,
        # MD, NEB, force constants -- all measured working), so those keep
        # running with the warning.
        relaxes_cell = (
            settings.nsw > 0
            and settings.ibrion >= 1
            and settings.ibrion not in {5, 6, 7, 8}
            and not neb_mode
            and settings.isif >= 3
        )
        if relaxes_cell:
            raise WorkdirInputError(
                f"Backend {config.mlp} does not provide stress, but "
                f"ISIF={settings.isif} requests a CELL relaxation whose "
                "optimizer needs the stress tensor on every step, so the run "
                "cannot proceed. Use ISIF<=2 for an ion-only relaxation, or "
                "select a stress-capable configuration (for MATRIS: "
                "MATRIS_TASK=efs)."
            )
        if root._stress_mode_from_isif(settings.stress_isif) != "none":
            print(
                f"Warning: backend {config.mlp} does not provide stress, so the "
                f"stress output ISIF={settings.stress_isif} asks for is omitted."
            )


# Fixed species coverage of hard-wired composition heads, by canonical MLP
# identity. CHGNet's head is 94 elements (H..Pu): any heavier element fails
# every forward pass with a torch shape RuntimeError; MatRIS carries a
# hard-wired 94-row atom embedding that IndexErrors for Z>=95 the same way.
# Deterministic, structural, decided entirely by the INPUT. Only backends
# whose coverage is VERIFIED are listed; others skip the static check (a
# loaded model that DECLARES its own coverage is still probed below).
_BACKEND_MAX_ATOMIC_NUMBER = {"CHGNET": 94, "MATRIS": 94}


def _check_model_declared_species_coverage(atoms, calculator) -> None:
    """Refuse species absent from a loaded model's own element table.

    MATGL-family coverage cannot be a max-Z number: the default
    M3GNet-PES-MatPES-PBE-2025.2 declares an 89-element set with HOLES below
    Z=94 (Po/At/Rn/Fr/Ra absent while Ac-Pu are present), and alpha-Po -- a
    real material -- failed every forward pass with KeyError, classified
    RETRYABLE exit 2. The loaded model itself declares the truth
    (``model.element_types``), so consult that declaration; models that
    declare nothing are skipped.
    """

    if calculator is None:
        return
    root = _root()
    candidates = list(root._calculator_candidates(calculator))
    for candidate in list(candidates):
        for attribute in ("calculator", "model", "potential"):
            try:
                nested = getattr(candidate, attribute, None)
            except Exception:
                continue
            if nested is not None and all(nested is not existing for existing in candidates):
                candidates.append(nested)
    declared: set[str] | None = None
    for candidate in candidates:
        try:
            element_types = getattr(candidate, "element_types", None)
        except Exception:
            continue
        if element_types is None:
            continue
        try:
            symbols = {str(symbol) for symbol in element_types}
        except Exception:
            continue
        if symbols:
            declared = symbols
            break
    if declared is None:
        # bam-torch's RACECalculator declares coverage by ATOMIC NUMBER, not
        # symbol: ``uniq_element`` is a {Z: index} dict built from the
        # training set, and the published BAM-MP-core table has the same
        # holed shape as matgl's element_types (89 entries spanning 1..94
        # with Po/At/Rn/Fr/Ra absent) -- alpha-Po died mid-forward-pass with
        # a raw KeyError, exit 2 (RETRYABLE) in server mode.
        from ase.data import chemical_symbols

        for candidate in candidates:
            try:
                uniq_element = getattr(candidate, "uniq_element", None)
            except Exception:
                continue
            if not isinstance(uniq_element, Mapping) or not uniq_element:
                continue
            try:
                symbols = {
                    chemical_symbols[int(z)]
                    for z in uniq_element
                    if 0 < int(z) < len(chemical_symbols)
                }
            except Exception:
                continue
            if symbols:
                declared = symbols
                break
    if declared is None:
        return
    try:
        needed = set(atoms.get_chemical_symbols())
    except Exception:
        return
    missing = sorted(needed - declared)
    if missing:
        raise WorkdirInputError(
            f"The structure contains elements ({', '.join(missing)}) that the "
            "loaded model does not declare in its element table; every "
            "forward pass fails for this input. Choose a model that covers "
            "these elements."
        )


def _check_backend_species_coverage(
    atoms, bcar, *, backend_tags=None, calculator=None
) -> None:
    """Refuse a structure the resident model structurally cannot compute.

    A POSCAR with Cm (Z=96) against the default CHGNET backend raised
    'RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x96 and 94x1)'
    on the first forward pass -- classified as calculation_error (exit 2,
    documented RETRYABLE), so a spec-following retry driver re-submitted the
    permanently uncomputable structure forever, paying a model forward per
    attempt, while one-shot exited 1 for the byte-identical workdir. The
    coverage is a fixed property of the model and the species a fixed
    property of the input, so it is decidable here, before anything runs.
    """

    root = _root()
    try:
        config = root.BackendConfig.from_mapping(
            _capability_backend_tags(bcar, backend_tags)
        )
        mlp_identity = str(config.mlp or "").strip().upper()
    except Exception:
        return  # selector problems get their own message elsewhere
    maximum = _BACKEND_MAX_ATOMIC_NUMBER.get(mlp_identity)
    if maximum is None:
        # No verified static table for this backend -- but a LOADED model that
        # declares its own element table (matgl) is still consulted.
        _check_model_declared_species_coverage(atoms, calculator)
        return
    try:
        numbers = list(atoms.get_atomic_numbers())
    except Exception:
        return
    offending = sorted(
        {int(z) for z in numbers if int(z) > maximum}
    )
    if offending:
        symbols = ", ".join(
            f"Z={z}" for z in offending
        )
        raise WorkdirInputError(
            f"The structure contains elements ({symbols}) beyond the "
            f"{mlp_identity} model's fixed coverage of atomic numbers up to "
            f"{maximum}; every forward pass fails for this input. Choose a "
            "backend that covers these elements."
        )
    _check_model_declared_species_coverage(atoms, calculator)


def run_workdir(
    workdir: str,
    *,
    calculator=None,
    bcar_tags: Mapping[str, str] | None = None,
    charge_base_dir: str | None = None,
    backend_tags: Mapping[str, str] | None = None,
) -> None:
    """Run one VASP-style calculation directory.

    When ``calculator`` is omitted this preserves the one-shot CLI behavior and
    constructs the configured backend. A supplied calculator is reset and
    reused, which is the execution path used by server mode.

    ``backend_tags`` describes the calculator that was ALREADY BUILT (the
    resident's effective configuration in server mode). It is consulted only to
    resolve backend capabilities for tags the request inherited rather than
    spelled out; nothing else in the run reads it, so a request's own BCAR keeps
    driving every output option exactly as before.
    """

    workdir_abs = os.path.abspath(workdir)
    caller_cwd = os.path.abspath(charge_base_dir or os.getcwd())
    if calculator is not None:
        _reset_calculator(calculator)

    poscar_path = os.path.join(workdir, "POSCAR")
    incar_path = os.path.join(workdir, "INCAR")
    potcar_path = os.path.join(workdir, "POTCAR")
    kpoints_path = os.path.join(workdir, "KPOINTS")
    bcar_path = os.path.join(workdir, "BCAR")

    root = _root()
    _print_unused_input_notices(workdir)

    incar = _read_workdir_input("Failed to read INCAR", lambda: root._load_incar(incar_path))
    if bcar_tags is not None:
        bcar = dict(bcar_tags)
        # Pre-parsed request tags (server mode): the hoisted parse suppressed
        # its unknown-tag warnings (warn_unknown_tags=False); emit them HERE,
        # the position one-shot's own BCAR parse warns at -- after the Note:
        # lines and only when the INCAR read has already succeeded.
        root._warn_unknown_bcar_tags(bcar)
    else:
        bcar = _read_workdir_input(
            "Failed to read BCAR",
            lambda: (
                root._reject_broken_input_link(bcar_path, "BCAR"),
                root.parse_key_value_file(bcar_path)
                if os.path.exists(bcar_path)
                else {},
            )[1],
        )

    write_energy_csv = root._should_write_energy_csv(bcar)
    write_lammps_traj = root._should_write_lammps_trajectory(bcar)
    write_pseudo_scf = root._should_write_pseudo_scf(bcar)
    write_chgcar = root._should_write_chgcar(bcar)
    pseudo_scf_settings = root._pseudo_scf_settings_from_incar(incar, enabled=write_pseudo_scf)
    root._warn_for_unsupported_incar_tags(
        incar,
        pseudo_scf_enabled=write_pseudo_scf,
        chgcar_enabled=write_chgcar,
    )
    settings = _read_workdir_input(
        "Invalid INCAR settings", lambda: root._load_incar_settings(incar)
    )
    neb_mode = root._is_neb_like_incar(incar)
    _check_backend_output_capabilities(
        bcar,
        settings,
        backend_tags=backend_tags,
        # The heuristic answers "NEB-flavored INCAR"; the gate needs "the
        # NEB branch will actually RUN", which additionally requires
        # numbered image directories (cli discovers them again below). A
        # flat workdir whose INCAR merely carried SPRING/LCLIMB/IMAGES
        # skipped the gate and died in the relaxation branch -- exit 2
        # (RETRYABLE) in server mode for a permanently invalid pair, while
        # the identical INCAR without the stray tag got the clean exit 1.
        neb_mode=bool(
            neb_mode and root._discover_neb_image_directories(workdir)
        ),
    )
    lammps_traj_interval = (
        _read_workdir_input(
            "Invalid LAMMPS_TRAJ_INTERVAL",
            lambda: root._get_lammps_trajectory_interval(bcar),
        )
        if write_lammps_traj
        else 1
    )
    _read_workdir_input(
        "Failed to read POTCAR",
        lambda: root._reject_broken_input_link(potcar_path, "POTCAR"),
    )
    potcar_for_structure = potcar_path if os.path.exists(potcar_path) else None
    input_paths = root._VaspInputPaths(
        incar_path=os.path.abspath(incar_path),
        potcar_path=os.path.abspath(potcar_path),
        kpoints_path=os.path.abspath(kpoints_path),
    )

    previous_charge_base_dir = os.environ.get(root._CHARGE_ENV_BASE_DIR_VAR)
    os.environ[root._CHARGE_ENV_BASE_DIR_VAR] = caller_cwd
    try:
        with root._active_pseudo_scf_settings(pseudo_scf_settings), root._active_vasp_input_paths(input_paths):
            if neb_mode:
                neb_image_dirs = root._discover_neb_image_directories(workdir)
                if neb_image_dirs:
                    # Wrapped like the flat path's call site (inside
                    # _structure_to_atoms): the POMASS reader's non-regular-file
                    # rejection is an input problem, and unwrapped it escaped as
                    # a bare ValueError -- exit 2 (documented RETRYABLE) in
                    # server mode and a raw traceback one-shot, for a POTCAR
                    # that is input_error / exit 1 in a flat workdir.
                    _read_workdir_input(
                        "Failed to read POTCAR",
                        lambda: root._warn_potcar_pomass_ignored(
                            potcar_for_structure, None
                        ),
                    )
                    root.run_neb_images(
                        workdir=workdir,
                        incar=incar,
                        settings=settings,
                        bcar=bcar,
                        potcar_path=potcar_for_structure,
                        write_energy_csv=write_energy_csv,
                        write_lammps_traj=write_lammps_traj,
                        lammps_traj_interval=lammps_traj_interval,
                        oszicar_pseudo_scf=write_pseudo_scf,
                        calculator=calculator,
                        backend_tags=backend_tags,
                    )
                    print("Calculation completed.")
                    return

            root._reject_unsupported_vtst_modes(incar)

            if not os.path.exists(poscar_path):
                if neb_mode:
                    print(
                        "POSCAR not found. In NEB mode provide either a top-level POSCAR or "
                        "numbered image directories (00, 01, ...)."
                    )
                else:
                    print("POSCAR not found.")
                # Message already printed to stdout above; mark reported so the
                # one-shot entrypoint does not echo it again to stderr (preserving
                # the exact single-line ``vpmdk --dir`` output).
                raise WorkdirInputError("POSCAR not found", reported=True)

            structure = _read_workdir_input(
                "Failed to read POSCAR",
                lambda: root.read_structure(poscar_path, potcar_for_structure),
            )

            def _structure_to_atoms():
                # Still the INPUT phase: the conversion and wrap() consume the
                # user's POSCAR lattice, so a degenerate/singular cell raises a raw
                # numpy.linalg.LinAlgError here. Outside the wrapper that surfaced
                # as calculation_error (exit 2, documented retryable) for a
                # permanently broken file, while the byte-identical problem caught
                # one line earlier inside read_structure is input_error (exit 1).
                converted = root.AseAtomsAdaptor.get_atoms(structure)
                root._apply_vasp_comment_from_structure(converted, structure)
                # Shared with the NEB band rule so a byte-identical POSCAR is
                # classified the same way in a flat workdir and in an image dir.
                root._validate_finite_geometry(converted)
                root._warn_potcar_pomass_ignored(potcar_for_structure, converted)
                _check_backend_species_coverage(
                    converted, bcar, backend_tags=backend_tags
                )
                converted.wrap()
                root._apply_initial_magnetization(converted, incar)
                return converted

            atoms = _read_workdir_input("Invalid POSCAR geometry", _structure_to_atoms)
            if write_chgcar:
                # Resolve the CHGCAR grid from the user's INCAR (ENCUT / NGX* /
                # NGXF* / PREC) NOW, on the INPUT structure, before anything is
                # computed. predict_charge_density resolves it internally at the
                # very end, so a missing ENCUT or a malformed NGXF surfaced from
                # deep inside the calculation as calculation_error (exit 2,
                # documented RETRYABLE) and a retry driver would resubmit a
                # permanently broken INCAR forever.
                #
                # It must run HERE rather than next to the CHGCAR write: `atoms`
                # is mutated in place by the relaxation/MD, so checking it after
                # the run fed a DIVERGED cell (NaN/inf from a blown-up structure)
                # into the grid math and _read_workdir_input rewrote that genuine
                # calculation failure into input_error/exit 1, blaming an INCAR
                # that is perfectly valid. Checking the input cell classifies the
                # user's INCAR, and a divergence later still surfaces from
                # predict_charge_density as the calculation failure it is.
                # This run WILL write CHGCAR at the end; fail on an
                # unwritable node now, before anything is computed (the
                # recorder's unconditional preflight was scoped down to the
                # always-written artifacts).
                def _preflight_chgcar():
                    with root._working_directory(workdir_abs):
                        root._require_writable_artifact_path("CHGCAR")

                _read_workdir_input(
                    "CHGCAR output path is unusable", _preflight_chgcar
                )
                _read_workdir_input(
                    "Invalid CHGCAR grid settings",
                    lambda: root.determine_vasp_fft_grid(atoms, incar),
                )
                # Validate CHARGE_* input before running the main calculation so
                # permanent configuration errors fail immediately.
                charge_options = _read_workdir_input(
                    "Invalid CHARGE_* option",
                    lambda: root._charge_density_options_from_bcar(bcar),
                )
            if calculator is None:
                calculator = _build_workdir_calculator(
                    bcar, structure=structure, workdir_abs=workdir_abs
                )
            # Second half of the species gate: the static table ran before the
            # calculator existed; a model that DECLARES its coverage (matgl's
            # element_types) is only consultable now.
            _check_model_declared_species_coverage(atoms, calculator)

            if settings.ibrion in {5, 6}:
                with root._working_directory(workdir_abs):
                    root.run_force_constants(
                        atoms,
                        calculator,
                        displacement=settings.potim,
                        nfree=settings.nfree if settings.nfree is not None else 2,
                        potim=settings.potim,
                        isif=settings.stress_isif,
                        ibrion=settings.ibrion,
                        use_symmetry=settings.ibrion == 6,
                        symprec=settings.symprec,
                        oszicar_pseudo_scf=write_pseudo_scf,
                        pstress=settings.pstress,
                        nsw=settings.nsw,
                    )
            elif settings.ibrion in {7, 8}:
                print(
                    "Warning: IBRION=7/8 are VASP DFPT modes. VPMDK cannot run "
                    "electronic DFPT; it will write a phonopy-compatible "
                    "finite-difference dynmat/hessian from MLP forces."
                )
                displacement = _read_workdir_input(
                    "Invalid FORCE_CONSTANTS_DISPLACEMENT",
                    lambda: root._force_constants_displacement_from_bcar(bcar),
                )
                with root._working_directory(workdir_abs):
                    root.run_force_constants(
                        atoms,
                        calculator,
                        displacement=displacement,
                        nfree=2,
                        isif=settings.stress_isif,
                        ibrion=settings.ibrion,
                        use_symmetry=settings.ibrion == 8,
                        symprec=settings.symprec,
                        oszicar_pseudo_scf=write_pseudo_scf,
                        pstress=settings.pstress,
                        nsw=settings.nsw,
                    )
            elif settings.nsw <= 0 or settings.ibrion < 0:
                with root._working_directory(workdir_abs):
                    root.run_single_point(
                        atoms,
                        calculator,
                        isif=settings.stress_isif,
                        oszicar_pseudo_scf=write_pseudo_scf,
                        pstress=settings.pstress,
                        nsw=settings.nsw,
                    )
            elif settings.ibrion == 0:
                with root._working_directory(workdir_abs):
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
                        oszicar_pseudo_scf=write_pseudo_scf,
                        write_lammps_traj=write_lammps_traj,
                        lammps_traj_interval=lammps_traj_interval,
                        pstress=settings.pstress,
                    )
            else:
                if settings.ibrion == 44:
                    # Stock VASP's improved-dimer transition-state search. The
                    # catch-all "other > 0 is a relaxation" branch silently ran
                    # a BFGS energy MINIMIZATION -- moving AWAY from the saddle
                    # point the user asked for -- with exit 0 and IBRION=44
                    # echoed in every artifact. The VTST spelling of the same
                    # method (ICHAIN=2) is already rejected; the stock spelling
                    # gets the same honest refusal.
                    raise root.UnsupportedInputError(
                        "IBRION=44 (improved-dimer transition-state search) is "
                        "not implemented; running it as a plain relaxation "
                        "would minimize away from the requested saddle point."
                    )
                with root._working_directory(workdir_abs):
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
                        neb_mode=neb_mode,
                        oszicar_pseudo_scf=write_pseudo_scf,
                    )
            if write_chgcar:
                with root._working_directory(workdir_abs):
                    charge_result = root.predict_charge_density(
                        atoms,
                        incar=incar,
                        reference=atoms,
                        **charge_options,
                    )
                    vasp_compat.write_chgcar(
                        "CHGCAR",
                        atoms,
                        charge_result.density,
                        spin_density=charge_result.spin_density,
                    )
    finally:
        if previous_charge_base_dir is None:
            os.environ.pop(root._CHARGE_ENV_BASE_DIR_VAR, None)
        else:
            os.environ[root._CHARGE_ENV_BASE_DIR_VAR] = previous_charge_base_dir

    print("Calculation completed.")


# Derived from the shared client list so a new client subcommand routes here
# automatically; "serve" is the one subcommand unique to the full CLI.
_SERVER_SUBCOMMANDS = frozenset({"serve", *CLIENT_SUBCOMMANDS})


def _legacy_main(argv: Sequence[str]) -> None:
    # the one-shot compatibility contract is a non-negotiable contract: the behavior of `vpmdk`
    # and `vpmdk --dir DIR` must not change by a single byte. `--help` is not a
    # subcommand, so it dispatches HERE -- which means this parser's description,
    # arguments and (absent) epilog are part of that byte-for-byte contract.
    # Subcommand discovery therefore must NOT be advertised from this parser; it
    # belongs to _server_parser, which only handles subcommand invocations.
    # test_legacy_help_output_is_byte_for_byte_unchanged locks this.
    parser = argparse.ArgumentParser(description="Run MLP with VASP style inputs")
    parser.add_argument("--dir", default=".", help="Input directory")
    args = parser.parse_args(list(argv))
    try:
        _root().run_workdir(args.dir)
    except (WorkdirInputError, UnsupportedInputError) as exc:
        # Surface the diagnostic before exiting: run_workdir raises
        # WorkdirInputError with a message naming the offending input (e.g.
        # "Invalid INCAR settings: invalid literal for int() ... 'not-a-number'").
        # Swallowing it would leave a malformed one-shot run exiting 1 with no
        # output at all, whereas server mode still reports it as input_error. When
        # the message was already printed before raising (exc.reported), stay
        # silent to avoid duplicating it (and to preserve exact --dir output).
        #
        # UnsupportedInputError is caught alongside it because it is the SAME
        # class of failure -- server.py classifies both as input_error -- and it is
        # raised OUTSIDE _read_workdir_input by design (VTST ICHAIN!=0,
        # unsupported NFREE), so it never becomes a WorkdirInputError. Without it
        # one-shot dumped a raw multi-frame traceback for the very conditions
        # `vpmdk run` reports as a clean one-line diagnostic.
        if not getattr(exc, "reported", False):
            print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None


def _server_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vpmdk",
        description="Run VPMDK calculations or manage a resident model server",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="start a resident calculator server")
    serve.add_argument("--dir", default=".", help="directory containing BCAR/POSCAR")
    serve.add_argument("--bcar", help="backend configuration file (overrides DIR/BCAR)")
    serve.add_argument("--socket", help="Unix socket path")
    serve.add_argument(
        "--idle-timeout",
        type=float,
        default=0.0,
        metavar="SEC",
        help="stop after SEC idle; 0 disables (default: 0)",
    )
    serve.add_argument("--daemon", action="store_true", help="run as a POSIX daemon")
    serve.add_argument(
        "--log-file",
        help="server log path (daemon default: <socket>.log)",
    )
    # Internal: set by the daemon's re-exec so the fresh interpreter reports
    # readiness on the inherited pipe instead of forking again.
    serve.add_argument(
        "--daemon-notify-fd",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )

    add_client_subcommands(subparsers)
    return parser


def _server_main(argv: Sequence[str]) -> int:
    from .server import serve_cli
    from vpmdk_client import client_cli, parse_client_args

    # Same usage-error mapping as the import-light client entry point: argparse's
    # sys.exit(2) collides with the server-mode exit-code contract's "retryable calculation
    # failure", and serve's own failures already report 1. Only the SUBCOMMAND
    # parser is remapped -- _legacy_main's parser is covered by the byte-for-byte
    # one-shot compatibility contract and is deliberately left untouched.
    args = parse_client_args(_server_parser(), list(argv))
    if args.command == "serve":
        return serve_cli(args)
    return client_cli(args)


def main(argv: Sequence[str] | None = None) -> int | None:
    """CLI entrypoint, preserving legacy parsing outside known subcommands."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in _SERVER_SUBCOMMANDS:
        return _server_main(arguments)
    return _legacy_main(arguments)
