"""Helpers for reading BCAR/POSCAR/POTCAR data and preparing structures."""

from __future__ import annotations

import os
import re
import stat as stat_module
from typing import Dict, Iterable, List

from pymatgen.io.vasp import Poscar, Potcar


def _require_regular_input_file(path: str, label: str) -> None:
    """Reject a non-regular input file BEFORE any open() can block on it.

    A FIFO planted as POSCAR/INCAR/BCAR blocks ``open()`` forever (a read
    opens waits for a writer), which wedged the resident worker permanently:
    status reported busy forever, queued jobs timed out, and even
    ``stop --force`` could not preempt the blocked open -- one pathological
    request directory took down the server, defeating the request-isolation contract
    and the 2.4 stop contract. The server's own pidfile/log paths already
    guard this with S_ISREG checks; the client-submitted input files were the
    unguarded half of the same surface. ``os.stat`` never blocks (no open),
    and following symlinks is deliberate: a symlink to a regular file is a
    legitimate input, a symlink to a FIFO is not.
    """

    try:
        mode = os.stat(path).st_mode
    except OSError:
        return  # missing/unreadable: let the opener report it
    if not stat_module.S_ISREG(mode):
        raise ValueError(
            f"{label} at {path} is not a regular file; refusing to open it "
            "(a FIFO would block the reader forever)."
        )

def _reject_broken_input_link(path: str, label: str) -> None:
    """Reject an unresolvable symlink where an optional input file may sit.

    ``os.path.exists`` FOLLOWS symlinks and returns False for a dangling or
    self-referential link, so every optional-input gate read a BROKEN
    INCAR/BCAR/POTCAR as an ABSENT one: a requested 200-step relaxation ran
    as a single point and a requested backend silently became the CHGNET
    default -- exit 0, plausible artifacts, no diagnostic. The directory
    entry exists (``os.path.lexists``), so this is a broken input, not an
    omitted one; the write side already distinguishes the two (a dangling
    OUTPUT symlink stays legal because ``open("w")`` creates the target --
    the write-side behavior -- which is exactly why the READ side must not treat
    the same shape as absence).
    """

    if os.path.lexists(path) and not os.path.exists(path):
        raise ValueError(
            f"{label} at {path} is a symlink that cannot be resolved "
            "(dangling or self-referential); refusing to treat a broken "
            f"link as an absent {label}. Fix or remove the link."
        )


_VASP_COMMENT_MAX_LENGTH = 40
_VASP_COMMENT_INFO_KEY = "vasp_comment"


def _normalize_vasp_comment(comment: object) -> str:
    """Return the VASP POSCAR/CONTCAR comment line."""

    text = str(comment)
    line = text.splitlines()[0] if text.splitlines() else ""
    return line[:_VASP_COMMENT_MAX_LENGTH]


def _read_vasp_comment(path: str) -> str:
    """Return the first POSCAR/CONTCAR line as VASP would preserve it."""

    with open(path, encoding="utf-8") as handle:
        return _normalize_vasp_comment(handle.readline().rstrip("\r\n"))


def _store_vasp_comment_on_structure(structure, comment: str) -> None:
    """Attach the source POSCAR/CONTCAR comment to a pymatgen structure."""

    try:
        setattr(structure, "_vpmdk_vasp_comment", comment)
    except Exception:
        pass

    try:
        properties = structure.properties
    except Exception:
        return
    try:
        properties[_VASP_COMMENT_INFO_KEY] = comment
        properties["comment"] = comment
    except Exception:
        pass


def _apply_vasp_comment_from_structure(atoms, structure) -> None:
    """Copy preserved POSCAR/CONTCAR comment metadata onto ASE atoms."""

    comment = getattr(structure, "_vpmdk_vasp_comment", None)
    if comment is None:
        try:
            comment = structure.properties.get(_VASP_COMMENT_INFO_KEY)
        except Exception:
            comment = None
    if comment is None:
        try:
            comment = structure.properties.get("comment")
        except Exception:
            comment = None
    if comment is None:
        return

    normalized = _normalize_vasp_comment(comment)
    atoms.info[_VASP_COMMENT_INFO_KEY] = normalized
    atoms.info["comment"] = normalized


_STATIC_BCAR_TAGS = frozenset(
    {
        "MLP",
        "NNP",
        "MODEL",
        "DEVICE",
        "WRITE_ENERGY_CSV",
        "WRITE_LAMMPS_TRAJ",
        "WRITE_PSEUDO_SCF",
        "WRITE_OSZICAR_PSEUDO_SCF",
        "WRITE_CHGCAR",
        "LAMMPS_TRAJ_INTERVAL",
        "FORCE_CONSTANTS_DISPLACEMENT",
        "PHONON_DISPLACEMENT",
        "CHARGE_MLP",
        "CHARGE_BACKEND",
        "CHARGE_MODEL",
        "CHARGE_DEVICE",
        "CHARGE_SOURCE_DIR",
        "CHARGE_PYTHON",
        "CHARGE_CUTOFF",
        "CHARGE_MAX_PROBES_PER_BATCH",
    }
)


def _known_bcar_tags() -> frozenset:
    """Return every BCAR tag some consumer actually reads.

    Assembled lazily so the authoritative sources stay single-homed: the
    backend construction vocabulary lives in server.BACKEND_CONFIGURATION_TAGS
    (root-exported) and the charge model-config/weighting families live as
    dicts in charge_density; only the output/selection tags are static here.
    """

    import sys as _sys

    known = set(_STATIC_BCAR_TAGS)
    root = _sys.modules.get("vpmdk_core")
    if root is not None:
        known.update(
            str(tag).upper()
            for tag in getattr(root, "BACKEND_CONFIGURATION_TAGS", ()) or ()
        )
    charge = _sys.modules.get("vpmdk_core.charge_density")
    if charge is None:
        try:
            from vpmdk_core import charge_density as charge  # type: ignore
        except Exception:
            charge = None
    if charge is not None:
        for mapping_name in (
            "_CHARGE_MODEL_CONFIG_TAGS",
            "_DEEPCDP_WEIGHTING_KEYS",
            "_DEEPCDP_OPTION_TAGS",
        ):
            known.update(
                str(tag).upper() for tag in getattr(charge, mapping_name, {}) or {}
            )
    return frozenset(known)


def _warn_unknown_bcar_tags(tags) -> None:
    """Warn once per BCAR tag outside the vocabulary any consumer reads.

    A separate helper so the SERVER can emit the warnings at one-shot's
    position: the request-BCAR parse is hoisted ahead of run_workdir, and
    warning at parse time put the line before the Note: lines and emitted it
    even for a request whose INCAR fails before one-shot ever reads BCAR --
    a 1.2 stdout divergence. run_workdir calls this for pre-parsed tags at
    the exact point the one-shot parse would have warned.
    """

    known_tags = _known_bcar_tags()
    for key in tags:
        if str(key).upper() not in known_tags:
            print(f"Warning: BCAR tag {key} is not recognized and will be ignored.")


def parse_key_value_file(path: str, *, warn_unknown_tags: bool = True) -> Dict[str, str]:
    """Parse simple key=value style file."""

    data: Dict[str, str] = {}
    _require_regular_input_file(path, "BCAR")
    # utf-8-sig: an editor-written BCAR can start with a UTF-8 BOM, which made the
    # FIRST key parse as "\ufeffMLP". The requested backend was then silently
    # replaced by the CHGNET default with no warning at all.
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            for comment in ("#", "!"):
                if comment in line:
                    line = line.split(comment, 1)[0]
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            data[k.strip().upper()] = v.strip()
    # Say so when a tag is outside the vocabulary every consumer reads: a
    # misspelled MODELL/DEVCIE/WRITE_CHGCARR was silently ignored, so the run
    # used the DEFAULT model/device and skipped the requested outputs with
    # exit 0 and no diagnostic -- while the byte-analogous INCAR typo warns
    # 'not supported and will be ignored'. Warn-don't-change: the tag stays
    # in the mapping exactly as documented, only the silence goes. Callers
    # that hoist this parse ahead of run_workdir (the server's request path)
    # pass warn_unknown_tags=False and let run_workdir warn at one-shot's
    # position instead.
    if warn_unknown_tags:
        _warn_unknown_bcar_tags(data)
    if "MLP" not in data and "NNP" in data:
        data["MLP"] = data["NNP"]
    if "DEVICE" in data:
        # torch device strings are lowercase-only ('CPU' raises RuntimeError),
        # while the server's backend-compatibility comparison already case-folds DEVICE --
        # so a VASP-style uppercase 'DEVICE = CPU' was ACCEPTED and computed by
        # the server (exit 0) but crashed the one-shot CLI on the byte-identical
        # directory (exit 1), a 1.2 divergence keyed on letter case. Folding at
        # the single BCAR parse point makes every reader (builders, resident
        # startup, request validation) see the same value. MODEL and the other
        # tags stay untouched: paths are case-sensitive.
        data["DEVICE"] = data["DEVICE"].lower()
    return data


def _resolve_mlp_tag(bcar_tags: Dict[str, str], *, default: str = "CHGNET") -> str:
    """Return selected BCAR potential tag using ``MLP`` with legacy ``NNP`` fallback."""

    if "MLP" in bcar_tags:
        mlp_value = str(bcar_tags.get("MLP", "")).strip()
        if not mlp_value:
            raise ValueError("BCAR tag MLP is present but empty.")
        return mlp_value.upper()

    if "NNP" in bcar_tags:
        nnp_value = str(bcar_tags.get("NNP", "")).strip()
        if not nnp_value:
            raise ValueError("BCAR tag NNP is present but empty.")
        return nnp_value.upper()

    return default.strip().upper()


def _flatten(values: Iterable[object]) -> List[float]:
    """Return flattened list of floats from nested sequences."""

    flattened: List[float] = []
    for item in values:
        if isinstance(item, (list, tuple)):
            flattened.extend(_flatten(item))
        else:
            try:
                flattened.append(float(item))
            except (TypeError, ValueError):
                continue
    return flattened


def _parse_magmom_values(value) -> List[float]:
    """Parse VASP-style MAGMOM definition into a list of floats."""

    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        return _flatten(value)

    text = str(value).strip()
    if not text:
        return []

    tokens = text.replace(",", " ").split()
    result: List[float] = []
    for token in tokens:
        if not token:
            continue
        if "*" in token:
            count_str, moment_str = token.split("*", 1)
            try:
                count = int(float(count_str))
            except (TypeError, ValueError, OverflowError):
                # OverflowError: int(float("inf"))/a huge repeat count raises it,
                # not ValueError. This branch tolerates a malformed MAGMOM token by
                # skipping it, so catch that class too rather than letting it escape.
                continue
            if count > 1_000_000:
                # Defense in depth behind _reject_huge_repeat_counts: values
                # arriving through the library API (not an INCAR file) must
                # not expand a multi-GB list either. The exponent spelling
                # ('1e10*2.0') escapes pymatgen's int() and detonated HERE.
                raise ValueError(
                    f"MAGMOM-style repeat token {token!r} has a count above "
                    "1000000; no structure has that many ions."
                )
            nested = _parse_magmom_values(moment_str)
            if not nested:
                try:
                    nested = [float(moment_str)]
                except (TypeError, ValueError):
                    continue
            if count * len(nested) > 1_000_000:
                # The per-count cap above bounds ONE nesting level, but the
                # nested spelling ('1000*1000*1000*1.0', or pymatgen's
                # documented count1*count2*value) MULTIPLIES levels, so the
                # product must be bounded too or the caps compose to 1e6^depth.
                raise ValueError(
                    f"MAGMOM-style repeat token {token!r} expands to more than "
                    "1000000 values; no structure has that many ions."
                )
            if len(result) + count * len(nested) > 1_000_000:
                # Sum cap across tokens, mirroring _reject_huge_repeat_counts'
                # total-expansion bound: many just-under-cap tokens must not
                # expand without bound either.
                raise ValueError(
                    "MAGMOM-style value expands to more than 1000000 values "
                    "in total; no structure has that many ions."
                )
            if len(nested) == 1:
                result.extend(nested * count)
            else:
                for _ in range(count):
                    result.extend(nested)
            continue
        try:
            result.append(float(token))
        except (TypeError, ValueError):
            continue
    return result


def _normalize_species_labels(symbols: Iterable[object]) -> List[str]:
    """Return species labels with POTCAR-style suffixes removed."""

    normalized: List[str] = []
    for symbol in symbols:
        text: str = ""
        if isinstance(symbol, str):
            text = symbol.strip()
        elif hasattr(symbol, "symbol"):
            text = str(getattr(symbol, "symbol", "")).strip()
        else:
            try:
                text = str(symbol).strip()
            except Exception:
                continue
        if not text:
            continue
        base = text.split("_", 1)[0].strip()
        normalized.append(base or text)
    return normalized


def _infer_type_map(structure) -> List[str]:
    """Infer a DeePMD type map from the provided structure when possible."""

    labels: List[str] = []
    for attr in ("site_symbols", "species"):
        symbols = getattr(structure, attr, None)
        if symbols:
            labels = _normalize_species_labels(symbols)
            if labels:
                break

    unique: List[str] = []
    for label in labels:
        if label and label not in unique:
            unique.append(label)

    return unique


def _expand_magmom_to_atoms(magmoms: List[float], atoms) -> List[float] | None:
    """Expand species MAGMOM values to per-atom list when necessary."""

    if not magmoms:
        return None

    num_atoms = len(atoms)
    if len(magmoms) == num_atoms:
        return magmoms

    symbols = atoms.get_chemical_symbols()
    species_counts: List[int] = []
    previous_symbol: str | None = None
    for symbol in symbols:
        if symbol == previous_symbol:
            species_counts[-1] += 1
        else:
            species_counts.append(1)
            previous_symbol = symbol

    if len(magmoms) == len(species_counts):
        expanded: List[float] = []
        for moment, count in zip(magmoms, species_counts):
            expanded.extend([moment] * count)
        return expanded

    return None


def _warn_magmom_is_inert(moments) -> None:
    if not moments or not any(abs(float(m)) > 1e-12 for m in moments):
        return
    print(
        "Warning: MAGMOM is attached as ASE initial magnetic moments, but no "
        "supported backend reads initial moments (they only PREDICT moments "
        "as outputs), so FM and AFM orderings produce identical results here."
    )


def _apply_initial_magnetization(atoms, incar) -> None:
    """Populate initial magnetic moments from INCAR when available."""

    if not hasattr(incar, "get"):
        return
    if "MAGMOM" not in incar:
        return

    raw = incar.get("MAGMOM")
    magmoms = _parse_magmom_values(raw)
    if not magmoms:
        return
    expanded = _expand_magmom_to_atoms(magmoms, atoms)
    if expanded is None or len(expanded) != len(atoms):
        print(
            "Warning: Unable to reconcile MAGMOM values with number of atoms; "
            "initial magnetic moments will not be set."
        )
        return
    atoms.set_initial_magnetic_moments(expanded)
    _warn_magmom_is_inert(expanded)


# Minimum perpendicular width of a periodic cell, in Angstrom. See the
# tiny-cell comment inside _validate_finite_geometry.
_MIN_PERIODIC_CELL_WIDTH = 0.5

# Maximum perpendicular width, in Angstrom (1e6 A = 0.1 mm -- >100x beyond any
# vacuum slab). The expanding mirror of the tiny-cell case: a diverged
# variable-cell CONTCAR in the 1e150+ range reused as POSCAR either crashed
# mid-run as a "retryable" exit 2 (LJ/scipy overflow) or, with CHGNet,
# COMPLETED with exit 0 and nan stress / inf volume that pymatgen parses
# without complaint.
_MAX_PERIODIC_CELL_WIDTH = 1.0e6

# Maximum cell VOLUME, in cubic Angstrom (1e9 = a 1000 A cube). The width cap
# above bounds each AXIS, but the neighbour-search cost pymatgen/backends pay
# is the BIN COUNT, which scales with the volume/cutoff^3 PRODUCT (the
# axis-vs-resource distinction, on the input geometry itself): a 20000 A cube with
# 2 atoms -- 50x under the width ceiling -- died in find_points_in_spheres
# asking for 152 GB (classified retryable exit 2), and a 4000 A cube wedged
# the resident worker beyond 200 s at 28.5 GB RSS, taking status/stop down
# with it. 1e9 A^3 is >100x beyond any MLP-relevant cell (a 1e6 A slab axis
# with ordinary 10 A cross-sections still passes at 1e8), while the worst
# accepted case keeps the bin count ~1e7 at a typical 5 A cutoff.
_MAX_PERIODIC_CELL_VOLUME = 1.0e9

# Per-axis companion to the volume cap for UNWRAPPED coordinate spans: the
# neighbour search replicates periodic images along each axis independently,
# so a SINGLE-axis excursion escapes any product-of-spans rule (the other
# floored factors stay 1) while its replication cost keeps growing linearly.
# Measured on a 3.87 A cell under an 8 GiB cap: a 3.9e6 A single-axis span
# completes in 2.3 s, 3.9e7 A is a MemoryError -- 1e7 A sits between the
# largest completing case and the smallest failing one, and the cost at the
# limit stays bounded (~2 GB). Axes wider than this in the CELL itself keep
# their own width as the limit.
_MAX_UNWRAPPED_AXIS_SPAN = 1.0e7

# Two ions closer than this (in Angstrom, minimum-image) are the same site: a
# duplicated POSCAR coordinate line, or fractional 0.0 and 1.0 coinciding
# under PBC. No physical pair sits below ~0.5 A (H2 is 0.74 A), and every
# tested backend deterministically returns nan for a coincident pair -- a
# permanent input classified as retryable exit 2 after a full model load.
_MIN_INTERATOMIC_DISTANCE = 0.01

def _reject_coincident_atoms(atoms) -> None:
    """Reject two ions occupying the same site, judged with minimum image.

    Grid-bucket algorithm, O(N) memory and time: the first version built
    ase.geometry.get_distances' all-pairs MIC matrix for anything up to 4096
    atoms, which cost ~17 GB RSS and ~17 s AT INPUT TIME for an ordinary
    4096-atom supercell -- and where the allocation failed, the MemoryError
    was rewritten into an input error for a perfectly fine structure. Here
    every atom is bucketed by its wrapped SCALED coordinate at a resolution
    of at least the threshold per axis (the width guard bounds every
    perpendicular width to >= 0.5 A, so a pair within the 0.01 A threshold
    always lands in the same or an adjacent bucket, with modular wrap for
    the periodic boundary), and only same/neighbor-bucket candidates get an
    exact minimum-image distance.
    """

    import numpy as np

    count = len(atoms)
    if count < 2:
        return
    cell = np.asarray(atoms.get_cell(), dtype=float)
    periodic = bool(cell.any())
    if periodic:
        scaled = np.mod(
            np.asarray(atoms.get_scaled_positions(wrap=False), dtype=float), 1.0
        )
        determinant = float(np.linalg.det(cell))
        widths = []
        for axis in range(3):
            face = float(
                np.linalg.norm(np.cross(cell[(axis + 1) % 3], cell[(axis + 2) % 3]))
            )
            widths.append(abs(determinant) / face if face else 1.0)
        # Bucket edge >= threshold along each axis (in cartesian terms).
        grid = np.array(
            [
                max(1, min(int(widths[axis] / _MIN_INTERATOMIC_DISTANCE), 10**6))
                for axis in range(3)
            ],
            dtype=int,
        )
        indices = np.minimum((scaled * grid).astype(int), grid - 1)
    else:
        positions = np.asarray(atoms.get_positions(), dtype=float)
        origin = positions.min(axis=0)
        indices = ((positions - origin) / _MIN_INTERATOMIC_DISTANCE).astype(int)
        grid = None

    buckets: dict[tuple, list[int]] = {}
    for atom_index, key in enumerate(map(tuple, indices)):
        buckets.setdefault(key, []).append(atom_index)

    def _separation(first: int, second: int) -> float:
        if periodic:
            delta = scaled[second] - scaled[first]
            delta -= np.round(delta)
            return float(np.linalg.norm(delta @ cell))
        return float(
            np.linalg.norm(
                np.asarray(atoms.get_positions()[second])
                - np.asarray(atoms.get_positions()[first])
            )
        )

    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    for key, members in buckets.items():
        for offset in offsets:
            if periodic:
                neighbor = tuple(
                    (key[axis] + offset[axis]) % int(grid[axis]) for axis in range(3)
                )
            else:
                neighbor = tuple(key[axis] + offset[axis] for axis in range(3))
            if neighbor < key:
                continue  # each bucket pair examined once
            others = buckets.get(neighbor)
            if not others:
                continue
            for i_pos, first in enumerate(members):
                candidates = (
                    members[i_pos + 1 :] if neighbor == key else others
                )
                for second in candidates:
                    if first == second:
                        continue
                    if _separation(first, second) < _MIN_INTERATOMIC_DISTANCE:
                        low, high = sorted((first + 1, second + 1))
                        raise ValueError(
                            f"ions {low} and {high} occupy the same site "
                            f"(separation below {_MIN_INTERATOMIC_DISTANCE} A "
                            "under periodic boundary conditions); every "
                            "backend returns nan for a coincident pair, so "
                            "the structure cannot be computed"
                        )


def _validate_finite_geometry(atoms) -> None:
    """Reject a non-finite or singular geometry while still in the INPUT phase.

    A ``nan``/``inf`` in a POSCAR lattice row or a fractional coordinate is
    typically a CONTCAR from a diverged upstream run being reused as the next
    POSCAR. Nothing downstream can detect it: ``Cell.complete()`` substitutes unit
    vectors only for an all-ZERO row, ``wrap()`` raises only for a FINITE singular
    cell, and a calculator given an atom at ``nan`` simply finds no neighbours --
    so the run reported exit 0 and ``Calculation completed.`` with a finite but
    physically meaningless energy, forces written as exactly 0.0, ``nan`` stress,
    and a CONTCAR that propagated the ``nan`` to the next stage. IBRION=2 was
    worse: zero forces read as immediate convergence.

    An entirely zero cell means "no cell given" (a legitimate molecular run) and
    stays allowed, matching the NEB band rule.
    """

    import numpy as np

    positions = np.asarray(atoms.get_positions(), dtype=float)
    if positions.size and not np.isfinite(positions).all():
        raise ValueError(
            "the atom positions contain non-finite values (nan/inf), so the "
            "structure has no well-defined geometry"
        )
    cell = np.asarray(atoms.get_cell(), dtype=float)
    if not np.isfinite(cell).all():
        raise ValueError(
            "the lattice contains non-finite values (nan/inf), so the structure "
            "has no well-defined geometry"
        )
    if cell.any():
        determinant = float(np.linalg.det(cell))
        if determinant == 0.0:
            raise ValueError(
                "the lattice is singular (zero cell volume), so the structure has "
                "no well-defined geometry"
            )
        # The determinant and face areas here are float64: a lattice in the
        # 1e150+ range has FINITE entries but overflows both to inf, making
        # width = inf/inf = nan -- and every nan comparison is False, which
        # silently disabled the width guard for exactly the expanding mirror
        # of the collapsed-cell case it was built for (measured: a 5.43e200
        # scale factor COMPLETED with exit 0, nan stress and inf volume under
        # CHGNet). Judge non-finiteness of the DERIVED quantities explicitly.
        if not np.isfinite(determinant):
            raise ValueError(
                "the lattice volume overflows a float (lattice magnitudes "
                "around 1e150 Angstrom or larger); check the POSCAR scale "
                "factor and lattice vectors"
            )
        # Tiny-but-finite cells are the huge-but-finite class in reverse: a
        # POSCAR scale-factor typo (0.01 instead of 1.0) or a collapsed
        # CONTCAR from a diverged variable-cell run passes both checks above,
        # and the backend's periodic-image enumeration then grows as
        # (cutoff/width)^3 -- measured: a 0.3 A cell wedged CHGNet's first
        # energy call beyond 280 s, and it worsens cubically from there, so a
        # resident worker hangs with no output and no exit. Judge the
        # PERPENDICULAR widths (|det| / |a_j x a_k|), which also catch a
        # near-degenerate cell whose vectors are individually long. No real
        # crystal is below ~2 A per axis; 0.5 A keeps a 4x margin while the
        # worst accepted case still completes in seconds.
        for axis in range(3):
            other = np.cross(cell[(axis + 1) % 3], cell[(axis + 2) % 3])
            face_area = float(np.linalg.norm(other))
            if face_area == 0.0:
                continue  # unreachable with det != 0, but stay safe
            width = abs(determinant) / face_area
            if not np.isfinite(width) or width > _MAX_PERIODIC_CELL_WIDTH:
                raise ValueError(
                    f"the lattice is {width:.4g} A wide along axis "
                    f"{axis + 1} (above the supported maximum of "
                    f"{_MAX_PERIODIC_CELL_WIDTH:g} A); check the POSCAR "
                    "scale factor and lattice vectors"
                )
            if width < _MIN_PERIODIC_CELL_WIDTH:
                raise ValueError(
                    f"the lattice is only {width:.4g} A wide along axis "
                    f"{axis + 1} (below the supported minimum of "
                    f"{_MIN_PERIODIC_CELL_WIDTH} A); check the POSCAR scale "
                    "factor and lattice vectors"
                )
        # Bound the RESOURCE (bin count ~ volume/cutoff^3), not just each
        # axis -- see _MAX_PERIODIC_CELL_VOLUME.
        if abs(determinant) > _MAX_PERIODIC_CELL_VOLUME:
            raise ValueError(
                f"the cell volume is {abs(determinant):.4g} A^3 (above the "
                f"supported maximum of {_MAX_PERIODIC_CELL_VOLUME:g} A^3): "
                "the neighbour search allocates bins proportional to the "
                "volume, so a cell this large exhausts memory or wedges the "
                "worker; check the POSCAR scale factor and lattice vectors"
            )
    _reject_coincident_atoms(atoms)


def _cleaned_poscar_lines(text: str) -> List[str]:
    """Return POSCAR lines exactly as pymatgen's reader indexes them.

    ``Poscar.from_str`` keeps only the first blank-line-delimited chunk (the
    structure block; velocities follow a blank line) and pushes every line
    through ``pymatgen.util.io_utils.clean_lines``, which TRUNCATES AT ``#`` and
    strips whitespace. Reading the raw file instead made this module disagree
    with the parser about which line 6 is and what it says -- see
    _poscar_declares_species.
    """

    chunks = re.split(r"\n\s*\n", text.rstrip(), flags=re.MULTILINE)
    if chunks and chunks[0] == "":
        # A leading blank line stays part of the block as an empty comment,
        # exactly as pymatgen re-attaches it.
        chunks.pop(0)
        if chunks:
            chunks[0] = "\n" + chunks[0]
    if not chunks:
        return []
    return [
        (line[: line.index("#")] if "#" in line else line).strip()
        for line in chunks[0].split("\n")
    ]


def _poscar_declares_species(poscar_path: str) -> bool:
    """Whether the POSCAR carries a species-name line (VASP 5 format).

    Line 6 of a POSCAR is either the species names (VASP 5) or the ion counts
    (VASP 4), so this is decidable from the file itself. It has to be: with real
    pymatgen a VASP-4 POSCAR does NOT yield empty ``site_symbols`` -- pymatgen
    fabricates ``['H', ...]`` and only emits a BadPoscarWarning on stderr -- so
    the ``not poscar.site_symbols`` branch below could never fire and a Si cell
    was silently computed as HYDROGEN, with CONTCAR rewritten to match.

    The decision must be made on the line THE PARSER sees. Reading the raw text
    let a perfectly ordinary comment (`` 2   # number of Si atoms``) flip the
    verdict: pymatgen truncates at ``#``, reads ``2``, and takes the VASP-4
    branch (fabricating hydrogen), while ``int('#')`` failed here and reported
    "VASP 5" -- which SKIPS every VASP-4 guard, so the Si cell was computed as
    H2 (-2.35 eV instead of -10.63 eV), exited 0, and wrote a CONTCAR whose
    species line is ``H``. Same class as the INCAR tokenizer mismatch fixed the
    round before: mirror the upstream reader instead of re-inventing it.
    """

    try:
        with open(poscar_path, encoding="utf-8", errors="surrogateescape") as handle:
            lines = _cleaned_poscar_lines(handle.read())
    except OSError:
        # Unreadable here means unreadable for Poscar.from_file too; let that
        # report the problem rather than guessing about the format.
        return True
    if len(lines) < 6:
        return True
    tokens = lines[5].split()
    if not tokens:
        # pymatgen's `[int(i) for i in lines[5].split()]` succeeds on an empty
        # list, i.e. it takes the VASP-4 branch. Say so, so the species still
        # have to come from a POTCAR rather than from fabricated names.
        return False
    # Counts are integers; species names never parse as numbers.
    for token in tokens:
        try:
            int(token)
        except ValueError:
            return True
    return False


def _apply_species_from_potcar(poscar, structure, symbols: List[str]):
    """Return ``structure`` relabelled to the POTCAR's species order, or ``None``.

    ``Poscar.site_symbols`` is a READ-ONLY property in real pymatgen, so the
    assignment this used to do raised AttributeError and the whole POSCAR/POTCAR
    reconciliation -- including the documented "Using POTCAR order" warning --
    turned into an input error (exit 1) for exactly the inputs it exists to
    repair. The test suite could not see it: conftest's Poscar stub exposes
    site_symbols as a plain attribute.

    The assignment is still attempted first so any object that does accept it
    keeps its behavior; the rebuild below is the real-pymatgen path.

    ``None`` means "the POTCAR order could not be applied". That case USED to
    return the input structure unchanged, which is only harmless when the POSCAR
    itself carries species names: for a VASP 4 POSCAR the unchanged structure is
    pymatgen's FABRICATED ['H', ...] cell, so a Cu band whose POTCAR listed two
    species while the POSCAR has one ion group was computed as hydrogen and
    reported as success. The caller decides -- refine, or reject.
    """

    try:
        poscar.site_symbols = symbols
        return poscar.structure
    except (AttributeError, TypeError):
        pass

    counts = [int(count) for count in (getattr(poscar, "natoms", None) or [])]
    if len(counts) != len(symbols):
        return None
    expanded: List[str] = []
    for count, symbol in zip(counts, symbols):
        expanded.extend([symbol] * count)
    if not expanded or len(expanded) != len(structure):
        return None
    try:
        relabelled = structure.copy()
        for index, symbol in enumerate(expanded):
            relabelled.replace(index, symbol)
    except Exception:
        return None
    return relabelled


# Bound the total POSCAR ion count before pymatgen allocates per-atom lists.
# The limit remains above expected large MLIP workloads.
_MAX_POSCAR_ION_COUNT = 10_000_000


def _reject_absurd_poscar_ion_counts(poscar_path: str) -> None:
    """Bound the declared ion count BEFORE pymatgen expands per-atom lists.

    Judged on the same cleaned lines the parser sees (_cleaned_poscar_lines):
    line 6 is the counts line for VASP 4, line 7 for VASP 5 -- check whichever
    of the two consists purely of integers, matching pymatgen's own format
    decision rather than re-inventing it.
    """

    try:
        with open(poscar_path, encoding="utf-8", errors="surrogateescape") as handle:
            lines = _cleaned_poscar_lines(handle.read())
    except OSError:
        return  # unreadable for Poscar.from_file too; let it report
    # pymatgen supports a MULTI-LINE species/counts block (written by VASP
    # itself for >20 species groups: symbol lines, then equally many count
    # lines) and scans lines[5+n] for n in 1..10 to find where the counts
    # start -- so counts can sit on line 8+, where a fixed (5, 6) window
    # never looked and the cap was bypassed. Mirror the parser: walk from
    # line 6, treat all-integer lines as counts (there may be several) and
    # keep a RUNNING total; stop at the first non-integer line after counts
    # have been seen (the Selective/Direct/Cartesian line), so coordinate
    # data is never miscounted.
    total = 0
    seen_counts = False
    counts_line_index = None
    for line_index in range(5, min(len(lines), 40)):
        tokens = lines[line_index].split()
        if not tokens:
            continue
        try:
            counts = [int(token) for token in tokens]
        except ValueError:
            if seen_counts:
                break  # the mode line after the counts block
            continue  # species-symbol line(s) before the counts
        seen_counts = True
        if counts_line_index is None:
            counts_line_index = line_index
        total += sum(count for count in counts if count > 0)
        if total > _MAX_POSCAR_ION_COUNT:
            raise ValueError(
                f"POSCAR declares {total} ions starting on line "
                f"{counts_line_index + 1}, above the supported maximum of "
                f"{_MAX_POSCAR_ION_COUNT}; check the ion-counts lines."
            )


def _warn_potcar_pomass_ignored(potcar_path: str | None, atoms) -> None:
    """Disclose that POTCAR POMASS does not set the masses VPMDK uses.

    An isotope-edited POTCAR (``POMASS = 2.014`` -- the canonical VASP way to
    run deuterium MD or phonons) ran at ASE's standard-isotope mass with exit
    0 and no disclosure: vibrational periods sqrt(2) off, diffusion
    coefficients and phonon frequencies silently wrong. Reading POMASS into
    the dynamics would change existing runs, so the divergence is DISCLOSED
    (the LCLIMB/ANDERSEN_PROB precedent). Pairs the Nth POMASS with the Nth
    TITEL element, matching the PSCTR block order.
    """

    if not potcar_path or not os.path.exists(potcar_path):
        return
    _require_regular_input_file(potcar_path, "POTCAR")
    try:
        from ase.data import atomic_masses, atomic_numbers

        titles: list[str] = []
        masses: list[float] = []
        with open(potcar_path, encoding="utf-8", errors="surrogateescape") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped.startswith("TITEL"):
                    parts = stripped.split()
                    titles.append(parts[3] if len(parts) > 3 else "")
                elif stripped.startswith("POMASS"):
                    token = stripped.split("=", 1)[-1].split(";")[0].strip()
                    try:
                        masses.append(float(token))
                    except ValueError:
                        masses.append(float("nan"))
        mismatched: list[str] = []
        for title, declared in zip(titles, masses):
            element = title.split("_", 1)[0].strip()
            number = atomic_numbers.get(element)
            if number is None or declared != declared:  # unknown / nan
                continue
            used = float(atomic_masses[number])
            if abs(declared - used) > 1e-2 * max(1.0, used):
                mismatched.append(
                    f"{element}: POTCAR POMASS={declared:g}, VPMDK uses {used:g}"
                )
        if mismatched:
            print(
                "Warning: POTCAR POMASS is not read; VPMDK takes atomic "
                "masses from ASE's element table. Differing entries: "
                + "; ".join(mismatched)
                + ". Isotope masses declared in POTCAR do NOT affect MD or "
                "phonon results here."
            )
    except OSError:
        return


def _poscar_declares_trailing_species(path: str) -> bool:
    """VASP-4 POSCAR with an element symbol at the END of each coordinate line.

    Mirrors ``Poscar.from_str``'s rule 3 (read from its source): when line 6
    is the counts line (VASP 4), the parser reads token index 3 (6 under
    Selective dynamics) of each of the first ``sum(counts)`` coordinate lines
    and uses them as the REAL species when all are valid element symbols --
    only when that also fails does it fabricate H/He/... names. Judging
    species on line 6 alone therefore rejected a format the parser reads
    correctly (exit 1 for a file HEAD computed fine).
    """

    try:
        with open(path, encoding="utf-8", errors="surrogateescape") as handle:
            lines = _cleaned_poscar_lines(handle.read())
        counts = [int(token) for token in lines[5].split()]
    except (OSError, ValueError, IndexError):
        return False
    if not counts or any(count < 0 for count in counts):
        return False
    position = 6
    token_index = 3
    try:
        pos_type = lines[position].split()[0]
    except IndexError:
        return False
    if pos_type[:1] in "sS":
        position += 1
        token_index = 6
    n_sites = sum(counts)
    coordinate_lines = lines[position + 1 : position + 1 + n_sites]
    if len(coordinate_lines) < n_sites or n_sites == 0:
        return False
    try:
        trailing = [line.split()[token_index] for line in coordinate_lines]
    except IndexError:
        return False
    return all(token in _ELEMENT_SYMBOLS for token in trailing)


from ase.data import chemical_symbols as _ase_chemical_symbols

_ELEMENT_SYMBOLS = frozenset(_ase_chemical_symbols[1:])


def _scan_poscar_species_counts(lines):
    """Return (symbol_tokens, count_tokens, next_index) mirroring the parser.

    Walks from line 6 exactly like _reject_absurd_poscar_ion_counts: alpha
    lines before the first all-integer line are species symbols, consecutive
    all-integer lines are counts, and the first non-integer line after counts
    is where the Selective/Direct/Cartesian block starts.
    """

    symbols: list[str] = []
    counts: list[str] = []
    index = 5
    seen_counts = False
    while index < min(len(lines), 40):
        tokens = lines[index].split()
        if not tokens:
            index += 1
            continue
        try:
            [int(token) for token in tokens]
        except ValueError:
            if seen_counts:
                break
            for token in tokens:
                # A species line may carry a Fortran '!' comment ('Si   !
                # silicon'): pymatgen keeps the junk tokens but its
                # zip-to-counts drops them harmlessly and HEAD ran the file
                # with the exactly correct composition -- counting them as
                # species rejected working decks. Only
                # LEADING valid element symbols (with pymatgen's VASP-6
                # hash/underscore normalization) are species; the first
                # non-element token ends the species portion of the line.
                normalized = token.split("/")[0].split("_")[0]
                if normalized in _ELEMENT_SYMBOLS:
                    symbols.append(token)
                else:
                    break
            index += 1
            continue
        counts.extend(tokens)
        seen_counts = True
        index += 1
    return symbols, counts, index


def _reject_mismatched_species_counts(poscar_path: str) -> None:
    """Reject a species line whose length disagrees with the counts line.

    pymatgen zips ``for idx, n_atom in enumerate(n_atoms):
    atomic_symbols.extend([symbols[idx]] * n_atom)`` -- with MORE symbols
    than counts the trailing species are silently DROPPED, so a hand-edited
    ``Si Ge`` over ``2`` computed a Si2 energy for a file that says SiGe,
    exit 0, and rewrote CONTCAR with the wrong chemistry. (The mirror typo,
    fewer symbols than counts, raised a bare IndexError; it gets the same
    clear diagnostic here.) VASP-4 files (no symbol line) are untouched.
    """

    try:
        with open(poscar_path, encoding="utf-8", errors="surrogateescape") as handle:
            lines = _cleaned_poscar_lines(handle.read())
    except OSError:
        return
    symbols, counts, _ = _scan_poscar_species_counts(lines)
    if not symbols or not counts:
        return  # VASP-4 (counts on line 6) or malformed enough to fail later
    if len(symbols) != len(counts):
        raise ValueError(
            f"POSCAR declares {len(symbols)} species ({' '.join(symbols)}) "
            f"but {len(counts)} ion-count entries ({' '.join(counts)}); the "
            "parser would silently drop the extra species and compute a "
            "different composition than the file declares. Make the two "
            "lines the same length."
        )


_SELECTIVE_DYNAMICS_TOKENS = frozenset({"T", "F"})


def _reject_ambiguous_selective_dynamics_tokens(poscar_path: str) -> None:
    """Reject selective-dynamics flags spelled other than bare ``T``/``F``.

    pymatgen reads the flags with an EXACT ``value == "T"`` comparison, so
    every other Fortran-logical TRUE spelling VASP itself accepts (``t``,
    ``TRUE``, ``.T.``, ``.TRUE.``) silently became False -- the atom the
    user marked FREE was FROZEN, exit 0, and CONTCAR rewrote the line as
    ``F F F``, the inverse of the input. The parsed mask is already boolean
    when the length guard below sees it, so the spelling must be judged on
    the raw text.
    """

    try:
        with open(poscar_path, encoding="utf-8", errors="surrogateescape") as handle:
            lines = _cleaned_poscar_lines(handle.read())
    except OSError:
        return
    symbols, counts, index = _scan_poscar_species_counts(lines)
    if not counts or index >= len(lines):
        return
    first_tokens = lines[index].split()
    if not first_tokens or first_tokens[0][:1] not in "sS":
        return  # no Selective dynamics block
    n_sites = 0
    for token in counts:
        try:
            value = int(token)
        except ValueError:
            continue
        if value > 0:
            n_sites += value
    coordinate_start = index + 2  # the Selective line, then the mode line
    for offset, line in enumerate(
        lines[coordinate_start : coordinate_start + n_sites]
    ):
        for token in line.split()[3:6]:
            if token not in _SELECTIVE_DYNAMICS_TOKENS:
                raise ValueError(
                    f"POSCAR selective-dynamics flag {token!r} on coordinate "
                    f"line {offset + 1} is not a bare T or F: the parser "
                    "reads flags with an exact comparison, so this spelling "
                    "would silently FREEZE an atom the file marks free. "
                    "Write the flags as T or F."
                )


def _reject_negative_scale_cartesian_poscar(poscar_path: str) -> None:
    """Reject the one POSCAR shape the parser reads as a DIFFERENT structure.

    VASP interprets a negative scale factor as the target cell VOLUME and
    applies the DERIVED factor (-scale/vol)**(1/3) to the lattice AND the
    Cartesian positions. pymatgen applies the derived factor to the lattice
    but multiplies Cartesian positions by the RAW negative volume number, so
    a VASP-legal deck (scale = -V, Cartesian) parsed to a completely
    different geometry: fractional 0.75 became -30.67, the energy was wrong
    by 0.36 eV, and CONTCAR persisted the corrupted positions -- exit 0, no
    warning, propagating to any continuation run. Direct-coordinate decks
    with the same negative scale parse correctly. NO scale<0 + Cartesian
    file is read correctly today, so rejecting cannot break a working
    input; the message names the two equivalent rewrites.
    """

    try:
        with open(poscar_path, encoding="utf-8", errors="surrogateescape") as handle:
            lines = _cleaned_poscar_lines(handle.read())
        scale = float(lines[1].split()[0])
    except (OSError, ValueError, IndexError):
        return  # unreadable/malformed enough for the parser to report itself
    if scale >= 0:
        return
    _, counts, index = _scan_poscar_species_counts(lines)
    if not counts or index >= len(lines):
        return
    tokens = lines[index].split()
    if not tokens:
        return
    mode = tokens[0][:1]
    if mode in "sS":
        if index + 1 >= len(lines):
            return
        next_tokens = lines[index + 1].split()
        if not next_tokens:
            return
        mode = next_tokens[0][:1]
    if mode in "cCkK":
        raise ValueError(
            "POSCAR combines a negative scale factor (VASP reads it as the "
            "target cell volume) with Cartesian coordinates; the parser "
            "scales the lattice by the derived factor but multiplies the "
            "Cartesian positions by the raw negative number, silently "
            "producing a different structure. Use Direct coordinates or a "
            "positive scale factor instead."
        )


def _reject_malformed_selective_dynamics(selective_dynamics) -> None:
    """Reject a selective-dynamics mask that is not exactly three flags.

    A coordinate line under ``Selective dynamics`` carrying fewer than three
    T/F tokens (a dropped column while hand-editing) parses in pymatgen as a
    short mask, and ``AseAtomsAdaptor`` recognizes only the eight exact
    3-element masks -- anything else silently becomes NO constraint, so the
    atom the user froze relaxes freely with exit 0 and a CONTCAR whose
    Selective dynamics block is gone. Real VASP's list-directed read would
    consume tokens from the next line instead; neither behavior is what the
    file says, so the mask is judged at input time.
    """

    if not selective_dynamics:
        return
    for site_index, mask in enumerate(selective_dynamics):
        flags = tuple(mask)
        if len(flags) != 3:
            raise ValueError(
                f"POSCAR selective-dynamics line {site_index + 1} carries "
                f"{len(flags)} T/F flags; VASP expects exactly three per "
                "coordinate line, and a short mask would silently drop the "
                "constraint entirely."
            )


def read_structure(poscar_path: str, potcar_path: str | None = None):
    """Read POSCAR and reconcile species with POTCAR if necessary."""

    _require_regular_input_file(poscar_path, "POSCAR")
    comment = _read_vasp_comment(poscar_path)
    _reject_absurd_poscar_ion_counts(poscar_path)
    _reject_mismatched_species_counts(poscar_path)
    _reject_ambiguous_selective_dynamics_tokens(poscar_path)
    _reject_negative_scale_cartesian_poscar(poscar_path)
    # check_for_potcar=False: real pymatgen otherwise globs *POTCAR* SIBLINGS
    # of the POSCAR and passes their symbols as default_names, silently
    # RELABELLING the species a VASP-5 POSCAR declares -- a leftover
    # POTCAR_Cu (no exact POTCAR present) made a declared-Si2 deck compute
    # Cu2, exit 0, 4.6 eV wrong, and the mirror shape spuriously rejected a
    # deck real VASP runs. Real VASP reads only the exact file POTCAR, and
    # read_structure's own reconciliation below (keyed on the exact
    # <workdir>/POTCAR) already covers every legitimate flow: VASP-5 match,
    # VASP-5 differ (warn + POTCAR order), VASP-4 + POTCAR (species from
    # POTCAR), VASP-4 without POTCAR (reject). The kwarg also removes the
    # parser's sibling-open side effect entirely, avoiding a FIFO hazard.
    try:
        poscar = Poscar.from_file(poscar_path, check_for_potcar=False)
    except TypeError:
        # A stub/legacy Poscar without the kwarg (it then has no sibling
        # glob either, so the behavior is already the wanted one).
        poscar = Poscar.from_file(poscar_path)
    _reject_malformed_selective_dynamics(getattr(poscar, "selective_dynamics", None))
    structure = poscar.structure
    declares_species = (
        _poscar_declares_species(poscar_path) and bool(poscar.site_symbols)
    ) or _poscar_declares_trailing_species(poscar_path)
    if potcar_path:
        _reject_broken_input_link(potcar_path, "POTCAR")
    if potcar_path and os.path.exists(potcar_path):
        _require_regular_input_file(potcar_path, "POTCAR")
        try:
            potcar = Potcar.from_file(potcar_path)
            potcar_symbols = getattr(potcar, "symbols", [])
        except Exception:
            potcar_symbols = []
        normalized_potcar_symbols = _normalize_species_labels(potcar_symbols)
        if normalized_potcar_symbols:
            if declares_species and len(poscar.site_symbols) == len(normalized_potcar_symbols):
                normalized_poscar_symbols = _normalize_species_labels(poscar.site_symbols)
                if normalized_poscar_symbols != normalized_potcar_symbols:
                    print(
                        "Warning: species in POSCAR and POTCAR differ. "
                        f"Using POTCAR order: {normalized_potcar_symbols}"
                    )
                    relabelled = _apply_species_from_potcar(
                        poscar, structure, normalized_potcar_symbols
                    )
                    # The POSCAR names the species itself here, so a refinement
                    # that cannot be applied leaves a structure that is still
                    # labelled with real elements.
                    if relabelled is not None:
                        structure = relabelled
                elif list(poscar.site_symbols) != normalized_potcar_symbols:
                    relabelled = _apply_species_from_potcar(
                        poscar, structure, normalized_potcar_symbols
                    )
                    if relabelled is not None:
                        structure = relabelled
            elif not declares_species:
                # VASP 4 layout: the POTCAR is the only source of species, exactly
                # as real VASP treats it.
                relabelled = _apply_species_from_potcar(
                    poscar, structure, normalized_potcar_symbols
                )
                if relabelled is None:
                    # Nothing to fall back on: the unchanged structure is
                    # pymatgen's fabricated ['H', ...]. Real VASP requires one
                    # POTCAR entry per ion group, so this POSCAR/POTCAR pair is
                    # simply inconsistent -- and the pair is only ever seen this
                    # way in a NEB image directory, where pymatgen's implicit
                    # same-directory POTCAR lookup cannot reach the band's
                    # POTCAR one level up. Reporting it beats computing the whole
                    # band as hydrogen and exiting 0.
                    ion_counts = [
                        int(count) for count in (getattr(poscar, "natoms", None) or [])
                    ]
                    raise ValueError(
                        "POSCAR has no species names (VASP 4 format) and the "
                        f"species {normalized_potcar_symbols} read from "
                        f"{potcar_path} cannot be matched to its "
                        f"{len(ion_counts)} ion group(s) {ion_counts}, so the "
                        "elements cannot be determined. Add the species line to "
                        "the POSCAR or provide a POTCAR with one entry per ion "
                        "group."
                    )
                structure = relabelled
        elif not declares_species:
            # A POTCAR that exists but yields no usable symbols (unreadable, or
            # rejected by pymatgen's validation) is not a species source either.
            # Without this, the fabricated ['H', ...] names survived exactly as they
            # did with no POTCAR at all -- the hole this check exists to close.
            raise ValueError(
                f"POSCAR has no species names (VASP 4 format) and no species could "
                f"be read from {potcar_path}, so the elements cannot be determined. "
                "Add the species line to the POSCAR or provide a readable POTCAR."
            )
    elif not declares_species:
        raise ValueError(
            "POSCAR has no species names (VASP 4 format) and no POTCAR was "
            "provided, so the elements cannot be determined. Add the species line "
            "to the POSCAR or provide a POTCAR."
        )
    _store_vasp_comment_on_structure(structure, comment)
    return structure
