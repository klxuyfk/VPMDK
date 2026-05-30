"""Helpers for reading BCAR/POSCAR/POTCAR data and preparing structures."""

from __future__ import annotations

import os
from typing import Dict, Iterable, List

from pymatgen.io.vasp import Poscar, Potcar

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


def parse_key_value_file(path: str) -> Dict[str, str]:
    """Parse simple key=value style file."""

    data: Dict[str, str] = {}
    with open(path) as f:
        for line in f:
            for comment in ("#", "!"):
                if comment in line:
                    line = line.split(comment, 1)[0]
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            data[k.strip().upper()] = v.strip()
    if "MLP" not in data and "NNP" in data:
        data["MLP"] = data["NNP"]
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
            except (TypeError, ValueError):
                continue
            nested = _parse_magmom_values(moment_str)
            if not nested:
                try:
                    nested = [float(moment_str)]
                except (TypeError, ValueError):
                    continue
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


def read_structure(poscar_path: str, potcar_path: str | None = None):
    """Read POSCAR and reconcile species with POTCAR if necessary."""

    comment = _read_vasp_comment(poscar_path)
    poscar = Poscar.from_file(poscar_path)
    structure = poscar.structure
    if potcar_path and os.path.exists(potcar_path):
        try:
            potcar = Potcar.from_file(potcar_path)
            potcar_symbols = getattr(potcar, "symbols", [])
        except Exception:
            potcar_symbols = []
        normalized_potcar_symbols = _normalize_species_labels(potcar_symbols)
        if normalized_potcar_symbols:
            if poscar.site_symbols and len(poscar.site_symbols) == len(normalized_potcar_symbols):
                normalized_poscar_symbols = _normalize_species_labels(poscar.site_symbols)
                if normalized_poscar_symbols != normalized_potcar_symbols:
                    print(
                        "Warning: species in POSCAR and POTCAR differ. "
                        f"Using POTCAR order: {normalized_potcar_symbols}"
                    )
                    poscar.site_symbols = normalized_potcar_symbols
                    structure = poscar.structure
                elif list(poscar.site_symbols) != normalized_potcar_symbols:
                    poscar.site_symbols = normalized_potcar_symbols
                    structure = poscar.structure
            elif not poscar.site_symbols:
                poscar.site_symbols = normalized_potcar_symbols
                structure = poscar.structure
    elif not poscar.site_symbols:
        print("Warning: POSCAR has no species names and no POTCAR provided.")
    _store_vasp_comment_on_structure(structure, comment)
    return structure
