"""INCAR-derived execution settings and related parsing helpers."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict

from pymatgen.io.vasp import Incar


KBAR_TO_EV_PER_A3 = 0.1 / 160.21766208


@dataclass(frozen=True)
class IncarSettings:
    """Container for the INCAR parameters that drive the simulation."""

    nsw: int = 0
    ibrion: int = -1
    ediffg: float = -0.02
    isif: int = 2
    stress_isif: int = 2
    pstress: float | None = None
    tebeg: float = 300.0
    teend: float = 300.0
    potim: float = 2.0
    mdalgo: int = 0
    smass: float | None = None
    thermostat_params: Dict[str, float] = field(default_factory=dict)

    @property
    def energy_tolerance(self) -> float | None:
        """Energy convergence threshold in eV when EDIFFG>0."""

        return self.ediffg if self.ediffg > 0 else None

    @property
    def force_limit(self) -> float:
        """Return ASE ``fmax`` argument derived from EDIFFG semantics."""

        if self.ediffg > 0:
            return -abs(self.ediffg)
        if self.ediffg < 0:
            return abs(self.ediffg)
        return 0.05


SUPPORTED_INCAR_TAGS = {
    "ISIF",
    "IBRION",
    "NSW",
    "EDIFFG",
    "PSTRESS",
    "TEBEG",
    "TEEND",
    "POTIM",
    "MDALGO",
    "SMASS",
    "ANDERSEN_PROB",
    "LANGEVIN_GAMMA",
    "CSVR_PERIOD",
    "NHC_NCHAINS",
    "MAGMOM",
    "IMAGES",
    "LCLIMB",
    "SPRING",
}

SUPPORTED_ISIF_VALUES = {0, 1, 2, 3, 4, 5, 6, 7, 8}

_NUMERIC_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def _load_incar(path: str):
    """Return ``Incar`` contents when available, falling back to ``{}``."""

    if os.path.exists(path):
        return Incar.from_file(path)
    return {}


def _warn_for_unsupported_incar_tags(incar, *, pseudo_scf_enabled: bool = False) -> None:
    """Emit warnings for INCAR options that are silently ignored."""

    import sys

    root = sys.modules["vpmdk_core"]
    supported_tags = SUPPORTED_INCAR_TAGS
    for key in getattr(incar, "keys", lambda: [])():
        if key in supported_tags:
            continue
        if pseudo_scf_enabled and key in root._PSEUDO_SCF_INCAR_TAGS:
            print(
                f"Warning: INCAR tag {key} does not affect the run and is used only "
                "for pseudo-SCF compatibility output"
            )
            continue
        if key not in supported_tags:
            print(f"INCAR tag {key} is not supported and will be ignored")


def _is_truthy_flag(value) -> bool:
    """Return whether ``value`` expresses a truthy INCAR-style flag."""

    if value is None:
        return False
    token = str(value).strip().strip(".").upper()
    return token in {"T", "TRUE", "1", "YES", "Y"}


def _is_neb_like_incar(incar) -> bool:
    """Detect whether INCAR appears to describe a NEB-style calculation."""

    if not hasattr(incar, "get"):
        return False

    images_value = incar.get("IMAGES")
    if images_value is not None:
        match = _NUMERIC_RE.search(str(images_value))
        if match is not None:
            try:
                if int(float(match.group(0))) > 0:
                    return True
            except ValueError:
                pass

    if "SPRING" in getattr(incar, "keys", lambda: [])():
        return True

    if _is_truthy_flag(incar.get("LCLIMB")):
        return True

    return False


def _parse_neb_image_count(incar) -> int | None:
    """Return ``IMAGES`` value when parseable and non-negative."""

    if not hasattr(incar, "get"):
        return None
    raw_value = incar.get("IMAGES")
    if raw_value is None:
        return None
    parsed = _parse_optional_float(raw_value, key="IMAGES")
    if parsed is None:
        return None
    count = int(parsed)
    if count < 0:
        print(f"Warning: IMAGES={raw_value} is invalid; ignoring NEB image count hint.")
        return None
    return count


def _parse_optional_float(value, *, key: str):
    """Attempt to convert ``value`` to ``float`` with warning on failure."""

    if value is None:
        return None
    candidate = value
    if isinstance(value, str):
        match = _NUMERIC_RE.search(value)
        if match is not None:
            candidate = match.group(0)
        else:
            candidate = value.strip()
    try:
        return float(candidate)
    except (TypeError, ValueError):
        print(f"Warning: Unable to parse {key}; ignoring value {value}")
        return None


def _normalize_isif(requested: int) -> int:
    """Map request to supported ISIF behaviour while preserving warnings."""

    if requested not in SUPPORTED_ISIF_VALUES:
        print(
            "Warning: ISIF="
            f"{requested} is not fully supported; defaulting to ISIF=2 behavior."
        )
        return 2
    if requested in (0, 1, 2):
        return 2
    return requested


def _extract_thermostat_parameters(incar) -> Dict[str, float]:
    """Collect thermostat keywords from ``incar`` with validation."""

    params: Dict[str, float] = {}
    keys = ("ANDERSEN_PROB", "LANGEVIN_GAMMA", "CSVR_PERIOD", "NHC_NCHAINS")
    for key in keys:
        if hasattr(incar, "__contains__") and key in incar:
            value = incar[key]
            if key == "NHC_NCHAINS":
                try:
                    coerced = int(float(value))
                except (TypeError, ValueError):
                    print(f"Warning: Unable to parse {key}; ignoring value {value}")
                    parsed = None
                else:
                    parsed = _parse_optional_float(coerced, key=key)
            else:
                parsed = _parse_optional_float(value, key=key)
            if parsed is not None:
                params[key] = float(parsed)
    return params


def _load_incar_settings(incar) -> IncarSettings:
    """Translate INCAR dictionary-like object into :class:`IncarSettings`."""

    if not hasattr(incar, "get"):
        return IncarSettings()

    nsw = int(float(incar.get("NSW", 0)))
    ibrion = int(float(incar.get("IBRION", -1)))
    ediffg = float(incar.get("EDIFFG", -0.02))
    pstress = None
    if "PSTRESS" in incar:
        pstress = _parse_optional_float(incar.get("PSTRESS", 0.0), key="PSTRESS")
    tebeg_default = 300.0
    tebeg_value = incar.get("TEBEG", tebeg_default)
    parsed_tebeg = _parse_optional_float(tebeg_value, key="TEBEG")
    tebeg = parsed_tebeg if parsed_tebeg is not None else tebeg_default

    teend_value = incar.get("TEEND", tebeg)
    parsed_teend = _parse_optional_float(teend_value, key="TEEND")
    teend = parsed_teend if parsed_teend is not None else tebeg
    potim = float(incar.get("POTIM", 2.0))
    smass = (
        _parse_optional_float(incar.get("SMASS"), key="SMASS")
        if "SMASS" in incar
        else None
    )
    mdalgo = int(float(incar.get("MDALGO", 0)))
    if mdalgo == 0 and smass is not None:
        if smass < 0:
            mdalgo = 3
        elif smass > 0:
            mdalgo = 2
    thermostat_params = _extract_thermostat_parameters(incar)
    default_isif = 0 if ibrion == 0 else 2
    requested_isif = int(float(incar.get("ISIF", default_isif)))
    normalized_isif = _normalize_isif(requested_isif)
    stress_isif = (
        requested_isif if requested_isif in SUPPORTED_ISIF_VALUES else normalized_isif
    )

    return IncarSettings(
        nsw=nsw,
        ibrion=ibrion,
        ediffg=ediffg,
        isif=normalized_isif,
        stress_isif=stress_isif,
        pstress=pstress,
        tebeg=tebeg,
        teend=teend,
        potim=potim,
        mdalgo=mdalgo,
        smass=smass,
        thermostat_params=thermostat_params,
    )


def _should_write_energy_csv(bcar_tags: Dict[str, str]) -> bool:
    """Return ``True`` when BCAR requests CSV output of ionic energies."""

    value = str(bcar_tags.get("WRITE_ENERGY_CSV", "0")).lower()
    return value in {"1", "true", "yes", "on"}


def _should_write_lammps_trajectory(bcar_tags: Dict[str, str]) -> bool:
    """Return ``True`` when BCAR requests LAMMPS-style trajectory output."""

    value = str(bcar_tags.get("WRITE_LAMMPS_TRAJ", "0")).lower()
    return value in {"1", "true", "yes", "on"}


def _should_write_pseudo_scf(bcar_tags: Dict[str, str]) -> bool:
    """Return ``True`` when BCAR requests pseudo electronic-step compatibility output."""

    raw = bcar_tags.get("WRITE_PSEUDO_SCF", bcar_tags.get("WRITE_OSZICAR_PSEUDO_SCF", "0"))
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _should_write_oszicar_pseudo_scf(bcar_tags: Dict[str, str]) -> bool:
    """Backward-compatible alias for :func:`_should_write_pseudo_scf`."""

    return _should_write_pseudo_scf(bcar_tags)


def _get_lammps_trajectory_interval(bcar_tags: Dict[str, str]) -> int:
    """Return the LAMMPS trajectory write interval requested in BCAR."""

    import sys

    raw = bcar_tags.get("LAMMPS_TRAJ_INTERVAL", "1")
    return_value = sys.modules["vpmdk_core"]._coerce_int_tag(raw, "LAMMPS_TRAJ_INTERVAL")
    if return_value <= 0:
        raise ValueError("LAMMPS_TRAJ_INTERVAL must be at least 1")
    return return_value
