"""VASP compatibility helpers kept separate from the main ASE-oriented API."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any


sys.modules.setdefault("vpmdk.compat.vasp", sys.modules[__name__])


@dataclass(frozen=True)
class VaspCompatConfig:
    """Compatibility-output settings used by the legacy wrappers and CLI."""

    enabled: bool = True
    write_pseudo_scf: bool = False
    write_contcar: bool = True
    write_xdatcar: bool = False
    write_lammps_traj: bool = False
    lammps_traj_interval: int = 1
    lammps_traj_path: str = "lammps.lammpstrj"
    neb_mode: bool = False
    neb_prev_positions: Any = None
    neb_next_positions: Any = None


@dataclass(frozen=True)
class VaspSinglePointConfig:
    """Advanced VASP metadata attached to a single-point library call."""

    isif: int | None = None
    ibrion: int = -1


@dataclass(frozen=True)
class VaspRelaxConfig:
    """Advanced VASP metadata attached to a relaxation library call."""

    isif: int | None = None
    stress_isif: int | None = None
    ibrion: int = 2


@dataclass(frozen=True)
class VaspMDConfig:
    """Advanced VASP metadata attached to an MD library call."""

    isif: int | None = 0
    mdalgo: int | None = None


def determine_vasp_fft_grid(reference, incar):
    """Return the VASP fine FFT grid using the core charge-density helper."""

    from ..charge_density import determine_vasp_fft_grid as _determine_vasp_fft_grid

    return _determine_vasp_fft_grid(reference, incar)


def write_chgcar(path, atoms, density, spin_density=None):
    """Write a VASP-like CHGCAR using the core charge-density helper."""

    from ..charge_density import write_chgcar as _write_chgcar

    return _write_chgcar(path, atoms, density, spin_density=spin_density)
