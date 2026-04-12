"""Single-point execution flow."""

from __future__ import annotations

import sys

from ..models import SinglePointConfig, VaspCompatConfig
from ..observers import PrintProgressObserver, VaspCompatObserver


def _root():
    return sys.modules["vpmdk_core"]


def run_single_point(
    atoms,
    calculator,
    *,
    isif: int | None = None,
    oszicar_pseudo_scf: bool = False,
    neb_mode: bool = False,
    neb_prev_positions=None,
    neb_next_positions=None,
):
    result = _root().single_point(
        atoms,
        calculator=calculator,
        config=SinglePointConfig(isif=isif),
        observer=[VaspCompatObserver(), PrintProgressObserver()],
        vasp_compat=VaspCompatConfig(
            enabled=True,
            write_pseudo_scf=oszicar_pseudo_scf,
            write_contcar=True,
            neb_mode=neb_mode,
            neb_prev_positions=neb_prev_positions,
            neb_next_positions=neb_next_positions,
        ),
    )
    return result.potential_energy
