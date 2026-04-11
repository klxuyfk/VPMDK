"""Single-point execution flow."""

from __future__ import annotations

import sys


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
    atoms.calc = _root()._resolve_calculator(calculator)
    recorder = _root()._initialize_vasp_compat_outputs(
        atoms,
        ibrion=-1,
        isif=isif,
        write_oszicar_pseudo_scf=oszicar_pseudo_scf,
        neb_mode=neb_mode,
        neb_prev_positions=neb_prev_positions,
        neb_next_positions=neb_next_positions,
    )
    energy = atoms.get_potential_energy()
    delta = 0.0
    kinetic_energy = 0.0
    temperature = 0.0
    try:
        kinetic_energy = float(atoms.get_kinetic_energy())
    except Exception:
        kinetic_energy = 0.0
    try:
        temperature = float(atoms.get_temperature())
    except Exception:
        temperature = 0.0
    _root()._record_vasp_compat_step(
        recorder,
        atoms,
        step_index=1,
        potential_energy=energy,
        total_energy=energy + kinetic_energy,
        kinetic_energy=kinetic_energy,
        temperature=temperature,
    )
    _root()._write_vasprun_xml(recorder, atoms)
    _root()._append_outcar_footer(recorder)
    _root().write("CONTCAR", atoms, direct=True)
    print(
        f"{1:4d} F= {_root()._format_energy_value(energy)} "
        f"E0= {_root()._format_energy_value(energy)}  d E ={_root()._format_energy_value(delta)}"
    )
    return energy
