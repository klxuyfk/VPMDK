"""CLI entrypoint for VPMDK."""

from __future__ import annotations

import argparse
import os
import sys

from .compat import vasp as vasp_compat


def _root():
    return sys.modules["vpmdk_core"]


def main():
    parser = argparse.ArgumentParser(description="Run MLP with VASP style inputs")
    parser.add_argument("--dir", default=".", help="Input directory")
    args = parser.parse_args()
    workdir = args.dir
    workdir_abs = os.path.abspath(workdir)
    caller_cwd = os.getcwd()

    poscar_path = os.path.join(workdir, "POSCAR")
    incar_path = os.path.join(workdir, "INCAR")
    potcar_path = os.path.join(workdir, "POTCAR")
    kpoints_path = os.path.join(workdir, "KPOINTS")
    bcar_path = os.path.join(workdir, "BCAR")

    root = _root()
    for fname in ["KPOINTS", "WAVECAR", "CHGCAR"]:
        if os.path.exists(os.path.join(workdir, fname)):
            print(f"Note: {fname} detected but not used in MLP calculations.")

    incar = root._load_incar(incar_path)
    bcar = root.parse_key_value_file(bcar_path) if os.path.exists(bcar_path) else {}

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
    settings = root._load_incar_settings(incar)
    neb_mode = root._is_neb_like_incar(incar)
    lammps_traj_interval = root._get_lammps_trajectory_interval(bcar) if write_lammps_traj else 1
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
                sys.exit(1)

            structure = root.read_structure(poscar_path, potcar_for_structure)
            atoms = root.AseAtomsAdaptor.get_atoms(structure)
            atoms.wrap()
            root._apply_initial_magnetization(atoms, incar)
            with root._working_directory(workdir_abs):
                calculator = root._build_calculator_from_tags(bcar, structure=structure)

            if settings.nsw <= 0 or settings.ibrion < 0:
                with root._working_directory(workdir_abs):
                    root.run_single_point(
                        atoms,
                        calculator,
                        isif=settings.stress_isif,
                        oszicar_pseudo_scf=write_pseudo_scf,
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
                    )
            else:
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
                        **root._charge_density_options_from_bcar(bcar),
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
