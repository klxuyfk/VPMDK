"""vpmdk: Run machine-learning potentials using VASP style inputs.

The utility consumes VASP-style inputs (POSCAR, INCAR, POTCAR, BCAR) and
executes single-point, relaxation, or molecular dynamics runs with the selected
neural-network potential.  Multiple ASE calculators are supported (CHGNet,
M3GNet/MatGL, MACE, MatterSim) and the expected VASP outputs such as CONTCAR
and OUTCAR-style energy logs are produced.
"""

import argparse
import os
import sys
from typing import Dict
import csv

from pymatgen.io.vasp import Incar, Poscar, Potcar
from pymatgen.io.ase import AseAtomsAdaptor

try:
    from chgnet.model import CHGNetCalculator
except Exception:  # pragma: no cover - optional dependency
    CHGNetCalculator = None  # type: ignore

try:
    from matgl.ext.ase import M3GNetCalculator
except Exception:  # pragma: no cover - optional dependency
    M3GNetCalculator = None  # type: ignore

try:
    from mace.calculators import MACECalculator
except Exception:  # pragma: no cover - optional dependency
    MACECalculator = None  # type: ignore

try:
    from mattersim.forcefield import MatterSimCalculator
except Exception:  # pragma: no cover - optional dependency
    MatterSimCalculator = None  # type: ignore

from ase import units
from ase.io import write
from ase.optimize import BFGS
from ase.constraints import UnitCellFilter, StrainFilter
from ase.md.verlet import VelocityVerlet
from ase.md import velocitydistribution


def parse_key_value_file(path: str) -> Dict[str, str]:
    """Parse simple key=value style file."""
    data: Dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.split('#')[0].strip()
            if not line or '=' not in line:
                continue
            k, v = line.split('=', 1)
            data[k.strip().upper()] = v.strip()
    return data


def read_structure(poscar_path: str, potcar_path: str | None = None):
    """Read POSCAR and reconcile species with POTCAR if necessary."""
    poscar = Poscar.from_file(poscar_path)
    structure = poscar.structure
    if potcar_path and os.path.exists(potcar_path):
        try:
            potcar = Potcar.from_file(potcar_path)
            potcar_symbols = potcar.symbols
        except Exception:
            potcar_symbols = []
        if potcar_symbols:
            # check consistency
            if poscar.site_symbols and len(poscar.site_symbols) == len(potcar_symbols):
                if list(poscar.site_symbols) != potcar_symbols:
                    print(
                        "Warning: species in POSCAR and POTCAR differ. "
                        f"Using POTCAR order: {potcar_symbols}"
                    )
                    poscar.site_symbols = potcar_symbols
                    structure = poscar.structure
            elif not poscar.site_symbols:
                poscar.site_symbols = potcar_symbols
                structure = poscar.structure
    else:
        if not poscar.site_symbols:
            print("Warning: POSCAR has no species names and no POTCAR provided.")
    return structure


def get_calculator(bcar_tags: Dict[str, str]):
    """Return ASE calculator based on BCAR tags."""
    nnp = bcar_tags.get("NNP", "CHGNET").upper()
    if nnp == "CHGNET":
        if CHGNetCalculator is None:
            raise RuntimeError("CHGNetCalculator not available. Install chgnet.")
        model_path = bcar_tags.get("MODEL")
        if model_path and os.path.exists(model_path):
            return CHGNetCalculator(model_path)
        return CHGNetCalculator()
    if nnp == "MATGL":
        if M3GNetCalculator is None:
            raise RuntimeError(
                "M3GNetCalculator not available. Install matgl and dependencies."
            )
        model_path = bcar_tags.get("MODEL")
        if model_path and os.path.exists(model_path):
            return M3GNetCalculator(model_path)
        return M3GNetCalculator()
    if nnp == "MACE":
        if MACECalculator is None:
            raise RuntimeError(
                "MACECalculator not available. Install mace-torch and dependencies."
            )
        model_path = bcar_tags.get("MODEL")
        device = bcar_tags.get("DEVICE")
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        if model_path and os.path.exists(model_path):
            return MACECalculator(model_path, device=device)
        return MACECalculator(device=device)
    if nnp == "MATTERSIM":
        if MatterSimCalculator is None:
            raise RuntimeError(
                "MatterSimCalculator not available. Install mattersim and dependencies."
            )
        model_path = bcar_tags.get("MODEL")
        if model_path and os.path.exists(model_path):
            return MatterSimCalculator(model_path)
        return MatterSimCalculator()
    raise ValueError(f"Unsupported NNP type: {nnp}")


def run_single_point(atoms, calculator):
    atoms.calc = calculator
    e = atoms.get_potential_energy()
    print(f"  energy  without entropy=     {e:10.6f} eV")
    return e


def run_relaxation(
    atoms,
    calculator,
    steps: int,
    fmax: float,
    write_energy_csv: bool = False,
    isif: int = 2,
):
    atoms.calc = calculator
    energies = []
    relax_object = atoms
    if isif == 3:
        relax_object = UnitCellFilter(atoms)
    elif isif == 6:
        relax_object = StrainFilter(atoms)
    dyn = BFGS(relax_object, logfile="OUTCAR")
    if write_energy_csv:
        dyn.attach(lambda: energies.append(atoms.get_potential_energy()))
    dyn.run(fmax=fmax, steps=steps)
    target_atoms = getattr(relax_object, "atoms", atoms)
    target_atoms.wrap()
    write("CONTCAR", target_atoms, direct=True)
    if write_energy_csv:
        with open("energy.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for energy in energies:
                writer.writerow([energy])
    return target_atoms.get_potential_energy()


def run_md(atoms, calculator, steps: int, temperature: float, timestep: float):
    atoms.calc = calculator
    velocitydistribution.MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    dyn = VelocityVerlet(atoms, timestep * units.fs, logfile="OUTCAR")
    for i in range(steps):
        dyn.run(1)
        atoms.wrap()
        write("XDATCAR", atoms, direct=True, append=i > 0)
    atoms.wrap()
    write("CONTCAR", atoms, direct=True)
    return atoms.get_potential_energy()


def main():
    parser = argparse.ArgumentParser(description="Run NNP with VASP style inputs")
    parser.add_argument("--dir", default=".", help="Input directory")
    args = parser.parse_args()
    workdir = args.dir

    poscar_path = os.path.join(workdir, "POSCAR")
    incar_path = os.path.join(workdir, "INCAR")
    potcar_path = os.path.join(workdir, "POTCAR")
    bcar_path = os.path.join(workdir, "BCAR")

    for fname in ["KPOINTS", "WAVECAR", "CHGCAR"]:
        if os.path.exists(os.path.join(workdir, fname)):
            print(f"Note: {fname} detected but not used in NNP calculations.")

    if not os.path.exists(poscar_path):
        print("POSCAR not found.")
        sys.exit(1)

    structure = read_structure(poscar_path, potcar_path if os.path.exists(potcar_path) else None)
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.wrap()

    incar = Incar.from_file(incar_path) if os.path.exists(incar_path) else {}
    bcar = parse_key_value_file(bcar_path) if os.path.exists(bcar_path) else {}

    supported = {"ISIF", "IBRION", "NSW", "EDIFFG", "TEBEG", "POTIM"}
    for k in getattr(incar, "keys", lambda: [])():
        if k not in supported:
            print(f"INCAR tag {k} is not supported and will be ignored")

    calculator = get_calculator(bcar)
    write_energy_csv = str(bcar.get("WRITE_ENERGY_CSV", "0")).lower() in ("1", "true", "yes", "on")

    nsw = int(incar.get("NSW", 0)) if hasattr(incar, "get") else 0
    ibrion = int(incar.get("IBRION", -1)) if hasattr(incar, "get") else -1
    ediffg = float(incar.get("EDIFFG", -0.02)) if hasattr(incar, "get") else -0.02
    requested_isif = int(incar.get("ISIF", 2)) if hasattr(incar, "get") else 2

    fallback_targets = {4: 3, 5: 3, 7: 6, 8: 2}
    fallback_messages = {
        4: (
            "Warning: ISIF=4 requires coupled stress constraints; "
            "falling back to combined ionic and cell relaxation (ISIF=3)."
        ),
        5: (
            "Warning: ISIF=5 fixes the cell shape while changing the volume; "
            "falling back to combined ionic and cell relaxation (ISIF=3)."
        ),
        7: (
            "Warning: ISIF=7 is not supported; falling back to cell-only relaxation (ISIF=6)."
        ),
        8: (
            "Warning: ISIF=8 enforces constant volume; "
            "falling back to ionic relaxation (ISIF=2)."
        ),
    }

    if requested_isif in fallback_targets:
        print(fallback_messages[requested_isif])
        effective_isif = fallback_targets[requested_isif]
    else:
        effective_isif = requested_isif

    if effective_isif in (0, 1, 2):
        isif = 2
    elif effective_isif in (3, 6):
        isif = effective_isif
    else:
        if requested_isif not in fallback_targets:
            print(
                "Warning: ISIF="
                f"{requested_isif} is not fully supported; defaulting to ISIF=2 behavior."
            )
        isif = 2

    if nsw <= 0:
        run_single_point(atoms, calculator)
    elif ibrion == 0:
        tebeg = float(incar.get("TEBEG", 300))
        potim = float(incar.get("POTIM", 2))
        run_md(atoms, calculator, nsw, tebeg, potim)
    else:
        run_relaxation(atoms, calculator, nsw, abs(ediffg), write_energy_csv, isif=isif)

    print("Calculation completed.")


if __name__ == "__main__":
    main()
