from __future__ import annotations

import os
from pathlib import Path
import sys

from ase.io import read
from pymatgen.io.vasp import Incar

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-vpmdk")

import vpmdk
import vpmdk.compat.vasp as vasp_compat


def main() -> None:
    atoms = read(HERE / "POSCAR")
    incar = Incar.from_file(HERE / "INCAR")

    result = vpmdk.predict_charge_density(
        atoms,
        incar=incar,
        reference=atoms,
        backend="CHARGE3NET",
    )

    output_path = HERE / "api_CHGCAR"
    vasp_compat.write_chgcar(
        output_path,
        atoms,
        result.density,
        spin_density=result.spin_density,
    )

    print(f"backend={result.backend}")
    print(f"grid_shape={result.grid_shape}")
    print(f"density_shape={result.density.shape}")
    print(f"wrote={output_path}")


if __name__ == "__main__":
    main()
