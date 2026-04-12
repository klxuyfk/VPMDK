from __future__ import annotations

from pathlib import Path
import sys

from ase.io import read


def _bootstrap_local_checkout() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_path = repo_root / "src"
    for candidate in (repo_root, src_path):
        text = str(candidate)
        if text not in sys.path:
            sys.path.insert(0, text)


_bootstrap_local_checkout()

import vpmdk


def main() -> None:
    here = Path(__file__).resolve().parent
    atoms = read(here / "POSCAR")

    result = vpmdk.single_point(
        atoms,
        mlp="CHGNET",
        device="cpu",
    )

    print(f"Energy: {result.potential_energy:.8f} eV")
    print("Forces (eV/Ang):")
    for force in result.forces:
        print(" ", " ".join(f"{value: .8f}" for value in force))

    if result.stress is not None:
        print("Stress (eV/Ang^3):")
        for row in result.stress:
            print(" ", " ".join(f"{value: .8f}" for value in row))


if __name__ == "__main__":
    main()
