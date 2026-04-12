from __future__ import annotations

from pathlib import Path
import sys

from ase.io import read, write


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

    result = vpmdk.relax(
        atoms,
        mlp="CHGNET",
        device="cpu",
        steps=5,
        fmax=0.05,
        relax_cell=False,
    )

    output_path = here / "relaxed.vasp"
    write(output_path, result.atoms, format="vasp", direct=True)

    print(f"Final energy: {result.potential_energy:.8f} eV")
    print(f"Recorded ionic steps: {len(result.steps)}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
