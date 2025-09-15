from pathlib import Path
import os, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unittest.mock import patch
from ase.calculators.emt import EMT
from pymatgen.io.vasp import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
import vpmdk

POSCAR_CONTENT = """Si2
1.0
        3.8669745922         0.0000000000         0.0000000000
        1.9334872961         3.3488982326         0.0000000000
        1.9334872961         1.1162994109         3.1573715331
   Si
    2
Direct
     0.749999979         0.749999983         0.749999997
     0.500000007         0.499999989         0.499999998
"""

INCAR_CONTENT = """Global Parameters
ISTART =  1            (Read existing wavefunction, if there)
ISPIN  =  1            (Non-Spin polarised DFT)
LREAL  = .FALSE.       (Projection operators: automatic)
LWAVE  = .TRUE.        (Write WAVECAR or not)
LCHARG = .TRUE.        (Write CHGCAR or not)
ADDGRID= .TRUE.        (Increase grid, helps GGA convergence)
Electronic Relaxation
ISMEAR =  0            (Gaussian smearing, metals:1)
SIGMA  =  0.05         (Smearing value in eV, metals:0.2)
NELM   =  90           (Max electronic SCF steps)
NELMIN =  6            (Min electronic SCF steps)
EDIFF  =  1E-08        (SCF energy convergence, in eV)
Ionic Relaxation
NSW    =  100          (Max ionic steps)
IBRION =  2            (Algorithm: 0-MD, 1-Quasi-New, 2-CG)
ISIF   =  3            (Stress/relaxation: 2-Ions, 3-Shape/Ions/V, 4-Shape/Ions)
EDIFFG = -1E-02        (Ionic convergence, eV/AA)
"""

BCAR_CONTENT = "NNP=CHGNET\n"

def test_relaxation_runs(tmp_path: Path):
    (tmp_path / "POSCAR").write_text(POSCAR_CONTENT)
    (tmp_path / "INCAR").write_text(INCAR_CONTENT)
    (tmp_path / "BCAR").write_text(BCAR_CONTENT)

    called = {}

    def fake_run_relaxation(atoms, calculator, steps, fmax, write_energy_csv=False):
        called["steps"] = steps
        called["fmax"] = fmax
        called["write"] = write_energy_csv
        return 0.0

    with patch("vpmdk.get_calculator", return_value=EMT()):
        with patch("vpmdk.run_relaxation", side_effect=fake_run_relaxation):
            with patch.object(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)]):
                vpmdk.main()

    assert called["steps"] == 100
    assert abs(called["fmax"] - 0.01) < 1e-6
    assert called["write"] is False


def test_all_potentials_give_same_relax(tmp_path: Path):
    (tmp_path / "POSCAR").write_text(POSCAR_CONTENT)
    (tmp_path / "INCAR").write_text(INCAR_CONTENT)

    results = []

    def fake_run_relaxation(atoms, calculator, steps, fmax, write_energy_csv=False):
        # store a copy of positions for comparison
        results.append(atoms.get_positions().copy())
        return 0.0

    potentials = ["CHGNET", "MATGL", "MACE", "MATTERSIM"]
    for pot in potentials:
        (tmp_path / "BCAR").write_text(f"NNP={pot}\n")
        with patch("vpmdk.get_calculator", return_value=EMT()):
            with patch("vpmdk.run_relaxation", side_effect=fake_run_relaxation):
                with patch.object(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)]):
                    vpmdk.main()

    for r in results[1:]:
        assert ((results[0] - r) ** 2).sum() < 1e-12


def test_energy_csv_flag(tmp_path: Path):
    (tmp_path / "POSCAR").write_text(POSCAR_CONTENT)
    (tmp_path / "INCAR").write_text(INCAR_CONTENT)
    (tmp_path / "BCAR").write_text(BCAR_CONTENT + "WRITE_ENERGY_CSV=1\n")

    called = {}

    def fake_run_relaxation(atoms, calculator, steps, fmax, write_energy_csv=False):
        called["write"] = write_energy_csv
        return 0.0

    with patch("vpmdk.get_calculator", return_value=EMT()):
        with patch("vpmdk.run_relaxation", side_effect=fake_run_relaxation):
            with patch.object(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)]):
                vpmdk.main()

    assert called["write"] is True


def test_fractional_coords_wrapped(tmp_path: Path):
    poscar = """Si
1.0
 1 0 0
 0 1 0
 0 0 1
 Si
 1
Direct
 1.1 0.2 0.3
"""
    (tmp_path / "POSCAR").write_text(poscar)
    (tmp_path / "INCAR").write_text("NSW=1\nIBRION=2\n")
    (tmp_path / "BCAR").write_text(BCAR_CONTENT)

    seen = {}

    def fake_run_relaxation(atoms, calculator, steps, fmax, write_energy_csv=False):
        seen["scaled"] = atoms.get_scaled_positions(wrap=False).copy()
        return 0.0

    with patch("vpmdk.get_calculator", return_value=EMT()):
        with patch("vpmdk.run_relaxation", side_effect=fake_run_relaxation):
            with patch.object(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)]):
                vpmdk.main()

    assert ((seen["scaled"] >= 0) & (seen["scaled"] < 1)).all()


def test_run_relaxation_wraps_on_write(tmp_path: Path):
    poscar = """Cu
1.0
 1 0 0
 0 1 0
 0 0 1
 Cu
 1
Direct
 1.2 0.3 0.4
"""
    structure = Poscar.from_str(poscar).structure
    atoms = AseAtomsAdaptor.get_atoms(structure)

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with patch("ase.optimize.bfgs.BFGS.run", lambda self, *a, **k: None):
            vpmdk.run_relaxation(atoms, EMT(), steps=0, fmax=0.01)
    finally:
        os.chdir(cwd)

    contcar = (tmp_path / "CONTCAR").read_text().splitlines()
    start = contcar.index("Direct") + 1
    coords = [list(map(float, line.split())) for line in contcar[start:start + len(atoms)]]
    for c in coords:
        assert all(0 <= x < 1 for x in c)
