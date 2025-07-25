from pathlib import Path
import os, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unittest.mock import patch
from ase.calculators.emt import EMT
import vp_modoki

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

    def fake_run_relaxation(atoms, calculator, steps, fmax):
        called["steps"] = steps
        called["fmax"] = fmax
        return 0.0

    with patch("vp_modoki.get_calculator", return_value=EMT()):
        with patch("vp_modoki.run_relaxation", side_effect=fake_run_relaxation):
            with patch.object(sys, "argv", ["vp_modoki.py", "--dir", str(tmp_path)]):
                vp_modoki.main()

    assert called["steps"] == 100
    assert abs(called["fmax"] - 0.01) < 1e-6


def test_all_potentials_give_same_relax(tmp_path: Path):
    (tmp_path / "POSCAR").write_text(POSCAR_CONTENT)
    (tmp_path / "INCAR").write_text(INCAR_CONTENT)

    results = []

    def fake_run_relaxation(atoms, calculator, steps, fmax):
        # store a copy of positions for comparison
        results.append(atoms.get_positions().copy())
        return 0.0

    potentials = ["CHGNET", "MATGL", "MACE", "MATTERSIM"]
    for pot in potentials:
        (tmp_path / "BCAR").write_text(f"NNP={pot}\n")
        with patch("vp_modoki.get_calculator", return_value=EMT()):
            with patch("vp_modoki.run_relaxation", side_effect=fake_run_relaxation):
                with patch.object(sys, "argv", ["vp_modoki.py", "--dir", str(tmp_path)]):
                    vp_modoki.main()

    for r in results[1:]:
        assert ((results[0] - r) ** 2).sum() < 1e-12
