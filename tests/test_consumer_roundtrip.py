"""End-to-end consumer round-trip harness.

Most vasprun/OUTCAR/CONTCAR assertions in this suite re-parse the files with
ElementTree or string matching -- i.e. they mirror the writer instead of
round-tripping through the reader on the other side. Several rounds of review
findings (R131 scstep, R133 selective dynamics, R135 NEB tangent) were
divergences between what VPMDK writes and what the CONSUMERS of those files
(ase.io and pymatgen) actually accept, invisible to writer-mirroring tests.

This module is the systematic net for that defect class: run the full
``run_workdir`` pipeline for each supported run mode and read every produced
artifact back through the real readers, asserting both parse success and
cross-artifact value consistency (vasprun final energy == OSZICAR E0 ==
OUTCAR energy; vasprun final structure == CONTCAR == XDATCAR last frame;
one ionic-step count across all artifacts).

The suite normally runs against conftest's pymatgen STUBS, which is exactly
how parser divergences kept escaping it -- so each scenario here executes in
a subprocess with ``VPMDK_TEST_REAL_PYMATGEN=1`` (the established escape
hatch) and the real libraries. The scenario logic lives in this same file
under ``__main__`` so the checked code and the launcher cannot drift apart.

Known, deliberately-excluded gap: OUTCAR is not readable by ``ase.io.read``
(it lacks the full POTCAR header ASE's parser requires); that is a recorded
deferred item, so OUTCAR is only checked textually here.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")

SCENARIOS = {
    # Single point: the minimal, most common request.
    "single-point": {"NSW": "0", "IBRION": "-1"},
    # Ionic relaxation, ions only. The structure is perturbed off symmetry
    # below so the forces are real and several steps actually run.
    "relax": {"NSW": "3", "IBRION": "2", "ISIF": "2", "EDIFFG": "-1E-10", "POTIM": "0.1"},
    # NVE MD (velocity Verlet).
    "md-nve": {"NSW": "3", "IBRION": "0", "MDALGO": "0", "POTIM": "1.0", "TEBEG": "300"},
    # Thermostatted MD (CSVR), the server-batch workhorse.
    "md-csvr": {
        "NSW": "3",
        "IBRION": "0",
        "MDALGO": "5",
        "POTIM": "1.0",
        "TEBEG": "300",
        "SMASS": "100",
    },
    # Selective dynamics through the full POSCAR->run->CONTCAR chain.
    "selective-dynamics": {
        "NSW": "2",
        "IBRION": "2",
        "ISIF": "2",
        "EDIFFG": "-1E-10",
        "POTIM": "0.1",
    },
}


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_artifacts_read_back_through_real_consumers(tmp_path: Path, scenario: str):
    env = {
        **os.environ,
        "VPMDK_TEST_REAL_PYMATGEN": "1",
        "PYTHONPATH": _SRC_DIR + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    completed = subprocess.run(
        [sys.executable, __file__, scenario, str(tmp_path)],
        capture_output=True,
        text=True,
        env=env,
    )
    if "ModuleNotFoundError: No module named 'pymatgen'" in completed.stderr:
        pytest.skip("real pymatgen is not installed")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "ROUNDTRIP OK" in completed.stdout, completed.stdout + completed.stderr


# --------------------------------------------------------------------------
# Everything below runs only in the subprocess, against the real libraries.
# --------------------------------------------------------------------------


def _prepare(workdir: Path, scenario: str) -> None:
    if scenario == "selective-dynamics":
        poscar = (
            "Cu4 selective\n"
            "1.0\n"
            "  3.6  0.0  0.0\n"
            "  0.0  3.6  0.0\n"
            "  0.0  0.0  3.6\n"
            "Cu\n"
            "4\n"
            "Selective dynamics\n"
            "Direct\n"
            "  0.02  0.01  0.03  T T F\n"
            "  0.00  0.50  0.50  T T T\n"
            "  0.50  0.00  0.50  T T T\n"
            "  0.50  0.50  0.00  F F F\n"
        )
    else:
        # Perturbed off the symmetric point so LJ forces are nonzero and a
        # relaxation genuinely moves (the unperturbed fixture sits at an
        # equilibrium where BFGS converges after one evaluation).
        poscar = (
            "Si2 fixture\n"
            "1.0\n"
            "  3.8669745922  0.0000000000  0.0000000000\n"
            "  1.9334872961  3.3488982326  0.0000000000\n"
            "  1.9334872961  1.1162994109  3.1573715331\n"
            "Si\n"
            "2\n"
            "Direct\n"
            "  0.78  0.73  0.76\n"
            "  0.50  0.50  0.50\n"
        )
    (workdir / "POSCAR").write_text(poscar)
    lines = [f"{key} = {value}" for key, value in SCENARIOS[scenario].items()]
    (workdir / "INCAR").write_text("\n".join(lines) + "\n")
    (workdir / "BCAR").write_text("MLP=CHGNET\nWRITE_ENERGY_CSV=0\n")


def _oszicar_e0_values(path: Path) -> list[float]:
    return [float(v) for v in re.findall(r"E0=\s*([-+.\dEe]+)", path.read_text())]


def _outcar_energies(path: Path) -> list[float]:
    return [
        float(v)
        for v in re.findall(r"energy\(sigma->0\)\s*=\s*([-+.\dEe]+)", path.read_text())
    ]


def _check_roundtrip(workdir: Path, scenario: str) -> None:
    import numpy as np
    import ase.io as ase_io
    from pymatgen.io.vasp.inputs import Poscar
    from pymatgen.io.vasp.outputs import Vasprun

    import vpmdk
    from ase.calculators.lj import LennardJones

    _prepare(workdir, scenario)
    natoms = 4 if scenario == "selective-dynamics" else 2
    vpmdk.run_workdir(
        str(workdir), calculator=LennardJones(sigma=2.0, epsilon=0.1, rc=6.0)
    )

    nsw = int(SCENARIOS[scenario]["NSW"])

    # --- vasprun.xml through ASE's reader ---
    images = ase_io.read(str(workdir / "vasprun.xml"), index=":")
    assert images, "vasprun.xml carries no ionic steps"
    steps = len(images)
    # MD runs exactly NSW steps. A relaxation records the INITIAL geometry as
    # its own step plus up to NSW optimizer moves (ASE counts moves, VASP
    # counts configurations -- so VPMDK can record NSW+1 where VASP records at
    # most NSW; a recorded, deliberate legacy behavior this harness pins
    # rather than judges), and it may converge earlier.
    if scenario.startswith("md"):
        assert steps == nsw, (steps, nsw)
    else:
        assert 1 <= steps <= max(1, nsw) + 1, (steps, nsw)
    final = images[-1]
    assert len(final) == natoms
    final_energy = final.get_potential_energy()
    assert np.isfinite(final_energy)
    forces = final.get_forces()
    assert forces.shape == (natoms, 3)
    assert np.all(np.isfinite(forces))

    # --- vasprun.xml through pymatgen's reader ---
    vasprun = Vasprun(
        str(workdir / "vasprun.xml"),
        parse_potcar_file=False,
        parse_dos=False,
        parse_eigen=False,
    )
    assert len(vasprun.ionic_steps) == steps, (len(vasprun.ionic_steps), steps)
    assert abs(float(vasprun.final_energy) - final_energy) < 1e-6

    # --- CONTCAR through both readers, consistent with vasprun's last frame ---
    contcar_ase = ase_io.read(str(workdir / "CONTCAR"))
    contcar_pmg = Poscar.from_file(str(workdir / "CONTCAR"), check_for_potcar=False)
    assert len(contcar_ase) == natoms
    assert len(contcar_pmg.structure) == natoms
    assert np.allclose(
        contcar_ase.get_scaled_positions(wrap=True),
        np.mod(contcar_pmg.structure.frac_coords, 1.0),
        atol=1e-6,
    ), "the two CONTCAR readers disagree"
    assert np.allclose(
        contcar_ase.get_scaled_positions(wrap=True),
        final.get_scaled_positions(wrap=True),
        atol=1e-5,
    ), "CONTCAR geometry != final vasprun geometry"

    # --- OSZICAR / OUTCAR agree with the machine-readable artifact ---
    e0 = _oszicar_e0_values(workdir / "OSZICAR")
    assert len(e0) == steps, (len(e0), steps)
    assert abs(e0[-1] - final_energy) < 1e-5
    outcar_energies = _outcar_energies(workdir / "OUTCAR")
    assert len(outcar_energies) == steps, (len(outcar_energies), steps)
    assert abs(outcar_energies[-1] - final_energy) < 1e-5

    # --- XDATCAR (when written) through ASE ---
    xdatcar = workdir / "XDATCAR"
    if xdatcar.exists():
        frames = ase_io.read(str(xdatcar), index=":")
        assert len(frames) == steps, (len(frames), steps)
        assert np.allclose(
            frames[-1].get_scaled_positions(wrap=True),
            final.get_scaled_positions(wrap=True),
            atol=1e-5,
        ), "XDATCAR last frame != final vasprun geometry"

    if scenario == "selective-dynamics":
        contcar_text = (workdir / "CONTCAR").read_text()
        assert "Selective dynamics" in contcar_text
        rows = [line.split() for line in contcar_text.splitlines()[9:13]]
        assert [row[3:6] for row in rows] == [
            ["T", "T", "F"],
            ["T", "T", "T"],
            ["T", "T", "T"],
            ["F", "F", "F"],
        ], contcar_text
        # The frozen coordinates did not move, and pymatgen reads the same
        # flags back (what makes `cp CONTCAR POSCAR` continuations safe).
        scaled = contcar_ase.get_scaled_positions(wrap=True)
        assert abs(scaled[0][2] - 0.03) < 1e-8
        assert np.allclose(scaled[3], [0.5, 0.5, 0.0], atol=1e-8)
        assert [list(row) for row in contcar_pmg.selective_dynamics] == [
            [True, True, False],
            [True, True, True],
            [True, True, True],
            [False, False, False],
        ]


if __name__ == "__main__":
    _scenario, _workdir = sys.argv[1], Path(sys.argv[2])
    _check_roundtrip(_workdir, _scenario)
    print("ROUNDTRIP OK")
