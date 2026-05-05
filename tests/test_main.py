from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms

import vpmdk
import vpmdk.compat.vasp as vasp_compat
from tests.conftest import DummyCalculator


def _shift_first_direct_position(poscar_text: str, delta: float) -> str:
    lines = poscar_text.splitlines()
    coord_start = None
    for index, line in enumerate(lines):
        if line.strip().lower().startswith(("direct", "cart")):
            coord_start = index + 1
            break
    if coord_start is None:
        raise AssertionError("test POSCAR does not contain a coordinate mode line")

    parts = lines[coord_start].split()
    parts[0] = f"{(float(parts[0]) + delta) % 1.0:.9f}"
    lines[coord_start] = "     " + "         ".join(parts)
    return "\n".join(lines) + "\n"


def _set_first_direct_position(poscar_text: str, value: float) -> str:
    lines = poscar_text.splitlines()
    coord_start = None
    for index, line in enumerate(lines):
        if line.strip().lower().startswith(("direct", "cart")):
            coord_start = index + 1
            break
    if coord_start is None:
        raise AssertionError("test POSCAR does not contain a coordinate mode line")

    parts = lines[coord_start].split()
    parts[0] = f"{value:.9f}"
    lines[coord_start] = "     " + "         ".join(parts)
    return "\n".join(lines) + "\n"


def _reconstruct_force_constants_from_vasprun(path: Path, num_atoms: int = 2):
    root = ET.parse(path).getroot()
    hessian_rows = [
        [float(value) for value in row.text.split()]
        for row in root.findall("./dynmat/varray[@name='hessian']/v")
    ]
    atomtype_rows = root.findall("./atominfo/array[@name='atomtypes']/set/rc")
    masses: list[float] = []
    for row in atomtype_rows:
        cells = row.findall("c")
        masses.extend([float(cells[2].text)] * int(cells[0].text))

    reconstructed = np.zeros((num_atoms, num_atoms, 3, 3), dtype=float)
    hessian = np.asarray(hessian_rows, dtype=float)
    for i in range(num_atoms):
        for j in range(num_atoms):
            reconstructed[i, j] = (
                -hessian[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3]
                * np.sqrt(masses[i] * masses[j])
            )
    return reconstructed


def _write_numbered_neb_poscars(run_dir: Path) -> None:
    poscar_text = (run_dir / "POSCAR").read_text()
    for image, delta in zip(("00", "01", "02"), (0.0, 0.01, 0.02)):
        image_dir = run_dir / image
        image_dir.mkdir()
        (image_dir / "POSCAR").write_text(
            _shift_first_direct_position(poscar_text, delta)
        )


class DummyNEBOptimizer:
    def __init__(self, obj, logfile=None):
        self.obj = obj
        self._callbacks = []

    def attach(self, callback, *args, **kwargs):
        self._callbacks.append((callback, args, kwargs))

    def run(self, *args, **kwargs):
        positions = self.obj.get_positions()
        self.obj.set_positions(positions + 0.01)
        for callback, cb_args, cb_kwargs in self._callbacks:
            callback(*cb_args, **cb_kwargs)
        return False


@pytest.mark.parametrize(
    "potential",
    [
        "CHGNET",
        "SEVENNET",
        "FLASHTP",
        "MATGL",
        "M3GNET",
        "MACE",
        "MATTERSIM",
        "MATLANTIS",
        "EQNORM",
        "MATRIS",
        "ALPHANET",
        "HIENET",
        "NEQUIX",
        "ALLEGRO",
        "NEQUIP",
        "ORB",
        "UPET",
        "TACE",
        "EQUFLASH",
        "FAIRCHEM",
        "FAIRCHEM_V2",
        "FAIRCHEM_V1",
        "GRACE",
        "DEEPMD",
    ],
)
def test_single_point_energy_for_all_potentials(
    tmp_path: Path,
    potential: str,
    prepare_inputs,
):
    extra_bcar: dict[str, str] = {}
    if potential in {"NEQUIP", "ALLEGRO", "DEEPMD", "FAIRCHEM_V1", "UPET", "TACE", "EQUFLASH"}:
        model_name = (
            "pet-oam-xl-v1.0.0.ckpt"
            if potential == "UPET"
            else (
                "tace-model.pt"
                if potential == "TACE"
                else ("equflash-model.ckpt" if potential == "EQUFLASH" else "nequip-model.pth")
            )
        )
        model_path = tmp_path / model_name
        model_path.write_text("dummy")
        extra_bcar["MODEL"] = str(model_path)

    prepare_inputs(
        tmp_path,
        potential=potential,
        incar_overrides={"NSW": "0"},
        extra_bcar=extra_bcar,
    )

    created: list[tuple[str, DummyCalculator]] = []

    def factory(name: str):
        calc = DummyCalculator()
        created.append((name, calc))
        return calc

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "CHGNetCalculator", lambda *a, **k: factory("CHGNET"))
    monkeypatch.setattr(vpmdk, "_build_sevennet_calculator", lambda *a, **k: factory("SEVENNET"))
    monkeypatch.setattr(vpmdk, "_build_flashtp_calculator", lambda *a, **k: factory("FLASHTP"))
    monkeypatch.setattr(
        vpmdk,
        "_build_m3gnet_calculator",
        lambda tags: factory(vpmdk._resolve_mlp_tag(tags, default="MATGL")),
    )
    monkeypatch.setattr(vpmdk, "MACECalculator", lambda *a, **k: factory("MACE"))
    monkeypatch.setattr(vpmdk, "MatterSimCalculator", lambda *a, **k: factory("MATTERSIM"))
    monkeypatch.setattr(vpmdk, "MatlantisEstimator", lambda *a, **k: object())
    monkeypatch.setattr(vpmdk, "MatlantisASECalculator", lambda *a, **k: factory("MATLANTIS"))
    monkeypatch.setattr(vpmdk, "_build_eqnorm_calculator", lambda *a, **k: factory("EQNORM"))
    monkeypatch.setattr(vpmdk, "MatRISCalculator", lambda *a, **k: factory("MATRIS"))
    monkeypatch.setattr(vpmdk, "_ensure_matris_named_model_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(vpmdk, "_build_alphanet_calculator", lambda *a, **k: factory("ALPHANET"))
    monkeypatch.setattr(vpmdk, "_build_hienet_calculator", lambda *a, **k: factory("HIENET"))
    monkeypatch.setattr(vpmdk, "_build_nequix_calculator", lambda *a, **k: factory("NEQUIX"))
    monkeypatch.setattr(vpmdk, "_build_allegro_calculator", lambda *a, **k: factory("ALLEGRO"))
    monkeypatch.setattr(vpmdk, "ORBCalculator", lambda *a, **k: factory("ORB"))
    monkeypatch.setattr(vpmdk, "ORB_PRETRAINED_MODELS", {vpmdk.DEFAULT_ORB_MODEL: lambda **_: "orb"})
    monkeypatch.setattr(vpmdk, "UPETCalculator", lambda *a, **k: factory("UPET"))
    monkeypatch.setattr(vpmdk, "TACEAseCalc", lambda *a, **k: factory("TACE"))
    monkeypatch.setattr(vpmdk, "_build_equflash_calculator", lambda *a, **k: factory("EQUFLASH"))
    monkeypatch.setattr(vpmdk, "_build_grace_calculator", lambda tags: factory("GRACE"))
    monkeypatch.setattr(vpmdk, "DeePMDCalculator", lambda *a, **k: factory("DEEPMD"))

    class _DummyFairChem:
        @classmethod
        def from_model_checkpoint(cls, *a, **k):
            return factory("FAIRCHEM")

    def fake_fairchem_builder(tags: dict[str, str]):
        mlp_tag = vpmdk._resolve_mlp_tag(tags, default="")
        name = "FAIRCHEM_V2" if mlp_tag == "FAIRCHEM_V2" else "FAIRCHEM"
        return factory(name)

    monkeypatch.setattr(vpmdk, "FAIRChemCalculator", _DummyFairChem)
    monkeypatch.setitem(vpmdk._CALCULATOR_BUILDERS, "FAIRCHEM", fake_fairchem_builder)
    monkeypatch.setitem(
        vpmdk._CALCULATOR_BUILDERS, "FAIRCHEM_V2", fake_fairchem_builder
    )

    class _DummyFairChemV1:
        def __init__(self, *a, **k):
            factory("FAIRCHEM_V1")

    monkeypatch.setattr(
        vpmdk, "_get_fairchem_v1_calculator_cls", lambda: _DummyFairChemV1
    )

    class DummyEstimatorMode:
        CRYSTAL = "CRYSTAL"

        @classmethod
        def __getitem__(cls, key):
            return getattr(cls, key)

    monkeypatch.setattr(vpmdk, "EstimatorCalcMode", DummyEstimatorMode)
    monkeypatch.setattr(
        vpmdk,
        "NequIPCalculator",
        SimpleNamespace(from_deployed_model=lambda *a, **k: factory("NEQUIP")),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert created and created[-1][0] == potential
    assert created[-1][1].called == 1
    assert (tmp_path / "CONTCAR").exists()


def test_main_transfers_magmom_to_atoms(tmp_path: Path, prepare_inputs, arrays_close):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "0", "MAGMOM": "1.25 -0.75"},
    )

    captured: dict[str, list[float]] = {}

    def capture_magmoms(atoms, calculator, **kwargs):
        captured["moments"] = list(atoms.get_initial_magnetic_moments())
        return 0.5

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(vpmdk, "run_single_point", capture_magmoms)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert "moments" in captured
    assert arrays_close(captured["moments"], [1.25, -0.75])


def test_fairchem_v1_predictor_tag_uses_predictor(tmp_path: Path):
    model_path = tmp_path / "fairchem-model.pt"
    model_path.write_text("dummy")

    class DummyPredictor:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "_get_fairchem_v1_predictor_cls", lambda: DummyPredictor)
    try:
        calc = vpmdk.get_calculator(
            vpmdk.BackendConfig(
                mlp="FAIRCHEM_V1",
                model=str(model_path),
                options={"FAIRCHEM_V1_PREDICTOR": "1"},
            )
        )
    finally:
        monkeypatch.undo()

    assert isinstance(calc, vpmdk._FairChemV1PredictorCalculator)


def test_fairchem_calculator_uses_bcar_overrides(tmp_path: Path, prepare_inputs):
    model_name = "esen-md-direct-all-omol"
    prepare_inputs(
        tmp_path,
        potential="FAIRCHEM",
        incar_overrides={"NSW": "0"},
        extra_bcar={
            "MODEL": model_name,
            "FAIRCHEM_TASK": "omol",
            "FAIRCHEM_INFERENCE_SETTINGS": "turbo",
            "DEVICE": "cuda",
        },
    )

    seen: dict[str, object] = {}

    class _DummyFairChem:
        @classmethod
        def from_model_checkpoint(
            cls,
            name_or_path,
            *,
            task_name=None,
            inference_settings="default",
            device=None,
            **_,
        ):
            seen.update(
                {
                    "name": name_or_path,
                    "task": task_name,
                    "settings": inference_settings,
                    "device": device,
                }
            )
            return DummyCalculator()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "FAIRChemCalculator", _DummyFairChem)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert seen == {
        "name": model_name,
        "task": "omol",
        "settings": "turbo",
        "device": "cuda",
    }


def test_fairchem_v1_builder_uses_bcar_overrides():
    seen: dict[str, object] = {}

    class _DummyFairChemV1:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def get_potential_energy(self, atoms=None, force_consistent=False):
            return 0.0

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        vpmdk, "_get_fairchem_v1_calculator_cls", lambda: _DummyFairChemV1
    )

    calculator = vpmdk.get_calculator(
        vpmdk.BackendConfig(
            mlp="FAIRCHEM_V1",
            model="checkpoint.pt",
            device="cpu",
            options={"FAIRCHEM_CONFIG": "config.yml"},
        )
    )

    monkeypatch.undo()

    assert isinstance(calculator, _DummyFairChemV1)
    assert seen == {
        "checkpoint_path": "checkpoint.pt",
        "cpu": True,
        "config_yml": "config.yml",
    }


def test_main_negative_ibrion_forces_single_point(tmp_path: Path, prepare_inputs):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "5", "IBRION": "-1"},
    )

    seen: dict[str, int] = {}

    def fake_single_point(atoms, calculator, **kwargs):
        seen["single_point"] = seen.get("single_point", 0) + 1
        return 0.5

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(vpmdk, "run_single_point", fake_single_point)

    def fail(*args, **kwargs):  # pragma: no cover - defensive guard
        raise AssertionError("Should not run MD or relaxation when IBRION<0")

    monkeypatch.setattr(vpmdk, "run_md", fail)
    monkeypatch.setattr(vpmdk, "run_relaxation", fail)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert seen.get("single_point") == 1


def test_main_ibrion7_writes_vasp_dynmat_for_phonopy_fc(
    tmp_path: Path, prepare_inputs
):
    stiffness = 3.25
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "1", "IBRION": "7", "ISIF": "2"},
        extra_bcar={"FORCE_CONSTANTS_DISPLACEMENT": "0.02"},
    )

    class HarmonicCalculator(Calculator):
        implemented_properties = ["energy", "forces", "stress"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            positions = atoms.get_positions()
            self.results = {
                "energy": 0.5 * stiffness * float(np.sum(positions * positions)),
                "forces": -stiffness * positions,
                "stress": np.zeros(6),
            }

    def fail(*args, **kwargs):  # pragma: no cover - defensive guard
        raise AssertionError("IBRION=7 should use force-constants mode")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: HarmonicCalculator())
    monkeypatch.setattr(vpmdk, "run_single_point", fail)
    monkeypatch.setattr(vpmdk, "run_relaxation", fail)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    root = ET.parse(tmp_path / "vasprun.xml").getroot()
    hessian_rows = [
        [float(value) for value in row.text.split()]
        for row in root.findall("./dynmat/varray[@name='hessian']/v")
    ]
    assert np.asarray(hessian_rows).shape == (6, 6)

    atomtype_rows = root.findall("./atominfo/array[@name='atomtypes']/set/rc")
    masses: list[float] = []
    for row in atomtype_rows:
        cells = row.findall("c")
        masses.extend([float(cells[2].text)] * int(cells[0].text))

    reconstructed = np.zeros((2, 2, 3, 3), dtype=float)
    hessian = np.asarray(hessian_rows, dtype=float)
    for i in range(2):
        for j in range(2):
            reconstructed[i, j] = (
                -hessian[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3]
                * np.sqrt(masses[i] * masses[j])
            )

    expected = np.zeros((2, 2, 3, 3), dtype=float)
    for atom_index in range(2):
        expected[atom_index, atom_index] = np.eye(3) * stiffness
    assert reconstructed == pytest.approx(expected)


def test_main_ibrion7_preserves_noncontiguous_atomtype_mass_order(
    tmp_path: Path, monkeypatch
):
    stiffness = 1.5
    (tmp_path / "POSCAR").write_text(
        """H_He_H
1.0
8.0 0.0 0.0
0.0 8.0 0.0
0.0 0.0 8.0
H He H
1 1 1
Direct
0.0 0.0 0.0
0.25 0.25 0.25
0.5 0.5 0.5
"""
    )
    (tmp_path / "INCAR").write_text("NSW = 1\nIBRION = 7\nISIF = 2\n")
    (tmp_path / "BCAR").write_text(
        "MLP=CHGNET\nFORCE_CONSTANTS_DISPLACEMENT=0.02\n"
    )

    class HarmonicCalculator(Calculator):
        implemented_properties = ["energy", "forces", "stress"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            positions = atoms.get_positions()
            self.results = {
                "energy": 0.5 * stiffness * float(np.sum(positions * positions)),
                "forces": -stiffness * positions,
                "stress": np.zeros(6),
            }

    monkeypatch.setattr(
        vpmdk,
        "_build_calculator_from_tags",
        lambda *_, **__: HarmonicCalculator(),
    )
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    vpmdk.main()

    root = ET.parse(tmp_path / "vasprun.xml").getroot()
    atomtype_rows = root.findall("./atominfo/array[@name='atomtypes']/set/rc")
    atomtypes = [
        (int(row.findall("c")[0].text), row.findall("c")[1].text)
        for row in atomtype_rows
    ]
    assert atomtypes == [(1, "H"), (1, "He"), (1, "H")]

    reconstructed = _reconstruct_force_constants_from_vasprun(
        tmp_path / "vasprun.xml",
        num_atoms=3,
    )
    expected = np.zeros((3, 3, 3, 3), dtype=float)
    for atom_index in range(3):
        expected[atom_index, atom_index] = np.eye(3) * stiffness
    assert reconstructed == pytest.approx(expected)


def test_run_force_constants_uses_raw_forces_with_constraints(
    tmp_path: Path, monkeypatch
):
    stiffness = 2.0
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.2, 0.0]],
        cell=[6.0, 6.0, 6.0],
        pbc=True,
    )
    atoms.set_constraint(FixAtoms(indices=[1]))

    class HarmonicCalculator(Calculator):
        implemented_properties = ["energy", "forces", "stress"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            positions = atoms.get_positions()
            self.results = {
                "energy": 0.5 * stiffness * float(np.sum(positions * positions)),
                "forces": -stiffness * positions,
                "stress": np.zeros(6),
            }

    monkeypatch.chdir(tmp_path)
    force_constants = vpmdk.run_force_constants(
        atoms,
        HarmonicCalculator(),
        displacement=0.02,
        nfree=2,
        ibrion=7,
    )

    expected = np.zeros((2, 2, 3, 3), dtype=float)
    for atom_index in range(2):
        expected[atom_index, atom_index] = np.eye(3) * stiffness
    assert force_constants == pytest.approx(expected)

    root = ET.parse(tmp_path / "vasprun.xml").getroot()
    recorded_forces = [
        [float(value) for value in row.text.split()]
        for row in root.findall("./calculation/varray[@name='forces']/v")
    ]
    assert recorded_forces[1] == pytest.approx([-stiffness, -0.2 * stiffness, 0.0])


def test_main_ibrion5_uses_potim_and_nfree2_for_finite_difference_fc(
    tmp_path: Path, prepare_inputs, monkeypatch
):
    stiffness = 2.5
    cubic = 7.0
    displacement = 0.04
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={
            "NSW": "1",
            "IBRION": "5",
            "ISIF": "2",
            "POTIM": str(displacement),
            "NFREE": "2",
        },
        extra_bcar={"FORCE_CONSTANTS_DISPLACEMENT": "0.001"},
    )

    class AnharmonicCalculator(Calculator):
        implemented_properties = ["energy", "forces", "stress"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            positions = atoms.get_positions()
            self.results = {
                "energy": float(
                    0.5 * stiffness * np.sum(positions * positions)
                    + 0.25 * cubic * np.sum(positions**4)
                ),
                "forces": -stiffness * positions - cubic * positions**3,
                "stress": np.zeros(6),
            }

    def fail(*args, **kwargs):  # pragma: no cover - defensive guard
        raise AssertionError("IBRION=5 should use finite-difference force-constants mode")

    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: AnharmonicCalculator())
    monkeypatch.setattr(vpmdk, "run_single_point", fail)
    monkeypatch.setattr(vpmdk, "run_relaxation", fail)
    monkeypatch.setattr(vpmdk, "run_md", fail)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    vpmdk.main()

    root = ET.parse(tmp_path / "vasprun.xml").getroot()
    assert root.findtext("./incar/i[@name='IBRION']") == "5"
    assert float(root.findtext("./incar/i[@name='POTIM']")) == pytest.approx(displacement)
    assert root.findtext("./incar/i[@name='NFREE']") == "2"

    hessian_rows = [
        [float(value) for value in row.text.split()]
        for row in root.findall("./dynmat/varray[@name='hessian']/v")
    ]
    atomtype_rows = root.findall("./atominfo/array[@name='atomtypes']/set/rc")
    masses: list[float] = []
    for row in atomtype_rows:
        cells = row.findall("c")
        masses.extend([float(cells[2].text)] * int(cells[0].text))

    reconstructed = np.zeros((2, 2, 3, 3), dtype=float)
    hessian = np.asarray(hessian_rows, dtype=float)
    for i in range(2):
        for j in range(2):
            reconstructed[i, j] = (
                -hessian[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3]
                * np.sqrt(masses[i] * masses[j])
            )

    structure = vpmdk.read_structure(str(tmp_path / "POSCAR"), None)
    atoms = vpmdk.AseAtomsAdaptor.get_atoms(structure)
    atoms.wrap()
    positions = atoms.get_positions()
    expected = np.zeros((2, 2, 3, 3), dtype=float)
    for atom_index in range(2):
        for axis in range(3):
            expected[atom_index, atom_index, axis, axis] = (
                stiffness + cubic * (3.0 * positions[atom_index, axis] ** 2 + displacement**2)
            )
    assert reconstructed == pytest.approx(expected)


def test_main_ibrion5_rejects_unsupported_nfree(
    tmp_path: Path, prepare_inputs, monkeypatch
):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "1", "IBRION": "5", "POTIM": "0.02", "NFREE": "3"},
    )
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])

    with pytest.raises(NotImplementedError, match="NFREE=1, NFREE=2, and NFREE=4"):
        vpmdk.main()


@pytest.mark.parametrize(
    ("nfree", "expected_diagonal"),
    [
        (
            1,
            lambda stiffness, cubic, positions, displacement, atom_index, axis: (
                stiffness
                + cubic
                * (
                    3.0 * positions[atom_index, axis] ** 2
                    + 3.0 * positions[atom_index, axis] * displacement
                    + displacement**2
                )
            ),
        ),
        (
            4,
            lambda stiffness, cubic, positions, displacement, atom_index, axis: (
                stiffness + 3.0 * cubic * positions[atom_index, axis] ** 2
            ),
        ),
    ],
)
def test_main_ibrion5_supports_nfree1_and_nfree4_stencils(
    tmp_path: Path, prepare_inputs, monkeypatch, nfree, expected_diagonal
):
    stiffness = 2.5
    cubic = 7.0
    displacement = 0.04
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={
            "NSW": "1",
            "IBRION": "5",
            "ISIF": "2",
            "POTIM": str(displacement),
            "NFREE": str(nfree),
        },
    )

    class AnharmonicCalculator(Calculator):
        implemented_properties = ["energy", "forces", "stress"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            positions = atoms.get_positions()
            self.results = {
                "energy": float(
                    0.5 * stiffness * np.sum(positions * positions)
                    + 0.25 * cubic * np.sum(positions**4)
                ),
                "forces": -stiffness * positions - cubic * positions**3,
                "stress": np.zeros(6),
            }

    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: AnharmonicCalculator())
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    vpmdk.main()

    root = ET.parse(tmp_path / "vasprun.xml").getroot()
    assert root.findtext("./incar/i[@name='NFREE']") == str(nfree)
    reconstructed = _reconstruct_force_constants_from_vasprun(tmp_path / "vasprun.xml")
    structure = vpmdk.read_structure(str(tmp_path / "POSCAR"), None)
    atoms = vpmdk.AseAtomsAdaptor.get_atoms(structure)
    atoms.wrap()
    positions = atoms.get_positions()

    expected = np.zeros((2, 2, 3, 3), dtype=float)
    for atom_index in range(2):
        for axis in range(3):
            expected[atom_index, atom_index, axis, axis] = expected_diagonal(
                stiffness,
                cubic,
                positions,
                displacement,
                atom_index,
                axis,
            )
    assert reconstructed == pytest.approx(expected)


@pytest.mark.parametrize(
    ("nfree", "max_force_calls"),
    [(1, 3), (2, 4), (4, 6)],
)
def test_main_ibrion6_uses_symmetry_reduced_atom_displacements(
    tmp_path: Path, prepare_inputs, monkeypatch, nfree, max_force_calls
):
    stiffness = 1.75
    calls = {"count": 0}
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={
            "NSW": "1",
            "IBRION": "6",
            "ISIF": "2",
            "POTIM": "0.03",
            "NFREE": str(nfree),
            "SYMPREC": "1e-5",
        },
    )

    class CountingHarmonicCalculator(Calculator):
        implemented_properties = ["energy", "forces", "stress"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            calls["count"] += 1
            positions = atoms.get_positions()
            self.results = {
                "energy": 0.5 * stiffness * float(np.sum(positions * positions)),
                "forces": -stiffness * positions,
                "stress": np.zeros(6),
            }

    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: CountingHarmonicCalculator())
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    vpmdk.main()

    root = ET.parse(tmp_path / "vasprun.xml").getroot()
    assert root.findtext("./incar/i[@name='IBRION']") == "6"
    assert root.findtext("./incar/i[@name='NFREE']") == str(nfree)
    assert calls["count"] <= max_force_calls

    hessian_rows = [
        [float(value) for value in row.text.split()]
        for row in root.findall("./dynmat/varray[@name='hessian']/v")
    ]
    atomtype_rows = root.findall("./atominfo/array[@name='atomtypes']/set/rc")
    masses: list[float] = []
    for row in atomtype_rows:
        cells = row.findall("c")
        masses.extend([float(cells[2].text)] * int(cells[0].text))

    reconstructed = np.zeros((2, 2, 3, 3), dtype=float)
    hessian = np.asarray(hessian_rows, dtype=float)
    for i in range(2):
        for j in range(2):
            reconstructed[i, j] = (
                -hessian[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3]
                * np.sqrt(masses[i] * masses[j])
            )

    expected = np.zeros((2, 2, 3, 3), dtype=float)
    for atom_index in range(2):
        expected[atom_index, atom_index] = np.eye(3) * stiffness
    assert reconstructed == pytest.approx(expected, abs=1e-8)


def test_main_ibrion7_warns_that_dfpt_is_finite_difference_compatibility(
    tmp_path: Path, prepare_inputs, monkeypatch, capsys
):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "1", "IBRION": "7", "ISIF": "2"},
    )
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    vpmdk.main()

    captured = capsys.readouterr()
    assert "IBRION=7/8 are VASP DFPT modes" in captured.out
    assert "finite-difference dynmat/hessian" in captured.out


def test_main_ibrion8_warns_and_uses_symmetry_reduction(
    tmp_path: Path, prepare_inputs, monkeypatch, capsys
):
    calls = {"count": 0}
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "1", "IBRION": "8", "ISIF": "2", "SYMPREC": "1e-5"},
        extra_bcar={"FORCE_CONSTANTS_DISPLACEMENT": "0.025"},
    )

    class CountingZeroForceCalculator(Calculator):
        implemented_properties = ["energy", "forces", "stress"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            calls["count"] += 1
            forces = atoms.get_positions() * 0.0
            self.results = {
                "energy": 0.5,
                "forces": forces,
                "stress": np.zeros(6),
            }

    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: CountingZeroForceCalculator())
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    vpmdk.main()

    captured = capsys.readouterr()
    assert "IBRION=7/8 are VASP DFPT modes" in captured.out
    assert calls["count"] <= 4
    root = ET.parse(tmp_path / "vasprun.xml").getroot()
    hessian_rows = [
        [float(value) for value in row.text.split()]
        for row in root.findall("./dynmat/varray[@name='hessian']/v")
    ]
    assert np.asarray(hessian_rows) == pytest.approx(np.zeros((6, 6)))


def test_build_grace_calculator_prefers_checkpoint(tmp_path: Path):
    model_path = tmp_path / "grace-model"
    model_path.write_text("dummy")

    captured: dict[str, object] = {}

    class DummyTP(DummyCalculator):
        def __init__(self, model, **kwargs):  # type: ignore[override]
            super().__init__()
            captured["model"] = model
            captured["kwargs"] = kwargs

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "TPCalculator", DummyTP)
    monkeypatch.setattr(vpmdk, "grace_fm", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(vpmdk, "GRACE_MODEL_NAMES", [])

    try:
        calc = vpmdk._build_grace_calculator(
            {
                "MODEL": str(model_path),
                "GRACE_PAD_NEIGHBORS_FRACTION": "0.1",
                "GRACE_PAD_ATOMS_NUMBER": "12",
                "GRACE_MAX_RECOMPILATION": "3",
                "GRACE_MIN_DIST": "1.5",
                "GRACE_FLOAT_DTYPE": "float32",
            }
        )
    finally:
        monkeypatch.undo()

    assert isinstance(calc, DummyTP)
    assert captured["model"] == str(model_path)
    assert captured["kwargs"] == {
        "pad_neighbors_fraction": 0.1,
        "pad_atoms_number": 12,
        "max_number_reduction_recompilation": 3,
        "min_dist": 1.5,
        "float_dtype": "float32",
    }


def test_build_grace_calculator_uses_foundation_model_when_available():
    monkeypatch = pytest.MonkeyPatch()
    selected: dict[str, object] = {}

    def fake_grace_fm(model, **kwargs):
        selected["model"] = model
        selected["kwargs"] = kwargs
        return DummyCalculator()

    monkeypatch.setattr(vpmdk, "grace_fm", fake_grace_fm)
    monkeypatch.setattr(vpmdk, "GRACE_MODEL_NAMES", ["GRACE-FOUNDATION", vpmdk.DEFAULT_GRACE_MODEL])
    monkeypatch.setattr(vpmdk, "TPCalculator", DummyCalculator)
    try:
        calc = vpmdk._build_grace_calculator({"MODEL": "GRACE-FOUNDATION"})
    finally:
        monkeypatch.undo()

    assert isinstance(calc, DummyCalculator)
    assert selected["model"] == "GRACE-FOUNDATION"
    assert selected["kwargs"] == {}


@pytest.mark.parametrize(
    "isif, expected, warning_fragment",
    [
        (0, 2, None),
        (1, 2, None),
        (2, 2, None),
        (3, 3, None),
        (4, 4, None),
        (5, 5, None),
        (6, 6, None),
        (7, 7, None),
        (8, 8, None),
    ],
)
def test_main_relaxation_respects_isif(
    tmp_path: Path, isif: int, expected: int, warning_fragment: str | None, prepare_inputs
):
    prepare_inputs(tmp_path, potential="CHGNET", incar_overrides={"NSW": "2", "ISIF": str(isif)})

    seen = {}

    def fake_run_relaxation(
        atoms,
        calculator,
        steps,
        fmax,
        write_energy_csv=False,
        isif=2,
        pstress=None,
        energy_tolerance=None,
        ibrion=2,
        stress_isif=None,
        neb_mode=False,
        oszicar_pseudo_scf=False,
    ):
        seen["isif"] = isif
        seen["pstress"] = pstress
        seen["ibrion"] = ibrion
        seen["stress_isif"] = stress_isif
        seen["neb_mode"] = neb_mode
        seen["oszicar_pseudo_scf"] = oszicar_pseudo_scf
        return 0.0

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(vpmdk, "run_relaxation", fake_run_relaxation)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    messages: list[str] = []

    def fake_print(*args, **kwargs):
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        messages.append(sep.join(str(a) for a in args) + end)

    monkeypatch.setattr("builtins.print", fake_print)
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert seen["isif"] == expected
    assert seen["stress_isif"] == isif
    if warning_fragment is None:
        assert not any("Warning: ISIF=" in message for message in messages)
    else:
        assert any(warning_fragment in message for message in messages)


def test_main_relaxation_invalid_isif_normalizes_stress_mode(tmp_path: Path, prepare_inputs):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "2", "ISIF": "-1"},
    )

    seen: dict[str, object] = {}
    messages: list[str] = []

    def fake_run_relaxation(
        atoms,
        calculator,
        steps,
        fmax,
        write_energy_csv=False,
        isif=2,
        pstress=None,
        energy_tolerance=None,
        ibrion=2,
        stress_isif=None,
        neb_mode=False,
        oszicar_pseudo_scf=False,
    ):
        seen["isif"] = isif
        seen["stress_isif"] = stress_isif
        return 0.0

    def fake_print(*args, **kwargs):
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        messages.append(sep.join(str(a) for a in args) + end)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(vpmdk, "run_relaxation", fake_run_relaxation)
    monkeypatch.setattr("builtins.print", fake_print)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert seen.get("isif") == 2
    assert seen.get("stress_isif") == 2
    assert any("defaulting to ISIF=2 behavior" in message for message in messages)


def test_main_relaxation_uses_energy_tolerance_for_positive_ediffg(
    tmp_path: Path, prepare_inputs
):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "4", "EDIFFG": "0.01"},
    )

    seen: dict[str, object] = {}

    def fake_run_relaxation(
        atoms,
        calculator,
        steps,
        fmax,
        write_energy_csv=False,
        isif=2,
        pstress=None,
        energy_tolerance=None,
        ibrion=2,
        stress_isif=None,
        neb_mode=False,
        oszicar_pseudo_scf=False,
    ):
        seen["fmax"] = fmax
        seen["energy_tolerance"] = energy_tolerance
        seen["ibrion"] = ibrion
        seen["stress_isif"] = stress_isif
        seen["neb_mode"] = neb_mode
        seen["oszicar_pseudo_scf"] = oszicar_pseudo_scf
        return 0.0

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(vpmdk, "run_relaxation", fake_run_relaxation)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert seen.get("energy_tolerance") == pytest.approx(0.01)
    assert seen.get("fmax") == pytest.approx(-0.01)


def test_main_enables_neb_mode_when_images_present(tmp_path: Path, prepare_inputs):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "2", "IMAGES": "3"},
    )

    seen: dict[str, object] = {}

    def fake_run_relaxation(
        atoms,
        calculator,
        steps,
        fmax,
        write_energy_csv=False,
        isif=2,
        pstress=None,
        energy_tolerance=None,
        ibrion=2,
        stress_isif=None,
        neb_mode=False,
        oszicar_pseudo_scf=False,
    ):
        seen["neb_mode"] = neb_mode
        seen["oszicar_pseudo_scf"] = oszicar_pseudo_scf
        return 0.0

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(vpmdk, "run_relaxation", fake_run_relaxation)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert seen.get("neb_mode") is True


def test_main_passes_pseudo_scf_flag_from_bcar(tmp_path: Path, prepare_inputs):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "2", "IBRION": "2"},
        extra_bcar={"WRITE_PSEUDO_SCF": "on"},
    )

    seen: dict[str, object] = {}

    def fake_run_relaxation(
        atoms,
        calculator,
        steps,
        fmax,
        write_energy_csv=False,
        isif=2,
        pstress=None,
        energy_tolerance=None,
        ibrion=2,
        stress_isif=None,
        neb_mode=False,
        oszicar_pseudo_scf=False,
    ):
        seen["oszicar_pseudo_scf"] = oszicar_pseudo_scf
        return 0.0

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(vpmdk, "run_relaxation", fake_run_relaxation)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert seen.get("oszicar_pseudo_scf") is True


def test_main_warns_that_pseudo_scf_incar_tags_are_ignored_by_default(
    tmp_path: Path, prepare_inputs
):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "0"},
    )

    messages: list[str] = []

    def fake_print(*args, **kwargs):
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        messages.append(sep.join(str(a) for a in args) + end)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(vpmdk, "run_single_point", lambda *_, **__: 0.0)
    monkeypatch.setattr("builtins.print", fake_print)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert any("INCAR tag NELM is not supported" in message for message in messages)
    assert any("INCAR tag NELMIN is not supported" in message for message in messages)
    assert any("INCAR tag EDIFF is not supported" in message for message in messages)


def test_main_warns_that_pseudo_scf_incar_tags_only_affect_compat_output_when_enabled(
    tmp_path: Path, prepare_inputs
):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "0"},
        extra_bcar={"WRITE_PSEUDO_SCF": "on"},
    )

    messages: list[str] = []

    def fake_print(*args, **kwargs):
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        messages.append(sep.join(str(a) for a in args) + end)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(vpmdk, "run_single_point", lambda *_, **__: 0.0)
    monkeypatch.setattr("builtins.print", fake_print)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert not any("INCAR tag NELM is not supported" in message for message in messages)
    assert not any("INCAR tag NELMIN is not supported" in message for message in messages)
    assert not any("INCAR tag EDIFF is not supported" in message for message in messages)
    assert any(
        "INCAR tag NELM does not affect the run and is used only for pseudo-SCF compatibility output"
        in message
        for message in messages
    )
    assert any(
        "INCAR tag NELMIN does not affect the run and is used only for pseudo-SCF compatibility output"
        in message
        for message in messages
    )
    assert any(
        "INCAR tag EDIFF does not affect the run and is used only for pseudo-SCF compatibility output"
        in message
        for message in messages
    )


def test_main_default_vasprun_does_not_echo_ignored_pseudo_scf_tags(
    tmp_path: Path, prepare_inputs
):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "0", "NELM": "37", "NELMIN": "4", "EDIFF": "5E-07"},
    )

    messages: list[str] = []

    def fake_print(*args, **kwargs):
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        messages.append(sep.join(str(a) for a in args) + end)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr("builtins.print", fake_print)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    root = ET.parse(tmp_path / "vasprun.xml").getroot()
    assert any("INCAR tag NELM is not supported" in message for message in messages)
    assert root.find("./incar/i[@name='NELM']") is None
    assert root.find("./parameters/separator[@name='electronic']/i[@name='NELM']").text.strip() == "60"


def test_main_pseudo_scf_uses_selected_run_incar_from_dir_argument(
    tmp_path: Path, prepare_inputs
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    prepare_inputs(
        run_dir,
        potential="CHGNET",
        incar_overrides={
            "NSW": "1",
            "IBRION": "2",
            "ISIF": "2",
            "NELM": "37",
            "NELMIN": "4",
            "EDIFF": "5E-07",
        },
        extra_bcar={"WRITE_PSEUDO_SCF": "on"},
    )
    (tmp_path / "INCAR").write_text("NELM = 12\nNELMIN = 1\nEDIFF = 1E-03\n")
    (run_dir / "KPOINTS").write_text("selected\n0\nMonkhorst-Pack\n2 2 2\n0 0 0\n")
    (tmp_path / "KPOINTS").write_text("cwd\n0\nGamma\n1 1 1\n0 0 0\n")

    class DummyBFGS:
        def __init__(self, obj, logfile=None):
            self.obj = obj
            self._callbacks = []

        def attach(self, callback, *args, **kwargs):
            self._callbacks.append(callback)

        def run(self, *args, **kwargs):
            target = getattr(self.obj, "atoms", self.obj)
            target.positions += 0.01
            for callback in self._callbacks:
                callback()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(vpmdk, "BFGS", DummyBFGS)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(run_dir)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    outcar = (run_dir / "OUTCAR").read_text()
    root = ET.parse(run_dir / "vasprun.xml").getroot()
    assert "NELM   =     37;" in outcar
    assert "NELM   =     12;" not in outcar
    assert "   NELM = 37" in outcar
    assert "   NELMIN = 4" in outcar
    assert "   EDIFF = 5E-07" in outcar
    assert "   NELM = 12" not in outcar
    assert "   NELMIN = 1" not in outcar
    assert "   EDIFF = 1E-03" not in outcar
    assert "k-points in reciprocal lattice and weights: Monkhorst-Pack" in outcar
    assert "k-points in reciprocal lattice and weights: Gamma" not in outcar
    assert root.find("./incar/i[@name='NELM']").text.strip() == "37"
    assert root.find("./incar/i[@name='NELMIN']").text.strip() == "4"
    assert root.find("./incar/i[@name='EDIFF']").text.strip() == "5.00000000E-07"
    assert not (tmp_path / "OUTCAR").exists()
    assert not (tmp_path / "vasprun.xml").exists()


def test_main_single_point_writes_contcar_into_selected_run_dir(
    tmp_path: Path, prepare_inputs
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    prepare_inputs(
        run_dir,
        potential="CHGNET",
        incar_overrides={"NSW": "0"},
    )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(run_dir)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert (run_dir / "CONTCAR").exists()
    assert (run_dir / "OUTCAR").exists()
    assert (run_dir / "OSZICAR").exists()
    assert (run_dir / "vasprun.xml").exists()
    assert not (tmp_path / "CONTCAR").exists()
    assert not (tmp_path / "OUTCAR").exists()
    assert not (tmp_path / "OSZICAR").exists()
    assert not (tmp_path / "vasprun.xml").exists()


def test_main_initializes_non_neb_calculator_from_selected_run_dir_for_relative_model_path(
    tmp_path: Path, prepare_inputs
):
    run_dir = tmp_path / "runs" / "single_model"
    run_dir.mkdir(parents=True)
    prepare_inputs(
        run_dir,
        potential="NEQUIP",
        incar_overrides={"NSW": "0"},
        extra_bcar={"MODEL": "./model/nequip.pth"},
    )

    model_dir = run_dir / "model"
    model_dir.mkdir()
    (model_dir / "nequip.pth").write_text("dummy")

    seen: dict[str, object] = {}

    def fake_get_calculator(tags, *, structure=None):
        seen["cwd"] = Path.cwd()
        seen["model"] = tags.get("MODEL")
        return DummyCalculator()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", fake_get_calculator)
    monkeypatch.setattr(vpmdk, "run_single_point", lambda *_, **__: 0.0)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", "runs/single_model"])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert seen.get("cwd") == run_dir
    assert seen.get("model") == "./model/nequip.pth"


def test_main_md_writes_outputs_into_selected_run_dir(tmp_path: Path, prepare_inputs):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    prepare_inputs(
        run_dir,
        potential="CHGNET",
        incar_overrides={"NSW": "2", "IBRION": "0", "TEBEG": "300", "POTIM": "1.0"},
    )

    class DummyDynamics:
        def run(self, n):
            assert n == 1

    def fake_selector(atoms, mdalgo, timestep, initial_temperature, smass, params):
        return DummyDynamics(), lambda temp: None

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(vpmdk, "_select_md_dynamics", fake_selector)
    monkeypatch.setattr(
        vpmdk.velocitydistribution,
        "MaxwellBoltzmannDistribution",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(run_dir)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert (run_dir / "CONTCAR").exists()
    assert (run_dir / "OUTCAR").exists()
    assert (run_dir / "OSZICAR").exists()
    assert (run_dir / "XDATCAR").exists()
    assert (run_dir / "vasprun.xml").exists()
    assert not (tmp_path / "CONTCAR").exists()
    assert not (tmp_path / "OUTCAR").exists()
    assert not (tmp_path / "OSZICAR").exists()
    assert not (tmp_path / "XDATCAR").exists()
    assert not (tmp_path / "vasprun.xml").exists()


def test_main_runs_neb_images_from_numbered_directories(tmp_path: Path, prepare_inputs):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "2", "ISIF": "2", "IMAGES": "1"},
    )

    _write_numbered_neb_poscars(tmp_path)

    seen: dict[str, object] = {}

    def fail(*args, **kwargs):  # pragma: no cover - defensive guard
        raise AssertionError("NEB relaxation should use the ASE NEB optimizer")

    class RecordingNEBOptimizer(DummyNEBOptimizer):
        def __init__(self, obj, logfile=None):
            super().__init__(obj, logfile=logfile)
            seen["optimizable_atoms"] = len(obj)
            seen["nimages"] = obj.nimages
            seen["spring"] = list(obj.k)
            seen["climb"] = obj.climb

        def run(self, *args, **kwargs):
            seen["steps"] = kwargs.get("steps")
            seen["fmax"] = kwargs.get("fmax")
            return super().run(*args, **kwargs)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(vpmdk, "BFGS", RecordingNEBOptimizer)
    monkeypatch.setattr(vpmdk, "run_relaxation", fail)
    monkeypatch.setattr(vpmdk, "run_single_point", fail)
    monkeypatch.setattr(vpmdk, "run_md", fail)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert seen["optimizable_atoms"] == 2
    assert seen["nimages"] == 3
    assert seen["spring"] == [5.0, 5.0]
    assert seen["climb"] is False
    assert seen["steps"] == 2
    assert seen["fmax"] == pytest.approx(0.01)
    for image in ("00", "01", "02"):
        assert (tmp_path / image / "OUTCAR").exists()
        assert (tmp_path / image / "CONTCAR").exists()


def test_run_neb_images_uses_parent_incar_for_pseudo_scf_settings(
    tmp_path: Path, prepare_inputs
):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={
            "NSW": "0",
            "IMAGES": "1",
            "NELM": "37",
            "NELMIN": "4",
            "NELMDL": "-3",
            "EDIFF": "5E-07",
        },
    )

    _write_numbered_neb_poscars(tmp_path)

    incar = vpmdk._load_incar(str(tmp_path / "INCAR"))
    settings = vpmdk._load_incar_settings(incar)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    try:
        vpmdk.run_neb_images(
            workdir=str(tmp_path),
            incar=incar,
            settings=settings,
            bcar={"POTENTIAL": "CHGNET"},
            potcar_path=str(tmp_path / "POTCAR"),
            write_energy_csv=False,
            write_lammps_traj=False,
            lammps_traj_interval=1,
            oszicar_pseudo_scf=True,
        )
    finally:
        monkeypatch.undo()

    outcar = (tmp_path / "00" / "OUTCAR").read_text()
    root = ET.parse(tmp_path / "00" / "vasprun.xml").getroot()
    assert "NELM   =     37;" in outcar
    assert "   NELM = 37" in outcar
    assert root.find("./incar/i[@name='NELM']").text.strip() == "37"
    assert root.find("./incar/i[@name='NELMIN']").text.strip() == "4"
    assert root.find("./incar/i[@name='NELMDL']").text.strip() == "-3"
    assert root.find("./incar/i[@name='EDIFF']").text.strip() == "5.00000000E-07"


def test_main_neb_runner_allows_missing_top_level_poscar(tmp_path: Path, prepare_inputs):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "2", "ISIF": "2", "IMAGES": "1"},
    )

    poscar_text = (tmp_path / "POSCAR").read_text()
    for image, delta in zip(("00", "01", "02"), (0.0, 0.01, 0.02)):
        image_dir = tmp_path / image
        image_dir.mkdir()
        (image_dir / "POSCAR").write_text(
            _shift_first_direct_position(poscar_text, delta)
        )
    (tmp_path / "POSCAR").unlink()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(vpmdk, "BFGS", DummyNEBOptimizer)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    for image in ("00", "01", "02"):
        assert (tmp_path / image / "OUTCAR").exists()


def test_main_neb_runner_dispatches_single_point_when_nsw_is_zero(
    tmp_path: Path, prepare_inputs
):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "0", "IMAGES": "1"},
    )

    _write_numbered_neb_poscars(tmp_path)

    seen: list[dict[str, object]] = []

    def fake_run_single_point(atoms, calculator, **kwargs):
        seen.append(
            {
                "cwd": Path.cwd().name,
                "neb_mode": kwargs.get("neb_mode"),
                "has_prev": kwargs.get("neb_prev_positions") is not None,
                "has_next": kwargs.get("neb_next_positions") is not None,
            }
        )
        return 0.0

    def fail(*args, **kwargs):  # pragma: no cover - defensive guard
        raise AssertionError("NEB single-point setup should not dispatch to MD/relaxation")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(vpmdk, "run_single_point", fake_run_single_point)
    monkeypatch.setattr(vpmdk, "run_md", fail)
    monkeypatch.setattr(vpmdk, "run_relaxation", fail)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert [item["cwd"] for item in seen] == ["00", "01", "02"]
    assert all(item["neb_mode"] is True for item in seen)
    assert [item["has_prev"] for item in seen] == [False, True, True]
    assert [item["has_next"] for item in seen] == [True, True, False]


def test_main_neb_runner_dispatches_single_point_when_ibrion_is_negative(
    tmp_path: Path, prepare_inputs
):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "2", "IBRION": "-1", "IMAGES": "1"},
    )

    _write_numbered_neb_poscars(tmp_path)

    seen: list[str] = []

    def fake_run_single_point(atoms, calculator, **kwargs):
        seen.append(Path.cwd().name)
        return 0.0

    def fail(*args, **kwargs):  # pragma: no cover - defensive guard
        raise AssertionError("Negative IBRION NEB setup should stay single-point")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(vpmdk, "run_single_point", fake_run_single_point)
    monkeypatch.setattr(vpmdk, "run_md", fail)
    monkeypatch.setattr(vpmdk, "run_relaxation", fail)
    monkeypatch.setattr(vpmdk, "BFGS", fail)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert seen == ["00", "01", "02"]


def test_main_neb_runner_rejects_ase_neb_without_moving_images(
    tmp_path: Path, prepare_inputs
):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "2", "IBRION": "2", "IMAGES": "1"},
    )

    poscar_text = (tmp_path / "POSCAR").read_text()
    for image, delta in zip(("00", "01"), (0.0, 0.02)):
        image_dir = tmp_path / image
        image_dir.mkdir()
        (image_dir / "POSCAR").write_text(
            _shift_first_direct_position(poscar_text, delta)
        )

    def fail(*args, **kwargs):  # pragma: no cover - defensive guard
        raise AssertionError("ASE NEB should be rejected before optimizer setup")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(vpmdk, "BFGS", fail)
    try:
        incar = vpmdk._load_incar(str(tmp_path / "INCAR"))
        with pytest.raises(RuntimeError, match="requires at least three"):
            vpmdk.run_neb_images(
                workdir=str(tmp_path),
                incar=incar,
                settings=vpmdk._load_incar_settings(incar),
                bcar={"MLP": "CHGNET"},
                potcar_path=None,
                write_energy_csv=False,
                write_lammps_traj=False,
                lammps_traj_interval=1,
                oszicar_pseudo_scf=False,
            )
    finally:
        monkeypatch.undo()


def test_main_neb_runner_rejects_unsupported_vtst_ts_mode_without_numbered_images(
    tmp_path: Path, prepare_inputs
):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "2", "IBRION": "2", "ICHAIN": "2"},
    )

    def fail(*args, **kwargs):  # pragma: no cover - defensive guard
        raise AssertionError("Unsupported VTST TS mode should not run relaxation")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", fail)
    monkeypatch.setattr(vpmdk, "run_relaxation", fail)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        with pytest.raises(NotImplementedError, match="ICHAIN=2"):
            vpmdk.main()
    finally:
        monkeypatch.undo()


def test_main_neb_runner_rejects_unsupported_vtst_ts_mode_before_per_image_dispatch(
    tmp_path: Path, prepare_inputs
):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "0", "IMAGES": "1", "ICHAIN": "2"},
    )

    _write_numbered_neb_poscars(tmp_path)

    def fail(*args, **kwargs):  # pragma: no cover - defensive guard
        raise AssertionError("Unsupported VTST TS mode should not dispatch images")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "run_single_point", fail)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        with pytest.raises(NotImplementedError, match="ICHAIN=2"):
            vpmdk.main()
    finally:
        monkeypatch.undo()


def test_main_neb_runner_single_point_writes_neb_projection_lines(
    tmp_path: Path, prepare_inputs
):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "0", "IMAGES": "1"},
    )

    poscar_text = (tmp_path / "POSCAR").read_text()
    for image in ("00", "01", "02"):
        image_dir = tmp_path / image
        image_dir.mkdir()
        (image_dir / "POSCAR").write_text(poscar_text)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    for image in ("00", "01", "02"):
        outcar = (tmp_path / image / "OUTCAR").read_text()
        assert "NEB: projections on to tangent" in outcar
        assert "CHAIN + TOTAL  (eV/Angst)" in outcar


def test_main_neb_runner_passes_neb_context_to_md(tmp_path: Path, prepare_inputs):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "2", "IBRION": "0", "IMAGES": "1"},
    )

    poscar_text = (tmp_path / "POSCAR").read_text()
    for image in ("00", "01", "02"):
        image_dir = tmp_path / image
        image_dir.mkdir()
        (image_dir / "POSCAR").write_text(poscar_text)

    seen: list[dict[str, object]] = []

    def fake_run_md(
        atoms,
        calculator,
        steps,
        temperature,
        timestep,
        *,
        mdalgo,
        teend=None,
        smass=None,
        thermostat_params=None,
        **kwargs,
    ):
        seen.append(
            {
                "cwd": Path.cwd().name,
                "steps": steps,
                "neb_mode": kwargs.get("neb_mode"),
                "has_prev": kwargs.get("neb_prev_positions") is not None,
                "has_next": kwargs.get("neb_next_positions") is not None,
            }
        )
        return 0.0

    def fail(*args, **kwargs):  # pragma: no cover - defensive guard
        raise AssertionError("NEB MD setup should not dispatch to relaxation/single-point")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(vpmdk, "run_md", fake_run_md)
    monkeypatch.setattr(vpmdk, "run_single_point", fail)
    monkeypatch.setattr(vpmdk, "run_relaxation", fail)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert [item["cwd"] for item in seen] == ["00", "01", "02"]
    assert all(item["steps"] == 2 for item in seen)
    assert all(item["neb_mode"] is True for item in seen)
    assert [item["has_prev"] for item in seen] == [False, True, True]
    assert [item["has_next"] for item in seen] == [True, True, False]


def test_main_neb_runner_writes_parent_aggregate_outputs(tmp_path: Path, prepare_inputs):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "1", "IBRION": "2", "ISIF": "2", "IMAGES": "1"},
    )

    _write_numbered_neb_poscars(tmp_path)

    class StressDummyCalculator(DummyCalculator):
        def calculate(self, atoms=None, properties=("energy",), system_changes=()):
            super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
            self.results["stress"] = np.zeros(6, dtype=float)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: StressDummyCalculator())
    monkeypatch.setattr(vpmdk, "BFGS", DummyNEBOptimizer)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert (tmp_path / "OUTCAR").exists()
    assert (tmp_path / "OSZICAR").exists()
    assert (tmp_path / "vasprun.xml").exists()
    outcar = (tmp_path / "OUTCAR").read_text()
    assert "NEB: projections on to tangent" in outcar
    assert "CHAIN + TOTAL  (eV/Angst)" in outcar
    root = ET.parse(tmp_path / "vasprun.xml").getroot()
    assert len(root.findall("calculation")) == 3


def test_main_neb_runner_parent_aggregate_supports_relative_workdir(
    tmp_path: Path, prepare_inputs
):
    run_dir = tmp_path / "runs" / "neb1"
    run_dir.mkdir(parents=True)
    prepare_inputs(
        run_dir,
        potential="CHGNET",
        incar_overrides={"NSW": "1", "IBRION": "2", "ISIF": "2", "IMAGES": "1"},
    )

    _write_numbered_neb_poscars(run_dir)

    class StressDummyCalculator(DummyCalculator):
        def calculate(self, atoms=None, properties=("energy",), system_changes=()):
            super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
            self.results["stress"] = np.zeros(6, dtype=float)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: StressDummyCalculator())
    monkeypatch.setattr(vpmdk, "BFGS", DummyNEBOptimizer)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", "runs/neb1"])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert (run_dir / "OUTCAR").exists()
    assert (run_dir / "OSZICAR").exists()
    assert (run_dir / "vasprun.xml").exists()
    root = ET.parse(run_dir / "vasprun.xml").getroot()
    assert len(root.findall("calculation")) == 3


def test_main_neb_runner_initializes_calculator_from_run_dir_for_relative_model_path(
    tmp_path: Path, prepare_inputs
):
    run_dir = tmp_path / "runs" / "neb_model"
    run_dir.mkdir(parents=True)
    prepare_inputs(
        run_dir,
        potential="NEQUIP",
        incar_overrides={"NSW": "1", "IBRION": "2", "IMAGES": "1"},
        extra_bcar={"MODEL": "./model/nequip.pth"},
    )

    _write_numbered_neb_poscars(run_dir)

    model_dir = run_dir / "model"
    model_dir.mkdir()
    (model_dir / "nequip.pth").write_text("dummy")

    seen_cwds: list[Path] = []
    seen_models: list[str | None] = []

    def fake_get_calculator(tags, *, structure=None):
        seen_cwds.append(Path.cwd())
        seen_models.append(tags.get("MODEL"))
        return DummyCalculator()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", fake_get_calculator)
    monkeypatch.setattr(vpmdk, "BFGS", DummyNEBOptimizer)
    monkeypatch.setattr(vpmdk, "_collect_neb_image_results", lambda *_, **__: [])
    monkeypatch.setattr(vpmdk, "_write_neb_parent_aggregate_outputs", lambda **_: None)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", "runs/neb_model"])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert seen_cwds == [run_dir, run_dir, run_dir]
    assert seen_models == ["./model/nequip.pth"] * 3


def test_main_neb_runner_evaluates_ase_neb_calculators_from_run_dir(
    tmp_path: Path, prepare_inputs
):
    run_dir = tmp_path / "runs" / "neb_eval"
    run_dir.mkdir(parents=True)
    prepare_inputs(
        run_dir,
        potential="CHGNET",
        incar_overrides={"NSW": "1", "IBRION": "2", "IMAGES": "1"},
    )

    _write_numbered_neb_poscars(run_dir)

    seen_cwds: list[Path] = []

    class CwdRecordingCalculator(DummyCalculator):
        def calculate(self, atoms=None, properties=("energy",), system_changes=()):
            seen_cwds.append(Path.cwd())
            super().calculate(
                atoms=atoms,
                properties=properties,
                system_changes=system_changes,
            )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        vpmdk,
        "_build_calculator_from_tags",
        lambda *_, **__: CwdRecordingCalculator(),
    )
    monkeypatch.setattr(vpmdk, "BFGS", DummyNEBOptimizer)
    monkeypatch.setattr(vpmdk, "_collect_neb_image_results", lambda *_, **__: [])
    monkeypatch.setattr(vpmdk, "_write_neb_parent_aggregate_outputs", lambda **_: None)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", "runs/neb_eval"])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert seen_cwds
    assert set(seen_cwds) == {run_dir}


def test_main_neb_runner_resolves_wrapped_calculators_for_ase_neb(
    tmp_path: Path, prepare_inputs
):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "1", "IBRION": "2", "IMAGES": "1"},
    )

    _write_numbered_neb_poscars(tmp_path)

    inner_calculators: list[DummyCalculator] = []

    class Wrapper:
        def __init__(self, calculator):
            self.calculator = calculator

    def fake_get_calculator(*args, **kwargs):
        calculator = DummyCalculator()
        inner_calculators.append(calculator)
        return Wrapper(calculator)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", fake_get_calculator)
    monkeypatch.setattr(vpmdk, "BFGS", DummyNEBOptimizer)
    monkeypatch.setattr(vpmdk, "_collect_neb_image_results", lambda *_, **__: [])
    monkeypatch.setattr(vpmdk, "_write_neb_parent_aggregate_outputs", lambda **_: None)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert len(inner_calculators) == 3
    assert all(calculator.called > 0 for calculator in inner_calculators)


def test_main_neb_runner_preserves_unwrapped_image_coordinates_for_ase_neb(
    tmp_path: Path, prepare_inputs
):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "1", "IBRION": "2", "IMAGES": "1"},
    )

    poscar_text = (tmp_path / "POSCAR").read_text()
    for image, x_position in zip(("00", "01", "02"), (0.95, 1.0, 1.05)):
        image_dir = tmp_path / image
        image_dir.mkdir()
        (image_dir / "POSCAR").write_text(
            _set_first_direct_position(poscar_text, x_position)
        )

    seen_scaled_x: list[float] = []

    class RecordingNEBOptimizer(DummyNEBOptimizer):
        def __init__(self, obj, logfile=None):
            super().__init__(obj, logfile=logfile)
            seen_scaled_x.extend(
                float(image.get_scaled_positions(wrap=False)[0, 0])
                for image in obj.images
            )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(vpmdk, "BFGS", RecordingNEBOptimizer)
    monkeypatch.setattr(vpmdk, "_collect_neb_image_results", lambda *_, **__: [])
    monkeypatch.setattr(vpmdk, "_write_neb_parent_aggregate_outputs", lambda **_: None)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert seen_scaled_x == pytest.approx([0.95, 1.0, 1.05])


def test_main_neb_runner_passes_absolute_potcar_to_collect_results(
    tmp_path: Path, prepare_inputs
):
    run_dir = tmp_path / "runs" / "neb2"
    run_dir.mkdir(parents=True)
    prepare_inputs(
        run_dir,
        potential="CHGNET",
        incar_overrides={"NSW": "1", "IBRION": "2", "ISIF": "2", "IMAGES": "1"},
    )
    (run_dir / "POTCAR").write_text("Si\n")

    _write_numbered_neb_poscars(run_dir)

    seen: dict[str, object] = {}

    def fake_collect(image_dirs, *, potcar_path=None):
        seen["cwd"] = Path.cwd()
        seen["potcar_path"] = potcar_path
        seen["image_dirs"] = list(image_dirs)
        return []

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(vpmdk, "BFGS", DummyNEBOptimizer)
    monkeypatch.setattr(vpmdk, "_collect_neb_image_results", fake_collect)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", "runs/neb2"])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert seen.get("cwd") == run_dir
    potcar_path = seen.get("potcar_path")
    assert isinstance(potcar_path, str)
    assert Path(potcar_path).is_absolute()
    assert Path(potcar_path).exists()


def test_main_passes_md_parameters_to_run_md(tmp_path: Path, prepare_inputs):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={
            "NSW": "3",
            "IBRION": "0",
            "TEBEG": "200",
            "TEEND": "400",
            "POTIM": "1.5",
            "MDALGO": "3",
            "SMASS": "-2.5",
            "LANGEVIN_GAMMA": "15.0",
        },
    )

    seen: dict[str, object] = {}

    def fake_run_md(
        atoms,
        calculator,
        steps,
        temperature,
        timestep,
        *,
        mdalgo,
        teend,
        smass,
        thermostat_params,
        **kwargs,
    ):
        write_lammps_traj = kwargs.pop("write_lammps_traj", False)
        lammps_traj_interval = kwargs.pop("lammps_traj_interval", 1)
        oszicar_pseudo_scf = kwargs.pop("oszicar_pseudo_scf", False)
        seen.update(
            {
                "steps": steps,
                "temperature": temperature,
                "timestep": timestep,
                "mdalgo": mdalgo,
                "teend": teend,
                "smass": smass,
                "thermostat": thermostat_params,
                "write_lammps_traj": write_lammps_traj,
                "lammps_traj_interval": lammps_traj_interval,
                "oszicar_pseudo_scf": oszicar_pseudo_scf,
            }
        )
        seen["unexpected_kwargs"] = kwargs
        return 0.0

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(vpmdk, "run_md", fake_run_md)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert seen["steps"] == 3
    assert seen["temperature"] == 200
    assert seen["timestep"] == 1.5
    assert seen["mdalgo"] == 3
    assert seen["teend"] == 400
    assert seen["smass"] == -2.5
    assert seen["thermostat"].get("LANGEVIN_GAMMA") == 15.0
    assert seen["write_lammps_traj"] is False
    assert seen["lammps_traj_interval"] == 1
    assert seen["oszicar_pseudo_scf"] is False


def test_main_defaults_to_langevin_when_smass_negative(tmp_path: Path, prepare_inputs):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "2", "IBRION": "0", "SMASS": "-3"},
    )

    seen: dict[str, object] = {}

    def fake_run_md(
        atoms,
        calculator,
        steps,
        temperature,
        timestep,
        *,
        mdalgo,
        smass,
        **kwargs,
    ):
        seen.update({"mdalgo": mdalgo, "smass": smass})
        return 0.0

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(vpmdk, "run_md", fake_run_md)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert seen["mdalgo"] == 3
    assert seen["smass"] == -3.0


def test_main_defaults_to_nose_when_smass_positive(tmp_path: Path, prepare_inputs):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "2", "IBRION": "0", "SMASS": "2.0"},
    )

    seen: dict[str, object] = {}

    def fake_run_md(
        atoms,
        calculator,
        steps,
        temperature,
        timestep,
        *,
        mdalgo,
        smass,
        **kwargs,
    ):
        seen.update({"mdalgo": mdalgo, "smass": smass})
        return 0.0

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(vpmdk, "run_md", fake_run_md)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert seen["mdalgo"] == 2
    assert seen["smass"] == 2.0


def test_main_writes_chgcar_when_requested(tmp_path: Path, prepare_inputs):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "0", "PREC": "N", "ENCUT": "400"},
        extra_bcar={"WRITE_CHGCAR": "1"},
    )

    seen: dict[str, object] = {}

    def fake_predict_charge_density(atoms, **kwargs):
        seen["incar"] = kwargs.get("incar")
        seen["reference"] = kwargs.get("reference")
        return vpmdk.ChargeDensityResult(
            atoms=atoms,
            density=np.ones((2, 2, 2), dtype=float),
            grid_shape=(2, 2, 2),
            backend="CHARGE3NET",
            spin_density=np.full((2, 2, 2), 0.5, dtype=float),
        )

    def fake_write_chgcar(path, atoms, density, **kwargs):
        seen["path"] = path
        seen["shape"] = tuple(density.shape)
        seen["n_atoms"] = len(atoms)
        seen["spin_shape"] = None if kwargs.get("spin_density") is None else tuple(kwargs["spin_density"].shape)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(vpmdk, "predict_charge_density", fake_predict_charge_density)
    monkeypatch.setattr(vasp_compat, "write_chgcar", fake_write_chgcar)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert seen["path"] == "CHGCAR"
    assert seen["shape"] == (2, 2, 2)
    assert seen["spin_shape"] == (2, 2, 2)
    assert seen["n_atoms"] == 2
    assert seen["incar"]["PREC"] == "N"
    assert seen["incar"]["ENCUT"] == "400"
    assert seen["reference"] is not None


def test_main_routes_chgcar_backend_from_charge_mlp_flag(tmp_path: Path, prepare_inputs):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "0", "PREC": "N", "ENCUT": "400"},
        extra_bcar={"WRITE_CHGCAR": "1", "CHARGE_MLP": "DeepDFT"},
    )

    seen: dict[str, object] = {}

    def fake_predict_charge_density(atoms, **kwargs):
        seen["backend"] = kwargs.get("backend")
        return vpmdk.ChargeDensityResult(
            atoms=atoms,
            density=np.ones((2, 2, 2), dtype=float),
            grid_shape=(2, 2, 2),
            backend="DEEPDFT",
        )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(vpmdk, "predict_charge_density", fake_predict_charge_density)
    monkeypatch.setattr(vasp_compat, "write_chgcar", lambda *_, **__: None)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert seen["backend"] == "DeepDFT"


def test_main_routes_chgcar_backend_to_deepcdp_from_charge_mlp_flag(
    tmp_path: Path,
    prepare_inputs,
):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "0", "PREC": "N", "ENCUT": "400"},
        extra_bcar={"WRITE_CHGCAR": "1", "CHARGE_MLP": "DeepCDP"},
    )

    seen: dict[str, object] = {}

    def fake_predict_charge_density(atoms, **kwargs):
        seen["backend"] = kwargs.get("backend")
        return vpmdk.ChargeDensityResult(
            atoms=atoms,
            density=np.ones((2, 2, 2), dtype=float),
            grid_shape=(2, 2, 2),
            backend="DEEPCDP",
        )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(vpmdk, "predict_charge_density", fake_predict_charge_density)
    monkeypatch.setattr(vasp_compat, "write_chgcar", lambda *_, **__: None)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert seen["backend"] == "DeepCDP"


def test_main_writes_chgcar_in_requested_directory_using_final_cell(
    tmp_path: Path,
    prepare_inputs,
):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "2", "IBRION": "2", "PREC": "N", "ENCUT": "400"},
        extra_bcar={"WRITE_CHGCAR": "1", "CHARGE_SOURCE_DIR": "relative-source"},
    )

    initial_structure = vpmdk.read_structure(str(tmp_path / "POSCAR"))
    initial_atoms = vpmdk.AseAtomsAdaptor.get_atoms(initial_structure)
    initial_atoms.wrap()
    final_cell = initial_atoms.get_cell().copy()
    final_cell[0, 0] *= 1.2
    final_cell[1, 1] *= 0.9
    final_cell[2, 2] *= 1.1

    caller_dir = tmp_path / "caller"
    caller_dir.mkdir()
    seen: dict[str, object] = {}

    def fake_run_relaxation(atoms, calculator, *args, **kwargs):
        atoms.set_cell(final_cell, scale_atoms=False)
        atoms.wrap()
        return 0.0

    def fake_predict_charge_density(atoms, **kwargs):
        seen["predict_cwd"] = Path.cwd()
        seen["reference_cell"] = np.array(kwargs["reference"].get_cell())
        seen["atoms_cell"] = np.array(atoms.get_cell())
        seen["source_dir"] = kwargs.get("source_dir")
        return vpmdk.ChargeDensityResult(
            atoms=atoms,
            density=np.ones((2, 2, 2), dtype=float),
            grid_shape=(2, 2, 2),
            backend="CHARGE3NET",
            spin_density=np.full((2, 2, 2), 0.25, dtype=float),
        )

    def fake_write_chgcar(path, atoms, density, **kwargs):
        seen["write_cwd"] = Path.cwd()
        seen["path"] = path
        seen["spin_shape"] = None if kwargs.get("spin_density") is None else tuple(kwargs["spin_density"].shape)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(caller_dir)
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(vpmdk, "run_relaxation", fake_run_relaxation)
    monkeypatch.setattr(vpmdk, "predict_charge_density", fake_predict_charge_density)
    monkeypatch.setattr(vasp_compat, "write_chgcar", fake_write_chgcar)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert seen["predict_cwd"] == tmp_path
    assert seen["write_cwd"] == tmp_path
    assert seen["path"] == "CHGCAR"
    assert seen["spin_shape"] == (2, 2, 2)
    assert seen["source_dir"] == "relative-source"
    assert np.allclose(seen["reference_cell"], seen["atoms_cell"])
    assert not np.allclose(seen["reference_cell"], np.array(initial_atoms.get_cell()))


def test_main_preserves_caller_relative_charge_env_paths_under_dir(
    tmp_path: Path,
    prepare_inputs,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    prepare_inputs(
        run_dir,
        potential="CHGNET",
        incar_overrides={"NSW": "0", "PREC": "N", "ENCUT": "400"},
        extra_bcar={"WRITE_CHGCAR": "1"},
    )

    caller_dir = tmp_path / "caller"
    caller_dir.mkdir()
    source_dir = caller_dir / "charge_src"
    source_dir.mkdir()
    model_path = caller_dir / "charge_model.pt"
    model_path.write_text("checkpoint")
    seen: dict[str, object] = {}

    def fake_predict_charge_density(atoms, **kwargs):
        seen["predict_cwd"] = Path.cwd()
        seen["charge_env_base_dir"] = os.environ.get(vpmdk._CHARGE_ENV_BASE_DIR_VAR)
        return vpmdk.ChargeDensityResult(
            atoms=atoms,
            density=np.ones((2, 2, 2), dtype=float),
            grid_shape=(2, 2, 2),
            backend="CHARGE3NET",
        )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(caller_dir)
    monkeypatch.setenv("VPMDK_CHARGE_SOURCE_DIR", "charge_src")
    monkeypatch.setenv("VPMDK_CHARGE_MODEL", "charge_model.pt")
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(vpmdk, "predict_charge_density", fake_predict_charge_density)
    monkeypatch.setattr(vasp_compat, "write_chgcar", lambda *_, **__: None)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(run_dir)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert seen["predict_cwd"] == run_dir
    assert seen["charge_env_base_dir"] == str(caller_dir)
