from __future__ import annotations

import os
import re
import subprocess
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


def test_construct_ase_neb_omits_shared_keyword_for_older_ase(monkeypatch):
    captured: dict[str, object] = {}

    class OldASEStyleNEB:
        def __init__(self, images, *, k, climb, method):
            captured.update(
                {"images": images, "k": k, "climb": climb, "method": method}
            )

    images = [object(), object(), object()]
    monkeypatch.setattr(vpmdk, "NEB", OldASEStyleNEB)

    neb = vpmdk._construct_ase_neb(
        images=images,
        spring_constant=5.0,
        climb=False,
        method="aseneb",
    )

    assert isinstance(neb, OldASEStyleNEB)
    assert captured == {
        "images": images,
        "k": 5.0,
        "climb": False,
        "method": "aseneb",
    }


def test_construct_ase_neb_enables_supported_resident_calculator_sharing(
    monkeypatch,
):
    captured: dict[str, object] = {}

    class NewASEStyleNEB:
        def __init__(
            self,
            images,
            *,
            k,
            climb,
            method,
            allow_shared_calculator=False,
        ):
            captured["allow_shared_calculator"] = allow_shared_calculator

    monkeypatch.setattr(vpmdk, "NEB", NewASEStyleNEB)

    vpmdk._construct_ase_neb(
        images=[object(), object(), object()],
        spring_constant=5.0,
        climb=True,
        method="aseneb",
        calculator=DummyCalculator(),
    )

    assert captured["allow_shared_calculator"] is True


def test_construct_ase_neb_proxies_resident_calculator_on_older_ase(monkeypatch):
    shared_calculator = DummyCalculator()
    images = [Atoms("H", positions=[[index * 0.1, 0.0, 0.0]]) for index in range(3)]
    for image in images:
        image.calc = shared_calculator

    class OldASEStyleNEB:
        def __init__(self, images, *, k, climb, method):
            calculators = [image.calc for image in images]
            assert len(set(calculators)) == len(calculators)

    monkeypatch.setattr(vpmdk, "NEB", OldASEStyleNEB)

    neb = vpmdk._construct_ase_neb(
        images=images,
        spring_constant=5.0,
        climb=False,
        method="aseneb",
        calculator=shared_calculator,
    )

    assert isinstance(neb, OldASEStyleNEB)
    assert all(image.calc is not shared_calculator for image in images)
    assert all(image.calc._calculator is shared_calculator for image in images)
    assert all(np.isfinite(image.get_potential_energy()) for image in images)


def test_construct_ase_neb_uses_proxies_instead_of_generic_kwargs(monkeypatch):
    captured: dict[str, object] = {}
    shared_calculator = DummyCalculator()
    images = [SimpleNamespace(calc=shared_calculator) for _ in range(3)]

    class KwargsOnlyNEB:
        def __init__(self, images, *, k, climb, method, **kwargs):
            captured["kwargs"] = kwargs
            captured["calculators"] = [image.calc for image in images]

    monkeypatch.setattr(vpmdk, "NEB", KwargsOnlyNEB)

    vpmdk._construct_ase_neb(
        images=images,
        spring_constant=5.0,
        climb=False,
        method="aseneb",
        calculator=shared_calculator,
    )

    calculators = captured["calculators"]
    assert captured["kwargs"] == {}
    assert len(set(calculators)) == len(calculators)


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
        "EQUIFORMER_V3",
        "FAIRCHEM",
        "FAIRCHEM_V2",
        "FAIRCHEM_V1",
        "GRACE",
        "DEEPMD",
        "BAM",
    ],
)
def test_single_point_energy_for_all_potentials(
    tmp_path: Path,
    potential: str,
    prepare_inputs,
):
    extra_bcar: dict[str, str] = {}
    if potential in {
        "NEQUIP",
        "ALLEGRO",
        "DEEPMD",
        "FAIRCHEM_V1",
        "UPET",
        "TACE",
        "EQUFLASH",
        "EQUIFORMER_V3",
        "BAM",
    }:
        model_name = (
            "BAM-MP-core.pkl"
            if potential == "BAM"
            else "pet-oam-xl-v1.0.0.ckpt"
            if potential == "UPET"
            else (
                "tace-model.pt"
                if potential == "TACE"
                else (
                    "equflash-model.ckpt"
                    if potential == "EQUFLASH"
                    else (
                        "equiformer-v3.pt"
                        if potential == "EQUIFORMER_V3"
                        else "nequip-model.pth"
                    )
                )
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
    monkeypatch.setattr(
        vpmdk,
        "_build_equiformer_v3_calculator",
        lambda *a, **k: factory("EQUIFORMER_V3"),
    )
    monkeypatch.setattr(vpmdk, "_build_grace_calculator", lambda tags: factory("GRACE"))
    monkeypatch.setattr(vpmdk, "DeePMDCalculator", lambda *a, **k: factory("DEEPMD"))
    monkeypatch.setattr(vpmdk, "BAMCalculator", lambda *a, **k: factory("BAM"))

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


def test_main_preserves_poscar_header_in_contcar(tmp_path: Path, prepare_inputs):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "0"},
    )
    header = "custom VASP system title"
    poscar_lines = (tmp_path / "POSCAR").read_text().splitlines()
    poscar_lines[0] = header
    (tmp_path / "POSCAR").write_text("\n".join(poscar_lines) + "\n")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert (tmp_path / "CONTCAR").read_text().splitlines()[0] == header


def test_write_vasp_structure_truncates_header_to_vasp_limit(tmp_path: Path, load_atoms):
    atoms = load_atoms()
    header = "0123456789" * 5
    atoms.info["vasp_comment"] = header

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    try:
        vpmdk._write_vasp_structure("CONTCAR", atoms, direct=True)
    finally:
        monkeypatch.undo()

    assert (tmp_path / "CONTCAR").read_text().splitlines()[0] == header[:40]


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


def test_fairchem_v1_builder_uses_bcar_overrides(tmp_path: Path):
    seen: dict[str, object] = {}
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_text("placeholder")

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
            model=str(checkpoint_path),
            device="cpu",
            options={"FAIRCHEM_CONFIG": "config.yml"},
        )
    )

    monkeypatch.undo()

    assert isinstance(calculator, _DummyFairChemV1)
    assert seen == {
        "checkpoint_path": str(checkpoint_path),
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
    tmp_path: Path, prepare_inputs, monkeypatch, capsys
):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "1", "IBRION": "5", "POTIM": "0.02", "NFREE": "3"},
    )
    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])

    with pytest.raises(SystemExit) as excinfo:
        vpmdk.main()
    assert excinfo.value.code == 1
    assert "NFREE=1, NFREE=2, and NFREE=4" in capsys.readouterr().err


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


def test_build_grace_calculator_uses_first_model_when_named_default_is_absent(
    monkeypatch: pytest.MonkeyPatch,
):
    selected: dict[str, object] = {}

    def fake_grace_fm(model, **kwargs):
        selected["model"] = model
        return DummyCalculator()

    monkeypatch.setattr(vpmdk, "grace_fm", fake_grace_fm)
    monkeypatch.setattr(
        vpmdk, "GRACE_MODEL_NAMES", ["GRACE-INSTALLED", "GRACE-OTHER"]
    )
    monkeypatch.setattr(vpmdk, "TPCalculator", DummyCalculator)

    calculator = vpmdk._build_grace_calculator({})

    assert isinstance(calculator, DummyCalculator)
    assert selected["model"] == "GRACE-INSTALLED"


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


def test_resident_neb_runs_with_older_ase_constructor(
    tmp_path: Path, prepare_inputs, monkeypatch
):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "1", "IBRION": "2", "IMAGES": "1"},
    )
    _write_numbered_neb_poscars(tmp_path)
    shared_calculator = DummyCalculator()
    actual_neb = vpmdk.NEB
    captured: dict[str, object] = {}

    class OldASEStyleNEB:
        def __new__(cls, images, *, k, climb, method):
            calculators = [image.calc for image in images]
            captured["calculators"] = calculators
            assert len(set(calculators)) == len(calculators)
            return actual_neb(
                images,
                k=k,
                climb=climb,
                method=method,
                allow_shared_calculator=False,
            )

    monkeypatch.setattr(vpmdk, "NEB", OldASEStyleNEB)
    monkeypatch.setattr(vpmdk, "BFGS", DummyNEBOptimizer)

    vpmdk.run_workdir(str(tmp_path), calculator=shared_calculator)

    calculators = captured["calculators"]
    assert len(set(calculators)) == 3
    assert all(calc._calculator is shared_calculator for calc in calculators)
    assert shared_calculator.called > 0
    for image in ("00", "01", "02"):
        assert (tmp_path / image / "OUTCAR").exists()


def test_malformed_incar_is_classified_as_input_error(tmp_path):
    # A malformed INCAR (NSW = not-a-number) must raise WorkdirInputError so both
    # one-shot mode (exit 1) and server mode (input_error) honor the documented
    # invalid-input contract instead of reporting a calculation_error (exit 2).
    (tmp_path / "INCAR").write_text("NSW = not-a-number\n")
    with pytest.raises(vpmdk.WorkdirInputError):
        vpmdk.run_workdir(str(tmp_path), calculator=DummyCalculator())


def test_malformed_poscar_is_classified_as_input_error(tmp_path):
    (tmp_path / "POSCAR").write_text("this is not a valid POSCAR file\n")
    with pytest.raises(vpmdk.WorkdirInputError):
        vpmdk.run_workdir(str(tmp_path), calculator=DummyCalculator())


def test_malformed_neb_image_poscar_is_classified_as_input_error(
    tmp_path: Path, prepare_inputs
):
    # A malformed NEB *image* POSCAR must be classified as input, exactly like a
    # malformed top-level POSCAR: the NEB branch reads image structures inside
    # run_neb_images, so those reads are wrapped (via _read_neb_image_structure)
    # and raise WorkdirInputError -> server input_error (exit 1), not the
    # calculation_error (exit 2) an unwrapped parse exception would produce.
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "0", "IBRION": "-1", "IMAGES": "1"},
    )
    _write_numbered_neb_poscars(tmp_path)
    (tmp_path / "01" / "POSCAR").write_text("garbage not a valid poscar\n@@@\n")
    with pytest.raises(vpmdk.WorkdirInputError):
        vpmdk.run_workdir(str(tmp_path), calculator=DummyCalculator())


def test_neb_images_with_inconsistent_atom_counts_are_input_error():
    # Adjacent NEB images with different atom counts are inconsistent user input.
    # _validate_neb_path must classify them as WorkdirInputError (input_error /
    # exit 1) via an explicit shape check, not let numpy's raw shape-mismatch
    # ValueError leak out as a calculation_error (exit 2). Consistent with the
    # duplicate-geometry check and the wrapped per-image structure reads.
    from vpmdk_core.runtime import neb as neb_module

    left = Atoms("H2", positions=[[0, 0, 0], [0.7, 0, 0]], cell=[10, 10, 10])
    right = Atoms("H3", positions=[[0, 0, 0], [0.7, 0, 0], [1.4, 0, 0]], cell=[10, 10, 10])
    with pytest.raises(vpmdk.WorkdirInputError, match="inconsistent atom counts"):
        neb_module._validate_neb_path([left, right])


@pytest.mark.parametrize(
    "incar_text",
    [
        "IMAGES = 1\nSPRING = -5\nNSW = 0\nIBRION = -1\n",  # single-point/MD branch
        "IMAGES = 1\nSPRING = -5\nNSW = 2\nIBRION = 2\n",  # ASE optimization branch
    ],
)
@pytest.mark.parametrize("defect", ["species-order"])
def test_inconsistent_neb_band_is_input_error_in_both_branches(
    tmp_path, incar_text, defect
):
    # ASE's own band-consistency rejections (atom count, pbc, species order) were
    # raised from root.NEB(...) with no wrapper, so the
    # optimization branch reported a permanently broken directory as
    # calculation_error (exit 2, documented RETRYABLE) while one-shot exits 1 --
    # and the single-point branch never checked at all, single-pointing every
    # image and writing a tangent computed between images describing different
    # systems as a successful run. Both branches must reject these as input. (The
    # CELL rule is branch-dependent -- see
    # test_slightly_different_neb_cells_only_block_the_optimizer -- because the
    # single-point branch evaluates each image in isolation.)
    def poscar(cell, x, species="H He"):
        return f"AB\n1.0\n{cell}\n{species}\n1 1\nCartesian\n0 0 0\n{x} 0 0\n"

    cell_a = "5 0 0\n0 5 0\n0 0 5"
    if defect == "cell":
        images = [
            poscar(cell_a, 1.0),
            poscar("5.2 0 0\n0 5 0\n0 0 5", 1.5),
            poscar(cell_a, 2.0),
        ]
    else:
        images = [
            poscar(cell_a, 1.0),
            poscar(cell_a, 1.5, species="He H"),
            poscar(cell_a, 2.0),
        ]

    (tmp_path / "INCAR").write_text(incar_text)
    (tmp_path / "BCAR").write_text("MLP = CHGNET\n")
    for index, text in enumerate(images):
        image_dir = tmp_path / f"0{index}"
        image_dir.mkdir()
        (image_dir / "POSCAR").write_text(text)

    with pytest.raises(vpmdk.WorkdirInputError):
        vpmdk.run_workdir(str(tmp_path), calculator=DummyCalculator())


def test_slightly_different_neb_cells_only_block_the_optimizer(tmp_path, capsys):
    cells = (
        "6.0000001 0 0\n0 6 0\n0 0 6",
        "6.0 0 0\n0 6 0\n0 0 6",
        "5.9999999 0 0\n0 6 0\n0 0 6",
    )

    def build(directory, incar_text):
        directory.mkdir(exist_ok=True)
        (directory / "INCAR").write_text(incar_text)
        (directory / "BCAR").write_text("MLP = CHGNET\n")
        for index, cell in enumerate(cells):
            image_dir = directory / f"0{index}"
            image_dir.mkdir()
            (image_dir / "POSCAR").write_text(
                f"X\n1.0\n{cell}\nH\n1\nCartesian\n{index * 0.5} 0 0\n"
            )

    single_point = tmp_path / "sp"
    build(single_point, "IMAGES = 1\nSPRING = -5\nNSW = 0\nIBRION = -1\n")
    vpmdk.run_workdir(str(single_point), calculator=DummyCalculator())
    assert "different periodic cell" in capsys.readouterr().out
    # Every image still produced its independent result.
    for index in range(3):
        assert (single_point / f"0{index}" / "OUTCAR").exists()

    optimization = tmp_path / "opt"
    build(optimization, "IMAGES = 1\nSPRING = -5\nNSW = 2\nIBRION = 2\n")
    with pytest.raises(vpmdk.WorkdirInputError, match="different periodic cell"):
        vpmdk.run_workdir(str(optimization), calculator=DummyCalculator())


def test_consistent_neb_band_still_runs_in_both_branches(tmp_path):
    # Guard the check above: the rules are copied from ASE, so a band ASE accepts
    # must still run in both branches.
    cell = "5 0 0\n0 5 0\n0 0 5"
    for index, x in enumerate((0.2, 0.3, 0.4)):
        image_dir = tmp_path / f"0{index}"
        image_dir.mkdir()
        (image_dir / "POSCAR").write_text(
            f"AB\n1.0\n{cell}\nH He\n1 1\nDirect\n0 0 0\n{x} 0 0\n"
        )
    (tmp_path / "BCAR").write_text("MLP = CHGNET\n")
    (tmp_path / "INCAR").write_text("IMAGES = 1\nSPRING = -5\nNSW = 0\nIBRION = -1\n")
    vpmdk.run_workdir(str(tmp_path), calculator=DummyCalculator())
    assert (tmp_path / "OUTCAR").exists()


def test_chgcar_grid_check_classifies_the_incar_not_a_diverged_run(tmp_path):
    # The grid pre-check must run on the INPUT structure, before anything is
    # computed. `atoms` is mutated in place by the relaxation/MD, so checking it
    # afterwards fed a DIVERGED cell (NaN from a blown-up structure) into the grid
    # math, and _read_workdir_input rewrote that genuine calculation failure into
    # input_error/exit 1 -- blaming an INCAR that is perfectly valid.
    import numpy as np

    poscar = "H\n1.0\n5 0 0\n0 5 0\n0 0 5\nH\n1\nCartesian\n0 0 0\n"

    class _Diverging(DummyCalculator):
        def get_forces(self, *args, **kwargs):
            return np.full((1, 3), np.nan)

        def get_stress(self, *args, **kwargs):
            return np.full(6, np.nan)

    # A genuinely unresolvable INCAR grid is still input_error -- and now fails
    # BEFORE the calculation runs, so no OUTCAR is produced.
    bad_incar = tmp_path / "bad"
    bad_incar.mkdir()
    (bad_incar / "INCAR").write_text("NSW = 0\nIBRION = -1\n")
    (bad_incar / "BCAR").write_text("MLP = CHGNET\nWRITE_CHGCAR = 1\n")
    (bad_incar / "POSCAR").write_text(poscar)
    with pytest.raises(vpmdk.WorkdirInputError, match="CHGCAR grid"):
        vpmdk.run_workdir(str(bad_incar), calculator=DummyCalculator())
    assert not (bad_incar / "OUTCAR").exists()

    # A valid INCAR whose RUN diverges is a calculation failure, not input.
    diverged = tmp_path / "diverged"
    diverged.mkdir()
    (diverged / "INCAR").write_text("ENCUT = 400\nIBRION = 2\nNSW = 1\nISIF = 3\n")
    (diverged / "BCAR").write_text("MLP = CHGNET\nWRITE_CHGCAR = 1\n")
    (diverged / "POSCAR").write_text(poscar)
    with pytest.raises(Exception) as excinfo:
        vpmdk.run_workdir(str(diverged), calculator=_Diverging())
    assert not isinstance(excinfo.value, vpmdk.WorkdirInputError)


@pytest.mark.parametrize("spelling", ["nan", "inf", "1e400"])
@pytest.mark.parametrize(
    "incar_mode",
    ["NSW = 3\nIBRION = 0\n", "NSW = 1\nIBRION = 5\n"],
)
def test_non_finite_potim_is_rejected_before_anything_runs(tmp_path, spelling, incar_mode):
    directory = tmp_path / f"wd-{spelling}-{abs(hash(incar_mode)) % 100}"
    directory.mkdir()
    (directory / "INCAR").write_text(incar_mode + f"POTIM = {spelling}\n")
    (directory / "BCAR").write_text("MLP = CHGNET\n")
    (directory / "POSCAR").write_text(
        "H\n1.0\n5 0 0\n0 5 0\n0 0 5\nH\n1\nCartesian\n0 0 0\n"
    )

    with pytest.raises(vpmdk.WorkdirInputError, match="POTIM"):
        vpmdk.run_workdir(str(directory), calculator=DummyCalculator())
    assert not (directory / "vasprun.xml").exists()
    assert not (directory / "OUTCAR").exists()


def test_non_finite_ediffg_warns_instead_of_silently_changing_fmax(tmp_path, capsys):
    # nan/inf makes BOTH `ediffg > 0` and `ediffg < 0` False, so force_limit
    # silently fell back to fmax=0.05 -- a convergence criterion the INCAR never
    # asked for, with no warning at all.
    from vpmdk_core.settings import incar as incar_module

    settings = incar_module._load_incar_settings({"NSW": 1, "IBRION": 2, "EDIFFG": float("nan")})
    assert settings.ediffg == -0.02
    assert settings.force_limit == 0.02
    assert "EDIFFG" in capsys.readouterr().out


def test_bcar_selector_errors_classify_the_same_in_flat_and_neb_dirs(tmp_path):
    # The two NEB per-image build sites were left behind when the flat path gained
    # the (ValueError, FileNotFoundError) -> WorkdirInputError guard, so the SAME
    # BCAR typo was a clean one-line diagnostic in a flat workdir and a raw
    # traceback in a NEB one. The rule now lives in one shared helper.
    poscar = "H\n1.0\n5 0 0\n0 5 0\n0 0 5\nH\n1\nCartesian\n{x} 0 0\n"

    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "INCAR").write_text("NSW = 0\nIBRION = -1\n")
    (flat / "POSCAR").write_text(poscar.format(x=0))
    (flat / "BCAR").write_text("MLP = CHGNETT\n")

    band = tmp_path / "band"
    band.mkdir()
    (band / "INCAR").write_text("IMAGES = 1\nSPRING = -5\nNSW = 10\nIBRION = 1\n")
    (band / "BCAR").write_text("MLP = CHGNETT\n")
    for index in range(3):
        image_dir = band / f"0{index}"
        image_dir.mkdir()
        (image_dir / "POSCAR").write_text(poscar.format(x=index))

    for directory in (flat, band):
        with pytest.raises(vpmdk.WorkdirInputError, match="Invalid BCAR backend settings"):
            vpmdk.run_workdir(str(directory))


def test_bcar_backend_selector_errors_are_input_not_a_traceback(tmp_path, capsys):
    # _build_calculator_from_tags was the only BCAR consumer in run_workdir not
    # routed through the input-error layer, so the MOST COMMON BCAR mistake still
    # dumped a raw multi-frame traceback while every sibling parse printed one
    # clean line. The catch is narrow on purpose: a missing backend PACKAGE raises
    # RuntimeError ("... not available. Install ..."), which is an environment
    # failure and must keep its own classification.
    poscar = "H\n1.0\n5 0 0\n0 5 0\n0 0 5\nH\n1\nCartesian\n0 0 0\n"

    def workdir(bcar_text: str) -> str:
        directory = tmp_path / f"wd{abs(hash(bcar_text)) % 10000}"
        directory.mkdir(exist_ok=True)
        (directory / "INCAR").write_text("NSW = 0\nIBRION = -1\n")
        (directory / "POSCAR").write_text(poscar)
        (directory / "BCAR").write_text(bcar_text)
        return str(directory)

    # Backend-independent selector errors (a MODEL-path case would first hit the
    # backend's own availability check, which is environment dependent).
    for bcar_text, expected in (
        ("MLP = BOGUS\n", "Unsupported MLP type"),
        ("MLP =\n", "present but empty"),
    ):
        with pytest.raises(vpmdk.WorkdirInputError) as excinfo:
            vpmdk.run_workdir(workdir(bcar_text))
        message = str(excinfo.value)
        assert "Invalid BCAR backend settings" in message
        assert expected in message  # the specific cause is preserved

    # A backend whose package is absent stays an environment error.
    with pytest.raises(RuntimeError) as excinfo:
        vpmdk.run_workdir(workdir("MLP = MATLANTIS\n"))
    assert not isinstance(excinfo.value, vpmdk.WorkdirInputError)
    assert "not available" in str(excinfo.value)


def test_unknown_charge_backend_is_input_error(tmp_path):
    # Every other CHARGE_* value is validated in the wrapped input phase, but the
    # backend SELECTOR was only rejected later at dispatch inside
    # predict_charge_density -- after the single point had already completed and
    # written OUTCAR/CONTCAR. A typo therefore escaped as a plain ValueError and
    # server mode reported calculation_error (exit 2, documented RETRYABLE), so a
    # retry driver resubmits a permanently broken BCAR forever.
    (tmp_path / "INCAR").write_text("NSW = 0\nIBRION = -1\nENCUT = 300\n")
    (tmp_path / "POSCAR").write_text(
        "H\n1.0\n5 0 0\n0 5 0\n0 0 5\nH\n1\nCartesian\n0 0 0\n"
    )
    (tmp_path / "BCAR").write_text(
        "MLP = CHGNET\nWRITE_CHGCAR = 1\nCHARGE_MLP = CHARGE3NE\n"
    )
    with pytest.raises(vpmdk.WorkdirInputError, match="charge-density backend"):
        vpmdk.run_workdir(str(tmp_path), calculator=DummyCalculator())

    # A documented alias is still accepted by the validation (it resolves to a
    # supported backend; whether that backend is installed is a separate concern).
    assert vpmdk._normalize_charge_backend_name("DEEP_DFT") in (
        vpmdk._SUPPORTED_CHARGE_BACKENDS
    )
    assert vpmdk._normalize_charge_backend_name(None) in (
        vpmdk._SUPPORTED_CHARGE_BACKENDS
    )


def test_zero_row_neb_cell_is_rejected_by_both_branches(tmp_path):
    # get_scaled_positions() is NOT a sufficient degeneracy probe: ASE's
    # Cell.complete() silently substitutes unit vectors for all-zero lattice rows,
    # so a POSCAR whose third vector is `0 0 0` sailed through the optimization
    # branch, ran the entire NEB relaxation, and only then died on a raw
    # AssertionError deep in the recorder setup -- while the single-point branch
    # and the flat workdir path both classify it as input.
    zero_row = "H\n1.0\n5 0 0\n0 5 0\n0 0 0\nH\n1\nDirect\n0.1 0.1 0.0\n"
    for index in range(3):
        image_dir = tmp_path / f"0{index}"
        image_dir.mkdir()
        (image_dir / "POSCAR").write_text(zero_row)
    (tmp_path / "BCAR").write_text("MLP = CHGNET\n")

    for incar_text in (
        "IMAGES = 1\nSPRING = -5\nNSW = 0\nIBRION = -1\n",
        "IMAGES = 1\nSPRING = -5\nNSW = 2\nIBRION = 3\n",
    ):
        (tmp_path / "INCAR").write_text(incar_text)
        with pytest.raises(vpmdk.WorkdirInputError):
            vpmdk.run_workdir(str(tmp_path), calculator=DummyCalculator())


def test_huge_incar_integer_is_tolerated_in_both_python_spellings():
    # Real pymatgen parses a several-hundred-digit INCAR literal into a Python
    # INT, and float(huge_int) raises OverflowError -- not ValueError -- so the
    # int path escaped from the VTST/NEB parse sites that run OUTSIDE
    # _read_workdir_input, surfacing as calculation_error (exit 2, documented
    # RETRYABLE) for a permanently broken INCAR. Only the STRING spelling reaches
    # the isfinite guard (float("9"*400) is inf), which is why the earlier
    # regression test missed this. Both spellings must be tolerated.
    from vpmdk_core.settings import incar as incar_module

    assert incar_module._parse_optional_float(int("9" * 400), key="ICHAIN") is None
    assert incar_module._parse_optional_float("9" * 400, key="ICHAIN") is None
    assert incar_module._parse_optional_float(3, key="ICHAIN") == 3.0
    assert incar_module._parse_optional_float("2.5", key="ICHAIN") == 2.5


def test_unreadable_optional_input_files_do_not_abort_the_run(tmp_path):
    # os.path.exists() only proves the entry is there, not that it can be read.
    # An unreadable (mode 000) or directory POTCAR/KPOINTS made the unguarded
    # open() raise while writing OUTCAR header metadata, which server mode
    # reported as calculation_error (exit 2, documented RETRYABLE) for a
    # permanent permission/type problem -- for KPOINTS, a file VPMDK explicitly
    # announces as "detected but not used". Both must degrade to "no data".
    from vpmdk_core.io import vasp_compat as vasp_compat_module

    directory = tmp_path / "POTCAR_as_dir"
    directory.mkdir()
    assert vasp_compat_module._extract_potcar_titles(str(directory)) == []
    assert vasp_compat_module._read_non_comment_lines(str(directory)) == []

    unreadable = tmp_path / "KPOINTS"
    unreadable.write_text("Automatic mesh\n")
    unreadable.chmod(0o000)
    try:
        assert vasp_compat_module._read_non_comment_lines(str(unreadable)) == []
        assert vasp_compat_module._extract_potcar_titles(str(unreadable)) == []
    finally:
        unreadable.chmod(0o644)
    # A readable file still parses.
    assert vasp_compat_module._read_non_comment_lines(str(unreadable)) == [
        "Automatic mesh"
    ]


@pytest.mark.parametrize(
    "incar_text",
    [
        "IMAGES = 1\nSPRING = -5\nNSW = 0\nIBRION = -1\n",  # single-point/MD branch
        "IMAGES = 1\nSPRING = -5\nNSW = 5\nIBRION = 3\n",  # ASE optimization branch
    ],
)
def test_degenerate_neb_image_lattice_is_input_error(tmp_path, incar_text):
    singular = "H\n1.0\n1 0 0\n2 0 0\n0 0 1\nH\n1\nCartesian\n0 0 0\n"
    healthy = "H\n1.0\n5 0 0\n0 5 0\n0 0 5\nH\n1\nCartesian\n0 0 0\n"

    (tmp_path / "INCAR").write_text(incar_text)
    (tmp_path / "BCAR").write_text("MLP = CHGNET\n")
    for index, text in ((0, healthy), (1, singular), (2, healthy)):
        image_dir = tmp_path / f"0{index}"
        image_dir.mkdir()
        (image_dir / "POSCAR").write_text(text)

    with pytest.raises(vpmdk.WorkdirInputError):
        vpmdk.run_workdir(str(tmp_path), calculator=DummyCalculator())


def test_healthy_neb_band_still_runs(tmp_path):
    # Guard the fix above: forcing the cell inversion inside the input wrapper
    # must not reject a legitimate band.
    (tmp_path / "INCAR").write_text("IMAGES = 1\nSPRING = -5\nNSW = 0\nIBRION = -1\n")
    (tmp_path / "BCAR").write_text("MLP = CHGNET\n")
    for index, shift in ((0, 0.0), (1, 0.5), (2, 1.0)):
        image_dir = tmp_path / f"0{index}"
        image_dir.mkdir()
        (image_dir / "POSCAR").write_text(
            f"H\n1.0\n5 0 0\n0 5 0\n0 0 5\nH\n1\nCartesian\n{shift} 0 0\n"
        )

    vpmdk.run_workdir(str(tmp_path), calculator=DummyCalculator())
    assert (tmp_path / "OUTCAR").exists()


def test_neb_atom_count_check_is_shared_by_both_run_branches():
    # run_neb_images has two branches: the ASE optimization path (NSW>0 and
    # IBRION>0) validated the band, while the single-point/MD path did not -- so a
    # directory with mismatched image atom counts was rejected as invalid input by
    # one branch and silently mis-computed by the other, writing an all-zero
    # TANGENT/CHAIN-FORCE summary that looks like a successful run. The SHAPE rule
    # is now shared; the duplicate-geometry rule stays exclusive to the optimizer,
    # where a zero tangent has no direction to project onto (identical adjacent
    # images are a legitimate single-point/MD input).
    import numpy as np

    from vpmdk_core.runtime import neb as neb_module

    mismatched = [np.zeros((2, 3)), np.ones((3, 3))]
    duplicated = [np.zeros((2, 3)), np.zeros((2, 3))]
    distinct = [np.zeros((2, 3)), np.ones((2, 3))]

    # Shared check (both branches): shape only.
    with pytest.raises(vpmdk.WorkdirInputError, match="inconsistent atom counts"):
        neb_module._validate_neb_image_shapes(mismatched)
    neb_module._validate_neb_image_shapes(duplicated)
    neb_module._validate_neb_image_shapes(distinct)

    # Optimizer-only check: the shared band rules PLUS duplicates. Real Atoms are
    # used because _validate_neb_path also enforces ASE's band-consistency rules
    # (species order, pbc, periodic cell), which a positions-only stub cannot
    # express.
    def image(positions):
        return Atoms("H" * len(positions), positions=positions, cell=[10, 10, 10])

    with pytest.raises(vpmdk.WorkdirInputError, match="inconsistent atom counts"):
        neb_module._validate_neb_path([image(p) for p in mismatched])
    with pytest.raises(vpmdk.WorkdirInputError, match="duplicate adjacent"):
        neb_module._validate_neb_path([image(p) for p in duplicated])
    neb_module._validate_neb_path([image(p) for p in distinct])


def test_malformed_lammps_traj_interval_is_classified_as_input_error(
    tmp_path: Path, prepare_inputs
):
    # LAMMPS_TRAJ_INTERVAL is parsed during the input phase; a malformed value
    # must raise WorkdirInputError (input_error / exit 1), matching the other
    # malformed-input tags, rather than propagating as a calculation_error.
    prepare_inputs(tmp_path, potential="CHGNET")
    (tmp_path / "BCAR").write_text(
        "MLP = CHGNET\nWRITE_LAMMPS_TRAJ = 1\nLAMMPS_TRAJ_INTERVAL = abc\n"
    )
    with pytest.raises(vpmdk.WorkdirInputError):
        vpmdk.run_workdir(str(tmp_path), calculator=DummyCalculator())


def test_huge_images_value_does_not_crash_neb_detection():
    # _is_neb_like_incar coerces IMAGES via int(float(...)), which raises
    # OverflowError (not ValueError) for a huge value (e.g. a several-hundred-digit
    # IMAGES). That must be tolerated like any other malformed IMAGES -- falling
    # back to SPRING/LCLIMB -- not propagate as an uncaught OverflowError (which a
    # server would mislabel calculation_error/exit 2 vs one-shot exit 1).
    class _FakeIncar(dict):
        def keys(self):
            return list(super().keys())

    huge = "9" * 400
    assert vpmdk._is_neb_like_incar(_FakeIncar({"IMAGES": huge})) is False
    # A huge IMAGES must not suppress NEB detection via other indicators.
    assert vpmdk._is_neb_like_incar(_FakeIncar({"IMAGES": huge, "SPRING": "-5"})) is True
    # A valid IMAGES still selects NEB.
    assert vpmdk._is_neb_like_incar(_FakeIncar({"IMAGES": "3"})) is True

    # _parse_neb_image_count (used by the NEB runtime for the image-count hint)
    # coerces via int(float(...)); inf/nan/huge must be ignored (return None),
    # not raise an uncaught OverflowError/ValueError, matching _is_neb_like_incar.
    for bad in (huge, "1e400", "nan", "-inf"):
        assert vpmdk._parse_neb_image_count(_FakeIncar({"IMAGES": bad})) is None
    assert vpmdk._parse_neb_image_count(_FakeIncar({"IMAGES": "3"})) == 3

    from vpmdk_core.settings import incar as _incar_mod
    from vpmdk_core.runtime import neb as _neb_mod

    for bad in (huge, "1e400", "nan", "-inf"):
        assert _incar_mod._parse_optional_float(bad, key="X") is None
        assert _incar_mod._parse_vtst_ichain(_FakeIncar({"ICHAIN": bad})) == 0
        assert _neb_mod._parse_neb_iopt(_FakeIncar({"IOPT": bad})) == 0
    # Valid values are still parsed.
    assert _incar_mod._parse_optional_float("2.5", key="X") == 2.5
    assert _incar_mod._parse_vtst_ichain(_FakeIncar({"ICHAIN": "2"})) == 2

    # Direct int(float(...)) "tolerate-a-malformed-value" branches must also catch
    # OverflowError (not just ValueError), so a non-finite value is ignored/
    # defaulted consistently with a non-numeric one instead of escaping.
    from vpmdk_core.io import inputs as _io_inputs

    for bad in ("inf", "9" * 400, "nan"):
        # NHC_NCHAINS: warn + ignore -> not recorded (MD uses the default chain).
        assert "NHC_NCHAINS" not in _incar_mod._extract_thermostat_parameters(
            _FakeIncar({"NHC_NCHAINS": bad})
        )
        # MAGMOM repeat count: malformed token is skipped, not crashed.
        assert _io_inputs._parse_magmom_values(f"{bad}*2.0") == []
    # Valid forms still work.
    assert _incar_mod._extract_thermostat_parameters(
        _FakeIncar({"NHC_NCHAINS": "4"})
    )["NHC_NCHAINS"] == 4.0
    assert _io_inputs._parse_magmom_values("2*1.5") == [1.5, 1.5]


def test_malformed_bcar_is_classified_as_input_error(tmp_path):
    # A malformed/unreadable BCAR is user input; run_workdir must raise
    # WorkdirInputError (exit 1 one-shot / input_error server), not let a bare
    # UnicodeDecodeError/OSError escape as an uncaught traceback (one-shot) or be
    # misclassified as calculation_error (server).
    (tmp_path / "INCAR").write_text("NSW = 0\nIBRION = -1\n")
    (tmp_path / "POSCAR").write_text(
        "H2\n1.0\n5 0 0\n0 5 0\n0 0 5\nH\n1\nCartesian\n0 0 0\n"
    )
    (tmp_path / "BCAR").write_bytes(b"MLP = CHGNET\n\xff\xfe not utf-8\n")
    with pytest.raises(vpmdk.WorkdirInputError):
        vpmdk.run_workdir(str(tmp_path), calculator=DummyCalculator())


def test_neb_layout_error_is_classified_as_input_error(
    tmp_path: Path, prepare_inputs
):
    # An NEB optimization directory with too few image directories is a user
    # layout problem: it must raise WorkdirInputError (input_error / exit 1),
    # consistent with the per-image structure reads, not a plain RuntimeError that
    # server mode would report as calculation_error (exit 2).
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "5", "IBRION": "2", "IMAGES": "1"},
    )
    for image, delta in (("00", 0.0), ("01", 0.02)):  # only two dirs (need >=3)
        image_dir = tmp_path / image
        image_dir.mkdir()
        (image_dir / "POSCAR").write_text(
            _shift_first_direct_position((tmp_path / "POSCAR").read_text(), delta)
        )
    with pytest.raises(vpmdk.WorkdirInputError):
        vpmdk.run_workdir(str(tmp_path), calculator=DummyCalculator())


def test_malformed_force_constants_displacement_is_input_error(tmp_path):
    # FORCE_CONSTANTS_DISPLACEMENT (IBRION 7/8) is a BCAR-tag input parse; a
    # malformed value must raise WorkdirInputError (input_error / exit 1), like
    # the other input tags, not surface as calculation_error (exit 2) in server
    # mode. Only the parse is wrapped; the force-constants run stays a calc.
    (tmp_path / "INCAR").write_text("IBRION = 7\nNSW = 1\n")
    (tmp_path / "POSCAR").write_text(
        "H2\n1.0\n5 0 0\n0 5 0\n0 0 5\nH\n1\nCartesian\n0 0 0\n"
    )
    (tmp_path / "BCAR").write_text(
        "MLP = CHGNET\nFORCE_CONSTANTS_DISPLACEMENT = abc\n"
    )
    with pytest.raises(vpmdk.WorkdirInputError):
        vpmdk.run_workdir(str(tmp_path), calculator=DummyCalculator())


def test_malformed_charge_option_is_input_error(tmp_path):
    # CHARGE_* charge-density options are BCAR-tag input parses; a malformed value
    # must raise WorkdirInputError (input_error / exit 1), not be misclassified as
    # calculation_error in server mode. The density prediction itself stays a calc.
    (tmp_path / "INCAR").write_text("NSW = 0\nIBRION = -1\n")
    (tmp_path / "POSCAR").write_text(
        "H2\n1.0\n5 0 0\n0 5 0\n0 0 5\nH\n1\nCartesian\n0 0 0\n"
    )
    (tmp_path / "BCAR").write_text(
        "MLP = CHGNET\nWRITE_CHGCAR = 1\nCHARGE_CUTOFF = abc\n"
    )
    with pytest.raises(vpmdk.WorkdirInputError):
        vpmdk.run_workdir(str(tmp_path), calculator=DummyCalculator())


@pytest.mark.parametrize(
    "incar_text",
    [
        "NSW = 0\nIBRION = -1\n",  # no ENCUT and no NGX*/NGXF*: grid unresolvable
        "NSW = 0\nIBRION = -1\nNGXF = abc\nNGYF = 1\nNGZF = 1\n",  # malformed grid tag
        "NSW = 0\nIBRION = -1\nPREC = Awesome\nENCUT = 300\n",  # unsupported PREC
    ],
)
def test_unresolvable_chgcar_grid_is_input_error(tmp_path, incar_text):
    # The CHGCAR grid comes from the user's INCAR (ENCUT / NGX* / NGXF* / PREC),
    # but only the adjacent BCAR tag parse was wrapped: the INCAR-derived grid
    # resolution happened deep inside predict_charge_density, so the same class of
    # user error surfaced as calculation_error (exit 2, documented RETRYABLE) and a
    # retry driver would resubmit a permanently broken INCAR forever.
    (tmp_path / "INCAR").write_text(incar_text)
    (tmp_path / "POSCAR").write_text(
        "H2\n1.0\n5 0 0\n0 5 0\n0 0 5\nH\n1\nCartesian\n0 0 0\n"
    )
    (tmp_path / "BCAR").write_text("MLP = CHGNET\nWRITE_CHGCAR = 1\n")
    with pytest.raises(vpmdk.WorkdirInputError):
        vpmdk.run_workdir(str(tmp_path), calculator=DummyCalculator())


def test_degenerate_poscar_lattice_is_input_error(tmp_path):
    # A geometrically invalid POSCAR lattice must be input_error (exit 1), never
    # the retryable calculation_error. Empirically the singular cell below is
    # caught inside read_structure (already wrapped), so this pins the
    # USER-VISIBLE contract; the conversion/wrap/MAGMOM steps that follow are
    # wrapped too, as defense in depth -- they consume the same user input, so
    # their failures belong on the same side of the classification boundary.
    (tmp_path / "INCAR").write_text("NSW = 0\nIBRION = -1\n")
    # A degenerate (zero-volume) cell: all three lattice vectors are collinear.
    (tmp_path / "POSCAR").write_text(
        "H\n1.0\n1 0 0\n2 0 0\n3 0 0\nH\n1\nCartesian\n0 0 0\n"
    )
    (tmp_path / "BCAR").write_text("MLP = CHGNET\n")
    with pytest.raises(vpmdk.WorkdirInputError):
        vpmdk.run_workdir(str(tmp_path), calculator=DummyCalculator())


def test_poscar_not_found_message_is_not_duplicated(tmp_path: Path, capsys):
    # Regression for the diagnostic-surfacing fix: run_workdir already prints
    # "POSCAR not found." to stdout before raising, so _legacy_main must NOT echo
    # it again to stderr (which would change exact ``vpmdk --dir`` output and
    # double the message). The message is marked reported=True for exactly this.
    (tmp_path / "INCAR").write_text("NSW = 0\nIBRION = -1\n")
    (tmp_path / "BCAR").write_text("MLP = CHGNET\n")
    with pytest.raises(SystemExit) as excinfo:
        vpmdk.main(["--dir", str(tmp_path)])
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert captured.out.count("POSCAR not found") == 1
    assert "POSCAR not found" not in captured.err


def test_legacy_main_reports_malformed_input_diagnostic(tmp_path: Path, capsys):
    (tmp_path / "INCAR").write_text("NSW = not-a-number\n")
    (tmp_path / "POSCAR").write_text(
        "H2\n1.0\n5 0 0\n0 5 0\n0 0 5\nH\n1\nCartesian\n0 0 0\n"
    )
    (tmp_path / "BCAR").write_text("MLP = CHGNET\n")
    with pytest.raises(SystemExit) as excinfo:
        vpmdk.main(["--dir", str(tmp_path)])
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "not-a-number" in (captured.out + captured.err).lower()


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
    tmp_path: Path, prepare_inputs, capsys
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
        # Clean exit-1 diagnostic, not a raw traceback (server mode reports the
        # same condition as input_error).
        with pytest.raises(SystemExit) as excinfo:
            vpmdk.main()
        assert excinfo.value.code == 1
        assert "ICHAIN=2" in capsys.readouterr().err
    finally:
        monkeypatch.undo()


def test_main_neb_runner_rejects_unsupported_vtst_ts_mode_before_per_image_dispatch(
    tmp_path: Path, prepare_inputs, capsys
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
        # Clean exit-1 diagnostic, not a raw traceback (server mode reports the
        # same condition as input_error).
        with pytest.raises(SystemExit) as excinfo:
            vpmdk.main()
        assert excinfo.value.code == 1
        assert "ICHAIN=2" in capsys.readouterr().err
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
            "NHC_PERIOD": "80",
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
    assert seen["thermostat"].get("NHC_PERIOD") == 80.0
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
    assert float(seen["incar"]["ENCUT"]) == 400.0
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
        assert seen["predict_cwd"] == run_dir
        assert seen["charge_env_base_dir"] == str(caller_dir)

        # Server mode runs in the server process cwd, but must retain the cwd
        # transmitted by the submitting client for relative environment paths.
        server_dir = tmp_path / "server-cwd"
        server_dir.mkdir()
        seen.clear()
        monkeypatch.chdir(server_dir)
        vpmdk.run_workdir(
            str(run_dir),
            calculator=DummyCalculator(),
            charge_base_dir=str(caller_dir),
        )
        assert seen["predict_cwd"] == run_dir
        assert seen["charge_env_base_dir"] == str(caller_dir)
    finally:
        monkeypatch.undo()


def test_non_finite_ediff_falls_back_to_the_documented_default(capsys):
    from vpmdk_core.io import vasp_compat

    for raw in (float("inf"), float("nan"), "1e400"):
        settings = vasp_compat._pseudo_scf_settings_from_incar(
            {"EDIFF": raw}, enabled=True
        )
        assert settings.ediff == 1.0e-4, raw
        assert "EDIFF" in capsys.readouterr().out


@pytest.mark.parametrize("period", ["0", "-100"])
def test_non_positive_csvr_period_is_an_input_error(period: str):
    from ase.build import bulk

    with pytest.raises(vpmdk.WorkdirInputError) as excinfo:
        vpmdk._select_md_dynamics(
            bulk("Si", "diamond", a=5.43),
            mdalgo=5,
            timestep=1.0,
            initial_temperature=300.0,
            smass=None,
            thermostat_params={"CSVR_PERIOD": float(period)},
        )

    assert "CSVR_PERIOD must be positive" in str(excinfo.value)


@pytest.mark.parametrize("corruption", ["lattice", "positions"])
def test_non_finite_poscar_geometry_is_an_input_error_in_both_layouts(
    tmp_path, corruption: str
):
    from ase.build import bulk
    from ase.io import write as ase_write

    from vpmdk_core.io import inputs as inputs_module

    source = tmp_path / "POSCAR"
    ase_write(str(source), bulk("Si", "diamond", a=5.43), format="vasp")
    lines = source.read_text().splitlines()
    if corruption == "lattice":
        lines[3] = "     0.0000000 nan 0.0000000"
    else:
        marker = next(
            index
            for index, line in enumerate(lines)
            if line.strip().lower().startswith(("direct", "cartesian"))
        )
        parts = lines[marker + 1].split()
        parts[0] = "nan"
        lines[marker + 1] = "  " + " ".join(parts)
    source.write_text("\n".join(lines) + "\n")

    structure = vpmdk.read_structure(str(source))
    atoms = vpmdk.AseAtomsAdaptor.get_atoms(structure)

    with pytest.raises(ValueError) as excinfo:
        inputs_module._validate_finite_geometry(atoms)
    assert "non-finite values" in str(excinfo.value)

    # An entirely zero cell still means "no cell given" (a legitimate molecular run).
    molecule = vpmdk.AseAtomsAdaptor.get_atoms(
        vpmdk.read_structure(str(source))
    )
    molecule.set_cell(np.zeros((3, 3)))
    molecule.set_positions(
        np.arange(len(molecule) * 3, dtype=float).reshape(-1, 3)
    )
    inputs_module._validate_finite_geometry(molecule)


def test_tiny_finite_cell_is_an_input_error_not_a_hang():
    from ase import Atoms

    from vpmdk_core.io import inputs as inputs_module

    tiny = Atoms("Cu2", scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
                 cell=np.eye(3) * 0.3, pbc=True)
    with pytest.raises(ValueError, match="below the supported minimum"):
        inputs_module._validate_finite_geometry(tiny)

    # A near-degenerate cell whose vectors are individually long but whose
    # perpendicular width has collapsed is the same failure mode.
    collapsed = Atoms(
        "Cu",
        positions=[[0.0, 0.0, 0.0]],
        cell=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [10.0, 10.0, 0.001]],
        pbc=True,
    )
    with pytest.raises(ValueError, match="below the supported minimum"):
        inputs_module._validate_finite_geometry(collapsed)

    # Ordinary cells -- including small-but-physical ones -- still pass.
    inputs_module._validate_finite_geometry(
        Atoms("Cu", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 2.0, pbc=True)
    )


def test_overflow_range_cell_is_an_input_error_not_a_silent_nan_run():
    from ase import Atoms

    from vpmdk_core.io import inputs as inputs_module

    overflowing = Atoms(
        "Si", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 1e154, pbc=True
    )
    with pytest.raises(ValueError, match="overflows a float"):
        inputs_module._validate_finite_geometry(overflowing)

    # Huge-but-representable is still absurd: nothing physical is wider than
    # 0.1 mm per axis.
    huge = Atoms(
        "Si", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 1e8, pbc=True
    )
    with pytest.raises(ValueError, match="above the supported maximum"):
        inputs_module._validate_finite_geometry(huge)

    # A large vacuum slab box stays legitimate.
    inputs_module._validate_finite_geometry(
        Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 500.0, pbc=True)
    )


def test_coincident_atoms_are_an_input_error_not_a_nan_run():
    from ase import Atoms

    from vpmdk_core.io import inputs as inputs_module

    duplicated = Atoms(
        "Si2",
        scaled_positions=[[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]],
        cell=np.eye(3) * 5.43,
        pbc=True,
    )
    with pytest.raises(ValueError, match="occupy the same site"):
        inputs_module._validate_finite_geometry(duplicated)

    # 0.0 and 1.0 are the same site under periodic boundary conditions.
    wrapped = Atoms(
        "Si2",
        scaled_positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.999999]],
        cell=np.eye(3) * 5.43,
        pbc=True,
    )
    with pytest.raises(ValueError, match="occupy the same site"):
        inputs_module._validate_finite_geometry(wrapped)

    # Real short bonds stay legitimate (H2 is 0.74 A).
    h2 = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        cell=np.eye(3) * 10.0,
        pbc=True,
    )
    inputs_module._validate_finite_geometry(h2)

    # Above the pairwise-matrix bound only EXACT duplicates are detected --
    # and they still are.
    many = Atoms(
        "H" + str(5000),
        scaled_positions=np.random.default_rng(7).uniform(0.0, 1.0, (5000, 3)),
        cell=np.eye(3) * 200.0,
        pbc=True,
    )
    inputs_module._validate_finite_geometry(many)
    positions = many.get_scaled_positions()
    positions[4999] = positions[0]
    many.set_scaled_positions(positions)
    with pytest.raises(ValueError, match="occupy the same site"):
        inputs_module._validate_finite_geometry(many)


def test_bcar_device_tag_is_case_folded_at_parse(tmp_path):
    path = tmp_path / "BCAR"
    path.write_text("MLP=CHGNET\nDEVICE = CUDA:0\nMODEL = /Models/CaseSensitive.pth\n")

    tags = vpmdk.parse_key_value_file(str(path))

    assert tags["DEVICE"] == "cuda:0"
    # Paths stay case-sensitive: only DEVICE is folded.
    assert tags["MODEL"] == "/Models/CaseSensitive.pth"


def test_corrupted_numeric_token_is_rejected_not_digit_extracted(tmp_path):
    from vpmdk_core.settings import incar as incar_module

    path = tmp_path / "INCAR"
    path.write_text("TEBEG = 5OO\n")
    with pytest.raises(ValueError, match="not a number"):
        incar_module._reject_truncated_integer_tags({"TEBEG": 5.0}, str(path))

    other = tmp_path / "INCAR2"
    other.write_text("NSW = 0x10\n")
    with pytest.raises(ValueError, match="not a number"):
        incar_module._reject_truncated_integer_tags({"NSW": 0}, str(other))

    # Non-numeric text for a genuinely non-numeric tag stays legal.
    system = tmp_path / "INCAR3"
    system.write_text("SYSTEM = D2O sample\n")
    incar_module._reject_truncated_integer_tags({"SYSTEM": "D2O sample"}, str(system))


def test_trailing_comma_scalar_values_stay_legal(tmp_path):
    from vpmdk_core.settings import incar as incar_module

    path = tmp_path / "INCAR"
    path.write_text("NSW = 3,\nTEBEG = 300,\nEDIFFG = -0.01,\n")
    incar_module._reject_truncated_integer_tags(
        {"NSW": 3, "TEBEG": 300.0, "EDIFFG": -0.01}, str(path)
    )

    # The comma must not weaken the guard: a MISREAD comma value still rejects.
    bad = tmp_path / "INCAR2"
    bad.write_text("NSW = 1e5,\n")
    with pytest.raises(ValueError):
        incar_module._reject_truncated_integer_tags({"NSW": 1}, str(bad))


def test_corrupted_token_for_untyped_numeric_tags_is_rejected(tmp_path):
    from vpmdk_core.settings import incar as incar_module

    path = tmp_path / "INCAR"
    path.write_text("CSVR_PERIOD = 5OO\n")
    with pytest.raises(ValueError, match="not a number"):
        incar_module._reject_truncated_integer_tags(
            {"CSVR_PERIOD": "5oo"}, str(path)
        )

    other = tmp_path / "INCAR2"
    other.write_text("LANGEVIN_GAMMA = 1O\n")
    with pytest.raises(ValueError, match="not a number"):
        incar_module._reject_truncated_integer_tags(
            {"LANGEVIN_GAMMA": [1]}, str(other)
        )

    # A digit-free unparseable value keeps the established warn-and-ignore
    # path of the thermostat readers (nothing can be extracted from it).
    inert = tmp_path / "INCAR3"
    inert.write_text("NHC_NCHAINS = abc\n")
    incar_module._reject_truncated_integer_tags({"NHC_NCHAINS": "abc"}, str(inert))

    # MAGMOM's 'N*value' mini-language legitimately fails float() and must
    # stay out of this rule.
    magmom = tmp_path / "INCAR4"
    magmom.write_text("MAGMOM = 4*1.0\n")
    incar_module._reject_truncated_integer_tags({"MAGMOM": [1.0] * 4}, str(magmom))


def test_huge_incar_repeat_counts_are_rejected_before_expansion(tmp_path):
    import time as time_module

    from vpmdk_core.settings import incar as incar_module

    for value in ("10000000000*1.0", "10000000000e0*2.0"):
        path = tmp_path / "INCAR"
        path.write_text(f"MAGMOM = {value}\n")
        started = time_module.monotonic()
        with pytest.raises(ValueError, match="repeat token"):
            incar_module._load_incar(str(path))
        assert time_module.monotonic() - started < 5.0

    # Ordinary MAGMOM repeats stay legal.
    path = tmp_path / "INCAR"
    path.write_text("MAGMOM = 4*1.0 2*-0.5\n")
    incar_module._load_incar(str(path))

    path.write_text("LANGEVIN_GAMMA = 3*10\n")
    with pytest.raises(ValueError, match="not a number"):
        incar_module._load_incar(str(path))

    # Defense in depth: the library-API expander enforces the same bound.
    from vpmdk_core.io import inputs as inputs_module

    with pytest.raises(ValueError, match="repeat token"):
        inputs_module._parse_magmom_values("2000000000*1.0")
    assert inputs_module._parse_magmom_values("4*1.0") == [1.0] * 4


def test_absurd_poscar_ion_counts_are_rejected_before_expansion(tmp_path):
    import time as time_module

    header = "Si\n1.0\n5 0 0\n0 5 0\n0 0 5\n"
    # VASP 5 (species line then counts) and VASP 4 (counts on line 6).
    for body in ("Si\n2000000000\nDirect\n0 0 0\n", "2000000000\nDirect\n0 0 0\n"):
        path = tmp_path / "POSCAR"
        path.write_text(header + body)
        started = time_module.monotonic()
        with pytest.raises(ValueError, match="supported maximum"):
            vpmdk.read_structure(str(path))
        assert time_module.monotonic() - started < 5.0

    # Ordinary files still read.
    path = tmp_path / "POSCAR"
    path.write_text(header + "Si\n1\nDirect\n0.1 0.2 0.3\n")
    assert vpmdk.read_structure(str(path)) is not None


def test_md_scalar_magnitudes_are_bounded(tmp_path):
    from vpmdk_core.settings import incar as incar_module

    for tag in ("TEBEG", "TEEND", "POTIM"):
        with pytest.raises(ValueError, match="supported magnitude"):
            incar_module._load_incar_settings({tag: 1e300, "IBRION": 0})
    with pytest.raises(ValueError, match="supported magnitude"):
        incar_module._load_incar_settings(
            {"IBRION": 0, "MDALGO": 3, "LANGEVIN_GAMMA": 1e300}
        )

    # The completing control stays legal.
    settings = incar_module._load_incar_settings({"TEBEG": 1e6, "IBRION": 0})
    assert settings.tebeg == 1e6


def test_smass_and_nhc_period_magnitudes_are_bounded():
    from vpmdk_core.settings import incar as incar_module

    for value in (1e300, -1e300):
        with pytest.raises(ValueError, match="supported magnitude"):
            incar_module._load_incar_settings({"IBRION": 0, "SMASS": value})
    with pytest.raises(ValueError, match="supported magnitude"):
        incar_module._load_incar_settings(
            {"IBRION": 0, "MDALGO": 2, "SMASS": 1.0, "NHC_PERIOD": 1e300}
        )

    settings = incar_module._load_incar_settings({"IBRION": 0, "SMASS": 100.0})
    assert settings.smass == 100.0


def test_fifo_input_files_are_rejected_not_hung(tmp_path):
    import time as time_module

    from vpmdk_core.io import inputs as inputs_module
    from vpmdk_core.settings import incar as incar_module

    fifo = tmp_path / "POSCAR"
    os.mkfifo(fifo)
    started = time_module.monotonic()
    with pytest.raises(ValueError, match="not a regular file"):
        vpmdk.read_structure(str(fifo))
    with pytest.raises(ValueError, match="not a regular file"):
        inputs_module.parse_key_value_file(str(fifo))
    with pytest.raises(ValueError, match="not a regular file"):
        incar_module._load_incar(str(fifo))
    assert time_module.monotonic() - started < 5.0

    # Symlinks to REGULAR files stay legitimate inputs.
    real = tmp_path / "real_bcar"
    real.write_text("MLP=CHGNET\n")
    link = tmp_path / "BCAR"
    link.symlink_to(real)
    assert inputs_module.parse_key_value_file(str(link))["MLP"] == "CHGNET"


def test_fifo_kpoints_and_output_artifacts_do_not_hang(tmp_path):
    import time as time_module

    from vpmdk_core.io import vasp_compat as vasp_compat_module

    fifo = tmp_path / "KPOINTS"
    os.mkfifo(fifo)
    started = time_module.monotonic()
    assert vasp_compat_module._read_non_comment_lines(str(fifo)) == []
    assert time_module.monotonic() - started < 5.0

    out_fifo = tmp_path / "OUTCAR"
    os.mkfifo(out_fifo)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    try:
        with pytest.raises(vpmdk.WorkdirInputError, match="not a regular file"):
            vasp_compat_module._require_writable_artifact_path("OUTCAR")
        # Directories keep their established IsADirectoryError classification.
        (tmp_path / "CONTCAR").mkdir()
        with pytest.raises(IsADirectoryError):
            vasp_compat_module._require_writable_artifact_path("CONTCAR")
        # A dangling symlink stays legal: open('w') creates the target.
        (tmp_path / "OSZICAR").symlink_to(tmp_path / "does-not-exist-yet")
        vasp_compat_module._require_writable_artifact_path("OSZICAR")
        # And the recorder initializer fails fast BEFORE any computation.
        from ase import Atoms

        with pytest.raises(vpmdk.WorkdirInputError, match="not a regular file"):
            vpmdk._initialize_vasp_compat_outputs(
                Atoms("H", positions=[[0.0, 0.0, 0.0]]), ibrion=-1
            )
    finally:
        monkeypatch.undo()


def test_free_text_system_titles_stay_legal(tmp_path):
    from vpmdk_core.settings import incar as incar_module

    for title in ("1D5 sample", "Infinity study", "NaN"):
        path = tmp_path / "INCAR"
        path.write_text(f"SYSTEM = {title}\nNSW = 2\n")
        incar_module._reject_truncated_integer_tags(
            {"SYSTEM": title, "NSW": 2}, str(path)
        )

    # Genuinely numeric tags keep every rejection.
    bad = tmp_path / "INCAR2"
    bad.write_text("TEBEG = Infinity\n")
    with pytest.raises(ValueError, match="finite"):
        incar_module._reject_truncated_integer_tags({"TEBEG": float("inf")}, str(bad))
    bad.write_text("EDIFFG = -1.0D-03\n")
    with pytest.raises(ValueError, match="would be read as"):
        incar_module._reject_truncated_integer_tags({"EDIFFG": -1.0}, str(bad))


def test_neb_image_artifacts_carry_the_pstress_convention(tmp_path):
    import numpy as np

    cell = "5 0 0\n0 5 0\n0 0 5"
    for index, x in enumerate((0.2, 0.3, 0.4)):
        image_dir = tmp_path / f"0{index}"
        image_dir.mkdir()
        (image_dir / "POSCAR").write_text(
            f"AB\n1.0\n{cell}\nH He\n1 1\nDirect\n0 0 0\n{x} 0 0\n"
        )
    (tmp_path / "BCAR").write_text("MLP = CHGNET\n")
    (tmp_path / "INCAR").write_text(
        "IMAGES = 1\nSPRING = -5\nNSW = 0\nIBRION = -1\nISIF = 3\nPSTRESS = 500\n"
    )

    class StressDummyCalculator(DummyCalculator):
        def calculate(self, atoms=None, properties=("energy",), system_changes=()):
            super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
            self.results["stress"] = np.zeros(6)

    vpmdk.run_workdir(str(tmp_path), calculator=StressDummyCalculator())

    image_outcar = (tmp_path / "01" / "OUTCAR").read_text()
    assert "Pullay stress =      500.00 kB" in image_outcar
    parent_outcar = (tmp_path / "OUTCAR").read_text()
    assert "Pullay stress =      500.00 kB" in parent_outcar


def test_ibrion_44_is_rejected_not_minimized(tmp_path):
    (tmp_path / "POSCAR").write_text(
        "Si\n1.0\n5 0 0\n0 5 0\n0 0 5\nSi\n1\nDirect\n0.1 0.1 0.1\n"
    )
    (tmp_path / "INCAR").write_text("IBRION = 44\nNSW = 3\nEDIFFG = -0.01\n")
    (tmp_path / "BCAR").write_text("MLP=CHGNET\n")

    with pytest.raises(vpmdk.UnsupportedInputError, match="IBRION=44"):
        vpmdk.run_workdir(str(tmp_path), calculator=DummyCalculator())


def test_fractional_integer_tags_are_rejected_not_floored(tmp_path):
    from vpmdk_core.settings import incar as incar_module

    path = tmp_path / "INCAR"
    path.write_text("NSW = 2.7\n")
    with pytest.raises(ValueError, match="not an integer"):
        incar_module._reject_truncated_integer_tags({"NSW": 2}, str(path))

    path.write_text("NSW = 100.\n")
    incar_module._reject_truncated_integer_tags({"NSW": 100}, str(path))


def test_lammps_trajectory_fifo_is_rejected_not_hung(tmp_path):
    import time as time_module

    from ase import Atoms

    fifo = tmp_path / "lammps.lammpstrj"
    os.mkfifo(fifo)
    atoms = Atoms(
        "H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 5.0, pbc=True
    )
    started = time_module.monotonic()
    with pytest.raises(vpmdk.WorkdirInputError, match="not a regular file"):
        vpmdk._write_lammps_trajectory_step(str(fifo), atoms, 0)
    assert time_module.monotonic() - started < 5.0


def test_client_write_line_survives_a_broken_pipe(tmp_path):
    import vpmdk_client as client_module

    read_fd, write_fd = os.pipe()
    os.close(read_fd)  # the consumer is gone
    stream = os.fdopen(write_fd, "w")
    try:
        client_module._write_line("line one", stream=stream)
        client_module._write_line("line two", stream=stream)  # still silent
    finally:
        import contextlib

        with contextlib.suppress(OSError):
            stream.close()


def test_spring_survives_pymatgen_int_typing(tmp_path):
    from vpmdk_core.settings import incar as incar_module

    path = tmp_path / "INCAR"
    path.write_text("SPRING = -5.5\n")
    mapping = {"SPRING": -5}  # what real pymatgen produces
    incar_module._repair_mistyped_real_tags(mapping, str(path))
    assert mapping["SPRING"] == -5.5
    # And the guard accepts the repaired mapping.
    incar_module._reject_truncated_integer_tags(mapping, str(path))

    # Integral SPRING stays as-is; the stub's string typing is tolerated.
    path.write_text("SPRING = -5\n")
    mapping = {"SPRING": -5}
    incar_module._repair_mistyped_real_tags(mapping, str(path))
    assert mapping["SPRING"] == -5
    text_mapping = {"SPRING": "-5.5"}
    incar_module._repair_mistyped_real_tags(text_mapping, str(path))
    assert text_mapping["SPRING"] == "-5.5"


def test_integer_semantic_float_typed_tags_reject_fractions(tmp_path):
    from vpmdk_core.settings import incar as incar_module

    for tag, value in (("NHC_NCHAINS", 2.7), ("IOPT", 7.5), ("ICHAIN", 0.7)):
        path = tmp_path / "INCAR"
        path.write_text(f"{tag} = {value}\n")
        with pytest.raises(ValueError, match="not an integer"):
            incar_module._reject_truncated_integer_tags({tag: value}, str(path))

    # Integral spellings stay legal.
    path = tmp_path / "INCAR"
    path.write_text("NHC_NCHAINS = 3.0\n")
    incar_module._reject_truncated_integer_tags({"NHC_NCHAINS": 3.0}, str(path))


def test_relative_socket_path_from_deleted_cwd_is_a_clean_error(monkeypatch):
    import vpmdk_protocol as protocol_module

    def deleted_cwd():
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(protocol_module.os, "getcwd", deleted_cwd)
    with pytest.raises(ValueError, match="no longer exists"):
        protocol_module.resolve_socket_path("rel.sock")
    # An absolute path needs no cwd and keeps working.
    assert protocol_module.resolve_socket_path("/tmp/abs.sock") == "/tmp/abs.sock"


def test_species_beyond_model_coverage_is_an_input_error(tmp_path):
    from ase import Atoms

    from vpmdk_core import cli as cli_module

    curium = Atoms("Cm2", positions=[[0, 0, 0], [0, 0, 2.0]])
    with pytest.raises(vpmdk.WorkdirInputError, match="Z=96"):
        cli_module._check_backend_species_coverage(curium, {"MLP": "CHGNET"})

    # Z=94 (Pu) is the last covered element; other backends skip the check.
    plutonium = Atoms("Pu", positions=[[0, 0, 0]])
    cli_module._check_backend_species_coverage(plutonium, {"MLP": "CHGNET"})
    cli_module._check_backend_species_coverage(curium, {"MLP": "MACE"})

    # End to end: the gate fires at input time, before any computation.
    (tmp_path / "POSCAR").write_text(
        "Cm2\n1.0\n6 0 0\n0 6 0\n0 0 6\nCm\n2\nDirect\n0.1 0.1 0.1\n0.5 0.5 0.5\n"
    )
    (tmp_path / "INCAR").write_text("NSW = 0\nIBRION = -1\n")
    (tmp_path / "BCAR").write_text("MLP=CHGNET\n")
    with pytest.raises(vpmdk.WorkdirInputError, match="coverage"):
        vpmdk.run_workdir(str(tmp_path), calculator=DummyCalculator())


def test_serve_stdout_death_cannot_override_the_exit_code(monkeypatch):
    import vpmdk_core.server as server_module

    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    broken = os.fdopen(write_fd, "w")
    monkeypatch.setattr(sys, "stdout", broken)
    print("buffered success line")  # sits in the buffer, reader is gone
    server_module._drain_stream_guarded(sys.stdout)  # must not raise
    # After the guard the stream points at /dev/null, so even a BARE flush --
    # the exact operation CPython's finalization performs -- cannot raise.
    print("late line")
    sys.stdout.flush()


def test_fractional_grid_and_pseudo_scf_tags_are_rejected(tmp_path):
    from vpmdk_core.settings import incar as incar_module

    for tag in ("NGXF", "NGX", "NELM", "NELMIN", "NELMDL"):
        path = tmp_path / "INCAR"
        path.write_text(f"{tag} = 100.5\n")
        with pytest.raises(ValueError, match="not an integer"):
            incar_module._reject_truncated_integer_tags({tag: 100}, str(path))

    path = tmp_path / "INCAR"
    path.write_text("NGXF = 100\n")
    incar_module._reject_truncated_integer_tags({"NGXF": 100}, str(path))


def test_species_gate_covers_matris_declared_sets_and_inheriting_requests():
    from ase import Atoms
    from types import SimpleNamespace

    from vpmdk_core import cli as cli_module

    americium = Atoms("Am", positions=[[0, 0, 0]])
    with pytest.raises(vpmdk.WorkdirInputError, match="Z=95"):
        cli_module._check_backend_species_coverage(americium, {"MLP": "MATRIS"})

    # (c) inheriting request: empty BCAR + resident tags select the backend.
    with pytest.raises(vpmdk.WorkdirInputError, match="Z=95"):
        cli_module._check_backend_species_coverage(
            americium, {}, backend_tags={"MLP": "MATRIS"}
        )
    # An inheriting request under a covered structure stays accepted.
    cli_module._check_backend_species_coverage(
        Atoms("Pu", positions=[[0, 0, 0]]), {}, backend_tags={"MLP": "MATRIS"}
    )

    # (b) model-declared element table (matgl shape): holes are respected.
    calculator = DummyCalculator()
    calculator.model = SimpleNamespace(element_types=("H", "He", "Bi"))
    polonium = Atoms("Po", positions=[[0, 0, 0]])
    with pytest.raises(vpmdk.WorkdirInputError, match="Po"):
        cli_module._check_model_declared_species_coverage(polonium, calculator)
    cli_module._check_model_declared_species_coverage(
        Atoms("Bi", positions=[[0, 0, 0]]), calculator
    )
    # A model that declares nothing is skipped.
    cli_module._check_model_declared_species_coverage(polonium, DummyCalculator())


def test_neb_without_lclimb_discloses_the_vtst_default(tmp_path, capsys):
    cell = "5 0 0\n0 5 0\n0 0 5"
    for index, x in enumerate((0.2, 0.3, 0.4)):
        image_dir = tmp_path / f"0{index}"
        image_dir.mkdir()
        (image_dir / "POSCAR").write_text(
            f"AB\n1.0\n{cell}\nH He\n1 1\nDirect\n0 0 0\n{x} 0 0\n"
        )
    (tmp_path / "BCAR").write_text("MLP = CHGNET\n")
    (tmp_path / "INCAR").write_text(
        "IMAGES = 1\nSPRING = -5\nNSW = 2\nIBRION = 1\nEDIFFG = -0.5\n"
    )
    vpmdk.run_workdir(str(tmp_path), calculator=DummyCalculator())
    out = capsys.readouterr().out
    assert "LCLIMB" in out and "VTST" in out and "climbing" in out

    # An explicit value -- either way -- stays silent.
    (tmp_path / "INCAR").write_text(
        "IMAGES = 1\nSPRING = -5\nNSW = 2\nIBRION = 1\nEDIFFG = -0.5\n"
        "LCLIMB = .TRUE.\n"
    )
    vpmdk.run_workdir(str(tmp_path), calculator=DummyCalculator())
    assert "Warning: LCLIMB" not in capsys.readouterr().out


def test_spring_repair_survives_real_pymatgen_setitem(tmp_path):
    from collections import UserDict

    from vpmdk_core.settings import incar as incar_module

    class RefloorDict(UserDict):
        def __setitem__(self, key, value):
            super().__setitem__(key, int(float(value)))  # proc_val analogue

    path = tmp_path / "INCAR"
    path.write_text("SPRING = -5.5\n")
    mapping = RefloorDict()
    dict.__setitem__(mapping.data, "SPRING", -5)
    incar_module._repair_mistyped_real_tags(mapping, str(path))
    assert mapping.get("SPRING") == -5.5


def test_tiny_positive_damping_times_are_rejected():
    with pytest.raises(vpmdk.WorkdirInputError, match="underflows"):
        vpmdk._estimate_tdamp(1e-300, 2.0, {})
    with pytest.raises(vpmdk.WorkdirInputError, match="underflows"):
        vpmdk._estimate_tdamp(None, 2.0, {"NHC_PERIOD": 1e-301})
    # Ordinary damping times keep working (with at most the stiffness warn).
    assert vpmdk._estimate_tdamp(100.0, 2.0, {}) == 100.0


def test_potcar_pomass_divergence_is_disclosed(tmp_path, capsys):
    from vpmdk_core.io import inputs as inputs_module

    potcar = tmp_path / "POTCAR"
    potcar.write_text(
        "  PAW_PBE H 15Jun2001\n"
        "  TITEL  = PAW_PBE H 15Jun2001\n"
        "  POMASS =    2.014; ZVAL   =    1.000\n"
    )
    inputs_module._warn_potcar_pomass_ignored(str(potcar), None)
    out = capsys.readouterr().out
    assert "POMASS" in out and "2.014" in out and "1.008" in out

    # A standard-mass POTCAR stays silent.
    potcar.write_text(
        "  TITEL  = PAW_PBE H 15Jun2001\n"
        "  POMASS =    1.000; ZVAL   =    1.000\n"
    )
    inputs_module._warn_potcar_pomass_ignored(str(potcar), None)
    assert "POMASS is not read" not in capsys.readouterr().out


def test_magmom_inertness_is_disclosed(capsys):
    from ase import Atoms

    from vpmdk_core.io import inputs as inputs_module

    atoms = Atoms("Fe2", positions=[[0, 0, 0], [0, 0, 2.0]])
    inputs_module._apply_initial_magnetization(atoms, {"MAGMOM": "3.0 -3.0"})
    out = capsys.readouterr().out
    assert "MAGMOM" in out and "identical results" in out

    # All-zero moments (the do-nothing spelling) stay silent.
    inputs_module._apply_initial_magnetization(atoms, {"MAGMOM": "0.0 0.0"})
    assert "identical results" not in capsys.readouterr().out


def test_serve_stdout_guard_survives_a_closed_fd(monkeypatch):
    import vpmdk_core.server as server_module

    monkeypatch.setattr(sys, "stdout", None)
    server_module._drain_stream_guarded(sys.stdout)  # must not raise


def test_device_index_edge_cases(tmp_path):
    import vpmdk_core.server as server_module

    assert server_module._resolve_backend_device("CHGNET", ":0") == ":0"
    assert server_module._resolve_backend_device("CHGNET", "cpu:1") == "cpu"
    assert server_module._resolve_backend_device("CHGNET", "cpu:0") == "cpu"
    assert server_module._resolve_backend_device("CHGNET", "cuda:0") == "cuda"
    assert server_module._resolve_backend_device("CHGNET", "cuda:1") == "cuda:1"


def test_spring_magnitude_is_bounded(tmp_path):
    with pytest.raises(vpmdk.WorkdirInputError, match="supported magnitude"):
        vpmdk._parse_neb_spring_constant({"SPRING": -1e300})
    assert vpmdk._parse_neb_spring_constant({"SPRING": -5.5}) == 5.5


def test_nsw_without_ibrion_discloses_the_vasp_default(capsys):
    from vpmdk_core.settings import incar as incar_module

    incar_module._load_incar_settings({"NSW": 5})
    out = capsys.readouterr().out
    assert "IBRION omitted" in out and "SINGLE POINT" in out

    incar_module._load_incar_settings({"NSW": 5, "IBRION": 2})
    assert "IBRION omitted" not in capsys.readouterr().out
    incar_module._load_incar_settings({"NSW": 0})
    assert "IBRION omitted" not in capsys.readouterr().out


def test_coincident_atom_guard_is_fast_and_lean_for_large_cells():
    import time as time_module

    import numpy as np
    from ase import Atoms

    from vpmdk_core.io import inputs as inputs_module

    rng = np.random.default_rng(11)
    atoms = Atoms(
        "H4096",
        scaled_positions=rng.uniform(0.0, 1.0, (4096, 3)),
        cell=np.eye(3) * 40.0,
        pbc=True,
    )
    started = time_module.monotonic()
    inputs_module._validate_finite_geometry(atoms)
    assert time_module.monotonic() - started < 5.0

    # Correctness is preserved at scale: a wrapped-boundary coincidence deep
    # in the list is still found.
    positions = atoms.get_scaled_positions()
    positions[4095] = np.mod(positions[7] + [0.0, 0.0, 1.0 - 1e-9], 1.0)
    atoms.set_scaled_positions(positions)
    with pytest.raises(ValueError, match="occupy the same site"):
        inputs_module._validate_finite_geometry(atoms)


def test_total_repeat_expansion_is_bounded(tmp_path):
    import time as time_module

    from vpmdk_core.settings import incar as incar_module

    path = tmp_path / "INCAR"
    path.write_text("MAGMOM = " + "1000000*1.0 " * 10 + "\n")
    started = time_module.monotonic()
    with pytest.raises(ValueError, match="expands to"):
        incar_module._load_incar(str(path))
    assert time_module.monotonic() - started < 5.0

    path.write_text("MAGMOM = 4*1.0 2*-0.5\n")
    incar_module._load_incar(str(path))


def test_nested_repeat_expansion_is_bounded(tmp_path):
    import time as time_module

    from vpmdk_core.settings import incar as incar_module

    for value in ("100000*100000*1.0", "1*999999999*2.0", "2*1000000000*1.0"):
        path = tmp_path / "INCAR"
        path.write_text(f"MAGMOM = {value}\n")
        started = time_module.monotonic()
        with pytest.raises(ValueError, match="repeat token"):
            incar_module._load_incar(str(path))
        assert time_module.monotonic() - started < 5.0

    # The legal nested spelling keeps parsing (pymatgen documents 2*3*1.0;
    # the conftest pymatgen stub does not expand repeats, so the expansion
    # itself is asserted on VPMDK's own expander below).
    path = tmp_path / "INCAR"
    path.write_text("MAGMOM = 2*3*1.0\n")
    incar_module._load_incar(str(path))

    # Defense in depth: VPMDK's own recursive expander applied its cap PER
    # NESTING LEVEL, so '1000*1000*1000*1.0' (1e9 entries) passed three
    # individually-legal levels. The product across levels is now bounded,
    # and so is the total across tokens.
    from vpmdk_core.io import inputs as inputs_module

    with pytest.raises(ValueError, match="expands to"):
        inputs_module._parse_magmom_values("1000*1000*1000*1.0")
    with pytest.raises(ValueError, match="expands to"):
        inputs_module._parse_magmom_values("1000000*1.0 1000000*2.0")
    assert inputs_module._parse_magmom_values("2*3*1.0") == [1.0] * 6


def test_finite_difference_displacement_underflow_is_rejected():
    from vpmdk_core.runtime import single as single_module

    for bad in (1e-30, 1e-8, 0.0, -0.01, float("nan"), float("inf")):
        with pytest.raises(vpmdk.WorkdirInputError, match="displacement"):
            single_module._finite_difference_force_constants(
                None, None, displacement=bad, nfree=2
            )
        with pytest.raises(vpmdk.WorkdirInputError, match="displacement"):
            single_module._finite_difference_force_response(
                None, 0, None, displacement=bad, nfree=2
            )
        with pytest.raises(vpmdk.WorkdirInputError, match="displacement"):
            single_module._symmetry_reduced_finite_difference_force_constants(
                None, None, displacement=bad, nfree=2
            )

    # The BCAR reader (IBRION=7/8 path) mirrors the same floor.
    with pytest.raises(ValueError, match="at least"):
        single_module._force_constants_displacement_from_bcar(
            {"FORCE_CONSTANTS_DISPLACEMENT": "1e-30"}
        )
    # The smallest displacement in real use keeps working.
    assert single_module._force_constants_displacement_from_bcar(
        {"FORCE_CONSTANTS_DISPLACEMENT": "0.001"}
    ) == 0.001


def test_pstress_magnitude_is_bounded(tmp_path):
    from vpmdk_core.settings import incar as incar_module

    with pytest.raises(ValueError, match="exceeds the supported magnitude"):
        incar_module._load_incar_settings({"PSTRESS": 1e300})

    settings = incar_module._load_incar_settings({"PSTRESS": 500.0})
    assert settings.pstress == 500.0


@pytest.mark.parametrize(
    "parsed,token,rejected",
    [
        ("NSW", "1e5", True),
        ("MDALGO", "1e400", True),
        ("IMAGES", "1e300", True),
        ("NSW", "100", False),
        ("NSW", "100.", False),
        ("NSW", "100 # comment stripped upstream", False),
    ],
)
def test_scientific_notation_integer_tags_are_rejected_not_truncated(
    tmp_path, parsed: str, token: str, rejected: bool
):
    from vpmdk_core.settings import incar as incar_module

    path = tmp_path / "INCAR"
    path.write_text(f"{parsed} = {token}\n")
    truncated = int(re.match(r"^-?[0-9]+", token).group(0))

    if rejected:
        with pytest.raises(ValueError):
            incar_module._reject_truncated_integer_tags({parsed: truncated}, str(path))
    else:
        incar_module._reject_truncated_integer_tags({parsed: truncated}, str(path))

    # Bool- and list-typed tags are not numbers and must be left alone. Use a
    # separate file: the guard scans the INCAR text itself, so a rejectable value
    # in the same file would fire regardless of the mapping passed in.
    other = tmp_path / "OTHER"
    other.write_text("LWAVE = .FALSE.\nMAGMOM = 4*1.0\nSYSTEM = D2O sample\n")
    incar_module._reject_truncated_integer_tags({"LWAVE": False}, str(other))
    incar_module._reject_truncated_integer_tags({"MAGMOM": [1.0, 1.0]}, str(other))


def test_scientific_notation_survives_real_pymatgen(tmp_path):
    # End-to-end with the REAL library (the suite otherwise stubs pymatgen), so this
    # pins the actual production behavior rather than the stub's.
    workdir = tmp_path / "run"
    workdir.mkdir()
    (workdir / "INCAR").write_text("NSW = 1e5\nIBRION = 2\n")
    script = (
        "import sys\n"
        "from vpmdk_core.settings import incar as m\n"
        "try:\n"
        "    m._load_incar(sys.argv[1])\n"
        "except ValueError as exc:\n"
        "    print('REJECTED', exc)\n"
        "else:\n"
        "    print('ACCEPTED')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(workdir / "INCAR")],
        capture_output=True,
        text=True,
        env={**os.environ, "VPMDK_TEST_REAL_PYMATGEN": "1"},
    )
    if "ModuleNotFoundError" in completed.stderr:
        pytest.skip("real pymatgen is not installed")
    assert "REJECTED" in completed.stdout, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    "tag,token,parsed,rejected",
    [
        ("NSW", "1D3", 1, True),
        ("EDIFFG", "-1.0D-03", -1.0, True),
        ("POTIM", "2.0D-03", 2.0, True),
        ("EDIFFG", "-1.0E-03", -0.001, False),
        ("POTIM", "2.0", 2.0, False),
        ("NSW", "1000", 1000, False),
    ],
)
def test_fortran_d_exponent_incar_values_are_rejected_not_truncated(
    tmp_path, tag: str, token: str, parsed, rejected: bool
):
    from vpmdk_core.settings import incar as incar_module

    path = tmp_path / "INCAR"
    path.write_text(f"{tag} = {token}\n")

    if rejected:
        with pytest.raises(ValueError):
            incar_module._reject_truncated_integer_tags({tag: parsed}, str(path))
    else:
        incar_module._reject_truncated_integer_tags({tag: parsed}, str(path))


@pytest.mark.parametrize(
    "body,rejected",
    [
        ("IBRION = 2 ; NSW = 50 ; EDIFFG = -1.0D-03\n", True),
        ("EDIFFG = -0.01 ; POTIM = 2.0D-01\n", True),
        # Backslash line continuations are joined by pymatgen too.
        ("IBRION = 2\nPOTIM = \\\n2.0D-01\n", True),
        # ...and correctly written compact INCARs must still be accepted.
        ("IBRION = 2 ; NSW = 50 ; EDIFFG = -1.0E-03\n", False),
        ("SYSTEM = D2O sample ; NSW = 20\n", False),
        # A value after a comment marker is not a tag at all for either reader.
        ("NSW = 20 # note ; EDIFFG = -1.0D-03\n", False),
        # A repeated tag keeps its LAST value in pymatgen; the earlier one must
        # not be compared against it.
        ("NSW = 5\nNSW = 20\n", False),
    ],
)
def test_multiple_incar_tags_on_one_line_are_checked_too(
    tmp_path, body: str, rejected: bool
):
    from vpmdk_core.settings import incar as incar_module

    path = tmp_path / "INCAR"
    path.write_text(body)

    if rejected:
        with pytest.raises(ValueError, match="read as"):
            incar_module._load_incar(str(path))
    else:
        assert dict(incar_module._load_incar(str(path))) is not None


def test_raw_incar_assignments_matches_pymatgens_own_tokenizer(tmp_path):
    # The guard can only compare a parsed value against what the user WROTE, so
    # the raw reader has to see the file the way Incar.from_str does: comments
    # stripped per line, '\'-continuations joined, several assignments per line,
    # quoted values kept whole, last occurrence wins.
    from vpmdk_core.settings import incar as incar_module

    path = tmp_path / "INCAR"
    path.write_text(
        'IBRION = 2 ; NSW = 50\n'
        "EDIFFG = -1.0E-03 # trailing ; POTIM = 9\n"
        "POTIM = \\\n"
        "  0.5\n"
        'SYSTEM = "a ; b"\n'
        "NSW = 60\n"
        "ISIF = 3 ! bang comment\n"
    )

    assert incar_module._raw_incar_assignments(str(path)) == {
        "IBRION": "2",
        "NSW": "60",
        "EDIFFG": "-1.0E-03",
        "POTIM": "0.5",
        "SYSTEM": "a ; b",
        "ISIF": "3",
    }


def test_semicolon_separated_incar_survives_real_pymatgen(tmp_path):
    # End-to-end with the REAL library: the suite's pymatgen stub keeps values as
    # strings, so only a subprocess can show that `IBRION = 2 ; NSW = 50 ;
    # EDIFFG = -1.0D-03` was accepted and read as EDIFFG=-1.0 (a relaxation that
    # stops at fmax<1.0 eV/A and writes an unconverged CONTCAR with exit 0),
    # while the byte-identical values one per line were rejected.
    script = (
        "import sys\n"
        "from vpmdk_core.settings import incar as m\n"
        "try:\n"
        "    parsed = m._load_incar(sys.argv[1])\n"
        "except ValueError as exc:\n"
        "    print('REJECTED', exc)\n"
        "else:\n"
        "    print('ACCEPTED', dict(parsed))\n"
    )

    def run(body: str, name: str) -> subprocess.CompletedProcess:
        path = tmp_path / name
        path.write_text(body)
        return subprocess.run(
            [sys.executable, "-c", script, str(path)],
            capture_output=True,
            text=True,
            env={**os.environ, "VPMDK_TEST_REAL_PYMATGEN": "1"},
        )

    mangled = run("IBRION = 2 ; NSW = 50 ; EDIFFG = -1.0D-03\n", "INCAR_semi")
    if "ModuleNotFoundError" in mangled.stderr:
        pytest.skip("real pymatgen is not installed")
    assert "REJECTED" in mangled.stdout, mangled.stdout + mangled.stderr

    # The int truncation needs the real library too: only real pymatgen types
    # NSW, and it reads `1e5` as 1 -- one ionic step instead of 100000.
    truncated = run("IBRION = 2; NSW = 1e5\n", "INCAR_e5")
    assert "REJECTED" in truncated.stdout, truncated.stdout + truncated.stderr

    ok = run("IBRION = 2 ; NSW = 50 ; EDIFFG = -1.0E-03\n", "INCAR_ok")
    assert "ACCEPTED" in ok.stdout, ok.stdout + ok.stderr
    assert "'EDIFFG': -0.001" in ok.stdout


def test_bcar_with_a_utf8_bom_still_selects_the_requested_backend(tmp_path):
    path = tmp_path / "BCAR"
    path.write_bytes("﻿MLP = MACE\nDEVICE = cpu\n".encode("utf-8"))

    tags = vpmdk.parse_key_value_file(str(path))

    assert list(tags) == ["MLP", "DEVICE"]
    assert vpmdk._resolve_mlp_tag(tags) == "MACE"


def test_out_of_range_symprec_is_an_input_error_not_a_retryable_failure():
    pytest.importorskip("spglib")
    from ase.build import bulk

    from vpmdk_core.runtime import single as single_module

    atoms = bulk("Si", "diamond", a=5.43)

    with pytest.raises(vpmdk.WorkdirInputError, match="SYMPREC"):
        single_module._symmetry_operations(atoms, symprec=1e5)

    operations = single_module._symmetry_operations(atoms, symprec=1e-5)
    assert operations


@pytest.mark.parametrize(
    "body,rejected",
    [
        # Tags pymatgen leaves as raw text reach VPMDK's own _NUMERIC_RE, which also
        # stops at the Fortran 'D', so the thermostat ran at 1 instead of 100.
        ("NHC_PERIOD = 1D2\n", True),
        ("CSVR_PERIOD = 5D1\n", True),
        # pymatgen can even turn a D exponent into a LIST.
        ("LANGEVIN_GAMMA = 1D1\n", True),
        ("NHC_PERIOD = 100\n", False),
        ("CSVR_PERIOD = 5E1\n", False),
        ("LANGEVIN_GAMMA = 1.0 2.0\n", False),
        # Non-numeric text that merely contains "D<digit>" must not be touched.
        ("SYSTEM = D2O sample\n", False),
        ("IOPT = 3\n", False),
    ],
)
def test_fortran_d_exponent_is_caught_for_untyped_incar_tags(
    tmp_path, body: str, rejected: bool
):
    from vpmdk_core.settings import incar as incar_module

    path = tmp_path / "INCAR"
    path.write_text(body)

    if rejected:
        with pytest.raises(ValueError, match="would be read as"):
            incar_module._load_incar(str(path))
    else:
        assert dict(incar_module._load_incar(str(path)))


def test_missing_backend_forces_are_reported_instead_of_written_as_zeros():
    from vpmdk_core.io import vasp_compat

    class ReturnsNone:
        def __len__(self):
            return 2

        def get_forces(self, apply_constraint=True):
            return None

    class Raises:
        def __len__(self):
            return 2

        def get_forces(self, apply_constraint=True):
            raise NotImplementedError("forces")

    with pytest.raises(RuntimeError, match="no usable forces"):
        vasp_compat._safe_get_forces(ReturnsNone())
    with pytest.raises(RuntimeError, match="per-atom forces"):
        vasp_compat._safe_get_forces(Raises())

    class Works:
        def __len__(self):
            return 2

        def get_forces(self, apply_constraint=True):
            return [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]

    assert vasp_compat._safe_get_forces(Works()).shape == (2, 3)


def test_energy_only_backend_is_rejected_before_anything_is_computed(tmp_path: Path):
    # Same defect, one layer up: MATRIS_TASK=e is a permanent INPUT problem, so
    # it must exit 1 up front rather than surface from deep inside the run as a
    # calculation failure (exit 2, documented RETRYABLE) after the whole
    # calculation has been paid for. The capability model derives forces=False
    # from the very tag that configures the backend.
    from vpmdk_core import cli as cli_module

    workdir = tmp_path / "energy_only"
    workdir.mkdir()
    (workdir / "INCAR").write_text("IBRION = -1\nNSW = 0\nISIF = 3\n")
    (workdir / "BCAR").write_text("MLP = MATRIS\nMATRIS_TASK = e\n")

    # No POSCAR on purpose: the gate must fire before any input is even read,
    # and certainly before a backend is built.
    with pytest.raises(vpmdk.WorkdirInputError, match="energy only"):
        vpmdk.run_workdir(str(workdir))
    assert not (workdir / "OUTCAR").exists()

    settings = vpmdk._load_incar_settings({"IBRION": 2, "NSW": 5, "ISIF": 2})

    # A force-capable but stress-less configuration keeps running -- ISIF
    # defaults to 2 for every relaxation, so rejecting it would break working
    # ion-only setups -- but the omission is announced instead of silent.
    cli_module._check_backend_output_capabilities(
        {"MLP": "MATRIS", "MATRIS_TASK": "ef"}, settings
    )
    cli_module._check_backend_output_capabilities({"MLP": "CHGNET"}, settings)


def test_energy_only_backend_gate_announces_a_missing_stress_block(
    tmp_path: Path, capsys
):
    from vpmdk_core import cli as cli_module

    settings = vpmdk._load_incar_settings({"IBRION": 2, "NSW": 5, "ISIF": 2})

    cli_module._check_backend_output_capabilities(
        {"MLP": "MATRIS", "MATRIS_TASK": "ef"}, settings
    )
    warned = capsys.readouterr().out
    assert "does not provide stress" in warned

    cli_module._check_backend_output_capabilities(
        {"MLP": "MATRIS", "MATRIS_TASK": "efs"}, settings
    )
    assert "does not provide stress" not in capsys.readouterr().out

    # An unknown backend must not be pre-empted here: the selector error belongs
    # to _build_workdir_calculator, with its own message.
    cli_module._check_backend_output_capabilities({"MLP": "NO_SUCH_BACKEND"}, settings)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "body,parsed,swallowed",
    [
        ("SYSTEM =\nIBRION = 2\nNSW = 50\n", {"SYSTEM": "IBRION = 2", "NSW": 50}, True),
        # A comment line in between does not help: comments are blanked first.
        ("SYSTEM =\n# note\nIBRION = 2\n", {"SYSTEM": "IBRION = 2"}, True),
        # Other doors of the same shape, including an int-typed swallower.
        ("IBRION = 0\nENCUT =\nNSW = 100\n", {"IBRION": 0, "ENCUT": "NSW = 100"}, True),
        ("NSW =\nIBRION = 2\n", {"NSW": "IBRION = 2"}, True),
        ("LWAVE =\nTEBEG = 900\nIBRION = 0\n", {"LWAVE": True, "IBRION": 0}, True),
        ("MAGMOM =\nTEBEG = 900\n", {"MAGMOM": [True, 900]}, True),
        ("LCHARG =\nTEEND = 900\nIBRION = 0\n", {"LCHARG": True, "IBRION": 0}, True),
        # ...and a bool tag that actually HAS a value swallows nothing.
        (
            "LWAVE = .FALSE.\nTEBEG = 900\n",
            {"LWAVE": False, "TEBEG": 900.0},
            False,
        ),
        # A blank tag on the LAST line swallows nothing; the parser just drops it.
        ("IBRION = 2\nNSW = 50\nSYSTEM =\n", {"IBRION": 2, "NSW": 50}, False),
        # Ordinary files, including the compact styles the previous round added.
        ("SYSTEM = test\nIBRION = 2\n", {"SYSTEM": "test", "IBRION": 2}, False),
        ("IBRION = 2 ; NSW = 50\n", {"IBRION": 2, "NSW": 50}, False),
        ("IBRION = 2\nPOTIM = \\\n0.5\n", {"IBRION": 2, "POTIM": 0.5}, False),
        ("NSW = 5\nNSW = 20\n", {"NSW": 20}, False),
        # A parser that splits lines differently than this reader (conftest's
        # Incar stub does not split on ';') must NOT look like a swallow: the
        # check compares ONE tag's raw and parsed value, never "a tag I can see
        # is missing", which would reject every compact INCAR on such a parser.
        ("IBRION = 2 ; NSW = 50 ; EDIFFG = -1.0E-03\n", {"IBRION": 2}, False),
        # A blank tag followed by a stray non-assignment line loses nothing (the
        # deferred `NSW =` / bare `50` input), so it stays accepted.
        ("NSW =\n50\nIBRION = 2\n", {"NSW": 50, "IBRION": 2}, False),
    ],
)
def test_incar_tag_swallowed_by_an_empty_value_is_rejected(
    tmp_path, body: str, parsed: dict, swallowed: bool
):
    from vpmdk_core.settings import incar as incar_module

    path = tmp_path / "INCAR"
    path.write_text(body)

    if swallowed:
        with pytest.raises(ValueError, match="empty value"):
            incar_module._reject_swallowed_incar_tags(parsed, str(path))
    else:
        incar_module._reject_swallowed_incar_tags(parsed, str(path))


def test_incar_swallowed_tag_survives_real_pymatgen(tmp_path):
    # End-to-end with the REAL library: the same INCAR with and without the
    # blank line, so the silent mode change is visible as a parsed-settings diff.
    script = (
        "import sys\n"
        "from vpmdk_core.settings import incar as m\n"
        "try:\n"
        "    parsed = m._load_incar(sys.argv[1])\n"
        "except ValueError as exc:\n"
        "    print('REJECTED', exc)\n"
        "else:\n"
        "    settings = m._load_incar_settings(parsed)\n"
        "    print('ACCEPTED ibrion=%d nsw=%d' % (settings.ibrion, settings.nsw))\n"
    )

    def run(body: str, name: str) -> subprocess.CompletedProcess:
        path = tmp_path / name
        path.write_text(body)
        return subprocess.run(
            [sys.executable, "-c", script, str(path)],
            capture_output=True,
            text=True,
            env={**os.environ, "VPMDK_TEST_REAL_PYMATGEN": "1"},
        )

    # The guard is COMPARISON-based, so it follows the installed parser.
    # pymatgen >= 2026 lets a blank value swallow the next assignment (the
    # silent mode change this test exists for) and the guard must reject;
    # older releases (2025.10.7 on Python 3.10 CI) keep SYSTEM='' and parse
    # the following tags intact, so nothing is swallowed and accepting the
    # written values IS the correct outcome there. Probe the library
    # explicitly (an outcome-based probe cannot discriminate in the
    # bool-swallower case, where a broken guard still prints the same
    # ibrion/nsw and only TEBEG is silently lost).
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pymatgen.io.vasp import Incar\n"
            "d = dict(Incar.from_str('SYSTEM =\\nIBRION = 2\\n'))\n"
            "print('SWALLOWS:', 'IBRION' not in d)\n",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "VPMDK_TEST_REAL_PYMATGEN": "1"},
    )
    if "ModuleNotFoundError" in probe.stderr:
        pytest.skip("real pymatgen is not installed")
    parser_swallows = "SWALLOWS: True" in probe.stdout

    swallowed = run("SYSTEM =\nIBRION = 2\nNSW = 50\nEDIFFG = -0.02\n", "INCAR_blank")
    if "ModuleNotFoundError" in swallowed.stderr:
        pytest.skip("real pymatgen is not installed")
    if parser_swallows:
        assert "REJECTED" in swallowed.stdout, swallowed.stdout + swallowed.stderr
        assert "SYSTEM" in swallowed.stdout and "IBRION" in swallowed.stdout
    else:
        assert "ACCEPTED ibrion=2 nsw=50" in swallowed.stdout, (
            swallowed.stdout + swallowed.stderr
        )

    ok = run("SYSTEM = test\nIBRION = 2\nNSW = 50\nEDIFFG = -0.02\n", "INCAR_ok")
    assert "ACCEPTED ibrion=2 nsw=50" in ok.stdout, ok.stdout + ok.stderr

    # The pre-fix behavior this pins down: without the guard the blank-value file
    # parses to ibrion=-1 (single point) while the byte-identical file with a
    # value relaxes.
    tail = run("IBRION = 2\nNSW = 50\nSYSTEM =\n", "INCAR_blank_last")
    assert "ACCEPTED ibrion=2 nsw=50" in tail.stdout, tail.stdout + tail.stderr

    typed = run("LWAVE =\nTEBEG = 900\nIBRION = 0\nNSW = 3\n", "INCAR_bool")
    if parser_swallows:
        assert "REJECTED" in typed.stdout, typed.stdout + typed.stderr
        assert "TEBEG" in typed.stdout and "LWAVE" in typed.stdout
    else:
        assert "ACCEPTED ibrion=0 nsw=3" in typed.stdout, (
            typed.stdout + typed.stderr
        )


def test_resident_neb_images_get_a_per_image_result_cache():
    from vpmdk_core.runtime import neb as neb_module

    delegate = DummyCalculator()
    image_a = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 3.0, pbc=True)
    image_b = Atoms("H", positions=[[0.5, 0.0, 0.0]], cell=np.eye(3) * 3.0, pbc=True)
    image_a.calc = neb_module._PerImageResultCache(delegate)
    image_b.calc = neb_module._PerImageResultCache(delegate)

    energy_a = image_a.get_potential_energy()
    forces_a = image_a.get_forces()
    image_b.get_potential_energy()
    evaluations = delegate.called
    # Interleaved access must NOT evict image A's cache (the defect): touching
    # A again after B costs zero delegate evaluations.
    assert image_a.get_potential_energy() == energy_a
    assert np.allclose(image_a.get_forces(), forces_a)
    assert delegate.called == evaluations
    # A genuine geometry change still recomputes.
    image_a.positions[0, 0] += 0.1
    image_a.get_potential_energy()
    assert delegate.called == evaluations + 1


def test_per_image_result_cache_stores_raw_forces_for_constrained_images():
    # The cache evaluates through a constraint-free copy: calculators cache RAW
    # forces (the constraint adjustment belongs to the Atoms layer, where the
    # caller applies it either way), so a constrained image must not poison its
    # calculator-level cache with adjusted forces.
    from ase.calculators.calculator import Calculator, all_changes
    from ase.constraints import FixAtoms
    from vpmdk_core.runtime import neb as neb_module

    class OnesCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {
                "energy": 1.0,
                "forces": np.ones((len(atoms), 3)),
            }

    image = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.8]],
        cell=np.eye(3) * 3.0,
        pbc=True,
    )
    image.set_constraint(FixAtoms(indices=[0]))
    image.calc = neb_module._PerImageResultCache(OnesCalculator())

    constrained = image.get_forces()
    assert np.allclose(constrained[0], 0.0)
    assert np.allclose(constrained[1], 1.0)
    assert np.allclose(image.calc.results["forces"], 1.0)


def test_non_regular_potcar_in_a_neb_workdir_is_an_input_error(
    tmp_path, prepare_inputs
):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "1", "IBRION": "2", "IMAGES": "1"},
    )
    _write_numbered_neb_poscars(tmp_path)
    potcar = tmp_path / "POTCAR"
    if potcar.exists():
        potcar.unlink()
    potcar.mkdir()

    with pytest.raises(vpmdk.WorkdirInputError, match="Failed to read POTCAR"):
        vpmdk.run_workdir(str(tmp_path), calculator=DummyCalculator())


def test_run_workdir_attaches_per_image_caches_for_a_resident_calculator(
    tmp_path, prepare_inputs, monkeypatch
):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "1", "IBRION": "2", "IMAGES": "1"},
    )
    _write_numbered_neb_poscars(tmp_path)
    from vpmdk_core.runtime import neb as neb_module

    shared_calculator = DummyCalculator()
    actual_neb = vpmdk.NEB
    captured: dict[str, object] = {}

    class CapturingNEB:
        def __new__(cls, images, **kwargs):
            captured["calculators"] = [image.calc for image in images]
            return actual_neb(images, **kwargs)

    monkeypatch.setattr(vpmdk, "NEB", CapturingNEB)
    monkeypatch.setattr(vpmdk, "BFGS", DummyNEBOptimizer)

    vpmdk.run_workdir(str(tmp_path), calculator=shared_calculator)

    calculators = captured["calculators"]
    assert len(calculators) == 3
    assert all(
        isinstance(calc, neb_module._PerImageResultCache) for calc in calculators
    )
    assert all(calc._calculator is shared_calculator for calc in calculators)


def test_model_declared_species_gate_fires_in_oneshot_neb(
    tmp_path, prepare_inputs, monkeypatch
):
    from types import SimpleNamespace

    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "1", "IBRION": "2", "IMAGES": "1"},
    )
    _write_numbered_neb_poscars(tmp_path)

    declared = DummyCalculator()
    declared.model = SimpleNamespace(element_types=("H", "He"))
    monkeypatch.setattr(
        vpmdk,
        "_build_workdir_calculator",
        lambda bcar, *, structure, workdir_abs: declared,
    )

    # Optimizer branch (_build_neb_images).
    with pytest.raises(vpmdk.WorkdirInputError, match="element table"):
        vpmdk.run_workdir(str(tmp_path))

    # Per-image single-point branch.
    incar_text = (tmp_path / "INCAR").read_text()
    (tmp_path / "INCAR").write_text(
        incar_text.replace("NSW = 1", "NSW = 0").replace("IBRION = 2", "IBRION = -1")
    )
    with pytest.raises(vpmdk.WorkdirInputError, match="element table"):
        vpmdk.run_workdir(str(tmp_path))


def test_unbalanced_incar_quote_swallowed_tags_are_rejected(tmp_path):
    from vpmdk_core.settings import incar as incar_module

    path = tmp_path / "INCAR"
    path.write_text(
        'SYSTEM = "Cu bulk\nISPIN = 2\nIBRION = 2\nNSW = 200\nGGA = "PE"\n'
    )
    swallowed_parse = {
        "SYSTEM": 'Cu bulk\nISPIN = 2\nIBRION = 2\nNSW = 200\nGGA ='
    }
    with pytest.raises(ValueError, match="spanning multiple lines"):
        incar_module._reject_swallowed_incar_tags(swallowed_parse, str(path))

    # A BALANCED quoted value stays legal (no newline in the parsed string).
    path.write_text('SYSTEM = "a ; b"\nNSW = 5\n')
    incar_module._reject_swallowed_incar_tags({"SYSTEM": "a ; b", "NSW": 5}, str(path))


def test_neb_projections_line_has_exactly_two_fields_for_vtst(
    tmp_path, prepare_inputs, monkeypatch
):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "1", "IBRION": "2", "IMAGES": "1"},
    )
    _write_numbered_neb_poscars(tmp_path)
    monkeypatch.setattr(vpmdk, "BFGS", DummyNEBOptimizer)

    vpmdk.run_workdir(str(tmp_path), calculator=DummyCalculator())

    for image in ("00", "01", "02"):
        outcar = (tmp_path / image / "OUTCAR").read_text()
        lines = [
            line
            for line in outcar.splitlines()
            if "NEB: projections on to tangent" in line
        ]
        assert lines, image
        for line in lines:
            fields = line.split("(spring, REAL)")[1].split()
            assert len(fields) == 2, (image, line)
            float(fields[0])
            float(fields[1])


def test_unbalanced_quote_is_rejected_for_bool_typed_swallowers(tmp_path):
    from vpmdk_core.settings import incar as incar_module

    path = tmp_path / "INCAR"
    path.write_text(
        'ISTART = 0\nLWAVE = ".FALSE.\nLCHARG = .FALSE.\nNSW = 200\n'
        'IBRION = 2\nLREAL = "Auto"\n'
    )
    swallowed_parse = {"ISTART": 0, "LWAVE": False, "LREAL": "Auto"}
    with pytest.raises(ValueError, match="unbalanced quote"):
        incar_module._reject_swallowed_incar_tags(swallowed_parse, str(path))

    # Balanced quoted values keep parsing (raw reader strips the quotes, so
    # no leading quote survives to the guard).
    path.write_text('SYSTEM = "a ; b"\nNSW = 5\n')
    incar_module._load_incar(str(path))

    path.write_text('SYSTEM = "run #3"\nIBRION = 2\nNSW = 5\nEDIFFG = -0.05\n')
    incar_module._reject_swallowed_incar_tags(
        {"SYSTEM": '"run', "IBRION": 2, "NSW": 5, "EDIFFG": -0.05}, str(path)
    )
    incar_module._load_incar(str(path))

    path.write_text('IBRION = 2\nNSW = 5\nSYSTEM = "Cu bulk\n')
    incar_module._reject_swallowed_incar_tags(
        {"IBRION": 2, "NSW": 5, "SYSTEM": '"Cu bulk'}, str(path)
    )
    incar_module._load_incar(str(path))


def test_force_constants_displacement_rejects_corrupted_tokens():
    from vpmdk_core.runtime import single as single_module

    for bad in ("1D-2", ".O1", "0.0.1", "abc"):
        with pytest.raises(ValueError, match="not a plain number"):
            single_module._force_constants_displacement_from_bcar(
                {"FORCE_CONSTANTS_DISPLACEMENT": bad}
            )

    # Plain spellings keep working.
    assert single_module._force_constants_displacement_from_bcar(
        {"FORCE_CONSTANTS_DISPLACEMENT": "1e-2"}
    ) == 0.01
    assert single_module._force_constants_displacement_from_bcar(
        {"FORCE_CONSTANTS_DISPLACEMENT": "0.02"}
    ) == 0.02


def test_absurd_cell_volume_is_rejected_at_input_time(tmp_path):
    from ase import Atoms

    from vpmdk_core.io import inputs as inputs_module

    huge = Atoms(
        "Si2",
        positions=[[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]],
        cell=np.eye(3) * 20000.0,
        pbc=True,
    )
    with pytest.raises(ValueError, match="cell volume"):
        inputs_module._validate_finite_geometry(huge)

    # An extreme-but-thin slab stays legal: the PRODUCT is what is bounded,
    # not another axis cap.
    slab = Atoms(
        "Si2",
        positions=[[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]],
        cell=np.diag([1.0e5, 10.0, 10.0]),
        pbc=True,
    )
    inputs_module._validate_finite_geometry(slab)


def test_potcar_siblings_are_inert_for_the_poscar_parse(tmp_path):
    from vpmdk_core.io import inputs as inputs_module

    poscar = tmp_path / "POSCAR"
    poscar.write_text(
        "Si2\n1.0\n5.4 0 0\n0 5.4 0\n0 0 5.4\nSi\n2\nDirect\n"
        "0.0 0.0 0.0\n0.25 0.25 0.25\n"
    )
    os.mkfifo(tmp_path / "POTCAR.bak")
    (tmp_path / "POTCARs").mkdir()
    (tmp_path / "POTCAR_Cu").write_text("not a POTCAR at all\n")

    inputs_module.read_structure(str(poscar))


def test_poscar_parse_does_not_consult_potcar_siblings(tmp_path, monkeypatch):
    from vpmdk_core.io import inputs as inputs_module

    captured: dict[str, object] = {}
    real_poscar = inputs_module.Poscar

    class RecordingPoscar:
        @classmethod
        def from_file(cls, path, **kwargs):
            captured.update(kwargs)
            return real_poscar.from_file(path)

    monkeypatch.setattr(inputs_module, "Poscar", RecordingPoscar)
    poscar = tmp_path / "POSCAR"
    poscar.write_text(
        "Si2\n1.0\n5.4 0 0\n0 5.4 0\n0 0 5.4\nSi\n2\nDirect\n"
        "0.0 0.0 0.0\n0.25 0.25 0.25\n"
    )
    inputs_module.read_structure(str(poscar))
    assert captured.get("check_for_potcar") is False


def test_vasp4_trailing_symbol_poscar_declares_species():
    from vpmdk_core.io import inputs as inputs_module
    import tempfile

    trailing = (
        "Si2 vasp4 trailing symbols\n1.0\n"
        "3.8669745922 0.0 0.0\n1.9334872961 3.3488982326 0.0\n"
        "1.9334872961 1.1162994109 3.1573715331\n2\nDirect\n"
        "0.749999979 0.749999983 0.749999997 Si\n"
        "0.500000007 0.499999989 0.499999998 Si\n"
    )
    bare = trailing.replace(" Si\n", "\n")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "POSCAR")
        with open(p, "w") as handle:
            handle.write(trailing)
        assert inputs_module._poscar_declares_trailing_species(p) is True
        with open(p, "w") as handle:
            handle.write(bare)
        # Genuinely undeterminable VASP-4 (no trailing symbols) still needs a
        # POTCAR -- the fabricated-H/He guard must keep firing.
        assert inputs_module._poscar_declares_trailing_species(p) is False


def test_multiline_species_counts_block_is_bounded(tmp_path):
    from vpmdk_core.io import inputs as inputs_module

    header = (
        "multi-line block\n1.0\n"
        "5.4 0 0\n0 5.4 0\n0 0 5.4\n"
    )
    path = tmp_path / "POSCAR"
    path.write_text(
        header + "Si\nGe\n2000000000\n2000000000\nDirect\n0 0 0\n0.5 0.5 0.5\n"
    )
    with pytest.raises(ValueError, match="declares"):
        inputs_module._reject_absurd_poscar_ion_counts(str(path))

    # A small multi-line block stays legal, and integer COORDINATE lines
    # after the mode line are never miscounted.
    path.write_text(
        header + "Si\nGe\n1\n1\nDirect\n0 0 0\n1 1 1\n"
    )
    inputs_module._reject_absurd_poscar_ion_counts(str(path))

    # The classic single-line layouts keep their bound.
    path.write_text(header + "Si\n2000000000\nDirect\n0 0 0\n")
    with pytest.raises(ValueError, match="declares"):
        inputs_module._reject_absurd_poscar_ion_counts(str(path))


def test_short_selective_dynamics_masks_are_rejected():
    from vpmdk_core.io import inputs as inputs_module

    with pytest.raises(ValueError, match="exactly three"):
        inputs_module._reject_malformed_selective_dynamics(
            [[False, False], [True, True]]
        )
    inputs_module._reject_malformed_selective_dynamics(
        [[False, False, False], [True, True, True]]
    )
    inputs_module._reject_malformed_selective_dynamics(None)
    inputs_module._reject_malformed_selective_dynamics([])


def test_poscar_edge_formats_survive_real_pymatgen(tmp_path):
    import subprocess as subprocess_module
    import sys as sys_module

    script = r"""
import os, sys
import vpmdk_core

d = sys.argv[1]
trailing = (
    "Si2 vasp4 trailing symbols\n1.0\n"
    "3.8669745922 0.0 0.0\n1.9334872961 3.3488982326 0.0\n"
    "1.9334872961 1.1162994109 3.1573715331\n2\nDirect\n"
    "0.749999979 0.749999983 0.749999997 Si\n"
    "0.500000007 0.499999989 0.499999998 Si\n"
)
p1 = os.path.join(d, "POSCAR")
open(p1, "w").write(trailing)
vpmdk_core.read_structure(p1)
print("TRAILING OK")

short = (
    "Si2 short mask\n1.0\n"
    "5.4 0 0\n0 5.4 0\n0 0 5.4\nSi\n2\nSelective dynamics\nDirect\n"
    "0.01 0.0 0.0 F F\n0.5 0.5 0.5 T T\n"
)
p2 = os.path.join(d, "sub")
os.makedirs(p2, exist_ok=True)
p2 = os.path.join(p2, "POSCAR")
open(p2, "w").write(short)
try:
    vpmdk_core.read_structure(p2)
    print("SHORTMASK NOT REJECTED")
except ValueError as exc:
    assert "exactly three" in str(exc), exc
    print("SHORTMASK OK")
"""
    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    env = {
        **os.environ,
        "PYTHONPATH": src_dir + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    completed = subprocess_module.run(
        [sys_module.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        env=env,
    )
    if "ModuleNotFoundError: No module named 'pymatgen'" in completed.stderr:
        pytest.skip("real pymatgen is not installed")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "TRAILING OK" in completed.stdout, completed.stdout + completed.stderr
    assert "SHORTMASK OK" in completed.stdout, completed.stdout + completed.stderr


def test_blank_following_tag_is_not_swallow_evidence(tmp_path):
    from vpmdk_core.settings import incar as incar_module

    path = tmp_path / "INCAR"
    path.write_text('SYSTEM = "run #3"\nNSW = 5\nIBRION = 2\nNPAR =\n')
    incar_module._reject_swallowed_incar_tags(
        {"SYSTEM": '"run', "NSW": 5, "IBRION": 2}, str(path)
    )

    # Detection power is kept: a REAL swallow still surfaces through the
    # non-blank neighbours of any blank tag inside the swallowed region.
    path.write_text(
        'LWAVE = ".FALSE.\nNPAR =\nNSW = 200\nIBRION = 2\nLREAL = "Auto"\n'
    )
    with pytest.raises(ValueError, match="unbalanced quote"):
        incar_module._reject_swallowed_incar_tags(
            {"LWAVE": False, "LREAL": "Auto"}, str(path)
        )


def test_broken_input_symlinks_are_rejected_not_treated_as_absent(tmp_path):
    from vpmdk_core.io import inputs as inputs_module
    from vpmdk_core.settings import incar as incar_module

    dangling = tmp_path / "INCAR"
    dangling.symlink_to(tmp_path / "shared" / "INCAR")
    with pytest.raises(ValueError, match="cannot be resolved"):
        incar_module._load_incar(str(dangling))

    self_loop = tmp_path / "BCAR"
    os.symlink("BCAR", self_loop)
    with pytest.raises(ValueError, match="cannot be resolved"):
        inputs_module._reject_broken_input_link(str(self_loop), "BCAR")

    # A genuinely absent path stays absent, and a symlink to a real file
    # stays legal.
    inputs_module._reject_broken_input_link(str(tmp_path / "MISSING"), "INCAR")
    target = tmp_path / "real-incar"
    target.write_text("NSW = 1\n")
    good = tmp_path / "GOOD"
    good.symlink_to(target)
    inputs_module._reject_broken_input_link(str(good), "INCAR")


def test_dangling_bcar_symlink_is_an_input_error(tmp_path, prepare_inputs):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "0", "IBRION": "-1"},
    )
    bcar = tmp_path / "BCAR"
    bcar.unlink()
    bcar.symlink_to(tmp_path / "nowhere" / "BCAR")
    with pytest.raises(vpmdk.WorkdirInputError, match="cannot be resolved"):
        vpmdk.run_workdir(str(tmp_path), calculator=DummyCalculator())


def test_nonbare_selective_dynamics_spellings_are_rejected(tmp_path):
    from vpmdk_core.io import inputs as inputs_module

    header = (
        "Si2 sd spellings\n1.0\n5.4 0 0\n0 5.4 0\n0 0 5.4\nSi\n2\n"
        "Selective dynamics\nDirect\n"
    )
    path = tmp_path / "POSCAR"
    for spelling in (".TRUE.", "TRUE", "t", ".T."):
        path.write_text(
            header + f"0.01 0.0 0.0 {spelling} {spelling} {spelling}\n"
            "0.5 0.5 0.5 T T T\n"
        )
        with pytest.raises(ValueError, match="bare T or F"):
            inputs_module._reject_ambiguous_selective_dynamics_tokens(str(path))

    # Bare T/F files (including VASP-4 trailing-symbol layouts) stay legal.
    path.write_text(
        header + "0.01 0.0 0.0 F F F\n0.5 0.5 0.5 T T T\n"
    )
    inputs_module._reject_ambiguous_selective_dynamics_tokens(str(path))
    path.write_text(
        "Si2 v4 sd\n1.0\n5.4 0 0\n0 5.4 0\n0 0 5.4\n2\n"
        "Selective dynamics\nDirect\n"
        "0.01 0.0 0.0 T T F Si\n0.5 0.5 0.5 T T T Si\n"
    )
    inputs_module._reject_ambiguous_selective_dynamics_tokens(str(path))


def test_species_counts_length_mismatch_is_rejected(tmp_path):
    from vpmdk_core.io import inputs as inputs_module

    header = "mismatch\n1.0\n5.4 0 0\n0 5.4 0\n0 0 5.4\n"
    path = tmp_path / "POSCAR"
    path.write_text(header + "Si Ge\n2\nDirect\n0 0 0\n0.5 0.5 0.5\n")
    with pytest.raises(ValueError, match="species"):
        inputs_module._reject_mismatched_species_counts(str(path))

    path.write_text(header + "Si\n1 1\nDirect\n0 0 0\n0.5 0.5 0.5\n")
    with pytest.raises(ValueError, match="species"):
        inputs_module._reject_mismatched_species_counts(str(path))

    # Legal layouts: classic, VASP-4 (no symbol line), and a matched
    # multi-line block.
    path.write_text(header + "Si Ge\n1 1\nDirect\n0 0 0\n0.5 0.5 0.5\n")
    inputs_module._reject_mismatched_species_counts(str(path))
    path.write_text(header + "2\nDirect\n0 0 0\n0.5 0.5 0.5\n")
    inputs_module._reject_mismatched_species_counts(str(path))
    path.write_text(
        header + "Si\nGe\n1\n1\nDirect\n0 0 0\n0.5 0.5 0.5\n"
    )
    inputs_module._reject_mismatched_species_counts(str(path))

    path.write_text(header + "Si   ! silicon\n2\nDirect\n0 0 0\n0.5 0.5 0.5\n")
    inputs_module._reject_mismatched_species_counts(str(path))
    path.write_text(
        header + "Si Ge   ! two species\n1 1\nDirect\n0 0 0\n0.5 0.5 0.5\n"
    )
    inputs_module._reject_mismatched_species_counts(str(path))


def test_requested_nsw_is_echoed_by_neb_images_and_force_constants(
    tmp_path, prepare_inputs, monkeypatch
):
    import xml.etree.ElementTree as ET

    def echoed_nsw(path):
        root_el = ET.parse(path).getroot()
        node = root_el.find("./incar/i[@name='NSW']")
        return None if node is None else node.text.strip()

    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "20", "IBRION": "-1", "IMAGES": "1"},
    )
    _write_numbered_neb_poscars(tmp_path)
    vpmdk.run_workdir(str(tmp_path), calculator=DummyCalculator())
    assert echoed_nsw(tmp_path / "01" / "vasprun.xml") == "20"
    assert echoed_nsw(tmp_path / "vasprun.xml") == "20"

    fc_dir = tmp_path / "fc"
    fc_dir.mkdir()
    prepare_inputs(
        fc_dir,
        potential="CHGNET",
        incar_overrides={"NSW": "20", "IBRION": "5", "POTIM": "0.015", "NFREE": "2"},
    )
    vpmdk.run_workdir(str(fc_dir), calculator=DummyCalculator())
    assert echoed_nsw(fc_dir / "vasprun.xml") == "20"


def test_nhc_nchains_reads_fortran_trailing_comma():
    settings = vpmdk._load_incar_settings(
        {"IBRION": 0, "NSW": 3, "MDALGO": 4, "NHC_NCHAINS": "5,"}
    )
    assert settings.thermostat_params["NHC_NCHAINS"] == 5.0


def test_negative_scale_with_cartesian_coordinates_is_rejected(tmp_path):
    from vpmdk_core.io import inputs as inputs_module

    header = (
        "Si2 negscale\n{scale}\n"
        "3.8669745922 0.0 0.0\n1.9334872961 3.3488982326 0.0\n"
        "1.9334872961 1.1162994109 3.1573715331\nSi\n2\n"
    )
    path = tmp_path / "POSCAR"
    cart = "3.867 2.232 2.368\n2.578 1.488 1.579\n"

    path.write_text(header.format(scale="-40.888") + "Cartesian\n" + cart)
    with pytest.raises(ValueError, match="negative scale factor"):
        inputs_module._reject_negative_scale_cartesian_poscar(str(path))

    # The Selective dynamics variant hides the mode one line lower.
    path.write_text(
        header.format(scale="-40.888")
        + "Selective dynamics\nCartesian\n"
        + "3.867 2.232 2.368 T T T\n2.578 1.488 1.579 T T T\n"
    )
    with pytest.raises(ValueError, match="negative scale factor"):
        inputs_module._reject_negative_scale_cartesian_poscar(str(path))

    # The combinations the parser reads correctly keep working: negative
    # scale + Direct, and positive scale + Cartesian.
    path.write_text(
        header.format(scale="-40.888") + "Direct\n0.75 0.75 0.75\n0.5 0.5 0.5\n"
    )
    inputs_module._reject_negative_scale_cartesian_poscar(str(path))
    path.write_text(header.format(scale="1.0") + "Cartesian\n" + cart)
    inputs_module._reject_negative_scale_cartesian_poscar(str(path))


def test_same_line_incar_assignments_without_separator_are_rejected(tmp_path):
    from vpmdk_core.settings import incar as incar_module

    path = tmp_path / "INCAR"
    for text, lost in (
        ("NSW = 5 IBRION = 2\nEDIFFG = -1e-8\n", "IBRION"),
        ("IBRION = 2 NSW = 5\nEDIFFG = -1e-8\n", "NSW"),
    ):
        path.write_text(text)
        with pytest.raises(ValueError, match=lost):
            incar_module._load_incar(str(path))

    path.write_text("NSW = 5; IBRION = 2\nEDIFFG = -1e-8\n")
    incar_module._load_incar(str(path))

    path.write_text("SYSTEM = NSW=100 study\nIBRION = 2\n")
    incar_module._load_incar(str(path))

    path.write_text(
        "NSW    = 3             (max ionic steps, ignored when IBRION=-1)\n"
        "IBRION = 2             (Algorithm: 0-MD, 1-Quasi-New, 2-CG)\n"
        "EDIFFG = -1E-02        (Ionic convergence, eV/AA)\n"
    )
    incar_module._load_incar(str(path))
    path.write_text("NSW = 3   (NSW=0 would be a single point)\nIBRION = 2\n")
    incar_module._load_incar(str(path))


def test_unknown_bcar_tags_are_warned_about(tmp_path, capsys):
    from vpmdk_core.io import inputs as inputs_module

    path = tmp_path / "BCAR"
    path.write_text(
        "MLP = CHGNET\nMODELL = /models/mine.pth\nDEVCIE = cuda\n"
        "WRITE_CHGCARR = 1\n"
    )
    tags = inputs_module.parse_key_value_file(str(path))
    out = capsys.readouterr().out
    for typo in ("MODELL", "DEVCIE", "WRITE_CHGCARR"):
        assert f"BCAR tag {typo} is not recognized" in out
        assert typo in tags  # preserved, exactly as documented

    # Every real vocabulary source stays silent: static output tags, the
    # backend construction vocabulary, and the charge model-config family.
    path.write_text(
        "MLP = SEVENNET\nMODEL = /m.pt\nDEVICE = cpu\nWRITE_ENERGY_CSV = 1\n"
        "SEVENNET_ENABLE_FLASH = 1\nCHARGE_NUM_INTERACTIONS = 3\n"
        "CHARGE_DEEPCDP_WEIGHTING_R0 = 1.5\nFORCE_CONSTANTS_DISPLACEMENT = 0.01\n"
        "CHARGE_DEEPCDP_RCUT = 6.0\nCHARGE_DEEPCDP_NMAX = 8\n"
        "CHARGE_DEEPCDP_LMAX = 6\nCHARGE_DEEPCDP_SIGMA = 0.3\n"
        "CHARGE_DEEPCDP_PERIODIC = 1\nCHARGE_DEEPCDP_SPECIES = Si\n"
        "CHARGE_DEEPCDP_ACTIVATION = relu\nCHARGE_DEEPCDP_METADATA = /m.json\n"
    )
    inputs_module.parse_key_value_file(str(path))
    assert "is not recognized" not in capsys.readouterr().out

    path.write_text("MLP = CHGNET\nMODELL = /x\n")
    inputs_module.parse_key_value_file(str(path), warn_unknown_tags=False)
    assert "is not recognized" not in capsys.readouterr().out


def test_serve_drains_stderr_like_stdout(monkeypatch, capsys):
    import vpmdk_core.server as server_module

    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    broken = os.fdopen(write_fd, "w")
    monkeypatch.setattr(sys, "stderr", broken)
    print("buffered warning line", file=sys.stderr)
    server_module._drain_stream_guarded(sys.stderr)  # must not raise
    print("late line", file=sys.stderr)
    sys.stderr.flush()  # the finalization operation; must not raise either

    # WIRING: serve_cli's finally must drain stderr too. Run it with a fresh
    # broken stderr carrying buffered bytes; afterwards the exact operation
    # CPython's finalization performs (a bare flush) must not raise, which is
    # only true if the finally pointed the stream at /dev/null.
    read_fd2, write_fd2 = os.pipe()
    os.close(read_fd2)
    broken2 = os.fdopen(write_fd2, "w")
    monkeypatch.setattr(sys, "stderr", broken2)
    print("buffered second line", file=sys.stderr)
    monkeypatch.setattr(server_module, "_serve_cli_inner", lambda args: 7)
    assert server_module.serve_cli(object()) == 7
    sys.stderr.flush()  # must not raise


def test_comma_separated_repeat_groups_are_bounded(tmp_path):
    import time as time_module

    from vpmdk_core.settings import incar as incar_module

    path = tmp_path / "INCAR"
    groups = ",".join(["1000000*1.0"] * 500)
    path.write_text(f"MAGMOM = {groups}\n")
    started = time_module.monotonic()
    with pytest.raises(ValueError, match="expands to"):
        incar_module._reject_huge_repeat_counts(str(path))
    assert time_module.monotonic() - started < 5.0

    # Legal comma-separated repeats keep parsing.
    path.write_text("MAGMOM = 2*1.0,3*-0.5\n")
    incar_module._reject_huge_repeat_counts(str(path))
    incar_module._load_incar(str(path))


def test_repeat_caps_apply_only_to_expanded_tags(tmp_path):
    from vpmdk_core.settings import incar as incar_module

    path = tmp_path / "INCAR"
    path.write_text("SYSTEM = 1000001*study\nNSW = 5\n")
    incar_module._reject_huge_repeat_counts(str(path))
    incar_module._load_incar(str(path))

    path.write_text("FOOTAG = 2000000*x\nNSW = 5\n")
    incar_module._reject_huge_repeat_counts(str(path))

    # Expanded tags keep every cap.
    path.write_text("MAGMOM = 10000000*1.0\n")
    with pytest.raises(ValueError, match="repeat token"):
        incar_module._reject_huge_repeat_counts(str(path))
    path.write_text("MAGMOM = " + ",".join(["1000000*1.0"] * 500) + "\n")
    with pytest.raises(ValueError, match="expands to"):
        incar_module._reject_huge_repeat_counts(str(path))


def test_conditional_artifacts_are_preflighted_only_when_written(
    tmp_path, prepare_inputs
):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "0", "IBRION": "-1"},
    )
    (tmp_path / "CHGCAR").mkdir()
    os.mkfifo(tmp_path / "XDATCAR")
    (tmp_path / "energy.csv").mkdir()

    # A static run without the flags must ignore all three nodes.
    vpmdk.run_workdir(str(tmp_path), calculator=DummyCalculator())

    # With WRITE_CHGCAR=1 the same directory fails UP FRONT as input error.
    (tmp_path / "BCAR").write_text("MLP = CHGNET\nWRITE_CHGCAR = 1\n")
    with pytest.raises(vpmdk.WorkdirInputError, match="CHGCAR"):
        vpmdk.run_workdir(str(tmp_path), calculator=DummyCalculator())


def test_stressless_backend_with_cell_relaxation_is_an_input_error():
    from vpmdk_core import cli as cli_module

    stressless = {"MLP": "MATRIS", "MATRIS_TASK": "ef"}
    for isif in (3, 4, 7):
        settings = vpmdk._load_incar_settings(
            {"IBRION": 2, "NSW": 3, "ISIF": isif}
        )
        with pytest.raises(vpmdk.WorkdirInputError, match="CELL relaxation"):
            cli_module._check_backend_output_capabilities(stressless, settings)

    # The branches where stress is only an OUTPUT keep running with the
    # warning (all measured working with a real MATRIS/ef resident):
    # ion-only relaxation, single point, MD, force constants and NEB.
    for incar in (
        {"IBRION": 2, "NSW": 3, "ISIF": 2},
        {"IBRION": -1, "NSW": 0, "ISIF": 3},
        {"IBRION": 0, "NSW": 3, "ISIF": 3},
        {"IBRION": 5, "NSW": 3, "ISIF": 3},
    ):
        cli_module._check_backend_output_capabilities(
            stressless, vpmdk._load_incar_settings(incar)
        )
    cli_module._check_backend_output_capabilities(
        stressless,
        vpmdk._load_incar_settings({"IBRION": 2, "NSW": 3, "ISIF": 3}),
        neb_mode=True,
    )

    # A stress-capable configuration is unaffected.
    cli_module._check_backend_output_capabilities(
        {"MLP": "MATRIS", "MATRIS_TASK": "efs"},
        vpmdk._load_incar_settings({"IBRION": 2, "NSW": 3, "ISIF": 3}),
    )


def test_repeat_guard_mirrors_the_pymatgen_tokenizer(tmp_path):
    from vpmdk_core.settings import incar as incar_module

    path = tmp_path / "INCAR"
    for value in (
        "(2000000000*1.0)",
        "x1000000*1.0x1000000*1.0",
        "2000000000*1.0",
        "[10000000*1.0]",
        ",".join(["1000000*1.0"] * 500),
    ):
        path.write_text(f"MAGMOM = {value}\n")
        with pytest.raises(ValueError):
            incar_module._reject_huge_repeat_counts(str(path))

    for value in ("4*1.0 2*-0.5", "2*3*1.0", "1.0 -1.0 1.0"):
        path.write_text(f"MAGMOM = {value}\n")
        incar_module._reject_huge_repeat_counts(str(path))
    # Free-text tags are still out of scope entirely.
    path.write_text("SYSTEM = (1000001*study)\n")
    incar_module._reject_huge_repeat_counts(str(path))


def test_newline_before_equals_reaches_every_raw_guard(tmp_path):
    import subprocess as subprocess_module
    import sys as sys_module

    from vpmdk_core.settings import incar as incar_module

    # Stub-safe in-process half: the repeat cap is raw-only, so the newline
    # door is detectable without the real parser.
    path = tmp_path / "INCAR"
    path.write_text("MAGMOM\n= 5000000*1.0\nNSW = 5\n")
    with pytest.raises(ValueError, match="repeat token|expands to"):
        incar_module._reject_huge_repeat_counts(str(path))

    # Plain files keep parsing in-process.
    path.write_text("NSW = 5\nIBRION = 2\n")
    incar_module._load_incar(str(path))

    # Real-pymatgen half (the conftest stub parses line-wise and never sees
    # the newline-crossing key, so the raw-vs-parsed guards are only
    # exercisable against the real library).
    script = r"""
import sys
from pymatgen.io.vasp import Incar
from vpmdk_core.settings import incar as m
import tempfile, os
d = sys.argv[1]
# The guards are COMPARISON-based, so they follow the installed parser:
# pymatgen >= 2026 compiles the key side as KEY\s*= (the whitespace crosses
# a newline, so 'NSW\n= 1e5' PARSES and the raw guards must see it), while
# older releases (e.g. 2025.10.7 on Python 3.10 CI) never match the
# newline-split key and silently DROP the tag instead -- a different door,
# with different correct outcomes. Probe the library instead of pinning one
# version's behavior.
key_crosses = "NSW" in dict(Incar.from_str("NSW\n= 7\n"))
print("KEYCROSS:", key_crosses)
p = os.path.join(d, "INCAR")
for text, label in (
    ("NSW\n= 1e5\nIBRION = 2\n", "scientific"),
    ("TEBEG\n= 5OO\nNSW = 3\n", "corrupted"),
    ("NSW\n= 5 IBRION = 2\n", "embedded"),
    ("SYSTEM =\nIBRION = 2\n", "blank-swallow"),
):
    open(p, "w").write(text)
    try:
        m._load_incar(p)
        print("NOT REJECTED:", label)
    except ValueError:
        print("REJECTED OK:", label)
open(p, "w").write("SPRING\n= -5.5\nIMAGES = 1\n")
inc = m._load_incar(p)
print("SPRING:", inc.get("SPRING"))
"""
    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    env = {
        **os.environ,
        "PYTHONPATH": src_dir + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    completed = subprocess_module.run(
        [sys_module.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        env=env,
    )
    if "ModuleNotFoundError: No module named 'pymatgen'" in completed.stderr:
        pytest.skip("real pymatgen is not installed")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    if "KEYCROSS: True" in completed.stdout:
        # The installed parser crosses the newline: every raw guard must see
        # what the parser sees, and SPRING parses to its value.
        for label in ("scientific", "corrupted", "embedded", "blank-swallow"):
            assert f"REJECTED OK: {label}" in completed.stdout, completed.stdout
        assert "SPRING: -5.5" in completed.stdout, completed.stdout
    else:
        # Older pymatgen never matches the newline-split key: the parser
        # itself drops the tag, so the newline door does not exist and the
        # comparison-based guards must NOT false-reject; the raw-only guards
        # (corrupted token, embedded assignment) still fire on the raw text.
        assert "KEYCROSS: False" in completed.stdout, completed.stdout
        assert "NOT REJECTED: scientific" in completed.stdout, completed.stdout
        assert "NOT REJECTED: blank-swallow" in completed.stdout, completed.stdout
        assert "REJECTED OK: corrupted" in completed.stdout, completed.stdout
        assert "REJECTED OK: embedded" in completed.stdout, completed.stdout
        assert "SPRING: None" in completed.stdout, completed.stdout


def test_stress_gate_keys_on_the_actual_neb_branch(tmp_path, prepare_inputs, monkeypatch):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={
            "NSW": "5",
            "IBRION": "2",
            "ISIF": "3",
            "SPRING": "-5.0",
        },
    )
    (tmp_path / "BCAR").write_text("MLP = MATRIS\nMATRIS_TASK = ef\n")

    # Flat workdir + stray NEB tag: the relaxation branch WILL run, so the
    # stress-less backend must be rejected up front.
    with pytest.raises(vpmdk.WorkdirInputError, match="CELL relaxation"):
        vpmdk.run_workdir(str(tmp_path), calculator=DummyCalculator())

    # A genuine NEB layout keeps the exclusion (image positions only; the
    # cell never relaxes there).
    _write_numbered_neb_poscars(tmp_path)
    monkeypatch.setattr(vpmdk, "BFGS", DummyNEBOptimizer)
    (tmp_path / "INCAR").write_text(
        "NSW = 2\nIBRION = 2\nISIF = 3\nSPRING = -5.0\nIMAGES = 1\n"
        "EDIFFG = -0.05\n"
    )
    vpmdk.run_workdir(str(tmp_path), calculator=DummyCalculator())


def test_bam_builder_requires_local_model_and_forwards_device(tmp_path, monkeypatch):
    # BAM-torch ships checkpoints (e.g. the published BAM-MP-core.pkl) as
    # plain local files with no named-model downloader, so MODEL is a
    # required local path like the NequIP/Allegro rule, and the builder
    # passes DEVICE through only when non-blank (RACECalculator selects cpu
    # itself when device is omitted).
    seen: dict[str, object] = {}

    def fake_bam(*, model, device=None):
        seen["model"] = model
        seen["device"] = device
        return DummyCalculator()

    monkeypatch.setattr(vpmdk, "BAMCalculator", fake_bam)

    with pytest.raises(ValueError, match="MODEL"):
        vpmdk._build_bam_calculator({"MLP": "BAM"})

    checkpoint = tmp_path / "BAM-MP-core.pkl"
    checkpoint.write_text("dummy")

    vpmdk._build_bam_calculator(
        {"MLP": "BAM", "MODEL": str(checkpoint), "DEVICE": "cpu"}
    )
    assert seen == {"model": str(checkpoint), "device": "cpu"}

    seen.clear()
    vpmdk._build_bam_calculator({"MLP": "BAM", "MODEL": str(checkpoint), "DEVICE": ""})
    assert seen == {"model": str(checkpoint), "device": None}

    monkeypatch.setattr(vpmdk, "BAMCalculator", None)
    with pytest.raises(RuntimeError, match="bam-torch"):
        vpmdk._build_bam_calculator({"MLP": "BAM", "MODEL": str(checkpoint)})


def test_build_grace_calculator_discloses_ignored_device(tmp_path: Path, capsys):
    model_path = tmp_path / "grace-model"
    model_path.write_text("dummy")

    class DummyTP(DummyCalculator):
        def __init__(self, model, **kwargs):  # type: ignore[override]
            super().__init__()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "TPCalculator", DummyTP)
    monkeypatch.setattr(vpmdk, "GRACE_MODEL_NAMES", [])
    try:
        vpmdk._build_grace_calculator({"MODEL": str(model_path)})
        assert "DEVICE" not in capsys.readouterr().out
        vpmdk._build_grace_calculator(
            {"MODEL": str(model_path), "DEVICE": "cuda"}
        )
        output = capsys.readouterr().out
        assert "GRACE ignores the DEVICE tag" in output
        assert "TensorFlow" in output
    finally:
        monkeypatch.undo()


def test_species_gate_reads_a_z_keyed_uniq_element_table(tmp_path, prepare_inputs, monkeypatch):
    from types import SimpleNamespace

    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "0", "IBRION": "-1"},
    )
    poscar_text = (tmp_path / "POSCAR").read_text().replace("Si", "Po")
    (tmp_path / "POSCAR").write_text(poscar_text)

    declared = DummyCalculator()
    # H..Bi table with the Po hole, the shape BAM-MP-core actually declares.
    declared.uniq_element = {z: i for i, z in enumerate(range(1, 84))}

    with pytest.raises(vpmdk.WorkdirInputError, match="element table"):
        vpmdk.run_workdir(str(tmp_path), calculator=declared)

    # A covered structure passes the same gate.
    covered = DummyCalculator()
    covered.uniq_element = {14: 0, 84: 1}
    vpmdk.run_workdir(str(tmp_path), calculator=covered)

    # A symbol-keyed element_types declaration still takes precedence over a
    # nested Z-keyed table.
    both = DummyCalculator()
    both.model = SimpleNamespace(element_types=("Po",), uniq_element={1: 0})
    vpmdk.run_workdir(str(tmp_path), calculator=both)


def test_neb_optimization_rejects_diverged_far_out_image_coordinates(
    tmp_path, prepare_inputs, monkeypatch
):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={
            "NSW": "2",
            "IBRION": "2",
            "IMAGES": "1",
            "SPRING": "-5.0",
            "EDIFFG": "-0.05",
        },
    )
    _write_numbered_neb_poscars(tmp_path)
    middle = tmp_path / "01" / "POSCAR"
    text = middle.read_text().splitlines()
    for index, line in enumerate(text):
        if line.strip().lower().startswith("direct"):
            fields = text[index + 1].split()
            # Two displaced axes, like a genuinely diverged geometry: a
            # single-axis excursion costs the neighbour search only linearly
            # (the same reasoning as the MD guard's per-axis floors) and is
            # deliberately not rejected.
            fields[0] = str(float(fields[0]) + 1000000.0)
            fields[1] = str(float(fields[1]) + 1000000.0)
            text[index + 1] = "  " + "  ".join(fields[:3])
            break
    middle.write_text("\n".join(text) + "\n")

    monkeypatch.setattr(vpmdk, "BFGS", DummyNEBOptimizer)
    with pytest.raises(vpmdk.WorkdirInputError, match="far outside the cell"):
        vpmdk.run_workdir(str(tmp_path), calculator=DummyCalculator())


def test_neb_guard_bounds_a_single_axis_divergence(tmp_path, prepare_inputs, monkeypatch):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={
            "NSW": "2",
            "IBRION": "2",
            "IMAGES": "1",
            "SPRING": "-5.0",
            "EDIFFG": "-0.05",
        },
    )
    _write_numbered_neb_poscars(tmp_path)
    middle = tmp_path / "01" / "POSCAR"
    text = middle.read_text().splitlines()
    for index, line in enumerate(text):
        if line.strip().lower().startswith("direct"):
            fields = text[index + 1].split()
            # ONE displaced fractional axis: ~3.9e7 A of cartesian span on a
            # single axis, product ~3.9e7 A^3 -- far under the volume cap.
            fields[0] = str(float(fields[0]) + 10000000.0)
            text[index + 1] = "  " + "  ".join(fields[:3])
            break
    middle.write_text("\n".join(text) + "\n")

    monkeypatch.setattr(vpmdk, "BFGS", DummyNEBOptimizer)
    with pytest.raises(vpmdk.WorkdirInputError, match="per axis"):
        vpmdk.run_workdir(str(tmp_path), calculator=DummyCalculator())
