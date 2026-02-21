from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

import vpmdk
from tests.conftest import DummyCalculator


@pytest.mark.parametrize(
    "potential",
    [
        "CHGNET",
        "SEVENNET",
        "MATGL",
        "M3GNET",
        "MACE",
        "MATTERSIM",
        "MATLANTIS",
        "ALLEGRO",
        "NEQUIP",
        "ORB",
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
    if potential in {"NEQUIP", "ALLEGRO", "DEEPMD", "FAIRCHEM_V1"}:
        model_path = tmp_path / "nequip-model.pth"
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
    monkeypatch.setattr(vpmdk, "SevenNetCalculator", lambda *a, **k: factory("SEVENNET"))
    monkeypatch.setattr(
        vpmdk,
        "_build_m3gnet_calculator",
        lambda tags: factory(vpmdk._resolve_mlp_tag(tags, default="MATGL")),
    )
    monkeypatch.setattr(vpmdk, "MACECalculator", lambda *a, **k: factory("MACE"))
    monkeypatch.setattr(vpmdk, "MatterSimCalculator", lambda *a, **k: factory("MATTERSIM"))
    monkeypatch.setattr(vpmdk, "MatlantisEstimator", lambda *a, **k: object())
    monkeypatch.setattr(vpmdk, "MatlantisASECalculator", lambda *a, **k: factory("MATLANTIS"))
    monkeypatch.setattr(vpmdk, "_build_allegro_calculator", lambda *a, **k: factory("ALLEGRO"))
    monkeypatch.setattr(vpmdk, "ORBCalculator", lambda *a, **k: factory("ORB"))
    monkeypatch.setattr(vpmdk, "ORB_PRETRAINED_MODELS", {vpmdk.DEFAULT_ORB_MODEL: lambda **_: "orb"})
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
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert created and created[-1][0] == potential
    assert created[-1][1].called == 1


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
    monkeypatch.setattr(vpmdk, "get_calculator", lambda *_, **__: DummyCalculator())
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
            {
                "MLP": "FAIRCHEM_V1",
                "MODEL": str(model_path),
                "FAIRCHEM_V1_PREDICTOR": "1",
            }
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
        {
            "MLP": "FAIRCHEM_V1",
            "MODEL": "checkpoint.pt",
            "FAIRCHEM_CONFIG": "config.yml",
            "DEVICE": "cpu",
        }
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
    monkeypatch.setattr(vpmdk, "get_calculator", lambda *_, **__: DummyCalculator())
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
    monkeypatch.setattr(vpmdk, "get_calculator", lambda *_, **__: DummyCalculator())
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
    monkeypatch.setattr(vpmdk, "get_calculator", lambda *_, **__: DummyCalculator())
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
    monkeypatch.setattr(vpmdk, "get_calculator", lambda *_, **__: DummyCalculator())
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
    monkeypatch.setattr(vpmdk, "get_calculator", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(vpmdk, "run_relaxation", fake_run_relaxation)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert seen.get("neb_mode") is True


def test_main_passes_oszicar_pseudo_scf_flag_from_bcar(tmp_path: Path, prepare_inputs):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "2", "IBRION": "2"},
        extra_bcar={"WRITE_OSZICAR_PSEUDO_SCF": "on"},
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
    monkeypatch.setattr(vpmdk, "get_calculator", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(vpmdk, "run_relaxation", fake_run_relaxation)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert seen.get("oszicar_pseudo_scf") is True


def test_main_runs_neb_images_from_numbered_directories(tmp_path: Path, prepare_inputs):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "2", "IMAGES": "1"},
    )

    poscar_text = (tmp_path / "POSCAR").read_text()
    for image in ("00", "01", "02"):
        image_dir = tmp_path / image
        image_dir.mkdir()
        (image_dir / "POSCAR").write_text(poscar_text)

    seen: list[dict[str, object]] = []

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
        neb_prev_positions=None,
        neb_next_positions=None,
        oszicar_pseudo_scf=False,
    ):
        seen.append(
            {
                "cwd": Path.cwd().name,
                "steps": steps,
                "neb_mode": neb_mode,
                "has_prev": neb_prev_positions is not None,
                "has_next": neb_next_positions is not None,
                "oszicar_pseudo_scf": oszicar_pseudo_scf,
            }
        )
        return 0.0

    def fail(*args, **kwargs):  # pragma: no cover - defensive guard
        raise AssertionError("NEB runner should dispatch to relaxation for this setup")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "get_calculator", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(vpmdk, "run_relaxation", fake_run_relaxation)
    monkeypatch.setattr(vpmdk, "run_single_point", fail)
    monkeypatch.setattr(vpmdk, "run_md", fail)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert [item["cwd"] for item in seen] == ["00", "01", "02"]
    assert all(item["neb_mode"] is True for item in seen)
    assert all(item["steps"] == 2 for item in seen)
    assert [item["has_prev"] for item in seen] == [False, True, True]
    assert [item["has_next"] for item in seen] == [True, True, False]
    assert all(item["oszicar_pseudo_scf"] is False for item in seen)


def test_main_neb_runner_allows_missing_top_level_poscar(tmp_path: Path, prepare_inputs):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "2", "IMAGES": "1"},
    )

    poscar_text = (tmp_path / "POSCAR").read_text()
    (tmp_path / "POSCAR").unlink()
    for image in ("00", "01", "02"):
        image_dir = tmp_path / image
        image_dir.mkdir()
        (image_dir / "POSCAR").write_text(poscar_text)

    seen: list[str] = []

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
        neb_prev_positions=None,
        neb_next_positions=None,
        oszicar_pseudo_scf=False,
    ):
        seen.append(Path.cwd().name)
        return 0.0

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "get_calculator", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(vpmdk, "run_relaxation", fake_run_relaxation)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert seen == ["00", "01", "02"]


def test_main_neb_runner_dispatches_single_point_when_nsw_is_zero(
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
    monkeypatch.setattr(vpmdk, "get_calculator", lambda *_, **__: DummyCalculator())
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
    monkeypatch.setattr(vpmdk, "get_calculator", lambda *_, **__: DummyCalculator())
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
    monkeypatch.setattr(vpmdk, "get_calculator", lambda *_, **__: DummyCalculator())
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

    poscar_text = (tmp_path / "POSCAR").read_text()
    for image in ("00", "01", "02"):
        image_dir = tmp_path / image
        image_dir.mkdir()
        (image_dir / "POSCAR").write_text(poscar_text)

    class StressDummyCalculator(DummyCalculator):
        def calculate(self, atoms=None, properties=("energy",), system_changes=()):
            super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
            self.results["stress"] = np.zeros(6, dtype=float)

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
    monkeypatch.setattr(vpmdk, "get_calculator", lambda *_, **__: StressDummyCalculator())
    monkeypatch.setattr(vpmdk, "BFGS", DummyBFGS)
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

    poscar_text = (run_dir / "POSCAR").read_text()
    for image in ("00", "01", "02"):
        image_dir = run_dir / image
        image_dir.mkdir()
        (image_dir / "POSCAR").write_text(poscar_text)

    class StressDummyCalculator(DummyCalculator):
        def calculate(self, atoms=None, properties=("energy",), system_changes=()):
            super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
            self.results["stress"] = np.zeros(6, dtype=float)

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
    monkeypatch.setattr(vpmdk, "get_calculator", lambda *_, **__: StressDummyCalculator())
    monkeypatch.setattr(vpmdk, "BFGS", DummyBFGS)
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

    poscar_text = (run_dir / "POSCAR").read_text()
    for image in ("00", "01", "02"):
        image_dir = run_dir / image
        image_dir.mkdir()
        (image_dir / "POSCAR").write_text(poscar_text)

    model_dir = run_dir / "model"
    model_dir.mkdir()
    (model_dir / "nequip.pth").write_text("dummy")

    seen_cwds: list[Path] = []
    seen_models: list[str | None] = []

    def fake_get_calculator(tags, *, structure=None):
        seen_cwds.append(Path.cwd())
        seen_models.append(tags.get("MODEL"))
        return DummyCalculator()

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
        neb_prev_positions=None,
        neb_next_positions=None,
        oszicar_pseudo_scf=False,
    ):
        return 0.0

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vpmdk, "get_calculator", fake_get_calculator)
    monkeypatch.setattr(vpmdk, "run_relaxation", fake_run_relaxation)
    monkeypatch.setattr(vpmdk, "_collect_neb_image_results", lambda *_, **__: [])
    monkeypatch.setattr(vpmdk, "_write_neb_parent_aggregate_outputs", lambda **_: None)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", "runs/neb_model"])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert seen_cwds == [run_dir, run_dir, run_dir]
    assert seen_models == ["./model/nequip.pth"] * 3


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

    poscar_text = (run_dir / "POSCAR").read_text()
    for image in ("00", "01", "02"):
        image_dir = run_dir / image
        image_dir.mkdir()
        (image_dir / "POSCAR").write_text(poscar_text)

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
        neb_prev_positions=None,
        neb_next_positions=None,
        oszicar_pseudo_scf=False,
    ):
        return 0.0

    def fake_collect(image_dirs, *, potcar_path=None):
        seen["cwd"] = Path.cwd()
        seen["potcar_path"] = potcar_path
        seen["image_dirs"] = list(image_dirs)
        return []

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vpmdk, "get_calculator", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(vpmdk, "run_relaxation", fake_run_relaxation)
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
    monkeypatch.setattr(vpmdk, "get_calculator", lambda *_, **__: DummyCalculator())
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
    monkeypatch.setattr(vpmdk, "get_calculator", lambda *_, **__: DummyCalculator())
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
    monkeypatch.setattr(vpmdk, "get_calculator", lambda *_, **__: DummyCalculator())
    monkeypatch.setattr(vpmdk, "run_md", fake_run_md)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert seen["mdalgo"] == 2
    assert seen["smass"] == 2.0
