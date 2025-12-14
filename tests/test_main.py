from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import vpmdk
from tests.conftest import DummyCalculator


@pytest.mark.parametrize(
    "potential",
    [
        "CHGNET",
        "MATGL",
        "M3GNET",
        "MACE",
        "MATTERSIM",
        "MATLANTIS",
        "ALLEGRO",
        "NEQUIP",
        "ORB",
        "FAIRCHEM",
        "GRACE",
    ],
)
def test_single_point_energy_for_all_potentials(
    tmp_path: Path,
    potential: str,
    prepare_inputs,
):
    extra_bcar: dict[str, str] = {}
    if potential in {"NEQUIP", "ALLEGRO"}:
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
    monkeypatch.setattr(
        vpmdk, "_build_m3gnet_calculator", lambda tags: factory(tags.get("NNP", "MATGL"))
    )
    monkeypatch.setattr(vpmdk, "MACECalculator", lambda *a, **k: factory("MACE"))
    monkeypatch.setattr(vpmdk, "MatterSimCalculator", lambda *a, **k: factory("MATTERSIM"))
    monkeypatch.setattr(vpmdk, "MatlantisEstimator", lambda *a, **k: object())
    monkeypatch.setattr(vpmdk, "MatlantisASECalculator", lambda *a, **k: factory("MATLANTIS"))
    monkeypatch.setattr(vpmdk, "_build_allegro_calculator", lambda *a, **k: factory("ALLEGRO"))
    monkeypatch.setattr(vpmdk, "ORBCalculator", lambda *a, **k: factory("ORB"))
    monkeypatch.setattr(vpmdk, "ORB_PRETRAINED_MODELS", {vpmdk.DEFAULT_ORB_MODEL: lambda **_: "orb"})
    monkeypatch.setattr(vpmdk, "_build_grace_calculator", lambda tags: factory("GRACE"))

    class _DummyFairChem:
        @classmethod
        def from_model_checkpoint(cls, *a, **k):
            return factory("FAIRCHEM")

    monkeypatch.setattr(vpmdk, "FAIRChemCalculator", _DummyFairChem)

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

    def capture_magmoms(atoms, calculator):
        captured["moments"] = list(atoms.get_initial_magnetic_moments())
        return 0.5

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "get_calculator", lambda *_: DummyCalculator())
    monkeypatch.setattr(vpmdk, "run_single_point", capture_magmoms)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert "moments" in captured
    assert arrays_close(captured["moments"], [1.25, -0.75])


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


def test_main_negative_ibrion_forces_single_point(tmp_path: Path, prepare_inputs):
    prepare_inputs(
        tmp_path,
        potential="CHGNET",
        incar_overrides={"NSW": "5", "IBRION": "-1"},
    )

    seen: dict[str, int] = {}

    def fake_single_point(atoms, calculator):
        seen["single_point"] = seen.get("single_point", 0) + 1
        return 0.5

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "get_calculator", lambda *_: DummyCalculator())
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
    ):
        seen["isif"] = isif
        seen["pstress"] = pstress
        return 0.0

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "get_calculator", lambda *_: DummyCalculator())
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
    if warning_fragment is None:
        assert not any("Warning: ISIF=" in message for message in messages)
    else:
        assert any(warning_fragment in message for message in messages)


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
    ):
        seen["fmax"] = fmax
        seen["energy_tolerance"] = energy_tolerance
        return 0.0

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "get_calculator", lambda *_: DummyCalculator())
    monkeypatch.setattr(vpmdk, "run_relaxation", fake_run_relaxation)
    monkeypatch.setattr(sys, "argv", ["vpmdk.py", "--dir", str(tmp_path)])
    try:
        vpmdk.main()
    finally:
        monkeypatch.undo()

    assert seen.get("energy_tolerance") == pytest.approx(0.01)
    assert seen.get("fmax") == pytest.approx(-0.01)


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
    ):
        seen.update(
            {
                "steps": steps,
                "temperature": temperature,
                "timestep": timestep,
                "mdalgo": mdalgo,
                "teend": teend,
                "smass": smass,
                "thermostat": thermostat_params,
            }
        )
        return 0.0

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vpmdk, "get_calculator", lambda *_: DummyCalculator())
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
