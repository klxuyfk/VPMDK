from __future__ import annotations

from pathlib import Path

import pytest

from pymatgen.io.vasp import Incar

import vpmdk


def test_incar_parsing_handles_case_whitespace_and_comments(tmp_path: Path):
    incar_content = """
    ! leading comment
    nsw = 5   ! ionic steps
      IBrIoN = 2 # relaxation mode
    """
    path = tmp_path / "INCAR"
    path.write_text(incar_content)

    incar = Incar.from_file(path)

    assert "NSW" in incar
    assert str(incar.get("NSW")) == "5"
    assert str(incar.get("IBRION")) == "2"


def test_bcar_parsing_handles_case_whitespace_and_comments(tmp_path: Path):
    bcar_content = """
    # initial comment
      nnp = mace   # inline comment
    Model = /path/to/model.nn  ! trailing comment
    WRITE_energy_csv = On
    """
    path = tmp_path / "BCAR"
    path.write_text(bcar_content)

    tags = vpmdk.parse_key_value_file(str(path))

    assert tags["NNP"] == "mace"
    assert tags["MODEL"] == "/path/to/model.nn"
    assert tags["WRITE_ENERGY_CSV"] == "On"


@pytest.mark.parametrize(
    "definition, expected",
    [
        ("2*1.5 0.25", [1.5, 1.5, 0.25]),
        ("1 2 3", [1.0, 2.0, 3.0]),
        ("", []),
        (None, []),
    ],
)
def test_parse_magmom_values(definition, expected, arrays_close):
    parsed = vpmdk._parse_magmom_values(definition)
    assert arrays_close(parsed, expected)


def test_read_structure_normalizes_potcar_species(tmp_path: Path):
    poscar_content = """Test structure
1.0
1 0 0
0 1 0
0 0 1
Y_sv O_h_GW
1 1
Direct
0.0 0.0 0.0
0.5 0.5 0.5
"""
    potcar_content = """Y_sv
O_h_GW
"""
    poscar_path = tmp_path / "POSCAR"
    potcar_path = tmp_path / "POTCAR"
    poscar_path.write_text(poscar_content)
    potcar_path.write_text(potcar_content)

    structure = vpmdk.read_structure(str(poscar_path), str(potcar_path))

    assert getattr(structure, "species", []) == ["Y", "O"]


def test_build_orb_calculator_uses_bcar_tags(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_model(**kwargs):
        captured.update(kwargs)
        return "model"

    class DummyCalculator:
        def __init__(self, model, device=None):
            self.model = model
            self.device = device

    monkeypatch.setattr(vpmdk, "ORBCalculator", DummyCalculator)
    monkeypatch.setattr(vpmdk, "ORB_PRETRAINED_MODELS", {"custom": fake_model})

    calculator = vpmdk._build_orb_calculator(
        {
            "NNP": "ORB",
            "MODEL": "weights.ckpt",
            "DEVICE": "cuda:1",
            "ORB_MODEL": "custom",
            "ORB_PRECISION": "float64",
            "ORB_COMPILE": "false",
        }
    )

    assert isinstance(calculator, DummyCalculator)
    assert calculator.device == "cuda:1"
    assert captured["weights_path"] == "weights.ckpt"
    assert captured["precision"] == "float64"
    assert captured["compile"] is False
