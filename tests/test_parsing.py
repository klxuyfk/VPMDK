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
