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
      mlp = mace   # inline comment
    Model = /path/to/model.nn  ! trailing comment
    WRITE_energy_csv = On
    """
    path = tmp_path / "BCAR"
    path.write_text(bcar_content)

    tags = vpmdk.parse_key_value_file(str(path))

    assert tags["MLP"] == "mace"
    assert tags["MODEL"] == "/path/to/model.nn"
    assert tags["WRITE_ENERGY_CSV"] == "On"


def test_bcar_parsing_maps_legacy_nnp_to_mlp(tmp_path: Path):
    path = tmp_path / "BCAR"
    path.write_text("NNP=CHGNET\n")

    tags = vpmdk.parse_key_value_file(str(path))

    assert tags["MLP"] == "CHGNET"
    assert tags["NNP"] == "CHGNET"


def test_bcar_parsing_prefers_mlp_over_legacy_nnp(tmp_path: Path):
    path = tmp_path / "BCAR"
    path.write_text("MLP=MATGL\nNNP=CHGNET\n")

    tags = vpmdk.parse_key_value_file(str(path))

    assert tags["MLP"] == "MATGL"
    assert tags["NNP"] == "CHGNET"


def test_get_calculator_accepts_legacy_nnp_tag(monkeypatch: pytest.MonkeyPatch):
    class DummyCHGNet:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(vpmdk, "CHGNetCalculator", DummyCHGNet)

    calculator = vpmdk.get_calculator({"NNP": "CHGNET"})

    assert isinstance(calculator, DummyCHGNet)


def test_get_calculator_rejects_explicit_empty_backend_tags(
    monkeypatch: pytest.MonkeyPatch,
):
    class DummyCHGNet:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(vpmdk, "CHGNetCalculator", DummyCHGNet)

    with pytest.raises(ValueError, match="MLP"):
        vpmdk.get_calculator({"MLP": ""})

    with pytest.raises(ValueError, match="NNP"):
        vpmdk.get_calculator({"NNP": "  "})


@pytest.mark.parametrize(
    "tags, expected",
    [
        ({"WRITE_OSZICAR_PSEUDO_SCF": "1"}, True),
        ({"WRITE_OSZICAR_PSEUDO_SCF": "on"}, True),
        ({"WRITE_PSEUDO_SCF": "yes"}, True),
        ({}, False),
        ({"WRITE_OSZICAR_PSEUDO_SCF": "off"}, False),
    ],
)
def test_should_write_oszicar_pseudo_scf(tags, expected):
    assert vpmdk._should_write_oszicar_pseudo_scf(tags) is expected


def test_collect_neb_image_results_prefers_contcar_for_geometry(tmp_path: Path):
    image_dir = tmp_path / "00"
    image_dir.mkdir()

    poscar_text = """Si2
1.0
        3.8669745922         0.0000000000         0.0000000000
        1.9334872961         3.3488982326         0.0000000000
        1.9334872961         1.1162994109         3.1573715331
   Si
    2
Direct
     0.750000000         0.750000000         0.750000000
     0.500000000         0.500000000         0.500000000
"""
    contcar_text = """Si2
1.0
        3.8669745922         0.0000000000         0.0000000000
        1.9334872961         3.3488982326         0.0000000000
        1.9334872961         1.1162994109         3.1573715331
   Si
    2
Direct
     0.250000000         0.750000000         0.750000000
     0.500000000         0.500000000         0.500000000
"""
    (image_dir / "POSCAR").write_text(poscar_text)
    (image_dir / "CONTCAR").write_text(contcar_text)

    results = vpmdk._collect_neb_image_results([str(image_dir)], potcar_path=None)

    assert len(results) == 1
    scaled = results[0].atoms.get_scaled_positions()
    assert scaled[0][0] == pytest.approx(0.25, rel=1e-12, abs=1e-12)


def test_collect_neb_image_results_raises_on_malformed_vasprun(tmp_path: Path):
    image_dir = tmp_path / "00"
    image_dir.mkdir()

    poscar_text = """Si2
1.0
        3.8669745922         0.0000000000         0.0000000000
        1.9334872961         3.3488982326         0.0000000000
        1.9334872961         1.1162994109         3.1573715331
   Si
    2
Direct
     0.750000000         0.750000000         0.750000000
     0.500000000         0.500000000         0.500000000
"""
    (image_dir / "POSCAR").write_text(poscar_text)
    (image_dir / "vasprun.xml").write_text("<modeling><calculation></modeling>")

    with pytest.raises(RuntimeError, match="Failed to parse NEB image vasprun.xml"):
        vpmdk._collect_neb_image_results([str(image_dir)], potcar_path=None)


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
            "MLP": "ORB",
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


def test_build_chgnet_calculator_respects_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    captured: dict[str, object] = {}

    class DummyCHGNet:
        def __init__(self, model_name=None, use_device=None, **_):
            captured.update({"model": model_name, "device": use_device})

    model_path = tmp_path / "chgnet.pt"
    model_path.write_text("dummy")

    monkeypatch.setattr(vpmdk, "CHGNetCalculator", DummyCHGNet)

    calculator = vpmdk._build_chgnet_calculator(
        {"MLP": "CHGNET", "MODEL": str(model_path), "DEVICE": "cpu"}
    )

    assert isinstance(calculator, DummyCHGNet)
    assert captured == {"model": str(model_path), "device": "cpu"}


def test_build_m3gnet_calculator_respects_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    captured: dict[str, object] = {}

    class DummyM3GNet:
        def __init__(self, model_path=None, *, potential=None, device=None, **_):
            captured.update(
                {"model": model_path, "potential": potential, "device": device}
            )

    model_path = tmp_path / "m3gnet.ckpt"
    model_path.write_text("dummy")

    monkeypatch.setattr(vpmdk, "M3GNetCalculator", DummyM3GNet)
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)

    calculator = vpmdk._build_m3gnet_calculator(
        {"MLP": "M3GNET", "MODEL": str(model_path), "DEVICE": "cuda:0"}
    )

    assert isinstance(calculator, DummyM3GNet)
    assert captured == {"model": str(model_path), "potential": None, "device": "cuda:0"}


def test_build_deepmd_calculator_infers_type_map(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    model_path = tmp_path / "graph.pb"
    model_path.write_text("dummy")

    captured: dict[str, object] = {}

    class DummyDeePMD:
        def __init__(self, model=None, **kwargs):
            captured["model"] = model
            captured["kwargs"] = kwargs

    structure = type("S", (), {"species": ["Si", "Si"], "site_symbols": ["Si"]})()

    monkeypatch.setattr(vpmdk, "DeePMDCalculator", DummyDeePMD)

    calculator = vpmdk._build_deepmd_calculator(
        {"MLP": "DEEPMD", "MODEL": str(model_path)}, structure=structure
    )

    assert isinstance(calculator, DummyDeePMD)
    assert captured["model"] == str(model_path)
    assert captured["kwargs"].get("type_map") == ["Si"]
