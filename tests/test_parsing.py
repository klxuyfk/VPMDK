from __future__ import annotations

import os
import subprocess
import sys
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


def test_get_calculator_accepts_upet_named_model(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_calc(**kwargs):
        captured.update(kwargs)
        return "upet"

    monkeypatch.setattr(vpmdk, "UPETCalculator", fake_calc)

    calculator = vpmdk.get_calculator(
        vpmdk.BackendConfig(mlp="UPET", model="pet-oam-xl")
    )

    assert calculator == "upet"
    assert captured["model"] == "pet-oam-xl"


def test_get_calculator_accepts_flashtp_backend(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_builder(tags):
        captured["tags"] = tags
        return "flashtp"

    monkeypatch.setattr(vpmdk, "_build_flashtp_calculator", fake_builder)

    calculator = vpmdk.get_calculator(
        vpmdk.BackendConfig(mlp="FlashTP", model="7net-0")
    )

    assert calculator == "flashtp"
    assert captured["tags"] == {"MLP": "FLASHTP", "MODEL": "7net-0"}


def test_get_calculator_accepts_equflash_backend(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_builder(tags):
        captured["tags"] = tags
        return "equflash"

    monkeypatch.setattr(vpmdk, "_build_equflash_calculator", fake_builder)

    calculator = vpmdk.get_calculator(
        vpmdk.BackendConfig(mlp="EquFlash", model="/tmp/equflash.ckpt")
    )

    assert calculator == "equflash"
    assert captured["tags"] == {"MLP": "EQUFLASH", "MODEL": "/tmp/equflash.ckpt"}


def test_get_calculator_accepts_eqnorm_named_model(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_ensure(model_name: str):
        captured["model_name"] = model_name
        return (
            {"model_name": "eqnorm", "model_variant": vpmdk.DEFAULT_EQNORM_MODEL},
            "/tmp/eqnorm-mptrj.pt",
        )

    def fake_safe_globals():
        captured["safe_globals"] = True

    def fake_calc(*, model_name, model_variant, device="cpu", compile=False):
        captured.update(
            {
                "calc_model_name": model_name,
                "calc_variant": model_variant,
                "device": device,
                "compile": compile,
            }
        )
        return "eqnorm"

    monkeypatch.setattr(vpmdk, "_ensure_eqnorm_named_model_checkpoint", fake_ensure)
    monkeypatch.setattr(vpmdk, "_ensure_eqnorm_torch_safe_globals", fake_safe_globals)
    monkeypatch.setattr(vpmdk, "EqnormCalculator", fake_calc)

    calculator = vpmdk.get_calculator(vpmdk.BackendConfig(mlp="EQNORM", model="eqnorm"))

    assert calculator == "eqnorm"
    assert captured["model_name"] == vpmdk.DEFAULT_EQNORM_MODEL
    assert captured["calc_variant"] == vpmdk.DEFAULT_EQNORM_MODEL


def test_get_calculator_accepts_hienet_named_model(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_ensure(model_name: str):
        captured["model_name"] = model_name
        return ({"display_name": vpmdk.DEFAULT_HIENET_MODEL}, "/tmp/HIENet-V3.pth")

    def fake_calc(*, model, file_type="checkpoint", device="cpu"):
        captured.update({"calc_model": model, "file_type": file_type, "device": device})
        return "hienet"

    monkeypatch.setattr(vpmdk, "_ensure_hienet_named_model_checkpoint", fake_ensure)
    monkeypatch.setattr(vpmdk, "HIENetCalculator", fake_calc)

    calculator = vpmdk.get_calculator(vpmdk.BackendConfig(mlp="HIENET", model="hienet"))

    assert calculator == "hienet"
    assert captured["model_name"] == vpmdk.DEFAULT_HIENET_MODEL
    assert captured["calc_model"] == "/tmp/HIENet-V3.pth"
    assert captured["file_type"] == "checkpoint"


def test_get_calculator_accepts_nequix_named_model(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class FakeNequixCalculator:
        URLS = {vpmdk.DEFAULT_NEQUIX_MODEL: "https://example.invalid/nequix-mp-1.nqx"}

        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(vpmdk, "NequixCalculator", FakeNequixCalculator)

    calculator = vpmdk.get_calculator(
        vpmdk.BackendConfig(mlp="NEQUIX", model="NEQUIX-MP-1")
    )

    assert isinstance(calculator, FakeNequixCalculator)
    assert captured["model_name"] == vpmdk.DEFAULT_NEQUIX_MODEL
    assert captured["backend"] == "jax"
    assert captured["use_kernel"] is False


def test_get_calculator_accepts_alphanet_named_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    captured: dict[str, object] = {}
    config_path = tmp_path / "matpes.json"
    config_path.write_text("{}")

    def fake_ensure(model_name: str):
        captured["model_name"] = model_name
        return ("/tmp/r2scan_1021.ckpt", str(config_path))

    def fake_load(config_file: str, *, precision: str, use_pbc: bool, compute_stress: bool):
        captured["config_file"] = config_file
        return "alpha-config"

    def fake_calc(*, ckpt_path, config, device="cpu", precision="32"):
        captured.update(
            {"ckpt_path": ckpt_path, "config": config, "device": device, "precision": precision}
        )
        return "alphanet"

    monkeypatch.setattr(vpmdk, "AlphaNetCalculator", fake_calc)
    monkeypatch.setattr(vpmdk, "_ensure_alphanet_named_model_files", fake_ensure)
    monkeypatch.setattr(vpmdk, "_load_alphanet_config", fake_load)

    calculator = vpmdk.get_calculator(
        vpmdk.BackendConfig(mlp="ALPHANET", model="AlphaNet-MATPES-r2scan")
    )

    assert calculator == "alphanet"
    assert captured["model_name"] == "AlphaNet-MATPES-r2scan"
    assert captured["ckpt_path"] == "/tmp/r2scan_1021.ckpt"
    assert captured["config"] == "alpha-config"


def test_get_calculator_forwards_structure_to_alphanet_builder(
    monkeypatch: pytest.MonkeyPatch,
):
    structure = object()
    captured: dict[str, object] = {}

    def fake_builder(tags, *, structure=None):
        captured["tags"] = tags
        captured["structure"] = structure
        return "alphanet"

    monkeypatch.setattr(vpmdk, "_build_alphanet_calculator", fake_builder)

    calculator = vpmdk.get_calculator(vpmdk.BackendConfig(mlp="ALPHANET"), structure=structure)

    assert calculator == "alphanet"
    assert captured["tags"] == {"MLP": "ALPHANET"}
    assert captured["structure"] is structure


def test_get_calculator_accepts_matris_named_model(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_ensure(model_name: str):
        captured["model_name"] = model_name
        return "/tmp/MatRIS_10M_MP.pth.tar"

    def fake_load(path: str, *, device: str | None):
        captured["load_path"] = path
        captured["load_device"] = device
        return "matris-model"

    def fake_instantiate(*, model, task="efs", device=None):
        captured.update({"model": model, "task": task, "device": device})
        return "matris"

    monkeypatch.setattr(vpmdk, "MatRISCalculator", object)
    monkeypatch.setattr(vpmdk, "_ensure_matris_named_model_checkpoint", fake_ensure)
    monkeypatch.setattr(vpmdk, "_load_matris_checkpoint_model", fake_load)
    monkeypatch.setattr(vpmdk, "_instantiate_matris_calculator", fake_instantiate)

    calculator = vpmdk.get_calculator(
        vpmdk.BackendConfig(mlp="MATRIS", model="matris_10m_mp")
    )

    assert calculator == "matris"
    assert captured["model_name"] == "matris_10m_mp"
    assert captured["load_path"] == "/tmp/MatRIS_10M_MP.pth.tar"
    assert captured["model"] == "matris-model"


def test_get_calculator_accepts_tace_named_model(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_calc(*, model, device=None):
        captured.update({"model": model, "device": device})
        return "tace"

    class DummyRegistry(dict):
        def list_models(self):
            return sorted(self)

    monkeypatch.setattr(vpmdk, "TACEAseCalc", fake_calc)
    monkeypatch.setattr(
        vpmdk,
        "tace_foundations",
        DummyRegistry({"TACE-v1-OAM-M": Path("/tmp/TACE-v1-OAM-M.pt")}),
    )

    calculator = vpmdk.get_calculator(
        vpmdk.BackendConfig(mlp="TACE", model="TACE-v1-OAM-M")
    )

    assert calculator == "tace"
    assert captured["model"] == "/tmp/TACE-v1-OAM-M.pt"


def test_get_calculator_rejects_empty_backend_name():
    with pytest.raises(ValueError, match="MLP"):
        vpmdk.get_calculator(vpmdk.BackendConfig(mlp=""))


@pytest.mark.parametrize(
    "tags, expected",
    [
        ({"WRITE_PSEUDO_SCF": "1"}, True),
        ({"WRITE_PSEUDO_SCF": "on"}, True),
        ({"WRITE_OSZICAR_PSEUDO_SCF": "yes"}, True),
        ({}, False),
        ({"WRITE_PSEUDO_SCF": "off"}, False),
    ],
)
def test_should_write_pseudo_scf(tags, expected):
    assert vpmdk._should_write_pseudo_scf(tags) is expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.0, "0.0E+00"),
        (5.0e-7, "0.5E-06"),
        (2.5e-6, "0.25E-05"),
        (1.0e-4, "0.1E-03"),
    ],
)
def test_format_outcar_ediff_preserves_value(value: float, expected: str):
    assert vpmdk._format_outcar_ediff(value) == expected


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

    assert [str(species) for species in getattr(structure, "species", [])] == [
        "Y",
        "O",
    ]


def test_parse_optional_float_accepts_pymatgen_singleton_list():
    assert vpmdk._parse_optional_float([15.0], key="LANGEVIN_GAMMA") == 15.0


def test_parse_optional_float_accepts_units_in_pymatgen_singleton_list():
    assert vpmdk._parse_optional_float(["2.0 fs"], key="POTIM") == 2.0


def test_build_orb_calculator_uses_bcar_tags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    captured: dict[str, object] = {}
    weights_path = tmp_path / "weights.ckpt"
    weights_path.write_text("placeholder")

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
            "MODEL": str(weights_path),
            "DEVICE": "cuda:1",
            "ORB_MODEL": "custom",
            "ORB_PRECISION": "float64",
            "ORB_COMPILE": "false",
        }
    )

    assert isinstance(calculator, DummyCalculator)
    assert calculator.device == "cuda:1"
    assert captured["weights_path"] == str(weights_path)
    assert captured["precision"] == "float64"
    assert captured["compile"] is False


def test_build_chgnet_calculator_respects_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    captured: dict[str, object] = {}

    class DummyCHGNet:
        @classmethod
        def from_file(cls, model_name=None, use_device=None, **_):
            captured.update({"model": model_name, "device": use_device})
            return cls()

    model_path = tmp_path / "chgnet.pt"
    model_path.write_text("dummy")

    monkeypatch.setattr(vpmdk, "CHGNetCalculator", DummyCHGNet)

    calculator = vpmdk._build_chgnet_calculator(
        {"MLP": "CHGNET", "MODEL": str(model_path), "DEVICE": "cpu"}
    )

    assert isinstance(calculator, DummyCHGNet)
    assert captured == {"model": str(model_path), "device": "cpu"}


def test_build_chgnet_calculator_forwards_graph_converter_algorithm(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    class DummyModel:
        pass

    class DummyCHGNet:
        def __init__(self, model=None, use_device=None, **_):
            captured.update({"model": model, "device": use_device})

    def fake_load(
        *,
        model_path: str | None,
        device: str | None,
        graph_converter_algorithm: str,
        model_reference=None,
    ):
        captured.update(
            {
                "load_model_path": model_path,
                "load_device": device,
                "graph_converter_algorithm": graph_converter_algorithm,
            }
        )
        return DummyModel()

    monkeypatch.setattr(vpmdk, "CHGNetCalculator", DummyCHGNet)
    monkeypatch.setattr(vpmdk, "_load_chgnet_model", fake_load)

    calculator = vpmdk._build_chgnet_calculator(
        {
            "MLP": "CHGNET",
            "DEVICE": "cuda:0",
            "CHGNET_GRAPH_CONVERTER_ALGORITHM": "fast",
        }
    )

    assert isinstance(calculator, DummyCHGNet)
    assert captured["load_model_path"] is None
    assert captured["load_device"] == "cuda:0"
    assert captured["graph_converter_algorithm"] == "fast"
    assert isinstance(captured["model"], DummyModel)
    assert captured["device"] == "cuda:0"


def test_load_chgnet_model_falls_back_to_legacy_load_signature(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}
    legacy_model = object()

    class DummyCHGNetModel:
        @classmethod
        def load(cls, model_name="0.3.0"):
            captured["model_name"] = model_name
            return legacy_model

    def fake_override(model, *, algorithm: str, backend_name: str):
        captured["override"] = {
            "model": model,
            "algorithm": algorithm,
            "backend_name": backend_name,
        }
        return "overridden-model"

    monkeypatch.setattr(vpmdk, "CHGNetModel", DummyCHGNetModel)
    monkeypatch.setattr(vpmdk, "_override_model_graph_converter_algorithm", fake_override)

    model = vpmdk._load_chgnet_model(
        model_path=None,
        device="cuda",
        graph_converter_algorithm="fast",
    )

    assert model == "overridden-model"
    assert captured["model_name"] == "0.3.0"
    assert captured["override"] == {
        "model": legacy_model,
        "algorithm": "fast",
        "backend_name": "CHGNet",
    }


def test_load_chgnet_model_reapplies_algorithm_when_from_file_ignores_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    captured: dict[str, object] = {}
    model_path = tmp_path / "chgnet.pt"
    model_path.write_text("dummy")
    legacy_model = object()

    class DummyCHGNetModel:
        @classmethod
        def from_file(cls, path, **kwargs):
            captured["path"] = path
            captured["kwargs"] = kwargs
            if "graph_converter_algorithm" in kwargs:
                raise TypeError("got multiple values for keyword argument 'graph_converter_algorithm'")
            return legacy_model

    def fake_override(model, *, algorithm: str, backend_name: str):
        captured["override"] = {
            "model": model,
            "algorithm": algorithm,
            "backend_name": backend_name,
        }
        return "overridden-model"

    monkeypatch.setattr(vpmdk, "CHGNetModel", DummyCHGNetModel)
    monkeypatch.setattr(vpmdk, "_override_model_graph_converter_algorithm", fake_override)

    model = vpmdk._load_chgnet_model(
        model_path=str(model_path),
        device="cuda",
        graph_converter_algorithm="fast",
    )

    assert model == "overridden-model"
    assert captured["path"] == str(model_path)
    assert captured["kwargs"] == {}
    assert captured["override"] == {
        "model": legacy_model,
        "algorithm": "fast",
        "backend_name": "CHGNet",
    }


def test_load_chgnet_model_forwards_named_model_to_load(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}
    named_model = object()

    class DummyCHGNetModel:
        @classmethod
        def load(cls, model_name="0.3.0", use_device=None, verbose=True):
            captured["load"] = {
                "model_name": model_name,
                "use_device": use_device,
                "verbose": verbose,
            }
            return named_model

    def fake_override(model, *, algorithm: str, backend_name: str):
        captured["override"] = {
            "model": model,
            "algorithm": algorithm,
            "backend_name": backend_name,
        }
        return "overridden-model"

    monkeypatch.setattr(vpmdk, "CHGNetModel", DummyCHGNetModel)
    monkeypatch.setattr(vpmdk, "_override_model_graph_converter_algorithm", fake_override)

    model = vpmdk._load_chgnet_model(
        model_path="0.2.0",
        device="cuda:0",
        graph_converter_algorithm="fast",
    )

    assert model == "overridden-model"
    assert captured["load"] == {
        "model_name": "0.2.0",
        "use_device": "cuda:0",
        "verbose": False,
    }
    assert captured["override"] == {
        "model": named_model,
        "algorithm": "fast",
        "backend_name": "CHGNet",
    }


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


def test_potcar_species_relabelling_works_with_real_pymatgen(tmp_path):
    # R129 (P1): `poscar.site_symbols = ...` is a READ-ONLY property in real
    # pymatgen, so the whole POSCAR/POTCAR species reconciliation -- including the
    # documented "Using POTCAR order" warning -- raised AttributeError and was
    # reported as invalid input (exit 1) for exactly the files it exists to repair.
    # conftest's Poscar STUB exposes site_symbols as a plain attribute, so the suite
    # could not see it; this test therefore runs against the real library.
    script = (
        "import sys, tempfile\n"
        "from pymatgen.io.vasp import Poscar\n"
        "from vpmdk_core.io.inputs import _apply_species_from_potcar\n"
        "d = tempfile.mkdtemp()\n"
        "open(d + '/POSCAR', 'w').write("
        "'Si2\\n1.0\\n 2.7 2.7 0.0\\n 0.0 2.7 2.7\\n 2.7 0.0 2.7\\nSi\\n2\\n"
        "Direct\\n 0.0 0.0 0.0\\n 0.25 0.25 0.25\\n')\n"
        "p = Poscar.from_file(d + '/POSCAR', check_for_potcar=False, "
        "read_velocities=False)\n"
        "structure = p.structure\n"
        "out = _apply_species_from_potcar(p, structure, ['Ge'])\n"
        "print('SPECIES', [str(site.specie) for site in out])\n"
        "print('ORIGINAL', [str(site.specie) for site in structure])\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "VPMDK_TEST_REAL_PYMATGEN": "1"},
    )
    if "ModuleNotFoundError" in completed.stderr:
        pytest.skip("real pymatgen is not installed")
    assert "SPECIES ['Ge', 'Ge']" in completed.stdout, (
        completed.stdout + completed.stderr
    )
    # The input structure must not be mutated in place.
    assert "ORIGINAL ['Si', 'Si']" in completed.stdout


def test_vasp4_poscar_without_potcar_is_rejected_not_computed_as_hydrogen(tmp_path):
    # R130 (P1): with real pymatgen a VASP-4 POSCAR (no species line -- a legitimate
    # VASP format) does NOT produce empty site_symbols: pymatgen fabricates
    # ['H', ...] and only emits a BadPoscarWarning on stderr. VPMDK's own
    # "no species names" branch was therefore unreachable, and a Si cell was
    # silently computed as HYDROGEN, with CONTCAR rewritten to match. Real VASP
    # takes the species from the POTCAR in this format, so without one the elements
    # genuinely cannot be determined.
    vasp4 = (
        "Si2 vasp4\n1.0\n 2.7 2.7 0.0\n 0.0 2.7 2.7\n 2.7 0.0 2.7\n"
        "2\nDirect\n 0.0 0.0 0.0\n 0.25 0.25 0.25\n"
    )
    script = (
        "import sys\n"
        "from vpmdk_core.io.inputs import read_structure, _poscar_declares_species\n"
        "path = sys.argv[1]\n"
        "print('DECLARES', _poscar_declares_species(path))\n"
        "try:\n"
        "    st = read_structure(path)\n"
        "except ValueError as exc:\n"
        "    print('REJECTED', exc)\n"
        "else:\n"
        "    print('ACCEPTED', [str(s.specie) for s in st])\n"
    )

    v4_path = tmp_path / "POSCAR_v4"
    v4_path.write_text(vasp4)
    v5_path = tmp_path / "POSCAR_v5"
    v5_path.write_text(vasp4.replace("2\nDirect", "Si\n2\nDirect"))

    def run(path):
        return subprocess.run(
            [sys.executable, "-c", script, str(path)],
            capture_output=True,
            text=True,
            env={**os.environ, "VPMDK_TEST_REAL_PYMATGEN": "1"},
        )

    v4 = run(v4_path)
    if "ModuleNotFoundError" in v4.stderr:
        pytest.skip("real pymatgen is not installed")
    assert "DECLARES False" in v4.stdout, v4.stdout + v4.stderr
    assert "REJECTED" in v4.stdout, v4.stdout + v4.stderr

    v5 = run(v5_path)
    assert "DECLARES True" in v5.stdout, v5.stdout + v5.stderr
    assert "ACCEPTED ['Si', 'Si']" in v5.stdout, v5.stdout + v5.stderr

    # Self-audit follow-up: a POTCAR that EXISTS but yields no usable symbols
    # (unreadable, or rejected by pymatgen's validation) is not a species source
    # either -- without this the fabricated ['H', ...] names survived exactly as
    # they did with no POTCAR at all.
    potcar = tmp_path / "POTCAR"
    potcar.write_text("this is not a valid POTCAR\n")
    script_with_potcar = script.replace(
        "read_structure(path)", "read_structure(path, sys.argv[2])"
    )

    def run_with_potcar(path):
        return subprocess.run(
            [sys.executable, "-c", script_with_potcar, str(path), str(potcar)],
            capture_output=True,
            text=True,
            env={**os.environ, "VPMDK_TEST_REAL_PYMATGEN": "1"},
        )

    assert "REJECTED" in run_with_potcar(v4_path).stdout
    assert "ACCEPTED ['Si', 'Si']" in run_with_potcar(v5_path).stdout


def _minimal_potcar_entry(symbol: str) -> str:
    """One POTCAR dataset real pymatgen accepts (header keywords + psp data)."""

    header = (
        f" PAW_PBE {symbol} 05Jan2001\n"
        "  11.0000000000000\n"
        " parameters from PSCTR are:\n"
        f"   VRHFIN ={symbol}: d10 p1\n"
        "   LEXCH  = PE\n"
        "   EATOM  =     0.0000 eV,       0.0000 Ry\n"
        "\n"
        f"   TITEL  = PAW_PBE {symbol} 05Jan2001\n"
        "   LULTRA =        F    use ultrasoft PP ?\n"
        "   IUNSCR =        1    unscreen: 0-lin 1-nonlin 2-no\n"
        "   RPACOR =    2.000    partial core radius\n"
        "   POMASS =   63.546; ZVAL   =   11.000    mass and valenz\n"
        "   RCORE  =    2.300    outmost cutoff radius\n"
        "   RWIGS  =    2.500; RWIGS  =    1.323    wigner-seitz radius (au A)\n"
        "   ENMAX  =  295.446; ENMIN  =  221.585 eV\n"
        "   RCLOC  =    1.680    cutoff for local pot\n"
        "   LCOR   =        T    correct aug charges\n"
        "   LPAW   =        T    paw PP\n"
        "   EAUG   =  649.837\n"
        "   RMAX   =    2.750    core radius for proj-oper\n"
        "   RAUG   =    1.300    factor for augmentation sphere\n"
        "   RDEP   =    2.363    radius for radial grids\n"
        "   QCUT   =   -4.216; QGAM   =    8.433    optimization parameters\n"
        " END of PSCTR-controll parameters\n"
        " local part\n"
    )
    data = "".join(
        "  " + "  ".join(f"{(row * 5 + col) * 0.1234567:18.12E}" for col in range(5)) + "\n"
        for row in range(6)
    )
    return header + data + " End of Dataset\n"


def test_vasp4_poscar_with_mismatching_potcar_species_count_is_rejected(tmp_path):
    # R131: R130 closed the "no POTCAR" and "unparseable POTCAR" variants of the
    # fabricated-hydrogen hole, but not the COUNT MISMATCH one. For a VASP-4
    # POSCAR (no species line) _apply_species_from_potcar returned the input
    # structure unchanged whenever the POTCAR's species count differed from the
    # number of ion groups -- and that unchanged structure is pymatgen's
    # fabricated ['H', ...]. In a NEB image directory, where pymatgen's implicit
    # same-directory POTCAR lookup cannot reach the band's POTCAR one level up,
    # a whole Cu band was therefore computed as hydrogen, CONTCARs were rewritten
    # with species line "H", and the run exited 0 with "Calculation completed.".
    # A POSCAR that DOES name its species keeps the lenient behavior: there the
    # unchanged structure is still labelled with real elements.
    image = tmp_path / "00"
    image.mkdir()
    body = (
        "1.0\n 3.615 0.0 0.0\n 0.0 3.615 0.0\n 0.0 0.0 3.615\n"
        "{species}4\nDirect\n 0.0 0.0 0.0\n 0.0 0.5 0.5\n 0.5 0.0 0.5\n 0.5 0.5 0.0\n"
    )
    (image / "POSCAR").write_text("Cu4 vasp4\n" + body.format(species=""))
    v5_image = tmp_path / "01"
    v5_image.mkdir()
    (v5_image / "POSCAR").write_text("Cu4 vasp5\n" + body.format(species="Cu\n"))

    two = tmp_path / "POTCAR_CuAg"
    two.write_text(_minimal_potcar_entry("Cu") + _minimal_potcar_entry("Ag"))
    one = tmp_path / "POTCAR_Cu"
    one.write_text(_minimal_potcar_entry("Cu"))

    script = (
        "import sys, warnings\n"
        "warnings.simplefilter('ignore')\n"
        "from vpmdk_core.io.inputs import read_structure\n"
        "try:\n"
        "    st = read_structure(sys.argv[1], sys.argv[2])\n"
        "except ValueError as exc:\n"
        "    print('REJECTED', exc)\n"
        "else:\n"
        "    print('ACCEPTED', [str(s.specie) for s in st])\n"
    )

    def run(poscar: Path, potcar: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-c", script, str(poscar), str(potcar)],
            capture_output=True,
            text=True,
            env={**os.environ, "VPMDK_TEST_REAL_PYMATGEN": "1"},
        )

    mismatch = run(image / "POSCAR", two)
    if "ModuleNotFoundError" in mismatch.stderr:
        pytest.skip("real pymatgen is not installed")
    assert "REJECTED" in mismatch.stdout, mismatch.stdout + mismatch.stderr
    assert "'Cu', 'Ag'" in mismatch.stdout

    # The matching POTCAR still resolves the species, which is the whole point of
    # the VASP-4 branch.
    matching = run(image / "POSCAR", one)
    assert "ACCEPTED ['Cu', 'Cu', 'Cu', 'Cu']" in matching.stdout, (
        matching.stdout + matching.stderr
    )

    # A VASP-5 POSCAR names its own species: an unusable POTCAR refinement must
    # NOT turn into a rejection there.
    declared = run(v5_image / "POSCAR", two)
    assert "ACCEPTED ['Cu', 'Cu', 'Cu', 'Cu']" in declared.stdout, (
        declared.stdout + declared.stderr
    )


@pytest.mark.parametrize(
    "line6,declares",
    [
        (" 2   # number of Si atoms", False),  # VASP 4 with a comment
        ("2", False),  # plain VASP 4
        ("Si  # species line", True),  # VASP 5 with a comment
        ("Si", True),  # plain VASP 5
        ("  2  2\t# two groups ", False),
        ("# the whole line is a comment", False),  # pymatgen reads [] counts
    ],
)
def test_poscar_format_is_decided_on_the_line_the_parser_sees(
    tmp_path, line6: str, declares: bool
):
    # R131 fixed the same class for INCAR (VPMDK's own tokenizer disagreed with
    # pymatgen's); R132 found it in the POSCAR classifier. pymatgen pushes every
    # POSCAR line through clean_lines(), which TRUNCATES AT '#', so
    # `` 2   # number of Si atoms`` reads as ``2`` -> VASP 4 -> fabricated
    # ['H', ...] names. Reading the RAW line here made int('#') fail, reported
    # "VASP 5", and SKIPPED every VASP-4 guard: the Si cell was computed as H2
    # (-2.35 eV instead of -10.63 eV), exited 0, and wrote a CONTCAR with
    # species line ``H``.
    from vpmdk_core.io.inputs import _poscar_declares_species

    path = tmp_path / f"POSCAR_{abs(hash(line6)) % 9973}"
    path.write_text(
        "Si2\n1.0\n 2.715 2.715 0.0\n 0.0 2.715 2.715\n 2.715 0.0 2.715\n"
        f"{line6}\nDirect\n 0.0 0.0 0.0\n 0.25 0.25 0.25\n"
    )

    assert _poscar_declares_species(str(path)) is declares


def test_cleaned_poscar_lines_matches_pymatgens_reader(tmp_path):
    from vpmdk_core.io.inputs import _cleaned_poscar_lines

    text = (
        "Si2 # a comment on the title\n"
        "1.0\n"
        " 2.715 2.715 0.0\n"
        " 0.0 2.715 2.715\n"
        " 2.715 0.0 2.715\n"
        " 2 # counts\n"
        "Direct\n"
        " 0.0 0.0 0.0\n"
        " 0.25 0.25 0.25\n"
        "\n"
        " 0.0 0.0 0.0\n"  # velocities: a separate chunk, dropped like pymatgen
        " 0.0 0.0 0.0\n"
    )
    path = tmp_path / "POSCAR"
    path.write_text(text)

    lines = _cleaned_poscar_lines(path.read_text())

    assert lines[0] == "Si2"
    assert lines[5] == "2"
    assert lines[6] == "Direct"
    assert len(lines) == 9  # structure block only


def test_vasp4_poscar_with_a_commented_counts_line_is_not_computed_as_hydrogen(tmp_path):
    # End-to-end with the REAL library (the suite stubs pymatgen), which is the
    # only place the fabricated ['H', ...] names appear at all.
    body = (
        "Si2\n1.0\n 2.715 2.715 0.0\n 0.0 2.715 2.715\n 2.715 0.0 2.715\n"
        "{line6}\nDirect\n 0.0 0.0 0.0\n 0.25 0.25 0.25\n"
    )
    v4 = tmp_path / "POSCAR_v4_comment"
    v4.write_text(body.format(line6=" 2   # number of Si atoms"))
    v5 = tmp_path / "POSCAR_v5_comment"
    v5.write_text(body.format(line6="Si  # species").replace("Direct", "2\nDirect"))

    script = (
        "import sys, warnings\n"
        "warnings.simplefilter('ignore')\n"
        "from vpmdk_core.io.inputs import read_structure, _poscar_declares_species\n"
        "path = sys.argv[1]\n"
        "print('DECLARES', _poscar_declares_species(path))\n"
        "try:\n"
        "    st = read_structure(path)\n"
        "except ValueError as exc:\n"
        "    print('REJECTED', exc)\n"
        "else:\n"
        "    print('ACCEPTED', [str(s.specie) for s in st])\n"
    )

    def run(path):
        return subprocess.run(
            [sys.executable, "-c", script, str(path)],
            capture_output=True,
            text=True,
            env={**os.environ, "VPMDK_TEST_REAL_PYMATGEN": "1"},
        )

    commented_v4 = run(v4)
    if "ModuleNotFoundError" in commented_v4.stderr:
        pytest.skip("real pymatgen is not installed")
    assert "DECLARES False" in commented_v4.stdout, (
        commented_v4.stdout + commented_v4.stderr
    )
    assert "REJECTED" in commented_v4.stdout, commented_v4.stdout + commented_v4.stderr

    commented_v5 = run(v5)
    assert "DECLARES True" in commented_v5.stdout, (
        commented_v5.stdout + commented_v5.stderr
    )
    assert "ACCEPTED ['Si', 'Si']" in commented_v5.stdout, (
        commented_v5.stdout + commented_v5.stderr
    )
