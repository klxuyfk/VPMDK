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


def test_get_calculator_accepts_upet_named_model(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_calc(**kwargs):
        captured.update(kwargs)
        return "upet"

    monkeypatch.setattr(vpmdk, "UPETCalculator", fake_calc)

    calculator = vpmdk.get_calculator({"MLP": "UPET", "MODEL": "pet-oam-xl"})

    assert calculator == "upet"
    assert captured["model"] == "pet-oam-xl"


def test_get_calculator_accepts_flashtp_backend(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_builder(tags):
        captured["tags"] = tags
        return "flashtp"

    monkeypatch.setattr(vpmdk, "_build_flashtp_calculator", fake_builder)

    calculator = vpmdk.get_calculator({"MLP": "FlashTP", "MODEL": "7net-0"})

    assert calculator == "flashtp"
    assert captured["tags"] == {"MLP": "FlashTP", "MODEL": "7net-0"}


def test_get_calculator_accepts_equflash_backend(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_builder(tags):
        captured["tags"] = tags
        return "equflash"

    monkeypatch.setattr(vpmdk, "_build_equflash_calculator", fake_builder)

    calculator = vpmdk.get_calculator({"MLP": "EquFlash", "MODEL": "/tmp/equflash.ckpt"})

    assert calculator == "equflash"
    assert captured["tags"] == {"MLP": "EquFlash", "MODEL": "/tmp/equflash.ckpt"}


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

    calculator = vpmdk.get_calculator({"MLP": "EQNORM", "MODEL": "eqnorm"})

    assert calculator == "eqnorm"
    assert captured["model_name"] == "eqnorm"
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

    calculator = vpmdk.get_calculator({"MLP": "HIENET", "MODEL": "hienet"})

    assert calculator == "hienet"
    assert captured["model_name"] == "hienet"
    assert captured["calc_model"] == "/tmp/HIENet-V3.pth"
    assert captured["file_type"] == "checkpoint"


def test_get_calculator_accepts_nequix_named_model(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class FakeNequixCalculator:
        URLS = {vpmdk.DEFAULT_NEQUIX_MODEL: "https://example.invalid/nequix-mp-1.nqx"}

        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(vpmdk, "NequixCalculator", FakeNequixCalculator)

    calculator = vpmdk.get_calculator({"MLP": "NEQUIX", "MODEL": "NEQUIX-MP-1"})

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

    calculator = vpmdk.get_calculator({"MLP": "ALPHANET", "MODEL": "AlphaNet-MATPES-r2scan"})

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

    calculator = vpmdk.get_calculator({"MLP": "ALPHANET"}, structure=structure)

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

    calculator = vpmdk.get_calculator({"MLP": "MATRIS", "MODEL": "matris_10m_mp"})

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

    calculator = vpmdk.get_calculator({"MLP": "TACE", "MODEL": "TACE-v1-OAM-M"})

    assert calculator == "tace"
    assert captured["model"] == "/tmp/TACE-v1-OAM-M.pt"


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

    def fake_load(*, model_path: str | None, device: str | None, graph_converter_algorithm: str):
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
