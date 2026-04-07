from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import vpmdk


def test_nequip_uses_compiled_model_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model_path = tmp_path / "model.pth"
    model_path.write_text("dummy")
    seen: dict[str, object] = {}

    def from_compiled_model(path, device=None):
        seen["path"] = path
        seen["device"] = device
        return "nequip-compiled"

    monkeypatch.setattr(
        vpmdk,
        "NequIPCalculator",
        SimpleNamespace(from_compiled_model=from_compiled_model),
    )

    calc = vpmdk._build_nequip_calculator({"MODEL": str(model_path), "DEVICE": "cuda"})

    assert calc == "nequip-compiled"
    assert seen == {"path": str(model_path), "device": "cuda"}


def test_matgl_load_model_path_is_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model_dir = tmp_path / "M3GNet-MP"
    model_dir.mkdir()
    seen: dict[str, object] = {}

    def fake_load_model(path):
        seen["load_path"] = path
        return "potential"

    def fake_calc(*args, **kwargs):
        seen["calc_args"] = args
        seen["calc_kwargs"] = kwargs
        return "calc"

    monkeypatch.setattr(vpmdk, "M3GNetCalculator", fake_calc)
    monkeypatch.setattr(vpmdk, "MatGLLoadModel", fake_load_model)
    monkeypatch.setattr(vpmdk, "LegacyM3GNetPotential", None)

    calc = vpmdk._build_m3gnet_calculator({"MODEL": str(model_dir)})

    assert calc == "calc"
    assert seen["load_path"] == str(model_dir)
    assert seen["calc_args"] == ("potential",)


def test_eqnorm_uses_checkpoint_path_and_bcar_tags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_path = tmp_path / "eqnorm-omat.pth"
    model_path.write_text("dummy")
    seen: dict[str, object] = {}

    @contextmanager
    def fake_stage(path: str, variant: str):
        seen["staged_path"] = path
        seen["variant"] = variant
        yield f"/tmp/{variant}.pt"

    def fake_safe_globals():
        seen["safe_globals"] = True

    def fake_calc(*, model_name, model_variant, device="cpu", compile=False):
        seen["model_name"] = model_name
        seen["model_variant"] = model_variant
        seen["device"] = device
        seen["compile"] = compile
        return "eqnorm"

    monkeypatch.setattr(vpmdk, "_temporarily_stage_eqnorm_local_checkpoint", fake_stage)
    monkeypatch.setattr(vpmdk, "_ensure_eqnorm_torch_safe_globals", fake_safe_globals)
    monkeypatch.setattr(vpmdk, "EqnormCalculator", fake_calc)

    calc = vpmdk._build_eqnorm_calculator(
        {"MODEL": str(model_path), "DEVICE": "cuda:0", "EQNORM_COMPILE": "true"}
    )

    assert calc == "eqnorm"
    assert seen == {
        "staged_path": str(model_path),
        "variant": "eqnorm-omat",
        "safe_globals": True,
        "model_name": "eqnorm",
        "model_variant": "eqnorm-omat",
        "device": "cuda:0",
        "compile": True,
    }


def test_eqnorm_accepts_named_model_and_defaults(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, object] = {}

    def fake_ensure(model_name: str):
        seen["model_name"] = model_name
        return (
            {"model_name": "eqnorm", "model_variant": vpmdk.DEFAULT_EQNORM_MODEL},
            "/tmp/eqnorm-mptrj.pt",
        )

    def fake_safe_globals():
        seen["safe_globals"] = True

    def fake_calc(*, model_name, model_variant, device="cpu", compile=False):
        seen["calc_model_name"] = model_name
        seen["calc_variant"] = model_variant
        seen["device"] = device
        seen["compile"] = compile
        return "eqnorm"

    monkeypatch.setattr(vpmdk, "_ensure_eqnorm_named_model_checkpoint", fake_ensure)
    monkeypatch.setattr(vpmdk, "_ensure_eqnorm_torch_safe_globals", fake_safe_globals)
    monkeypatch.setattr(vpmdk, "EqnormCalculator", fake_calc)

    calc = vpmdk._build_eqnorm_calculator({})

    assert calc == "eqnorm"
    assert seen == {
        "model_name": vpmdk.DEFAULT_EQNORM_MODEL,
        "safe_globals": True,
        "calc_model_name": "eqnorm",
        "calc_variant": vpmdk.DEFAULT_EQNORM_MODEL,
        "device": "cpu",
        "compile": False,
    }


def test_eqnorm_restores_named_cache_after_local_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    local_checkpoint = tmp_path / "custom-eqnorm.pt"
    local_checkpoint.write_text("local-weights")
    cache_dir = tmp_path / ".cache" / "eqnorm"
    cache_dir.mkdir(parents=True)
    named_cache_path = cache_dir / "eqnorm-omat.pt"
    named_cache_path.write_text("official-weights")
    seen: dict[str, object] = {}

    original_expanduser = vpmdk.os.path.expanduser

    def fake_expanduser(path: str) -> str:
        if path == "~/.cache/eqnorm":
            return str(cache_dir)
        return original_expanduser(path)

    def fake_safe_globals():
        seen["safe_globals"] = True

    def fake_calc(*, model_name, model_variant, device="cpu", compile=False):
        seen["loaded_contents"] = named_cache_path.read_text()
        return "eqnorm"

    monkeypatch.setattr(vpmdk.os.path, "expanduser", fake_expanduser)
    monkeypatch.setattr(vpmdk, "_ensure_eqnorm_torch_safe_globals", fake_safe_globals)
    monkeypatch.setattr(vpmdk, "EqnormCalculator", fake_calc)

    calc = vpmdk._build_eqnorm_calculator(
        {"MODEL": str(local_checkpoint), "EQNORM_VARIANT": "eqnorm-omat"}
    )

    assert calc == "eqnorm"
    assert seen["safe_globals"] is True
    assert seen["loaded_contents"] == "local-weights"
    assert named_cache_path.read_text() == "official-weights"
    assert not list(cache_dir.glob(".eqnorm-omat.vpmdk-backup-*.pt"))


def test_eqnorm_missing_checkpoint_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(vpmdk, "EqnormCalculator", object)

    missing_path = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError, match="not found"):
        vpmdk._build_eqnorm_calculator({"MODEL": str(missing_path)})


def test_eqnorm_requires_variant_for_unknown_local_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(vpmdk, "EqnormCalculator", object)

    model_path = tmp_path / "custom-model.pt"
    model_path.write_text("dummy")
    with pytest.raises(ValueError, match="EQNORM_VARIANT"):
        vpmdk._build_eqnorm_calculator({"MODEL": str(model_path)})


def test_hienet_accepts_named_model_and_defaults(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, object] = {}

    def fake_ensure(model_name: str):
        seen["model_name"] = model_name
        return ({"display_name": vpmdk.DEFAULT_HIENET_MODEL}, "/tmp/HIENet-V3.pth")

    def fake_calc(*, model, file_type="checkpoint", device="cpu"):
        seen["calc_model"] = model
        seen["file_type"] = file_type
        seen["device"] = device
        return "hienet"

    monkeypatch.setattr(vpmdk, "_ensure_hienet_named_model_checkpoint", fake_ensure)
    monkeypatch.setattr(vpmdk, "HIENetCalculator", fake_calc)

    calc = vpmdk._build_hienet_calculator({})

    assert calc == "hienet"
    assert seen == {
        "model_name": vpmdk.DEFAULT_HIENET_MODEL,
        "calc_model": "/tmp/HIENet-V3.pth",
        "file_type": "checkpoint",
        "device": "cpu",
    }


def test_hienet_uses_checkpoint_path_and_bcar_tags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_path = tmp_path / "custom-hienet.ckpt"
    model_path.write_text("dummy")
    seen: dict[str, object] = {}

    def fake_calc(*, model, file_type="checkpoint", device="cpu"):
        seen["model"] = model
        seen["file_type"] = file_type
        seen["device"] = device
        return "hienet"

    monkeypatch.setattr(vpmdk, "HIENetCalculator", fake_calc)

    calc = vpmdk._build_hienet_calculator(
        {
            "MODEL": str(model_path),
            "HIENET_FILE_TYPE": "checkpoint",
            "DEVICE": "cuda:0",
        }
    )

    assert calc == "hienet"
    assert seen == {
        "model": str(model_path),
        "file_type": "checkpoint",
        "device": "cuda:0",
    }


def test_hienet_missing_checkpoint_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(vpmdk, "HIENetCalculator", object)

    missing_path = tmp_path / "missing.pth"
    with pytest.raises(FileNotFoundError, match="not found"):
        vpmdk._build_hienet_calculator({"MODEL": str(missing_path)})


def test_hienet_invalid_file_type_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(vpmdk, "HIENetCalculator", object)

    with pytest.raises(ValueError, match="HIENET_FILE_TYPE"):
        vpmdk._build_hienet_calculator({"HIENET_FILE_TYPE": "weights"})


def test_nequix_accepts_named_model_and_defaults(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, object] = {}

    class FakeNequixCalculator:
        URLS = {
            vpmdk.DEFAULT_NEQUIX_MODEL: "https://example.invalid/nequix-mp-1.nqx",
            "nequix-oam-1": "https://example.invalid/nequix-oam-1.nqx",
        }

        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(vpmdk, "NequixCalculator", FakeNequixCalculator)

    calc = vpmdk._build_nequix_calculator({})

    assert isinstance(calc, FakeNequixCalculator)
    assert seen == {
        "model_name": vpmdk.DEFAULT_NEQUIX_MODEL,
        "backend": "jax",
        "use_kernel": False,
        "use_compile": False,
        "capacity_multiplier": 1.1,
    }


def test_nequix_uses_checkpoint_path_and_torch_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_path = tmp_path / "nequix-oam-1.nqx"
    model_path.write_text("dummy")
    seen: dict[str, object] = {}

    class FakeModel:
        def to(self, device):
            seen["moved_to"] = str(device)
            return self

        def eval(self):
            seen["eval_called"] = True

    class FakeNequixCalculator(vpmdk.Calculator):
        def __init__(self, **kwargs):
            raise AssertionError("torch path should not call NequixCalculator.__init__")

    def fake_from_pretrained(*, model_name=None, model_path=None, backend="jax", use_kernel=True):
        seen["from_pretrained"] = {
            "model_name": model_name,
            "model_path": model_path,
            "backend": backend,
            "use_kernel": use_kernel,
        }
        return FakeModel(), {"atomic_numbers": [8, 14], "cutoff": 6.0}

    def fake_atomic_numbers_to_indices(atomic_numbers):
        seen["atomic_numbers"] = list(atomic_numbers)
        return {number: index for index, number in enumerate(atomic_numbers)}

    original_import_module = vpmdk.importlib.import_module

    def fake_import_module(name: str):
        if name == "nequix.calculator":
            return SimpleNamespace(from_pretrained=fake_from_pretrained)
        if name == "nequix.data":
            return SimpleNamespace(atomic_numbers_to_indices=fake_atomic_numbers_to_indices)
        return original_import_module(name)

    monkeypatch.setattr(vpmdk, "NequixCalculator", FakeNequixCalculator)
    monkeypatch.setattr(vpmdk.importlib, "import_module", fake_import_module)

    calc = vpmdk._build_nequix_calculator(
        {
            "MODEL": str(model_path),
            "DEVICE": "cpu",
            "NEQUIX_BACKEND": "torch",
            "NEQUIX_USE_KERNEL": "true",
            "NEQUIX_USE_COMPILE": "true",
            "NEQUIX_CAPACITY_MULTIPLIER": "1.25",
        }
    )

    assert isinstance(calc, FakeNequixCalculator)
    assert seen["from_pretrained"] == {
        "model_path": str(model_path),
        "model_name": "nequix-oam-1",
        "backend": "torch",
        "use_kernel": True,
    }
    assert seen["atomic_numbers"] == [8, 14]
    assert seen["moved_to"] == "cpu"
    assert seen["eval_called"] is True
    assert str(calc.device) == "cpu"
    assert calc.backend == "torch"
    assert calc.cutoff == 6.0
    assert calc._capacity_multiplier == 1.25


def test_nequix_missing_checkpoint_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(vpmdk, "NequixCalculator", object)

    missing_path = tmp_path / "missing.nqx"
    with pytest.raises(FileNotFoundError, match="not found"):
        vpmdk._build_nequix_calculator({"MODEL": str(missing_path)})


def test_nequix_invalid_backend_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(vpmdk, "NequixCalculator", object)

    with pytest.raises(ValueError, match="NEQUIX_BACKEND"):
        vpmdk._build_nequix_calculator({"NEQUIX_BACKEND": "metal"})


def test_alphanet_uses_checkpoint_path_and_bcar_tags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_path = tmp_path / "alphanet.ckpt"
    config_path = tmp_path / "config.json"
    model_path.write_text("dummy")
    config_path.write_text("{}")
    seen: dict[str, object] = {}

    def fake_load(config_file: str, *, precision: str, use_pbc: bool, compute_stress: bool):
        seen["config_file"] = config_file
        seen["precision"] = precision
        seen["use_pbc"] = use_pbc
        seen["compute_stress"] = compute_stress
        return "alpha-config"

    def fake_calc(*, ckpt_path, config, device="cpu", precision="32"):
        seen["ckpt_path"] = ckpt_path
        seen["config"] = config
        seen["device"] = device
        seen["calc_precision"] = precision
        return "alphanet"

    monkeypatch.setattr(vpmdk, "AlphaNetCalculator", fake_calc)
    monkeypatch.setattr(vpmdk, "_load_alphanet_config", fake_load)

    calc = vpmdk._build_alphanet_calculator(
        {
            "MODEL": str(model_path),
            "ALPHANET_CONFIG": str(config_path),
            "ALPHANET_PRECISION": "float64",
            "DEVICE": "cuda:0",
        },
        structure=object(),
    )

    assert calc == "alphanet"
    assert seen == {
        "config_file": str(config_path),
        "precision": "64",
        "use_pbc": False,
        "compute_stress": False,
        "ckpt_path": str(model_path),
        "config": "alpha-config",
        "device": "cuda:0",
        "calc_precision": "64",
    }


def test_alphanet_accepts_named_model_and_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    seen: dict[str, object] = {}
    config_path = tmp_path / "matpes.json"
    config_path.write_text("{}")

    def fake_ensure(model_name: str):
        seen["model_name"] = model_name
        return ("/tmp/r2scan_1021.ckpt", str(config_path))

    def fake_load(config_file: str, *, precision: str, use_pbc: bool, compute_stress: bool):
        seen["config_file"] = config_file
        seen["precision"] = precision
        seen["use_pbc"] = use_pbc
        seen["compute_stress"] = compute_stress
        return "alpha-config"

    def fake_calc(*, ckpt_path, config, device="cpu", precision="32"):
        seen["ckpt_path"] = ckpt_path
        seen["config"] = config
        seen["device"] = device
        seen["calc_precision"] = precision
        return "alphanet"

    monkeypatch.setattr(vpmdk, "AlphaNetCalculator", fake_calc)
    monkeypatch.setattr(vpmdk, "_ensure_alphanet_named_model_files", fake_ensure)
    monkeypatch.setattr(vpmdk, "_load_alphanet_config", fake_load)

    calc = vpmdk._build_alphanet_calculator({}, structure=SimpleNamespace(lattice=object()))

    assert calc == "alphanet"
    assert seen == {
        "model_name": vpmdk.DEFAULT_ALPHANET_MODEL,
        "config_file": str(config_path),
        "precision": "32",
        "use_pbc": True,
        "compute_stress": True,
        "ckpt_path": "/tmp/r2scan_1021.ckpt",
        "config": "alpha-config",
        "device": "cpu",
        "calc_precision": "32",
    }


def test_alphanet_missing_checkpoint_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(vpmdk, "AlphaNetCalculator", object)

    missing_path = tmp_path / "missing.ckpt"
    with pytest.raises(FileNotFoundError, match="not found"):
        vpmdk._build_alphanet_calculator({"MODEL": str(missing_path)})


def test_alphanet_requires_config_for_local_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(vpmdk, "AlphaNetCalculator", object)

    model_path = tmp_path / "alphanet.ckpt"
    model_path.write_text("dummy")
    with pytest.raises(ValueError, match="ALPHANET_CONFIG"):
        vpmdk._build_alphanet_calculator({"MODEL": str(model_path)})


def test_matris_uses_checkpoint_path_and_bcar_tags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_path = tmp_path / "MatRIS_10M_OAM.pth.tar"
    model_path.write_text("dummy")
    seen: dict[str, object] = {}

    def fake_load(path: str, *, device: str | None):
        seen["load_path"] = path
        seen["load_device"] = device
        return "matris-model"

    def fake_instantiate(*, model, task: str, device: str | None):
        seen["model"] = model
        seen["task"] = task
        seen["device"] = device
        return "matris"

    monkeypatch.setattr(vpmdk, "MatRISCalculator", object)
    monkeypatch.setattr(vpmdk, "_load_matris_checkpoint_model", fake_load)
    monkeypatch.setattr(vpmdk, "_instantiate_matris_calculator", fake_instantiate)

    calc = vpmdk._build_matris_calculator(
        {"MODEL": str(model_path), "DEVICE": "cuda:0", "MATRIS_TASK": "efsm"}
    )

    assert calc == "matris"
    assert seen == {
        "load_path": str(model_path),
        "load_device": "cuda:0",
        "model": "matris-model",
        "task": "efsm",
        "device": "cuda:0",
    }


def test_override_model_graph_converter_algorithm_rebuilds_converter():
    calls: list[dict[str, object]] = []

    class DummyConverter:
        def __init__(
            self,
            *,
            atom_graph_cutoff: float,
            bond_graph_cutoff: float,
            algorithm: str = "legacy",
        ):
            calls.append(
                {
                    "atom_graph_cutoff": atom_graph_cutoff,
                    "bond_graph_cutoff": bond_graph_cutoff,
                    "algorithm": algorithm,
                }
            )
            self.atom_graph_cutoff = atom_graph_cutoff
            self.bond_graph_cutoff = bond_graph_cutoff
            self.algorithm = algorithm
            self.on_isolated_atoms = "warn"

        def set_isolated_atom_response(self, value: str):
            self.on_isolated_atoms = value

    model = SimpleNamespace(
        graph_converter=DummyConverter(atom_graph_cutoff=6, bond_graph_cutoff=3)
    )
    model.graph_converter.set_isolated_atom_response("error")

    updated_model = vpmdk._override_model_graph_converter_algorithm(
        model,
        algorithm="fast",
        backend_name="MatRIS",
    )

    assert updated_model is model
    assert calls[-1] == {
        "atom_graph_cutoff": 6,
        "bond_graph_cutoff": 3,
        "algorithm": "fast",
    }
    assert model.graph_converter.algorithm == "fast"
    assert model.graph_converter.on_isolated_atoms == "error"


def test_override_model_graph_converter_algorithm_supports_make_graph_switch(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_module = SimpleNamespace(make_graph=object(), __package__=None)

    class DummyConverter:
        def __init__(
            self,
            *,
            atom_graph_cutoff: float,
            line_graph_cutoff: float,
            verbose: bool = False,
        ):
            self.atom_graph_cutoff = atom_graph_cutoff
            self.line_graph_cutoff = line_graph_cutoff
            self.verbose = verbose
            self.algorithm = "fast" if fake_module.make_graph is not None else "legacy"
            self.on_isolated_atoms = "warn"

        def set_isolated_atom_response(self, value: str):
            self.on_isolated_atoms = value

    monkeypatch.setattr(vpmdk.inspect, "getmodule", lambda cls: fake_module)

    model = SimpleNamespace(
        graph_converter=DummyConverter(atom_graph_cutoff=6, line_graph_cutoff=4)
    )
    model.graph_converter.set_isolated_atom_response("error")

    vpmdk._override_model_graph_converter_algorithm(
        model,
        algorithm="legacy",
        backend_name="MatRIS",
    )
    assert model.graph_converter.algorithm == "legacy"
    assert model.graph_converter.on_isolated_atoms == "error"

    vpmdk._override_model_graph_converter_algorithm(
        model,
        algorithm="fast",
        backend_name="MatRIS",
    )
    assert model.graph_converter.algorithm == "fast"
    assert model.graph_converter.on_isolated_atoms == "error"


def test_matris_checkpoint_path_applies_graph_converter_algorithm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_path = tmp_path / "MatRIS_10M_OAM.pth.tar"
    model_path.write_text("dummy")

    class DummyConverter:
        def __init__(
            self,
            *,
            atom_graph_cutoff: float,
            bond_graph_cutoff: float,
            algorithm: str = "legacy",
        ):
            self.atom_graph_cutoff = atom_graph_cutoff
            self.bond_graph_cutoff = bond_graph_cutoff
            self.algorithm = algorithm
            self.on_isolated_atoms = "warn"

        def set_isolated_atom_response(self, value: str):
            self.on_isolated_atoms = value

    model = SimpleNamespace(
        graph_converter=DummyConverter(atom_graph_cutoff=6, bond_graph_cutoff=3)
    )
    seen: dict[str, object] = {}

    def fake_load(path: str, *, device: str | None):
        seen["load_path"] = path
        seen["load_device"] = device
        return model

    def fake_instantiate(*, model, task: str, device: str | None):
        seen["algorithm"] = model.graph_converter.algorithm
        seen["task"] = task
        seen["device"] = device
        return "matris"

    monkeypatch.setattr(vpmdk, "MatRISCalculator", object)
    monkeypatch.setattr(vpmdk, "_load_matris_checkpoint_model", fake_load)
    monkeypatch.setattr(vpmdk, "_instantiate_matris_calculator", fake_instantiate)

    calc = vpmdk._build_matris_calculator(
        {
            "MODEL": str(model_path),
            "DEVICE": "cuda:0",
            "MATRIS_TASK": "efsm",
            "MATRIS_GRAPH_CONVERTER_ALGORITHM": "fast",
        }
    )

    assert calc == "matris"
    assert seen == {
        "load_path": str(model_path),
        "load_device": "cuda:0",
        "algorithm": "fast",
        "task": "efsm",
        "device": "cuda:0",
    }


def test_matris_downloads_named_model_and_defaults(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, object] = {}

    def fake_ensure(model_name: str):
        seen["model_name"] = model_name
        return "/tmp/MatRIS_10M_OAM.pth.tar"

    def fake_load(path: str, *, device: str | None):
        seen["load_path"] = path
        seen["load_device"] = device
        return "matris-model"

    def fake_instantiate(*, model, task: str, device: str | None):
        seen["model"] = model
        seen["task"] = task
        seen["device"] = device
        return "matris"

    monkeypatch.setattr(vpmdk, "MatRISCalculator", object)
    monkeypatch.setattr(vpmdk, "_ensure_matris_named_model_checkpoint", fake_ensure)
    monkeypatch.setattr(vpmdk, "_load_matris_checkpoint_model", fake_load)
    monkeypatch.setattr(vpmdk, "_instantiate_matris_calculator", fake_instantiate)

    calc = vpmdk._build_matris_calculator({"DEVICE": "cpu"})

    assert calc == "matris"
    assert seen == {
        "model_name": vpmdk.DEFAULT_MATRIS_MODEL,
        "load_path": "/tmp/MatRIS_10M_OAM.pth.tar",
        "load_device": "cpu",
        "model": "matris-model",
        "task": "efs",
        "device": "cpu",
    }


def test_matris_unknown_named_model_falls_back_to_upstream_calculator(
    monkeypatch: pytest.MonkeyPatch,
):
    seen: dict[str, object] = {}

    def fake_calc(*, model, task="efs", device=None, graph_converter_algorithm=None):
        seen.update(
            {
                "model": model,
                "task": task,
                "device": device,
                "graph_converter_algorithm": graph_converter_algorithm,
            }
        )
        return "matris"

    monkeypatch.setattr(vpmdk, "MatRISCalculator", fake_calc)
    monkeypatch.setattr(vpmdk, "_ensure_matris_named_model_checkpoint", lambda model: None)

    calc = vpmdk._build_matris_calculator(
        {
            "MODEL": "custom-model",
            "MATRIS_TASK": "efsm",
            "DEVICE": "cpu",
            "MATRIS_GRAPH_CONVERTER_ALGORITHM": "legacy",
        }
    )

    assert calc == "matris"
    assert seen == {
        "model": "custom-model",
        "task": "efsm",
        "device": "cpu",
        "graph_converter_algorithm": "legacy",
    }


def test_matris_unknown_named_model_falls_back_when_constructor_lacks_algorithm(
    monkeypatch: pytest.MonkeyPatch,
):
    seen: dict[str, object] = {}

    class DummyCalculator:
        def __init__(self, *, model, task="efs", device=None):
            seen["init"] = {"model": model, "task": task, "device": device}
            self.model = "legacy-model"

    def fake_override(model, *, algorithm: str, backend_name: str):
        seen["override"] = {
            "model": model,
            "algorithm": algorithm,
            "backend_name": backend_name,
        }
        return "updated-model"

    monkeypatch.setattr(vpmdk, "MatRISCalculator", DummyCalculator)
    monkeypatch.setattr(vpmdk, "_ensure_matris_named_model_checkpoint", lambda model: None)
    monkeypatch.setattr(vpmdk, "_override_model_graph_converter_algorithm", fake_override)

    calc = vpmdk._build_matris_calculator(
        {
            "MODEL": "custom-model",
            "MATRIS_TASK": "efsm",
            "DEVICE": "cpu",
            "MATRIS_GRAPH_CONVERTER_ALGORITHM": "fast",
        }
    )

    assert isinstance(calc, DummyCalculator)
    assert seen["init"] == {"model": "custom-model", "task": "efsm", "device": "cpu"}
    assert seen["override"] == {
        "model": "legacy-model",
        "algorithm": "fast",
        "backend_name": "MatRIS",
    }
    assert calc.model == "updated-model"


def test_matris_missing_checkpoint_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(vpmdk, "MatRISCalculator", object)

    missing_path = tmp_path / "missing.pth.tar"
    with pytest.raises(FileNotFoundError, match="not found"):
        vpmdk._build_matris_calculator({"MODEL": str(missing_path)})


def test_upet_uses_checkpoint_path_and_bcar_tags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_path = tmp_path / "pet-oam-xl-v1.0.0.ckpt"
    model_path.write_text("dummy")
    seen: dict[str, object] = {}

    def fake_calc(**kwargs):
        seen.update(kwargs)
        return "upet"

    monkeypatch.setattr(vpmdk, "UPETCalculator", fake_calc)

    calc = vpmdk._build_upet_calculator(
        {
            "MODEL": str(model_path),
            "DEVICE": "cuda:0",
            "UPET_NON_CONSERVATIVE": "true",
        }
    )

    assert calc == "upet"
    assert seen == {
        "checkpoint_path": str(model_path),
        "device": "cuda:0",
        "non_conservative": True,
    }


def test_upet_accepts_named_model_and_version(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, object] = {}

    def fake_calc(**kwargs):
        seen.update(kwargs)
        return "upet"

    monkeypatch.setattr(vpmdk, "UPETCalculator", fake_calc)

    calc = vpmdk._build_upet_calculator(
        {"MODEL": "pet-oam-xl", "UPET_VERSION": "1.0.0", "DEVICE": "cpu"}
    )

    assert calc == "upet"
    assert seen == {"model": "pet-oam-xl", "version": "1.0.0", "device": "cpu"}


def test_upet_missing_checkpoint_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(vpmdk, "UPETCalculator", lambda **kwargs: None)

    missing_path = tmp_path / "missing.ckpt"
    with pytest.raises(FileNotFoundError, match="not found"):
        vpmdk._build_upet_calculator({"MODEL": str(missing_path)})


def test_tace_uses_checkpoint_path_and_bcar_tags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_path = tmp_path / "tace-model.pt"
    model_path.write_text("dummy")
    seen: dict[str, object] = {}

    def fake_calc(
        *,
        model,
        device=None,
        dtype=None,
        fidelity_idx=None,
        spin_on=None,
        neighborlist_backend=None,
    ):
        seen.update(
            {
                "model": model,
                "device": device,
                "dtype": dtype,
                "fidelity_idx": fidelity_idx,
                "spin_on": spin_on,
                "neighborlist_backend": neighborlist_backend,
            }
        )
        return "tace"

    monkeypatch.setattr(vpmdk, "TACEAseCalc", fake_calc)

    calc = vpmdk._build_tace_calculator(
        {
            "MODEL": str(model_path),
            "DEVICE": "cuda:0",
            "TACE_DTYPE": "float32",
            "TACE_FIDELITY_IDX": "2",
            "TACE_SPIN_ON": "true",
            "TACE_NEIGHBORLIST_BACKEND": "ase",
        }
    )

    assert calc == "tace"
    assert seen == {
        "model": str(model_path),
        "device": "cuda:0",
        "dtype": "float32",
        "fidelity_idx": 2,
        "spin_on": True,
        "neighborlist_backend": "ase",
    }


def test_tace_accepts_named_model_and_level_alias(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, object] = {}

    def fake_calc(*, model, device=None, level=None):
        seen.update({"model": model, "device": device, "level": level})
        return "tace"

    class DummyRegistry(dict):
        def list_models(self):
            return sorted(self)

    monkeypatch.setattr(vpmdk, "TACEAseCalc", fake_calc)
    monkeypatch.setattr(
        vpmdk,
        "tace_foundations",
        DummyRegistry({"TACE-v1-OMat24-M": Path("/tmp/TACE-v1-OMat24-M.pt")}),
    )

    calc = vpmdk._build_tace_calculator(
        {"MODEL": "TACE-v1-OMat24-M", "TACE_LEVEL": "1", "DEVICE": "cpu"}
    )

    assert calc == "tace"
    assert seen == {
        "model": "/tmp/TACE-v1-OMat24-M.pt",
        "device": "cpu",
        "level": 1,
    }


def test_tace_missing_checkpoint_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(vpmdk, "TACEAseCalc", lambda **kwargs: None)

    missing_path = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError, match="not found"):
        vpmdk._build_tace_calculator({"MODEL": str(missing_path)})


def test_deepmd_head_is_forwarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model_path = tmp_path / "model.pt"
    model_path.write_text("dummy")
    seen: dict[str, object] = {}

    def fake_calc(*, model, **kwargs):
        seen["model"] = model
        seen.update(kwargs)
        return "deepmd"

    monkeypatch.setattr(vpmdk, "DeePMDCalculator", fake_calc)

    calc = vpmdk._build_deepmd_calculator(
        {"MODEL": str(model_path), "DEEPMD_HEAD": "myhead"}
    )

    assert calc == "deepmd"
    assert seen["model"] == str(model_path)
    assert seen["head"] == "myhead"


def test_deepmd_requires_model_path(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(vpmdk, "DeePMDCalculator", lambda *a, **k: None)

    with pytest.raises(ValueError, match="requires MODEL"):
        vpmdk._build_deepmd_calculator({})


def test_deepmd_missing_model_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(vpmdk, "DeePMDCalculator", lambda *a, **k: None)

    missing_path = tmp_path / "missing.pb"
    with pytest.raises(FileNotFoundError, match="not found"):
        vpmdk._build_deepmd_calculator({"MODEL": str(missing_path)})
