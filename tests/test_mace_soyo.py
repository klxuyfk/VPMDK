"""MACE-SOYO model selection and resident configuration regressions."""
from pathlib import Path
from types import SimpleNamespace

import pytest

import vpmdk
from vpmdk_core.backends import mace_soyo
from vpmdk_core.server import (
    VPMDKServer, backend_identity, validate_request_backend, BackendConfigurationMismatch,
)


def test_package_and_head_are_forwarded(tmp_path, monkeypatch):
    package = tmp_path / "model.pt2"
    package.touch()
    monkeypatch.setattr(mace_soyo, "_get_mace_soyo_calculator_cls", lambda: lambda **kw: kw)
    result = vpmdk.build_calculator(vpmdk.BackendConfig(
        mlp="MACE_SOYO", model=str(package), device="cuda:0",
        options={"MACE_SOYO_HEAD": "mptraj"},
    ))
    assert result == {"package_path": str(package), "device": "cuda:0",
                      "head": "mptraj", "use_d3": False}


@pytest.mark.parametrize("model", [None, "missing.pt2"])
def test_model_is_required_and_missing_paths_fail(model):
    with pytest.raises((ValueError, FileNotFoundError)):
        mace_soyo._build_mace_soyo_calculator({"MODEL": model, "DEVICE": "cuda"})


def test_unknown_format_is_rejected(tmp_path):
    checkpoint = tmp_path / "model.model"
    checkpoint.touch()
    with pytest.raises(ValueError, match=".pth checkpoint"):
        mace_soyo._build_mace_soyo_calculator({"MODEL": str(checkpoint), "DEVICE": "cuda"})


def test_cpu_is_rejected(tmp_path):
    package = tmp_path / "model.pt2"
    package.touch()
    with pytest.raises(ValueError, match="CUDA device"):
        mace_soyo._build_mace_soyo_calculator({"MODEL": str(package), "DEVICE": "cpu"})


def test_dependency_error(tmp_path, monkeypatch):
    package = tmp_path / "model.pt2"
    package.touch()
    monkeypatch.setattr(mace_soyo, "_get_mace_soyo_calculator_cls", lambda: None)
    with pytest.raises(RuntimeError, match="MACE-SOYO is not available"):
        mace_soyo._build_mace_soyo_calculator({"MODEL": str(package), "DEVICE": "cuda"})


def test_upstream_load_failure_is_not_replaced(tmp_path, monkeypatch):
    package = tmp_path / "model.pt2"
    package.touch()
    def fail(**kwargs):
        raise RuntimeError("incompatible AOTI package")
    monkeypatch.setattr(mace_soyo, "_get_mace_soyo_calculator_cls", lambda: fail)
    with pytest.raises(RuntimeError, match="incompatible AOTI"):
        mace_soyo._build_mace_soyo_calculator({"MODEL": str(package), "DEVICE": "cuda"})


@pytest.mark.parametrize("lowercase", [False, True])
def test_resident_records_selected_head(tmp_path, lowercase):
    package = tmp_path / "model.pt2"
    package.touch()
    tags = {"MLP": "MACE_SOYO", "MODEL": str(package), "DEVICE": "cuda"}
    if lowercase:
        tags = {key.lower(): value for key, value in tags.items()}
    server = VPMDKServer(str(tmp_path / "server.sock"),
        SimpleNamespace(head="mptraj", device="cuda", cueq_method="naive"), tags,
        backend_base_dir=str(tmp_path))
    assert server.backend["effective_configuration"]["MACE_SOYO_HEAD"] == "mptraj"
    assert server.backend["effective_configuration"]["MACE_SOYO_CUEQ_METHOD"] == "naive"
    validate_request_backend(server.backend, {"MACE_SOYO_HEAD": "mptraj", "MACE_SOYO_CUEQ_METHOD": "NAIVE"}, request_base_dir=str(tmp_path))
    for changed in [{"MACE_SOYO_HEAD": "other"}, {"MACE_SOYO_CUEQ_METHOD": "native"}]:
        with pytest.raises(BackendConfigurationMismatch):
            validate_request_backend(server.backend, changed, request_base_dir=str(tmp_path))


def test_head_is_a_resident_setting(tmp_path):
    package = tmp_path / "model.pt2"
    package.touch()
    tags = {"MLP": "MACE_SOYO", "MODEL": str(package), "DEVICE": "cuda"}
    first = backend_identity({**tags, "MACE_SOYO_HEAD": "mptraj"}, base_dir=str(tmp_path))
    second = backend_identity({**tags, "MACE_SOYO_HEAD": "other"}, base_dir=str(tmp_path))
    assert first["effective_configuration"] != second["effective_configuration"]


@pytest.mark.parametrize("positional", [False, True])
def test_portable_method_override_is_explicit_and_restored(positional):
    from vpmdk_core.backends.mace_soyo_common import _naive_segmented_polynomials
    class Polynomial:
        def __init__(self, polynomial, method="uniform_1d", **kwargs):
            self.method = method
            self.m = SimpleNamespace(
                _buffers={"existing": 1}, _non_persistent_buffers_set=set()
            )
            self.m.named_modules = lambda: [("", self.m)]
            if method == "naive":
                self.m._buffers["derived"] = 2
            if method == "uniform_1d":
                self.m.buffer_num_segments = [3]
                self.m.operand_extent = 4
        def state_dict(self):
            return {f"m.{key}": value for key, value in self.m._buffers.items()}
    original = Polynomial.__init__
    with pytest.raises(RuntimeError, match="stop"):
        with _naive_segmented_polynomials(Polynomial):
            module = Polynomial(None, "uniform_1d") if positional else Polynomial(None, method="uniform_1d")
            assert module.method == "naive"
            assert module.m.buffer_num_segments == [3]
            assert module.m.operand_extent == 4
            assert module.m._non_persistent_buffers_set == {"derived"}
            raise RuntimeError("stop")
    assert Polynomial.__init__ is original
    assert Polynomial(None).method == "uniform_1d"



@pytest.mark.parametrize("available", [False, True])
def test_backend_discovery(monkeypatch, available):
    monkeypatch.setattr(vpmdk, "_get_mace_soyo_calculator_cls", lambda: object if available else None)
    spec = next(spec for spec in vpmdk.list_backends() if spec.name == "MACE_SOYO")
    assert spec.available is available
    assert spec.default_model is None
    assert spec.capabilities.energy and spec.capabilities.forces and spec.capabilities.stress



def test_direct_checkpoint_and_explicit_method(tmp_path, monkeypatch):
    from vpmdk_core.backends import mace_soyo_eager
    checkpoint = tmp_path / "model.pth"
    checkpoint.touch()
    monkeypatch.setattr(mace_soyo, "_get_mace_soyo_calculator_cls", lambda: object)
    monkeypatch.setattr(mace_soyo_eager, "MaceSoyoEagerCalculator", lambda path, **kw: (path, kw))
    path, options = mace_soyo._build_mace_soyo_calculator({
        "MODEL": str(checkpoint), "DEVICE": "cuda", "MACE_SOYO_CUEQ_METHOD": "naive",
        "MACE_SOYO_HEAD": "mptraj",
    })
    assert path == str(checkpoint)
    assert options == {"device": "cuda", "head": "mptraj", "cueq_method": "naive"}


def test_compiled_method_cannot_be_changed(tmp_path, monkeypatch):
    package = tmp_path / "model.pt2"
    package.touch()
    monkeypatch.setattr(mace_soyo, "_get_mace_soyo_calculator_cls", lambda: object)
    with pytest.raises(ValueError, match="fixes the method at export"):
        mace_soyo._build_mace_soyo_calculator({"MODEL": str(package), "DEVICE": "cuda", "MACE_SOYO_CUEQ_METHOD": "naive"})


@pytest.mark.parametrize("shared", [False, True])
def test_legacy_linear_aliases_preserve_evaluator_and_restore(shared):
    from vpmdk_core.backends.mace_soyo_common import _legacy_cueq_linear_aliases
    class Linear:
        def __init__(self):
            self.evaluator = object()
            self.fallback = self.evaluator if shared else object()
            self.child = SimpleNamespace(_modules={"m": self.evaluator, "fallback": self.fallback})
        def modules(self):
            return [self.child]
    original = Linear.__init__
    with pytest.raises(RuntimeError, match="stop"):
        with _legacy_cueq_linear_aliases(Linear):
            model = Linear()
            assert model.child._modules["m"] is model.evaluator
            if shared:
                assert model.child.fallback is model.evaluator
                assert "fallback" not in model.child._modules
            else:
                assert model.child._modules["fallback"] is model.fallback
            raise RuntimeError("stop")
    assert Linear.__init__ is original


@pytest.mark.parametrize("version", ["0.6.1", "0.12.0", "0.13.0"])
def test_legacy_alias_rule_is_version_scoped(monkeypatch, version):
    import importlib.metadata
    import sys
    from contextlib import nullcontext
    from types import ModuleType
    from vpmdk_core.backends import mace_soyo_common as common
    calls = []
    module = ModuleType("cuequivariance_torch")
    module.Linear = object
    monkeypatch.setitem(sys.modules, "cuequivariance_torch", module)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: version)
    def legacy(cls):
        calls.append(cls)
        return nullcontext()
    monkeypatch.setattr(common, "_legacy_cueq_linear_aliases", legacy)
    with common._cueq_checkpoint_compat():
        pass
    assert calls == ([object] if version == "0.6.1" else [])
