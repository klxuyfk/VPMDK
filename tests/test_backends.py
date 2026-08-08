from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import vpmdk
from vpmdk_core.api import _BASE_CAPABILITIES
from vpmdk_core.backends import misc as backend_misc


_P = vpmdk.BackendModelPolicy
_EXPECTED_BACKEND_MODEL_POLICY = {
    "BAM": _P(required=True, local_only=True),
    "CHGNET": _P(allow_named=True),
    "MATGL": _P(
        default_attribute="DEFAULT_MATGL_MODEL",
        allow_named=True,
        delegate_registry_ids=True,
    ),
    "M3GNET": _P(
        default_attribute="DEFAULT_MATGL_MODEL",
        allow_named=True,
        delegate_registry_ids=True,
    ),
    "MACE": _P(local_only=True),
    "MATTERSIM": _P(allow_named=True),
    "MATLANTIS": _P(
        default_value="v8.0.0", allow_local=False, allow_named=True
    ),
    "EQNORM": _P(
        default_attribute="DEFAULT_EQNORM_MODEL", named_resolver="eqnorm"
    ),
    "MATRIS": _P(
        default_attribute="DEFAULT_MATRIS_MODEL",
        allow_named=True,
        known_names_attribute="_MATRIS_NAMED_MODEL_DOWNLOADS",
    ),
    "ALPHANET": _P(
        default_attribute="DEFAULT_ALPHANET_MODEL", named_resolver="alphanet"
    ),
    "HIENET": _P(
        default_attribute="DEFAULT_HIENET_MODEL", named_resolver="hienet"
    ),
    "NEQUIX": _P(
        default_attribute="DEFAULT_NEQUIX_MODEL", named_resolver="nequix"
    ),
    "SEVENNET": _P(
        default_attribute="DEFAULT_SEVENNET_MODEL", allow_named=True
    ),
    "FLASHTP": _P(
        default_attribute="DEFAULT_SEVENNET_MODEL", allow_named=True
    ),
    "ALLEGRO": _P(required=True, local_only=True),
    "NEQUIP": _P(required=True, local_only=True),
    "ORB": _P(local_only=True, allow_remote_uri=True),
    "UPET": _P(required=True, allow_named=True),
    "TACE": _P(required=True, allow_named=True),
    "EQUFLASH": _P(required=True, local_only=True),
    "EQUIFORMER_V3": _P(required=True, local_only=True),
    "FAIRCHEM": _P(
        default_attribute="DEFAULT_FAIRCHEM_MODEL",
        allow_named=True,
        delegate_unresolved=True,
    ),
    "FAIRCHEM_V2": _P(
        default_attribute="DEFAULT_FAIRCHEM_MODEL",
        allow_named=True,
        delegate_unresolved=True,
    ),
    "ESEN": _P(
        default_attribute="DEFAULT_FAIRCHEM_MODEL",
        allow_named=True,
        delegate_unresolved=True,
    ),
    "FAIRCHEM_V1": _P(
        required=True, allow_named=True, delegate_unresolved=True
    ),
    "GRACE": _P(named_resolver="grace", resolver_supplies_default=True),
    "DEEPMD": _P(required=True, local_only=True),
}


def test_backend_model_policy_matrix_covers_every_builtin_backend():
    assert set(_BASE_CAPABILITIES) == set(_EXPECTED_BACKEND_MODEL_POLICY)
    assert set(vpmdk._BACKEND_MODEL_POLICIES) == set(
        _EXPECTED_BACKEND_MODEL_POLICY
    )

    assert dict(vpmdk._BACKEND_MODEL_POLICIES) == _EXPECTED_BACKEND_MODEL_POLICY


@pytest.mark.parametrize(
    ("backend", "expected_model"),
    [
        ("CHGNET", None),
        ("MACE", None),
        ("MATTERSIM", None),
        ("ORB", None),
        ("MATGL", vpmdk.DEFAULT_MATGL_MODEL),
        ("M3GNET", vpmdk.DEFAULT_MATGL_MODEL),
        ("EQNORM", vpmdk.DEFAULT_EQNORM_MODEL),
        ("ALPHANET", vpmdk.DEFAULT_ALPHANET_MODEL),
        ("HIENET", vpmdk.DEFAULT_HIENET_MODEL),
        ("NEQUIX", vpmdk.DEFAULT_NEQUIX_MODEL),
        ("MATRIS", vpmdk.DEFAULT_MATRIS_MODEL),
        ("SEVENNET", vpmdk.DEFAULT_SEVENNET_MODEL),
        ("FLASHTP", vpmdk.DEFAULT_SEVENNET_MODEL),
        ("FAIRCHEM", vpmdk.DEFAULT_FAIRCHEM_MODEL),
        ("FAIRCHEM_V2", vpmdk.DEFAULT_FAIRCHEM_MODEL),
        ("ESEN", vpmdk.DEFAULT_FAIRCHEM_MODEL),
        ("MATLANTIS", "v8.0.0"),
    ],
)
def test_shared_model_resolver_classifies_backend_defaults(
    backend: str,
    expected_model: str | None,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)

    reference = vpmdk._resolve_backend_model_reference(backend, None)

    assert reference == vpmdk.ModelReference(
        vpmdk.ModelReferenceKind.DEFAULT,
        expected_model,
        explicit=False,
    )


@pytest.mark.parametrize(
    "backend",
    [
        "NEQUIP",
        "ALLEGRO",
        "EQUFLASH",
        "EQUIFORMER_V3",
        "DEEPMD",
        "UPET",
        "TACE",
        "FAIRCHEM_V1",
    ],
)
def test_shared_model_resolver_requires_model_for_required_backends(backend: str):
    with pytest.raises(ValueError, match=rf"{backend} requires MODEL"):
        vpmdk._resolve_backend_model_reference(backend, None)


@pytest.mark.parametrize(
    "backend",
    [
        "CHGNET",
        "MACE",
        "ORB",
        "MATGL",
        "M3GNET",
        "MATTERSIM",
        "EQNORM",
        "ALPHANET",
        "HIENET",
        "NEQUIX",
        "MATRIS",
        "SEVENNET",
        "FLASHTP",
        "UPET",
        "EQUFLASH",
        "TACE",
        "FAIRCHEM",
        "FAIRCHEM_V2",
        "ESEN",
        "EQUIFORMER_V3",
        "FAIRCHEM_V1",
        "GRACE",
        "DEEPMD",
        "NEQUIP",
        "ALLEGRO",
    ],
)
def test_shared_model_resolver_classifies_existing_local_models(
    backend: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)
    model_path = tmp_path / "model.checkpoint"
    model_path.write_text("placeholder")

    reference = vpmdk._resolve_backend_model_reference(backend, str(model_path))

    assert reference == vpmdk.ModelReference(
        vpmdk.ModelReferenceKind.LOCAL_PATH,
        str(model_path),
        explicit=True,
        identity=str(model_path.resolve()),
    )


@pytest.mark.parametrize(
    "backend",
    [
        "MACE",
        "ORB",
        "NEQUIP",
        "ALLEGRO",
        "EQUFLASH",
        "EQUIFORMER_V3",
        "DEEPMD",
    ],
)
def test_shared_model_resolver_rejects_unknown_local_only_models(
    backend: str,
    tmp_path: Path,
):
    with pytest.raises(FileNotFoundError, match=rf"{backend} MODEL path not found"):
        vpmdk._resolve_backend_model_reference(
            backend,
            "unknown-model",
            base_dir=str(tmp_path),
        )


@pytest.mark.parametrize(
    "backend",
    [
        "CHGNET",
        "MATGL",
        "M3GNET",
        "MATTERSIM",
        "MATRIS",
        "SEVENNET",
        "FLASHTP",
        "UPET",
        "TACE",
        "FAIRCHEM",
        "FAIRCHEM_V2",
        "ESEN",
        "MATLANTIS",
    ],
)
def test_shared_model_resolver_preserves_upstream_named_models(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)

    reference = vpmdk._resolve_backend_model_reference(backend, "registry-model")

    assert reference == vpmdk.ModelReference(
        vpmdk.ModelReferenceKind.NAMED_MODEL,
        "registry-model",
        explicit=True,
    )


@pytest.mark.parametrize("backend", ["EQNORM", "ALPHANET", "HIENET"])
def test_shared_model_resolver_rejects_unknown_static_registry_names(backend: str):
    with pytest.raises(ValueError, match="Unsupported"):
        vpmdk._resolve_backend_model_reference(backend, "unknown-registry-model")


def test_shared_model_resolver_falls_back_for_unknown_grace_registry_name(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(vpmdk, "GRACE_MODEL_NAMES", ["GRACE-INSTALLED"])

    reference = vpmdk._resolve_backend_model_reference("GRACE", "GRACE-UNKNOWN")

    assert reference == vpmdk.ModelReference(
        vpmdk.ModelReferenceKind.NAMED_MODEL,
        "GRACE-INSTALLED",
        explicit=True,
    )
    # The resolver stays side-effect free: the resident server calls it for
    # every status/run request, so warning here would append a daemon-log line
    # per request. The substitution is reported when a calculator is built.
    assert capsys.readouterr().out == ""


def test_grace_builder_warns_once_about_an_unknown_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(vpmdk, "GRACE_MODEL_NAMES", ["GRACE-INSTALLED"])
    monkeypatch.setattr(vpmdk, "TPCalculator", lambda *a, **k: object())
    monkeypatch.setattr(vpmdk, "grace_fm", lambda name, **kwargs: f"grace:{name}")

    calculator = vpmdk._build_grace_calculator({"MODEL": "GRACE-UNKNOWN"})

    assert calculator == "grace:GRACE-INSTALLED"
    assert capsys.readouterr().out == (
        "Warning: Unknown GRACE model 'GRACE-UNKNOWN', using default "
        "GRACE-INSTALLED instead.\n"
    )


def test_grace_builder_rejects_omitted_default_when_registry_empty(
    monkeypatch: pytest.MonkeyPatch,
):
    # Omitted MODEL falls back to DEFAULT_GRACE_MODEL, which cannot be validated
    # against an empty foundation registry. Forwarding it to grace_fm risks a
    # silently-substituted model, so fail clearly instead.
    called: list[str] = []

    monkeypatch.setattr(vpmdk, "GRACE_MODEL_NAMES", [])
    monkeypatch.setattr(vpmdk, "TPCalculator", lambda *a, **k: object())
    monkeypatch.setattr(
        vpmdk, "grace_fm", lambda name, **kwargs: called.append(name) or object()
    )

    with pytest.raises(RuntimeError, match="no enumerable foundation models"):
        vpmdk._build_grace_calculator({})

    assert called == []


def test_grace_builder_rejects_unvalidated_explicit_name_when_registry_empty(
    monkeypatch: pytest.MonkeyPatch,
):
    # grace_fm present but the foundation registry is empty (version skew): an
    # explicit unknown name cannot be validated, so it must not be forwarded to
    # grace_fm (which might silently load a substituted model). Fail clearly.
    called: list[str] = []

    monkeypatch.setattr(vpmdk, "GRACE_MODEL_NAMES", [])
    monkeypatch.setattr(vpmdk, "TPCalculator", lambda *a, **k: object())
    monkeypatch.setattr(
        vpmdk, "grace_fm", lambda name, **kwargs: called.append(name) or object()
    )

    with pytest.raises(FileNotFoundError, match="GRACE model not found"):
        vpmdk._build_grace_calculator({"MODEL": "totally-unknown-model"})

    assert called == []


def test_grace_builder_does_not_warn_for_a_case_variant_known_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    # A known model given in non-canonical case resolves to its canonical
    # spelling; that is a match, not a substitution, so no "using default"
    # warning must be printed.
    monkeypatch.setattr(vpmdk, "GRACE_MODEL_NAMES", ["GRACE-2L-OMAT"])
    monkeypatch.setattr(vpmdk, "TPCalculator", lambda *a, **k: object())
    monkeypatch.setattr(vpmdk, "grace_fm", lambda name, **kwargs: f"grace:{name}")

    calculator = vpmdk._build_grace_calculator({"MODEL": "grace-2l-omat"})

    assert calculator == "grace:GRACE-2L-OMAT"
    assert capsys.readouterr().out == ""


def test_grace_policy_resolver_is_the_single_default_source(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str | None] = []

    def resolve(model_value=None):
        calls.append(model_value)
        return "GRACE-INSTALLED" if model_value is None else None

    monkeypatch.setattr(vpmdk, "_resolve_grace_foundation_model", resolve)

    default_reference = vpmdk._resolve_backend_model_reference("GRACE", None)
    assert default_reference.value == "GRACE-INSTALLED"
    assert calls == [None]

    calls.clear()
    explicit_reference = vpmdk._resolve_backend_model_reference(
        "GRACE", "GRACE-UNKNOWN"
    )
    assert explicit_reference.value == "GRACE-INSTALLED"
    assert calls == [None, "GRACE-UNKNOWN"]


def test_grace_named_model_without_foundation_loader_raises_not_found(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(vpmdk, "GRACE_MODEL_NAMES", [])
    monkeypatch.setattr(vpmdk, "TPCalculator", lambda *a, **k: object())
    monkeypatch.setattr(vpmdk, "grace_fm", None)

    with pytest.raises(FileNotFoundError, match="GRACE model not found"):
        vpmdk._build_grace_calculator({"MODEL": "GRACE-2L-OMAT"})


def test_grace_omitted_model_without_foundation_loader_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
):
    # No MODEL supplied and grace_fm unavailable: there is no default to fall
    # back on. This is an environment problem (RuntimeError), not a missing
    # named model, and must not mislabel the default checkpoint as 'not found'.
    monkeypatch.setattr(vpmdk, "GRACE_MODEL_NAMES", [])
    monkeypatch.setattr(vpmdk, "TPCalculator", lambda *a, **k: object())
    monkeypatch.setattr(vpmdk, "grace_fm", None)

    with pytest.raises(RuntimeError, match="foundation loader"):
        vpmdk._build_grace_calculator({})


@pytest.mark.parametrize(
    "backend",
    ["FAIRCHEM", "FAIRCHEM_V2", "ESEN", "FAIRCHEM_V1"],
)
def test_shared_model_resolver_delegates_path_shaped_upstream_selectors(
    backend: str, tmp_path: Path
):
    reference = vpmdk._resolve_backend_model_reference(
        backend,
        "provider/runtime-model.ckpt",
        base_dir=str(tmp_path),
    )

    assert reference == vpmdk.ModelReference(
        vpmdk.ModelReferenceKind.NAMED_MODEL,
        "provider/runtime-model.ckpt",
        explicit=True,
    )


@pytest.mark.parametrize("backend", ["MATGL", "M3GNET"])
def test_matgl_delegates_slash_registry_id_but_rejects_path_typo(
    backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)

    # A slash-qualified Hugging Face identifier (no checkpoint suffix) is a
    # registry id delegated to matgl.load_model, not a filesystem path.
    reference = vpmdk._resolve_backend_model_reference(
        backend, "owner/model", base_dir=str(tmp_path)
    )
    assert reference == vpmdk.ModelReference(
        vpmdk.ModelReferenceKind.NAMED_MODEL, "owner/model", explicit=True
    )

    # Clear filesystem-path shapes (absolute, multi-segment, dot-relative, or a
    # checkpoint suffix) are NOT registry ids and stay strict missing-path
    # errors, preserving the confirmed strict-MODEL handling for real paths.
    for path_like in (
        "weights.pt",
        "subdir/weights.pt",
        "/abs/model_dir",
        "/home/u/models/mp_run",
        "a/b/c",
        "./rel/model",
        "../up/model",
    ):
        with pytest.raises(FileNotFoundError, match="MODEL path not found"):
            vpmdk._resolve_backend_model_reference(
                backend, path_like, base_dir=str(tmp_path)
            )


@pytest.mark.parametrize(
    "backend",
    [
        "CHGNET",
        "MATGL",
        "M3GNET",
        "MACE",
        "MATTERSIM",
        "EQNORM",
        "MATRIS",
        "ALPHANET",
        "HIENET",
        "NEQUIX",
        "SEVENNET",
        "FLASHTP",
        "ALLEGRO",
        "NEQUIP",
        "ORB",
        "UPET",
        "TACE",
        "EQUFLASH",
        "EQUIFORMER_V3",
        "GRACE",
        "DEEPMD",
    ],
)
def test_shared_model_resolver_rejects_missing_path_shaped_models(
    backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)

    with pytest.raises(FileNotFoundError, match="MODEL path not found"):
        vpmdk._resolve_backend_model_reference(
            backend,
            "provider/missing-model.ckpt",
            base_dir=str(tmp_path),
        )


@pytest.mark.parametrize(
    ("spec_backend", "spec_label"),
    [("EQNORM", "Eqnorm"), ("ALPHANET", "AlphaNet"), ("HIENET", "HIENet")],
)
def test_spec_resolver_backends_enumerate_available_names_for_unknown_model(
    spec_backend: str, spec_label: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)

    with pytest.raises(
        ValueError, match=rf"Unsupported {spec_label} model .*Available:"
    ):
        vpmdk._resolve_backend_model_reference(
            spec_backend, f"{spec_backend.lower()}-typo-v0"
        )


@pytest.mark.parametrize("suffix", [".yaml", ".yml", ".json"])
def test_config_suffixes_do_not_shape_named_models(
    suffix: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # A named model identifier that merely ends in a config-file extension must
    # not be mis-classified as a missing local checkpoint for an allow_named,
    # non-delegating backend; only checkpoint extensions gate the path branch.
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)

    reference = vpmdk._resolve_backend_model_reference(
        "SEVENNET", f"some-named-model{suffix}", base_dir=str(tmp_path)
    )

    assert reference.kind is vpmdk.ModelReferenceKind.NAMED_MODEL
    assert reference.value == f"some-named-model{suffix}"


@pytest.mark.parametrize("suffix", [".pt", ".pth", ".ckpt", ".nqx"])
def test_checkpoint_suffixes_still_shape_missing_paths(
    suffix: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # A non-existent value ending in a checkpoint extension is still treated as a
    # mistyped local path for a non-delegating backend.
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)

    with pytest.raises(FileNotFoundError, match="MODEL path not found"):
        vpmdk._resolve_backend_model_reference(
            "SEVENNET", f"missing-checkpoint{suffix}", base_dir=str(tmp_path)
        )


@pytest.mark.parametrize(
    "uri",
    [
        "https://example.invalid/orb-weights.ckpt",
        "s3://bucket/orb-weights.ckpt",
        "hf://org/model/weights.ckpt",
    ],
)
def test_orb_remote_uri_model_is_delegated(
    uri: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # ORB weights_path accepts remote URIs (orb-models downloads them); a
    # scheme-qualified reference is delegated, not rejected as a missing path.
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)

    reference = vpmdk._resolve_backend_model_reference(
        "ORB", uri, base_dir=str(tmp_path)
    )

    assert reference.kind is vpmdk.ModelReferenceKind.NAMED_MODEL
    assert reference.value == uri


def test_orb_missing_local_path_still_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # A non-existent local path (no scheme) is a typo and must still error early
    # under the strict MODEL-resolution policy.
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)

    with pytest.raises(FileNotFoundError, match="MODEL path not found"):
        vpmdk._resolve_backend_model_reference(
            "ORB", "./missing-orb-weights.ckpt", base_dir=str(tmp_path)
        )


def test_remote_uri_needs_opt_in_allow_remote_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Remote-URI delegation is opt-in: a local_only backend that cannot download
    # (MACE) still rejects a remote URI as an unresolved local path.
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)

    with pytest.raises(FileNotFoundError, match="MODEL path not found"):
        vpmdk._resolve_backend_model_reference(
            "MACE", "https://example.invalid/weights.ckpt", base_dir=str(tmp_path)
        )


def test_orb_remote_uri_reaches_the_model_factory(
    monkeypatch: pytest.MonkeyPatch,
):
    # End-to-end: an explicit remote MODEL overrides the factory's bundled
    # default weights and is forwarded to the loader as weights_path.
    seen: dict[str, object] = {}

    def factory(*, weights_path, device, precision, compile, train):
        seen["weights_path"] = weights_path
        return object()

    monkeypatch.setattr(
        vpmdk, "ORBCalculator", lambda model, device=None: SimpleNamespace(model=model)
    )
    monkeypatch.setattr(
        vpmdk, "ORB_PRETRAINED_MODELS", {vpmdk.DEFAULT_ORB_MODEL: factory}
    )

    vpmdk._build_orb_calculator(
        {"MODEL": "https://example.invalid/orb-weights.ckpt", "DEVICE": "cpu"}
    )

    assert seen["weights_path"] == "https://example.invalid/orb-weights.ckpt"


def test_shared_model_resolver_never_treats_matlantis_path_shape_as_local(
    tmp_path: Path,
):
    reference = vpmdk._resolve_backend_model_reference(
        "MATLANTIS",
        "provider/version.ckpt",
        base_dir=str(tmp_path),
    )

    assert reference == vpmdk.ModelReference(
        vpmdk.ModelReferenceKind.NAMED_MODEL,
        "provider/version.ckpt",
        explicit=True,
    )


def test_shared_model_resolver_treats_matlantis_versions_as_opaque(
    tmp_path: Path,
):
    version_entry = tmp_path / "v8.0.0"
    version_entry.mkdir()

    reference = vpmdk._resolve_backend_model_reference(
        "MATLANTIS",
        "v8.0.0",
        base_dir=str(tmp_path),
    )

    assert reference == vpmdk.ModelReference(
        vpmdk.ModelReferenceKind.NAMED_MODEL,
        "v8.0.0",
        explicit=True,
    )


def test_matlantis_forwards_version_even_when_same_named_path_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (tmp_path / "v8.0.0").mkdir()
    seen: dict[str, object] = {}

    class CalcMode:
        PBE = "PBE-mode"

    def estimator(**kwargs):
        seen.update(kwargs)
        return "estimator"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vpmdk, "EstimatorCalcMode", CalcMode)
    monkeypatch.setattr(vpmdk, "MatlantisEstimator", estimator)
    monkeypatch.setattr(
        vpmdk,
        "MatlantisASECalculator",
        lambda value: ("calculator", value),
    )

    calculator = vpmdk._build_matlantis_calculator({"MODEL": "v8.0.0"})

    assert calculator == ("calculator", "estimator")
    assert seen == {
        "model_version": "v8.0.0",
        "priority": 50,
        "calc_mode": "PBE-mode",
    }


@pytest.mark.parametrize(
    "backend",
    [
        "MACE",
        "MATGL",
        "ALPHANET",
        "EQNORM",
        "MATRIS",
        "NEQUIX",
        "SEVENNET",
        "FAIRCHEM_V1",
    ],
)
def test_shared_model_resolver_preserves_symlink_for_loaders(
    backend: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)
    target_dir = tmp_path / "target"
    link_dir = tmp_path / "links"
    target_dir.mkdir()
    link_dir.mkdir()
    target = target_dir / "model.ckpt"
    target.write_text("placeholder")
    model_link = link_dir / "model.ckpt"
    model_link.symlink_to(target)

    reference = vpmdk._resolve_backend_model_reference(backend, str(model_link))

    assert reference.kind is vpmdk.ModelReferenceKind.LOCAL_PATH
    assert reference.value == str(model_link)
    assert reference.identity == str(target.resolve())


def test_nequix_delegates_named_models_when_registry_metadata_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
    seen: dict[str, object] = {}

    class Calculator:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(vpmdk, "NequixCalculator", Calculator)

    calculator = vpmdk._build_nequix_calculator(
        {"MODEL": "upstream-custom-model"}
    )

    assert isinstance(calculator, Calculator)
    assert seen["model_name"] == "upstream-custom-model"


def test_nequix_rejects_unknown_name_when_registry_metadata_is_available(
    monkeypatch: pytest.MonkeyPatch,
):
    class Calculator:
        URLS = {"known-model": "https://example.invalid/known.nqx"}

    monkeypatch.setattr(vpmdk, "NequixCalculator", Calculator)

    with pytest.raises(ValueError, match="Unsupported Nequix model"):
        vpmdk._resolve_backend_model_reference("NEQUIX", "unknown-model")


def test_nequix_default_identity_is_canonicalized_like_explicit_request(
    monkeypatch: pytest.MonkeyPatch,
):
    # The resident-default NEQUIX identity must be canonicalized the same way an
    # explicit request is, so a request naming the default is not falsely
    # rejected as a backend mismatch over registry-key casing.
    class Calculator:
        URLS = {"Nequix-MP-1": "https://example.invalid/nequix.nqx"}

    monkeypatch.setattr(vpmdk, "NequixCalculator", Calculator)
    monkeypatch.setattr(vpmdk, "DEFAULT_NEQUIX_MODEL", "nequix-mp-1")

    default_reference = vpmdk._resolve_backend_model_reference("NEQUIX", None)
    explicit_reference = vpmdk._resolve_backend_model_reference(
        "NEQUIX", "nequix-mp-1"
    )

    assert default_reference.value == "Nequix-MP-1"
    assert explicit_reference.value == "Nequix-MP-1"
    assert default_reference.value == explicit_reference.value


def test_nequix_default_model_is_validated_against_registry(
    monkeypatch: pytest.MonkeyPatch,
):
    # An omitted MODEL falls back to DEFAULT_NEQUIX_MODEL, which must still be
    # validated against installed metadata (like an explicit name) so a registry
    # or version mismatch raises a clear error instead of deferring to a cryptic
    # loader failure.
    class Calculator:
        URLS = {"known-model": "https://example.invalid/known.nqx"}

    monkeypatch.setattr(vpmdk, "NequixCalculator", Calculator)
    monkeypatch.setattr(
        vpmdk, "DEFAULT_NEQUIX_MODEL", "nequix-missing-from-registry"
    )

    with pytest.raises(ValueError, match="Unsupported Nequix model"):
        vpmdk._build_nequix_calculator({})


@pytest.mark.parametrize("loader_result", [None, False, ""])
def test_chgnet_named_model_loader_must_return_requested_model(
    loader_result,
    monkeypatch: pytest.MonkeyPatch,
):
    calculator_calls: list[dict[str, object]] = []

    class ModelLoader:
        @classmethod
        def load(cls, *args, **kwargs):
            assert kwargs.get("model_name") == "chgnet-named-model"
            return loader_result

    def calculator(**kwargs):
        calculator_calls.append(kwargs)
        return object()

    monkeypatch.setattr(vpmdk, "CHGNetModel", ModelLoader)
    monkeypatch.setattr(vpmdk, "CHGNetCalculator", calculator)

    with pytest.raises(RuntimeError, match="loader returned no model"):
        vpmdk._build_chgnet_calculator(
            {"MODEL": "chgnet-named-model", "DEVICE": "cpu"}
        )

    assert calculator_calls == []


def test_mace_missing_explicit_model_does_not_use_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[tuple[object, ...]] = []

    def calculator(*args, **kwargs):
        calls.append(args)
        return object()

    monkeypatch.setattr(vpmdk, "MACECalculator", calculator)

    with pytest.raises(FileNotFoundError, match="MACE MODEL path not found"):
        vpmdk._build_mace_calculator(
            {"MODEL": str(tmp_path / "named-or-missing-model"), "DEVICE": "cpu"}
        )

    assert calls == []


def test_mace_existing_explicit_model_is_forwarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_path = tmp_path / "mace.model"
    model_path.write_text("placeholder")
    seen: dict[str, object] = {}

    def calculator(model, **kwargs):
        seen.update({"model": model, "kwargs": kwargs})
        return "calculator"

    monkeypatch.setattr(vpmdk, "MACECalculator", calculator)

    result = vpmdk._build_mace_calculator(
        {"MODEL": str(model_path), "DEVICE": "cuda"}
    )

    assert result == "calculator"
    assert seen == {"model": str(model_path), "kwargs": {"device": "cuda"}}


@pytest.mark.parametrize("selection", ["default", "local"])
@pytest.mark.parametrize("supports_device", [True, False])
def test_mace_model_classification_and_device_signature_matrix(
    selection: str,
    supports_device: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    model_path = tmp_path / "mace.model"
    model_path.write_text("placeholder")

    if supports_device:
        class Calculator:
            def __init__(self, model=None, *, device=None):
                self.model = model
                self.device = device
    else:
        class Calculator:
            def __init__(self, model=None):
                self.model = model
                self.device = None

    monkeypatch.setattr(vpmdk, "MACECalculator", Calculator)
    tags = {"DEVICE": "cuda"}
    if selection == "local":
        tags["MODEL"] = str(model_path)

    calculator = vpmdk._build_mace_calculator(tags)

    assert calculator.model == (str(model_path) if selection == "local" else None)
    assert calculator.device == ("cuda" if supports_device else None)


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


def test_matgl_2_pes_calculator_name_is_supported():
    calculator_cls = object()

    assert (
        vpmdk._select_matgl_calculator_class(
            SimpleNamespace(PESCalculator=calculator_cls)
        )
        is calculator_cls
    )


def test_matgl_pes_calculator_loads_default_potential(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, object] = {}

    class DummyPESCalculator:
        def __init__(self, potential, **kwargs):
            seen["potential"] = potential
            seen["kwargs"] = kwargs

    def fake_load_model(model_identifier):
        seen["model_identifier"] = model_identifier
        return "default-potential"

    monkeypatch.setattr(vpmdk, "M3GNetCalculator", DummyPESCalculator)
    monkeypatch.setattr(vpmdk, "MatGLLoadModel", fake_load_model)
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)

    calculator = vpmdk._build_m3gnet_calculator({"DEVICE": "cpu"})

    assert isinstance(calculator, DummyPESCalculator)
    assert seen == {
        "model_identifier": vpmdk.DEFAULT_MATGL_MODEL,
        "potential": "default-potential",
        "kwargs": {"device": "cpu"},
    }


@pytest.mark.parametrize("selection", ["default", "local", "named"])
@pytest.mark.parametrize("supports_device", [True, False])
def test_matgl_model_classification_and_device_signature_matrix(
    selection: str,
    supports_device: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    model_path = tmp_path / "matgl-model"
    model_path.mkdir()
    named_model = "M3GNet-registry-model"

    if supports_device:
        class Calculator:
            def __init__(self, potential, *, device=None):
                self.potential = potential
                self.device = device
    else:
        class Calculator:
            def __init__(self, potential):
                self.potential = potential
                self.device = None

    def load_model(identifier):
        return SimpleNamespace(identifier=identifier)

    monkeypatch.setattr(vpmdk, "M3GNetCalculator", Calculator)
    monkeypatch.setattr(vpmdk, "MatGLLoadModel", load_model)
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)
    tags = {"DEVICE": "cuda"}
    expected_identifier = vpmdk.DEFAULT_MATGL_MODEL
    if selection == "local":
        tags["MODEL"] = str(model_path)
        expected_identifier = str(model_path)
    elif selection == "named":
        tags["MODEL"] = named_model
        expected_identifier = named_model

    calculator = vpmdk._build_m3gnet_calculator(tags)

    assert calculator.potential.identifier == expected_identifier
    assert calculator.device == ("cuda" if supports_device else None)


def test_matgl_registry_model_name_loads_the_requested_potential(
    monkeypatch: pytest.MonkeyPatch,
):
    requested_model = "M3GNet-MP-2018.6.1-Eform"
    requested_potential = object()
    seen: dict[str, object] = {}

    class DummyPESCalculator:
        def __init__(self, potential, **kwargs):
            seen["potential"] = potential
            seen["kwargs"] = kwargs

    def fake_load_model(model_identifier):
        seen["model_identifier"] = model_identifier
        return requested_potential

    monkeypatch.setattr(vpmdk, "M3GNetCalculator", DummyPESCalculator)
    monkeypatch.setattr(vpmdk, "MatGLLoadModel", fake_load_model)
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)

    calculator = vpmdk._build_m3gnet_calculator(
        {"MODEL": requested_model, "DEVICE": "cpu"}
    )

    assert isinstance(calculator, DummyPESCalculator)
    assert seen == {
        "model_identifier": requested_model,
        "potential": requested_potential,
        "kwargs": {"device": "cpu"},
    }


def test_matgl_potential_is_moved_to_requested_device(
    monkeypatch: pytest.MonkeyPatch,
):
    # Modern MatGL's PESCalculator takes no device argument and does not move the
    # potential, so a DEVICE request must relocate the loaded potential itself
    # before construction -- otherwise placement would silently diverge from the
    # device server status reports.
    moves: list[str] = []

    class MovablePotential:
        def to(self, device):
            moves.append(device)
            return self

    potential = MovablePotential()

    class DummyPESCalculator:
        def __init__(self, potential, **kwargs):
            self.potential = potential

    monkeypatch.setattr(vpmdk, "M3GNetCalculator", DummyPESCalculator)
    monkeypatch.setattr(vpmdk, "MatGLLoadModel", lambda identifier: potential)
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)

    calculator = vpmdk._build_m3gnet_calculator(
        {"MODEL": "provider/registry-model", "DEVICE": "cpu"}
    )

    assert calculator.potential is potential
    assert moves == ["cpu"]


def test_matgl_potential_move_tolerates_missing_to(monkeypatch: pytest.MonkeyPatch):
    # A loaded object without a torch-style .to (older/stub loaders) must not
    # crash the builder; it is used as-is.
    potential = object()

    class DummyPESCalculator:
        def __init__(self, potential, **kwargs):
            self.potential = potential

    monkeypatch.setattr(vpmdk, "M3GNetCalculator", DummyPESCalculator)
    monkeypatch.setattr(vpmdk, "MatGLLoadModel", lambda identifier: potential)
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)

    calculator = vpmdk._build_m3gnet_calculator({"DEVICE": "cpu"})

    assert calculator.potential is potential


def test_matgl_local_path_device_move_failure_is_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # A device-placement failure for an existing local checkpoint must fail
    # loudly, not be caught by _construct_matgl_identifier's load fallback (which
    # would rebuild on CPU while status reports the requested device).
    model_path = tmp_path / "matgl-model"
    model_path.mkdir()

    class DeviceRejectingPotential:
        def to(self, device):
            raise RuntimeError("No CUDA GPUs are available")

    fallback_calls: list[object] = []

    class DummyCalc:
        def __init__(self, model, **kwargs):
            fallback_calls.append((model, kwargs))

    monkeypatch.setattr(vpmdk, "M3GNetCalculator", DummyCalc)
    monkeypatch.setattr(
        vpmdk, "MatGLLoadModel", lambda identifier: DeviceRejectingPotential()
    )
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)

    with pytest.raises(RuntimeError, match="No CUDA GPUs are available"):
        vpmdk._build_m3gnet_calculator(
            {"MODEL": str(model_path), "DEVICE": "cuda"}
        )

    # The local-path fallback must not have silently rebuilt the calculator.
    assert fallback_calls == []


def test_matgl_local_path_fallback_moves_potential_to_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # When potential-based construction fails and the local-path fallback
    # rebuilds from the checkpoint path (dropping the device kwarg on the
    # TypeError retry), the resulting calculator's potential must still be moved
    # to the requested device, keeping the fail-loud device guarantee.
    model_path = tmp_path / "matgl-model"
    model_path.mkdir()
    path_moves: list[str] = []

    class _LoadedPotential:
        def to(self, device):
            return self

    class _PathPotential:
        def to(self, device):
            path_moves.append(device)
            return self

    path_potential = _PathPotential()

    class MatGLCalc:
        def __init__(self, arg, **kwargs):
            if isinstance(arg, _LoadedPotential):
                raise ValueError("this release wants a path, not a potential")
            if "device" in kwargs:
                raise TypeError("device not accepted")
            self.potential = path_potential

    monkeypatch.setattr(vpmdk, "M3GNetCalculator", MatGLCalc)
    monkeypatch.setattr(vpmdk, "MatGLLoadModel", lambda identifier: _LoadedPotential())
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)

    calc = vpmdk._build_m3gnet_calculator(
        {"MODEL": str(model_path), "DEVICE": "cpu"}
    )

    assert isinstance(calc, MatGLCalc)
    assert calc.potential is path_potential
    assert path_moves == ["cpu"]


def test_move_matgl_calculator_reassigns_non_inplace_potential():
    # If a calculator's potential.to() returns a new relocated module instead of
    # mutating in place, the calculator must be repointed at it, or it would keep
    # running on the original (unmoved) potential while status reports otherwise.
    from vpmdk_core.backends.m3gnet import (
        _move_matgl_calculator_potential_to_device,
    )

    moved = object()

    class _Potential:
        def to(self, device):
            return moved

    class _Calc:
        def __init__(self):
            self.potential = _Potential()

    calc = _Calc()
    _move_matgl_calculator_potential_to_device(calc, "cuda")

    assert calc.potential is moved


def test_mace_moves_model_when_device_parameter_unsupported(
    monkeypatch: pytest.MonkeyPatch,
):
    # When MACECalculator has no device parameter, the requested device must
    # still be honored by moving the loaded model(s), not silently dropped.
    moves: list[str] = []

    class _Model:
        def to(self, device):
            moves.append(device)
            return self

    model = _Model()

    class MACECalc:  # no device parameter
        def __init__(self, arg=None):
            self.models = [model]

    monkeypatch.setattr(vpmdk, "MACECalculator", MACECalc)

    calculator = vpmdk._build_mace_calculator({"DEVICE": "cuda"})

    assert isinstance(calculator, MACECalc)
    assert moves == ["cuda"]


def test_mace_repoints_non_inplace_moved_model(monkeypatch: pytest.MonkeyPatch):
    # A non-in-place .to on a MACE model must repoint calculator.models, matching
    # the matgl sibling, or the calculator keeps the original unmoved model.
    moved = object()

    class _Model:
        def to(self, device):
            return moved

    class MACECalc:  # no device parameter
        def __init__(self, arg=None):
            self.models = [_Model()]

    monkeypatch.setattr(vpmdk, "MACECalculator", MACECalc)

    calculator = vpmdk._build_mace_calculator({"DEVICE": "cuda"})

    assert calculator.models == [moved]


def test_matgl_named_model_requires_registry_loader(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[object, ...]] = []

    def calculator(*args, **kwargs):
        calls.append(args)
        return object()

    monkeypatch.setattr(vpmdk, "M3GNetCalculator", calculator)
    monkeypatch.setattr(vpmdk, "MatGLLoadModel", None)
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)

    with pytest.raises(RuntimeError, match="matgl.load_model is unavailable"):
        vpmdk._build_m3gnet_calculator({"MODEL": "M3GNet-registry-model"})

    assert calls == []


@pytest.mark.parametrize("supports_device", [True, False])
def test_matgl_local_identifier_without_loader_device_signature_matrix(
    supports_device: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    model_path = tmp_path / "direct-matgl-checkpoint"
    model_path.mkdir()

    if supports_device:
        class Calculator:
            def __init__(self, model, *, device=None):
                self.model = model
                self.device = device
    else:
        class Calculator:
            def __init__(self, model):
                self.model = model
                self.device = None

    monkeypatch.setattr(vpmdk, "M3GNetCalculator", Calculator)
    monkeypatch.setattr(vpmdk, "MatGLLoadModel", None)
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)

    calculator = vpmdk._build_m3gnet_calculator(
        {"MODEL": str(model_path), "DEVICE": "cuda"}
    )

    assert calculator.model == str(model_path)
    assert calculator.device == ("cuda" if supports_device else None)


@pytest.mark.parametrize("loader_result", [None, False, ""])
def test_matgl_registry_loader_must_return_a_potential(
    monkeypatch: pytest.MonkeyPatch, loader_result
):
    requested_model = "M3GNet-MP-unresolved"
    monkeypatch.setattr(vpmdk, "MatGLLoadModel", lambda model: loader_result)
    monkeypatch.setattr(
        vpmdk,
        "M3GNetCalculator",
        lambda *args, **kwargs: pytest.fail(
            "calculator received an invalid loader result"
        ),
    )
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)

    with pytest.raises(RuntimeError, match="loader returned no model"):
        vpmdk._build_m3gnet_calculator({"MODEL": requested_model})


def test_require_loaded_model_accepts_a_falsy_but_valid_model():
    # A validly-loaded model may be falsy under bool() (e.g. a container-style
    # potential whose __len__ is 0 at construction). It must pass through
    # unchanged rather than be rejected as an empty loader result.
    class EmptyContainerPotential:
        def __len__(self):
            return 0

    potential = EmptyContainerPotential()

    assert (
        vpmdk._require_loaded_model(
            potential, backend_name="X", model="named-model"
        )
        is potential
    )


@pytest.mark.parametrize("sentinel", [None, False, "", "   "])
def test_require_loaded_model_rejects_no_result_sentinels(sentinel):
    with pytest.raises(RuntimeError, match="loader returned no model"):
        vpmdk._require_loaded_model(sentinel, backend_name="X", model=None)


def test_matgl_model_load_error_preserves_original_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_dir = tmp_path / "dgl-model"
    model_dir.mkdir()

    def fail_load(path):
        raise ModuleNotFoundError("No module named 'dgl'")

    monkeypatch.setattr(vpmdk, "M3GNetCalculator", object())
    monkeypatch.setattr(vpmdk, "MatGLLoadModel", fail_load)
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)

    with pytest.raises(RuntimeError, match="Unable to load MatGL model") as exc_info:
        vpmdk._build_m3gnet_calculator({"MODEL": str(model_dir), "DEVICE": "cuda"})

    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)


def test_matgl_loader_failure_falls_back_to_direct_checkpoint_constructor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_dir = tmp_path / "legacy-checkpoint"
    model_dir.mkdir()
    seen: dict[str, object] = {}

    def fail_load(path):
        raise ValueError("unsupported by matgl.load_model")

    class DirectCheckpointCalculator:
        def __init__(self, model, **kwargs):
            seen["model"] = model
            seen["kwargs"] = kwargs

    monkeypatch.setattr(vpmdk, "M3GNetCalculator", DirectCheckpointCalculator)
    monkeypatch.setattr(vpmdk, "MatGLLoadModel", fail_load)
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)

    calculator = vpmdk._build_m3gnet_calculator(
        {"MODEL": str(model_dir), "DEVICE": "cpu"}
    )

    assert isinstance(calculator, DirectCheckpointCalculator)
    assert seen == {"model": str(model_dir), "kwargs": {"device": "cpu"}}


def test_matgl_potential_construction_failure_falls_back_to_model_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_dir = tmp_path / "legacy-checkpoint"
    model_dir.mkdir()
    calls: list[tuple[object, dict[str, object]]] = []

    def fake_load_model(path):
        return "loaded-potential"

    class PathCompatibleCalculator:
        def __init__(self, model, **kwargs):
            calls.append((model, dict(kwargs)))
            if model == "loaded-potential" or kwargs:
                raise TypeError("unsupported constructor form")

    monkeypatch.setattr(vpmdk, "M3GNetCalculator", PathCompatibleCalculator)
    monkeypatch.setattr(vpmdk, "MatGLLoadModel", fake_load_model)
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)

    calculator = vpmdk._build_m3gnet_calculator(
        {"MODEL": str(model_dir), "DEVICE": "cuda"}
    )

    assert isinstance(calculator, PathCompatibleCalculator)
    assert calls == [
        ("loaded-potential", {"device": "cuda"}),
        ("loaded-potential", {}),
        (str(model_dir), {"device": "cuda"}),
        (str(model_dir), {}),
    ]


def test_matgl_loaded_potential_retries_without_device_before_path_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_dir = tmp_path / "loaded-checkpoint"
    model_dir.mkdir()
    loaded_potential = object()
    calls: list[tuple[object, dict[str, object]]] = []

    class PotentialOnlyCalculator:
        def __init__(self, potential, **kwargs):
            calls.append((potential, dict(kwargs)))
            if isinstance(potential, str):
                raise TypeError("a loaded potential is required")
            if kwargs:
                raise TypeError("device is unsupported")

    monkeypatch.setattr(vpmdk, "M3GNetCalculator", PotentialOnlyCalculator)
    monkeypatch.setattr(
        vpmdk, "MatGLLoadModel", lambda model: loaded_potential
    )
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)

    calculator = vpmdk._build_m3gnet_calculator(
        {"MODEL": str(model_dir), "DEVICE": "cuda"}
    )

    assert isinstance(calculator, PotentialOnlyCalculator)
    assert calls == [
        (loaded_potential, {"device": "cuda"}),
        (loaded_potential, {}),
    ]


def test_matgl_default_potential_retries_without_device(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[object, dict[str, object]]] = []

    class NoDeviceCalculator:
        def __init__(self, potential, **kwargs):
            calls.append((potential, dict(kwargs)))
            if kwargs:
                raise TypeError("device is unsupported")

    monkeypatch.setattr(vpmdk, "M3GNetCalculator", NoDeviceCalculator)
    monkeypatch.setattr(vpmdk, "MatGLLoadModel", lambda model: "default-potential")
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)

    calculator = vpmdk._build_m3gnet_calculator({"DEVICE": "cuda"})

    assert isinstance(calculator, NoDeviceCalculator)
    assert calls == [
        ("default-potential", {"device": "cuda"}),
        ("default-potential", {}),
    ]


def test_legacy_m3gnet_missing_explicit_model_does_not_load_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    load_calls: list[tuple[object, ...]] = []

    class LegacyModel:
        @classmethod
        def load(cls, *args):
            load_calls.append(args)
            return object()

    class LegacyPotential:
        @classmethod
        def from_checkpoint(cls, path):
            return cls()

    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", True)
    monkeypatch.setattr(vpmdk, "LegacyM3GNet", LegacyModel)
    monkeypatch.setattr(vpmdk, "LegacyM3GNetPotential", LegacyPotential)
    monkeypatch.setattr(vpmdk, "M3GNetCalculator", lambda **kwargs: object())

    with pytest.raises(
        FileNotFoundError, match="Legacy M3GNet MODEL path not found"
    ):
        vpmdk._build_m3gnet_calculator(
            {"MODEL": str(tmp_path / "named-or-missing-model"), "DEVICE": "cpu"}
        )

    assert load_calls == []


def test_legacy_m3gnet_named_foundation_model_is_loaded_not_rejected(
    monkeypatch: pytest.MonkeyPatch,
):
    # Legacy M3GNet resolves foundation-model preset names through its own
    # loader, so a bare (non-path) MODEL name must reach the loader instead of
    # being rejected as a missing local path by a local-only policy.
    seen: dict[str, object] = {}

    class LegacyModel:
        @classmethod
        def load(cls, *args):
            seen["load"] = args
            return object()

    class LegacyPotential:
        def __init__(self, model):
            self.model = model

    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", True)
    monkeypatch.setattr(vpmdk, "LegacyM3GNet", LegacyModel)
    monkeypatch.setattr(vpmdk, "LegacyM3GNetPotential", LegacyPotential)
    monkeypatch.setattr(
        vpmdk, "M3GNetCalculator", lambda **kwargs: SimpleNamespace(**kwargs)
    )

    calculator = vpmdk._build_m3gnet_calculator(
        {"MODEL": "M3GNet-MP-2021.2.8-PES", "DEVICE": "cpu"}
    )

    assert seen["load"] == ("M3GNet-MP-2021.2.8-PES",)
    assert isinstance(calculator.potential, LegacyPotential)


@pytest.mark.parametrize("selection", ["default", "local"])
@pytest.mark.parametrize("supports_device", [True, False])
def test_legacy_m3gnet_model_classification_and_device_signature_matrix(
    selection: str,
    supports_device: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    model_path = tmp_path / "legacy.model"
    model_path.write_text("placeholder")
    default_model = object()

    class LegacyModel:
        @classmethod
        def load(cls, *args):
            assert args == ()
            return default_model

    class LegacyPotential:
        def __init__(self, selected_model):
            self.selection = selected_model

        @classmethod
        def from_checkpoint(cls, path):
            potential = cls(None)
            potential.selection = path
            return potential

    if supports_device:
        class Calculator:
            def __init__(self, *, potential, device=None):
                self.potential = potential
                self.device = device
    else:
        class Calculator:
            def __init__(self, *, potential):
                self.potential = potential
                self.device = None

    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", True)
    monkeypatch.setattr(vpmdk, "LegacyM3GNet", LegacyModel)
    monkeypatch.setattr(vpmdk, "LegacyM3GNetPotential", LegacyPotential)
    monkeypatch.setattr(vpmdk, "M3GNetCalculator", Calculator)
    tags = {"DEVICE": "cuda"}
    if selection == "local":
        tags["MODEL"] = str(model_path)

    calculator = vpmdk._build_m3gnet_calculator(tags)

    expected_selection = str(model_path) if selection == "local" else default_model
    assert calculator.potential.selection is expected_selection or (
        calculator.potential.selection == expected_selection
    )
    assert calculator.device == ("cuda" if supports_device else None)


def test_legacy_m3gnet_invalid_explicit_model_does_not_load_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_path = tmp_path / "invalid.model"
    model_path.write_text("invalid")
    load_calls: list[tuple[object, ...]] = []

    class LegacyModel:
        @classmethod
        def load(cls, *args):
            load_calls.append(args)
            raise ValueError("invalid legacy model")

    class LegacyPotential:
        def __init__(self, model):
            self.model = model

        @classmethod
        def from_checkpoint(cls, path):
            raise ValueError("invalid checkpoint")

    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", True)
    monkeypatch.setattr(vpmdk, "LegacyM3GNet", LegacyModel)
    monkeypatch.setattr(vpmdk, "LegacyM3GNetPotential", LegacyPotential)
    monkeypatch.setattr(vpmdk, "M3GNetCalculator", lambda **kwargs: object())

    with pytest.raises(RuntimeError, match="Unable to load requested legacy"):
        vpmdk._build_m3gnet_calculator({"MODEL": str(model_path)})

    assert load_calls == [(str(model_path),)]


def test_legacy_m3gnet_empty_loader_results_do_not_reach_calculator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_path = tmp_path / "empty.model"
    model_path.write_text("placeholder")
    load_calls: list[tuple[object, ...]] = []

    class LegacyModel:
        @classmethod
        def load(cls, *args):
            load_calls.append(args)
            return None

    class LegacyPotential:
        @classmethod
        def from_checkpoint(cls, path):
            return None

    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", True)
    monkeypatch.setattr(vpmdk, "LegacyM3GNet", LegacyModel)
    monkeypatch.setattr(vpmdk, "LegacyM3GNetPotential", LegacyPotential)
    monkeypatch.setattr(
        vpmdk,
        "M3GNetCalculator",
        lambda **kwargs: pytest.fail("calculator received an empty potential"),
    )

    with pytest.raises(RuntimeError, match="Unable to load requested legacy"):
        vpmdk._build_m3gnet_calculator({"MODEL": str(model_path)})

    assert load_calls == [(str(model_path),)]


def test_legacy_m3gnet_explicit_checkpoint_is_forwarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_path = tmp_path / "legacy.model"
    model_path.write_text("placeholder")
    potential = object()
    seen: dict[str, object] = {}

    class LegacyPotential:
        @classmethod
        def from_checkpoint(cls, path):
            seen["path"] = path
            return potential

    def calculator(**kwargs):
        seen["calculator_kwargs"] = kwargs
        return "calculator"

    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", True)
    monkeypatch.setattr(vpmdk, "LegacyM3GNet", None)
    monkeypatch.setattr(vpmdk, "LegacyM3GNetPotential", LegacyPotential)
    monkeypatch.setattr(vpmdk, "M3GNetCalculator", calculator)

    result = vpmdk._build_m3gnet_calculator(
        {"MODEL": str(model_path), "DEVICE": "cpu"}
    )

    assert result == "calculator"
    assert seen == {
        "path": str(model_path),
        "calculator_kwargs": {"potential": potential, "device": "cpu"},
    }


def test_legacy_m3gnet_omitted_model_uses_default(
    monkeypatch: pytest.MonkeyPatch,
):
    loaded_model = object()
    potential = object()
    load_calls: list[tuple[object, ...]] = []

    class LegacyModel:
        @classmethod
        def load(cls, *args):
            load_calls.append(args)
            return loaded_model

    def legacy_potential(model):
        assert model is loaded_model
        return potential

    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", True)
    monkeypatch.setattr(vpmdk, "LegacyM3GNet", LegacyModel)
    monkeypatch.setattr(vpmdk, "LegacyM3GNetPotential", legacy_potential)
    monkeypatch.setattr(
        vpmdk,
        "M3GNetCalculator",
        lambda **kwargs: ("calculator", kwargs),
    )

    result = vpmdk._build_m3gnet_calculator({"DEVICE": "cpu"})

    assert result == (
        "calculator",
        {"potential": potential, "device": "cpu"},
    )
    assert load_calls == [()]


def test_mattersim_forwards_device_and_optional_tags(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, object] = {}

    class FakeMatterSimCalculator:
        def __init__(self, *, device="cuda", compute_stress=True, stress_weight=None):
            seen.update(
                {
                    "device": device,
                    "compute_stress": compute_stress,
                    "stress_weight": stress_weight,
                }
            )

    monkeypatch.setattr(vpmdk, "MatterSimCalculator", FakeMatterSimCalculator)

    calc = vpmdk._build_mattersim_calculator(
        {
            "DEVICE": "cpu",
            "MATTERSIM_COMPUTE_STRESS": "false",
            "MATTERSIM_STRESS_WEIGHT": "0.5",
        }
    )

    assert isinstance(calc, FakeMatterSimCalculator)
    assert seen == {
        "device": "cpu",
        "compute_stress": False,
        "stress_weight": 0.5,
    }


def test_mattersim_physics_tag_rejects_kwargs_only_signature(
    monkeypatch: pytest.MonkeyPatch,
):
    # A MatterSimCalculator whose __init__ takes load_path plus a **kwargs
    # catch-all (no EXPLICIT compute_stress) would silently swallow and drop a
    # requested MATTERSIM_COMPUTE_STRESS, computing without stress. The physics
    # gate must use _callable_declares_parameter (not _supports, which is True for
    # **kwargs) so it raises "does not accept" instead of dropping the tag.
    class KwargsForwardingMatterSimCalculator:
        def __init__(self, load_path=None, **kwargs):
            self.load_path = load_path
            self.kwargs = kwargs

    monkeypatch.setattr(
        vpmdk, "MatterSimCalculator", KwargsForwardingMatterSimCalculator
    )

    with pytest.raises(RuntimeError, match="MATTERSIM_COMPUTE_STRESS"):
        vpmdk._build_mattersim_calculator(
            {"DEVICE": "cpu", "MATTERSIM_COMPUTE_STRESS": "true"}
        )


def test_mattersim_physics_tag_accepts_one_verified_forwarding_hop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkpoint = tmp_path / "m.pth"
    checkpoint.write_text("placeholder")

    class Upstream:
        def __init__(self, *, device=None, compute_stress=None, stress_weight=None):
            self.compute_stress = compute_stress
            self.stress_weight = stress_weight

        @classmethod
        def from_checkpoint(cls, load_path, *, device=None, **kwargs):
            built = cls(device=device, **kwargs)
            built.load_path = load_path
            return built

    monkeypatch.setattr(vpmdk, "MatterSimCalculator", Upstream)
    built = vpmdk._build_mattersim_calculator(
        {
            "MODEL": str(checkpoint),
            "MATTERSIM_COMPUTE_STRESS": "1",
            "MATTERSIM_STRESS_WEIGHT": "0.5",
            "DEVICE": "cpu",
        }
    )
    assert built.compute_stress is True
    assert built.stress_weight == 0.5

    # from_checkpoint forwards, but __init__ ALSO only has **kwargs: the tag has
    # nowhere to land, so it must raise instead of being swallowed.
    class NoDeclaringTarget:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        @classmethod
        def from_checkpoint(cls, load_path, **kwargs):
            return cls(**kwargs)

    monkeypatch.setattr(vpmdk, "MatterSimCalculator", NoDeclaringTarget)
    with pytest.raises(RuntimeError, match="MATTERSIM_COMPUTE_STRESS"):
        vpmdk._build_mattersim_calculator(
            {"MODEL": str(checkpoint), "MATTERSIM_COMPUTE_STRESS": "1", "DEVICE": "cpu"}
        )


def test_langevin_gamma_accepts_vasp_per_species_lists():
    # VASP takes one LANGEVIN_GAMMA per POTCAR species, so real pymatgen returns a
    # list with one entry per species. Unwrapping only a singleton discarded every
    # multi-species INCAR, and the Langevin setup then silently used its 1.0 default
    # friction -- different dynamics than requested. A uniform list unwraps; a
    # genuinely per-species list warns explicitly and uses the first value.
    from vpmdk_core.settings import incar as incar_module

    def gamma(value):
        return incar_module._extract_thermostat_parameters(
            {"LANGEVIN_GAMMA": value}
        ).get("LANGEVIN_GAMMA")

    assert gamma(10.0) == 10.0
    assert gamma([10.0]) == 10.0
    assert gamma([10.0, 10.0, 10.0]) == 10.0  # multi-species, uniform
    assert gamma([10.0, 5.0]) == 10.0  # per-species: first value, with a warning
    assert gamma("abc") is None  # genuinely malformed still ignored

    # Sibling scalar thermostat tags are unaffected.
    parameters = incar_module._extract_thermostat_parameters(
        {"ANDERSEN_PROB": 0.5, "CSVR_PERIOD": 40, "NHC_NCHAINS": 3, "NHC_PERIOD": 40}
    )
    assert parameters == {
        "ANDERSEN_PROB": 0.5,
        "CSVR_PERIOD": 40.0,
        "NHC_NCHAINS": 3.0,
        "NHC_PERIOD": 40.0,
    }


def test_tace_fidelity_tag_is_never_silently_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # TACE_FIDELITY_IDX / TACE_LEVEL selects WHICH DFT level the model predicts,
    # so dropping it silently changes the physics. Gating with
    # _callable_supports_parameter (True for a bare **kwargs) let the value be
    # absorbed and ignored -- energies from a different fidelity head, no warning
    # -- and the old if/elif had no else, so a build exposing neither spelling
    # discarded the tag outright. Both must raise instead.
    checkpoint = tmp_path / "tace.model"
    checkpoint.write_text("placeholder")

    class KwargsOnlyTACE:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(vpmdk, "TACEAseCalc", KwargsOnlyTACE)
    for tag in ("TACE_FIDELITY_IDX", "TACE_LEVEL"):
        with pytest.raises(RuntimeError, match=tag):
            vpmdk._build_tace_calculator({"MODEL": str(checkpoint), tag: "2"})

    # Without the tag the same build is fine (nothing to drop).
    assert vpmdk._build_tace_calculator({"MODEL": str(checkpoint)}) is not None

    # A build that DECLARES either spelling still receives the value.
    class DeclaresFidelity:
        def __init__(self, model=None, device=None, fidelity_idx=None, **kwargs):
            self.fidelity_idx = fidelity_idx

    class DeclaresLevel:
        def __init__(self, model=None, device=None, level=None, **kwargs):
            self.level = level

    monkeypatch.setattr(vpmdk, "TACEAseCalc", DeclaresFidelity)
    built = vpmdk._build_tace_calculator(
        {"MODEL": str(checkpoint), "TACE_FIDELITY_IDX": "2"}
    )
    assert built.fidelity_idx == 2
    monkeypatch.setattr(vpmdk, "TACEAseCalc", DeclaresLevel)
    built = vpmdk._build_tace_calculator({"MODEL": str(checkpoint), "TACE_LEVEL": "3"})
    assert built.level == 3


def test_matris_graph_converter_falls_back_when_kwarg_is_not_declared(
    monkeypatch: pytest.MonkeyPatch,
):
    # Selecting the kwarg path via _callable_supports_parameter (True for a bare
    # **kwargs) passed graph_converter_algorithm into a catch-all that discarded
    # it AND skipped the explicit _override_model_graph_converter_algorithm
    # fallback below, because that branch returns early -- so the model quietly
    # used its default converter while `status` advertised the requested one.
    applied: list[str] = []

    def fake_override(model, *, algorithm, backend_name):
        applied.append(algorithm)
        return model

    class KwargsOnlyMatRIS:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.model = object()

    class DeclaringMatRIS:
        def __init__(
            self, model=None, task=None, device=None, graph_converter_algorithm=None, **kw
        ):
            self.graph_converter_algorithm = graph_converter_algorithm
            self.model = object()

    monkeypatch.setattr(
        vpmdk, "_override_model_graph_converter_algorithm", fake_override
    )
    tags = {"MODEL": "an-unregistered-matris-name", "MATRIS_GRAPH_CONVERTER": "fast"}

    monkeypatch.setattr(vpmdk, "MatRISCalculator", KwargsOnlyMatRIS)
    calculator = vpmdk._build_matris_calculator(dict(tags))
    assert "graph_converter_algorithm" not in calculator.kwargs
    assert applied == ["fast"]  # the explicit override ran instead

    applied.clear()
    monkeypatch.setattr(vpmdk, "MatRISCalculator", DeclaringMatRIS)
    calculator = vpmdk._build_matris_calculator(dict(tags))
    assert calculator.graph_converter_algorithm == "fast"
    assert applied == []  # kwarg path, no double application


def test_blank_device_does_not_break_matgl_module_placement():
    # A present-but-blank `DEVICE =` resolves to "" (_resolve_device autodetects
    # only for None), and module.to("") raises "Device string must not be empty".
    # MATGL is deliberately outside the server's blank->cpu normalization, so the
    # relocation helper must treat a blank like an omitted device and leave
    # placement alone -- as the pre-relocation code effectively did.
    from vpmdk_core.backends.m3gnet import _move_module_to_device

    class Module:
        def __init__(self):
            self.moved_to = None

        def to(self, device):
            if isinstance(device, str) and not device.strip():
                raise RuntimeError("Device string must not be empty")
            self.moved_to = device
            return self

    for device in (None, "", "   "):
        module = Module()
        assert _move_module_to_device(module, device) is module
        assert module.moved_to is None
    module = Module()
    assert _move_module_to_device(module, "cpu") is module
    assert module.moved_to == "cpu"


def test_mattersim_uses_local_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model_path = tmp_path / "mattersim.pth"
    model_path.write_text("dummy")
    seen: dict[str, object] = {}

    class FakeMatterSimCalculator:
        def __init__(self, *, device=None):
            pass

        @classmethod
        def from_checkpoint(cls, load_path, **kwargs):
            seen["load_path"] = load_path
            seen["kwargs"] = kwargs
            return "mattersim"

    monkeypatch.setattr(vpmdk, "MatterSimCalculator", FakeMatterSimCalculator)

    calc = vpmdk._build_mattersim_calculator(
        {"MODEL": str(model_path), "DEVICE": "cuda:0"}
    )

    assert calc == "mattersim"
    assert seen == {"load_path": str(model_path), "kwargs": {"device": "cuda:0"}}


@pytest.mark.parametrize(
    ("supports_device", "expected_kwargs"),
    [(True, {"device": "cuda:0"}), (False, {})],
)
def test_mattersim_named_model_and_device_signature_matrix(
    supports_device: bool,
    expected_kwargs: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
):
    seen: dict[str, object] = {}

    if supports_device:

        class FakeMatterSimCalculator:
            def __init__(self, *, device=None):
                pass

            @classmethod
            def from_checkpoint(cls, selector, *, device=None):
                seen.update({"selector": selector, "kwargs": {"device": device}})
                return "mattersim"
    else:

        class FakeMatterSimCalculator:
            def __init__(self, *, device=None):
                pass

            @classmethod
            def from_checkpoint(cls, selector):
                seen.update({"selector": selector, "kwargs": {}})
                return "mattersim"

    monkeypatch.setattr(vpmdk, "MatterSimCalculator", FakeMatterSimCalculator)

    calculator = vpmdk._build_mattersim_calculator(
        {"MODEL": "mattersim-v1.0.0-5M", "DEVICE": "cuda:0"}
    )

    assert calculator == "mattersim"
    assert seen == {
        "selector": "mattersim-v1.0.0-5M",
        "kwargs": expected_kwargs,
    }


def test_mattersim_missing_path_fails_before_upstream_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[str] = []

    class FakeMatterSimCalculator:
        @classmethod
        def from_checkpoint(cls, selector):
            calls.append(selector)
            return object()

    monkeypatch.setattr(vpmdk, "MatterSimCalculator", FakeMatterSimCalculator)
    missing_path = tmp_path / "missing-mattersim.pth"

    with pytest.raises(FileNotFoundError, match="MATTERSIM MODEL path not found"):
        vpmdk._build_mattersim_calculator({"MODEL": str(missing_path)})

    assert calls == []


def test_mattersim_named_model_uses_legacy_load_path_api(
    monkeypatch: pytest.MonkeyPatch,
):
    seen: dict[str, object] = {}

    class LegacyMatterSimCalculator:
        def __init__(self, *, load_path=None, device=None):
            seen.update({"load_path": load_path, "device": device})

    monkeypatch.setattr(vpmdk, "MatterSimCalculator", LegacyMatterSimCalculator)

    calculator = vpmdk._build_mattersim_calculator(
        {"MODEL": "mattersim-v1.0.0-5M", "DEVICE": "cpu"}
    )

    assert isinstance(calculator, LegacyMatterSimCalculator)
    assert seen == {"load_path": "mattersim-v1.0.0-5M", "device": "cpu"}


def test_mattersim_named_model_rejects_unsupported_legacy_api(
    monkeypatch: pytest.MonkeyPatch,
):
    constructor_calls: list[object] = []

    class PotentialOnlyMatterSimCalculator:
        def __init__(self, potential):
            constructor_calls.append(potential)

    monkeypatch.setattr(
        vpmdk, "MatterSimCalculator", PotentialOnlyMatterSimCalculator
    )

    with pytest.raises(
        RuntimeError,
        match="from_checkpoint and load_path are unavailable",
    ):
        vpmdk._build_mattersim_calculator(
            {"MODEL": "mattersim-v1.0.0-5M", "DEVICE": "cpu"}
        )

    assert constructor_calls == []


def test_mattersim_rejects_unsupported_physics_tags(
    monkeypatch: pytest.MonkeyPatch,
):
    # MATTERSIM_COMPUTE_STRESS/STRESS_WEIGHT come only from an explicit BCAR tag.
    # Silently dropping one changes what the run computes, so an installed
    # from_checkpoint that cannot honor it must fail loudly.
    class FakeMatterSimCalculator:
        @classmethod
        def from_checkpoint(cls, selector, *, device=None):
            return "mattersim"

    monkeypatch.setattr(vpmdk, "MatterSimCalculator", FakeMatterSimCalculator)

    with pytest.raises(RuntimeError, match="MATTERSIM_COMPUTE_STRESS"):
        vpmdk._build_mattersim_calculator(
            {
                "MODEL": "mattersim-v1.0.0-5M",
                "DEVICE": "cpu",
                "MATTERSIM_COMPUTE_STRESS": "false",
            }
        )

    # DEVICE is resolved automatically and only affects placement, so a loader
    # that picks its own device is still accepted.
    assert (
        vpmdk._build_mattersim_calculator(
            {"MODEL": "mattersim-v1.0.0-5M", "DEVICE": "cpu"}
        )
        == "mattersim"
    )


@pytest.mark.parametrize(
    "tags",
    [
        # Omitted MODEL -> plain constructor branch.
        {"DEVICE": "cpu", "MATTERSIM_COMPUTE_STRESS": "true"},
        # Explicit local checkpoint -> positional constructor branch.
        {"MODEL": "__LOCAL__", "DEVICE": "cpu", "MATTERSIM_STRESS_WEIGHT": "0.5"},
    ],
)
def test_mattersim_physics_tags_are_never_dropped_on_any_branch(
    tags: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # The guard must cover every construction branch: identical tags raising in
    # one path and being silently ignored in another is exactly the silent
    # physics change the guard exists to prevent.
    class PlainMatterSimCalculator:
        def __init__(self, *args, device=None):
            pass

    monkeypatch.setattr(vpmdk, "MatterSimCalculator", PlainMatterSimCalculator)

    resolved = dict(tags)
    if resolved.get("MODEL") == "__LOCAL__":
        checkpoint = tmp_path / "model.pth"
        checkpoint.write_text("weights")
        resolved["MODEL"] = str(checkpoint)

    with pytest.raises(RuntimeError, match="MATTERSIM_"):
        vpmdk._build_mattersim_calculator(resolved)


def test_mattersim_kwargs_only_constructor_does_not_absorb_load_path(
    monkeypatch: pytest.MonkeyPatch,
):
    # A ``**kwargs``-only constructor does not truly support ``load_path``; the
    # checkpoint would be silently swallowed into ``**kwargs`` and the default
    # model loaded. Such a build must raise instead of building a wrong model.
    constructor_calls: list[dict[str, object]] = []

    class KwargsOnlyMatterSimCalculator:
        def __init__(self, **kwargs):
            constructor_calls.append(kwargs)

    monkeypatch.setattr(
        vpmdk, "MatterSimCalculator", KwargsOnlyMatterSimCalculator
    )

    with pytest.raises(
        RuntimeError,
        match="from_checkpoint and load_path are unavailable",
    ):
        vpmdk._build_mattersim_calculator(
            {"MODEL": "mattersim-v1.0.0-5M", "DEVICE": "cpu"}
        )

    assert constructor_calls == []


def test_dynamic_simple_backend_forwards_local_and_named_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[tuple[object, ...]] = []

    class PluginCalculator:
        def __init__(self, *args):
            calls.append(args)

    monkeypatch.setitem(
        vpmdk._SIMPLE_CALCULATORS,
        "PLUGIN",
        ("PluginCalculator", "plugin unavailable"),
    )
    monkeypatch.setattr(
        vpmdk, "PluginCalculator", PluginCalculator, raising=False
    )

    local_model = tmp_path / "plugin.model"
    local_model.write_text("placeholder")
    vpmdk._build_calculator_from_tags(
        {"MLP": "PLUGIN", "MODEL": str(local_model)}
    )
    vpmdk._build_calculator_from_tags(
        {"MLP": "PLUGIN", "MODEL": "plugin-preset"}
    )
    vpmdk._build_calculator_from_tags({"MLP": "PLUGIN"})

    assert calls == [
        (str(local_model),),
        ("plugin-preset",),
        (),
    ]


def test_dynamic_simple_backend_rejects_missing_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    constructor_calls: list[tuple[object, ...]] = []

    class PluginCalculator:
        def __init__(self, *args):
            constructor_calls.append(args)

    monkeypatch.setitem(
        vpmdk._SIMPLE_CALCULATORS,
        "PLUGIN",
        ("PluginCalculator", "plugin unavailable"),
    )
    monkeypatch.setattr(
        vpmdk, "PluginCalculator", PluginCalculator, raising=False
    )

    with pytest.raises(FileNotFoundError, match="PLUGIN MODEL path not found"):
        vpmdk._build_calculator_from_tags(
            {"MLP": "PLUGIN", "MODEL": str(tmp_path / "missing.model")}
        )

    assert constructor_calls == []


def test_grace_unknown_name_warns_and_loads_effective_fallback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    seen: dict[str, object] = {}

    def grace_fm(model, **kwargs):
        seen.update({"model": model, "kwargs": kwargs})
        return "grace"

    monkeypatch.setattr(vpmdk, "TPCalculator", object)
    monkeypatch.setattr(vpmdk, "GRACE_MODEL_NAMES", ["GRACE-INSTALLED"])
    monkeypatch.setattr(vpmdk, "grace_fm", grace_fm)

    calculator = vpmdk._build_grace_calculator(
        {"MODEL": "grace-2l-mp", "GRACE_FLOAT_DTYPE": "float32"}
    )

    assert calculator == "grace"
    assert seen == {
        "model": "GRACE-INSTALLED",
        "kwargs": {"float_dtype": "float32"},
    }
    assert capsys.readouterr().out == (
        "Warning: Unknown GRACE model 'grace-2l-mp', using default "
        "GRACE-INSTALLED instead.\n"
    )


def test_build_sevennet_calculator_uses_new_backend_and_tags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_path = tmp_path / "sevennet.ckpt"
    model_path.write_text("dummy")
    seen: dict[str, object] = {}

    class FakeSevenNetCalculator:
        def __init__(
            self,
            *,
            model,
            device="auto",
            file_type="checkpoint",
            modal=None,
            enable_cueq=None,
            enable_flash=None,
            enable_oeq=None,
            **_,
        ):
            seen.update(
                {
                    "model": model,
                    "device": device,
                    "file_type": file_type,
                    "modal": modal,
                    "enable_cueq": enable_cueq,
                    "enable_flash": enable_flash,
                    "enable_oeq": enable_oeq,
                }
            )

    monkeypatch.setattr(vpmdk, "SevenNetCalculator", FakeSevenNetCalculator)
    monkeypatch.setattr(vpmdk, "_SEVENNET_PACKAGE", "sevenn")
    monkeypatch.setattr(vpmdk, "_is_sevennet_flash_available", lambda: True)

    calc = vpmdk._build_sevennet_calculator(
        {
            "MODEL": str(model_path),
            "DEVICE": "cuda:0",
            "SEVENNET_MODAL": "mpa",
            "SEVENNET_ENABLE_FLASH": "1",
        }
    )

    assert isinstance(calc, FakeSevenNetCalculator)
    assert seen == {
        "model": str(model_path),
        "device": "cuda:0",
        "file_type": "checkpoint",
        "modal": "mpa",
        "enable_cueq": False,
        "enable_flash": True,
        "enable_oeq": False,
    }


def test_build_flashtp_calculator_forces_flash(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, object] = {}

    class FakeSevenNetCalculator:
        def __init__(
            self,
            *,
            model,
            device="auto",
            file_type="checkpoint",
            modal=None,
            enable_cueq=None,
            enable_flash=None,
            enable_oeq=None,
            **_,
        ):
            seen.update(
                {
                    "model": model,
                    "device": device,
                    "file_type": file_type,
                    "modal": modal,
                    "enable_cueq": enable_cueq,
                    "enable_flash": enable_flash,
                    "enable_oeq": enable_oeq,
                }
            )

    monkeypatch.setattr(vpmdk, "SevenNetCalculator", FakeSevenNetCalculator)
    monkeypatch.setattr(vpmdk, "_SEVENNET_PACKAGE", "sevenn")
    monkeypatch.setattr(vpmdk, "_is_sevennet_flash_available", lambda: True)

    calc = vpmdk._build_flashtp_calculator({"DEVICE": "cuda"})

    assert isinstance(calc, FakeSevenNetCalculator)
    assert seen == {
        "model": vpmdk.DEFAULT_SEVENNET_MODEL,
        "device": "cuda",
        "file_type": "checkpoint",
        "modal": None,
        "enable_cueq": False,
        "enable_flash": True,
        "enable_oeq": False,
    }


def test_build_flashtp_rejects_conflicting_accelerators(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeSevenNetCalculator:
        def __init__(self, **_):
            raise AssertionError("constructor should not be reached")

    monkeypatch.setattr(vpmdk, "SevenNetCalculator", FakeSevenNetCalculator)
    monkeypatch.setattr(vpmdk, "_SEVENNET_PACKAGE", "sevenn")

    with pytest.raises(ValueError, match="MLP=FLASHTP"):
        vpmdk._build_flashtp_calculator({"SEVENNET_ENABLE_CUEQ": "1"})


def test_build_flashtp_supports_partial_sevennet_accelerators(
    monkeypatch: pytest.MonkeyPatch,
):
    seen: dict[str, object] = {}

    class FakeSevenNetCalculator:
        def __init__(
            self,
            *,
            model,
            device="auto",
            file_type="checkpoint",
            enable_cueq=None,
            enable_flash=None,
            **_,
        ):
            seen.update(
                {
                    "model": model,
                    "device": device,
                    "file_type": file_type,
                    "enable_cueq": enable_cueq,
                    "enable_flash": enable_flash,
                }
            )

    monkeypatch.setattr(vpmdk, "SevenNetCalculator", FakeSevenNetCalculator)
    monkeypatch.setattr(vpmdk, "_SEVENNET_PACKAGE", "sevenn")
    monkeypatch.setattr(vpmdk, "_is_sevennet_flash_available", lambda: True)

    calc = vpmdk._build_flashtp_calculator({"MODEL": "7net-0", "DEVICE": "cuda"})

    assert isinstance(calc, FakeSevenNetCalculator)
    assert seen == {
        "model": "7net-0",
        "device": "cuda",
        "file_type": "checkpoint",
        "enable_cueq": False,
        "enable_flash": True,
    }


def test_build_sevennet_flash_requires_checkpoint(monkeypatch: pytest.MonkeyPatch):
    class FakeSevenNetCalculator:
        def __init__(self, **_):
            raise AssertionError("constructor should not be reached")

    monkeypatch.setattr(vpmdk, "SevenNetCalculator", FakeSevenNetCalculator)
    monkeypatch.setattr(vpmdk, "_SEVENNET_PACKAGE", "sevenn")
    monkeypatch.setattr(vpmdk, "_is_sevennet_flash_available", lambda: True)

    with pytest.raises(ValueError, match="SEVENNET_FILE_TYPE=checkpoint"):
        vpmdk._build_sevennet_calculator(
            {
                "MODEL": "7net-0",
                "SEVENNET_FILE_TYPE": "torchscript",
                "SEVENNET_ENABLE_FLASH": "1",
            }
        )


def test_build_sevennet_rejects_unsupported_oeq(monkeypatch: pytest.MonkeyPatch):
    class FakeSevenNetCalculator:
        def __init__(self, *, model, device="auto", file_type="checkpoint", **_):
            self.model = model
            self.device = device
            self.file_type = file_type

    monkeypatch.setattr(vpmdk, "SevenNetCalculator", FakeSevenNetCalculator)
    monkeypatch.setattr(vpmdk, "_SEVENNET_PACKAGE", "sevenn")

    with pytest.raises(RuntimeError, match="SEVENNET_ENABLE_OEQ"):
        vpmdk._build_sevennet_calculator({"MODEL": "7net-0", "SEVENNET_ENABLE_OEQ": "1"})


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
    expected_device = vpmdk._resolve_device(None) or "cpu"

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
        "device": expected_device,
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
    expected_device = vpmdk._resolve_device(None) or "cpu"

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
        "device": expected_device,
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
            super().__init__()
            seen["init_kwargs"] = dict(kwargs)
            self.model = FakeModel()
            self.device = None
            self.backend = kwargs["backend"]
            self.cutoff = 6.0
            self._capacity_multiplier = kwargs["capacity_multiplier"]

    monkeypatch.setattr(vpmdk, "NequixCalculator", FakeNequixCalculator)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(device=lambda value: value))

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
    assert seen["init_kwargs"] == {
        "model_path": str(model_path),
        "model_name": "nequix-oam-1",
        "backend": "torch",
        "use_kernel": True,
        "use_compile": True,
        "capacity_multiplier": 1.25,
    }
    assert seen["moved_to"] == "cpu"
    assert seen["eval_called"] is True
    assert str(calc.device) == "cpu"
    assert calc.backend == "torch"
    assert calc.cutoff == 6.0
    assert calc._capacity_multiplier == 1.25


def test_nequix_torch_backend_preserves_constructor_default_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_path = tmp_path / "nequix-oam-1.nqx"
    model_path.write_text("dummy")

    class FakeModel:
        def to(self, device):
            raise AssertionError("model should not move without DEVICE")

        def eval(self):
            raise AssertionError("model should not eval without DEVICE")

    class FakeNequixCalculator(vpmdk.Calculator):
        def __init__(self, **kwargs):
            super().__init__()
            self.model = FakeModel()
            self.device = "cpu"
            self.backend = kwargs["backend"]

    monkeypatch.setattr(vpmdk, "NequixCalculator", FakeNequixCalculator)

    calc = vpmdk._build_nequix_calculator(
        {
            "MODEL": str(model_path),
            "NEQUIX_BACKEND": "torch",
        }
    )

    assert calc.device == "cpu"


def test_nequix_jax_backend_ignores_device_override_for_torch_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_path = tmp_path / "nequix-oam-1.nqx"
    model_path.write_text("dummy")

    class FakeModel:
        def to(self, device):
            raise AssertionError("jax backend should not move torch model")

        def eval(self):
            raise AssertionError("jax backend should not eval torch model")

    class FakeNequixCalculator(vpmdk.Calculator):
        def __init__(self, **kwargs):
            super().__init__()
            self.model = FakeModel()
            self.device = "jax-default"
            self.backend = kwargs["backend"]

    monkeypatch.setattr(vpmdk, "NequixCalculator", FakeNequixCalculator)

    calc = vpmdk._build_nequix_calculator(
        {
            "MODEL": str(model_path),
            "NEQUIX_BACKEND": "jax",
            "DEVICE": "cpu",
        }
    )

    assert calc.backend == "jax"
    assert calc.device == "jax-default"


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


def test_alphanet_infers_config_next_to_checkpoint_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    target_dir = tmp_path / "target"
    input_dir = tmp_path / "input"
    target_dir.mkdir()
    input_dir.mkdir()
    target = target_dir / "alphanet.ckpt"
    target.write_text("checkpoint")
    model_link = input_dir / "alphanet.ckpt"
    model_link.symlink_to(target)
    config_path = input_dir / "alphanet.json"
    config_path.write_text("{}")
    seen: dict[str, object] = {}

    def fake_load(
        config_file: str,
        *,
        precision: str,
        use_pbc: bool,
        compute_stress: bool,
    ):
        seen["config_file"] = config_file
        return "alpha-config"

    def fake_calc(*, ckpt_path, config, device="cpu", precision="32"):
        seen["ckpt_path"] = ckpt_path
        return "alphanet"

    monkeypatch.setattr(vpmdk, "AlphaNetCalculator", fake_calc)
    monkeypatch.setattr(vpmdk, "_load_alphanet_config", fake_load)

    calculator = vpmdk._build_alphanet_calculator(
        {"MODEL": str(model_link), "DEVICE": "cpu"}
    )

    assert calculator == "alphanet"
    assert seen == {
        "config_file": str(config_path),
        "ckpt_path": str(model_link),
    }


def test_alphanet_accepts_named_model_and_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    seen: dict[str, object] = {}
    expected_device = vpmdk._resolve_device(None) or "cpu"
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
        "device": expected_device,
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


def test_override_model_graph_converter_algorithm_rejects_silent_fallback():
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
            self.algorithm = "legacy"
            self.on_isolated_atoms = "warn"

        def set_isolated_atom_response(self, value: str):
            self.on_isolated_atoms = value

    model = SimpleNamespace(
        graph_converter=DummyConverter(atom_graph_cutoff=6, bond_graph_cutoff=3)
    )

    with pytest.raises(RuntimeError, match="requested 'fast' but initialized 'legacy'"):
        vpmdk._override_model_graph_converter_algorithm(
            model,
            algorithm="fast",
            backend_name="CHGNet",
        )


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


def test_upet_cuda_defaults_to_cpu_neighborlist_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_path = tmp_path / "pet-oam-xl-v1.0.0.ckpt"
    model_path.write_text("dummy")

    class FakeUPETCalculator:
        implemented_properties = ["energy", "forces", "stress"]

        def get_potential_energy(self, *_, **__):
            return 0.0

    monkeypatch.setattr(vpmdk, "UPETCalculator", lambda **_: FakeUPETCalculator())

    calc = vpmdk._build_upet_calculator({"MODEL": str(model_path), "DEVICE": "cuda"})

    assert calc.__class__.__name__ == "_UPETNeighborListDeviceProxy"
    assert calc.neighborlist_device == "cpu"
    assert calc.calculator.__class__ is FakeUPETCalculator


def test_upet_neighborlist_device_can_follow_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_path = tmp_path / "pet-oam-xl-v1.0.0.ckpt"
    model_path.write_text("dummy")

    class FakeUPETCalculator:
        implemented_properties = ["energy", "forces", "stress"]

        def get_potential_energy(self, *_, **__):
            return 0.0

    monkeypatch.setattr(vpmdk, "UPETCalculator", lambda **_: FakeUPETCalculator())

    calc = vpmdk._build_upet_calculator(
        {
            "MODEL": str(model_path),
            "DEVICE": "cuda",
            "UPET_NEIGHBORLIST_DEVICE": "model",
        }
    )

    assert calc.__class__ is FakeUPETCalculator


class _FakeMetatomicSystem:
    def __init__(self, device: str | SimpleNamespace):
        if isinstance(device, str):
            device = SimpleNamespace(type=device)
        self.device = device

    def to(self, *, device):
        return _FakeMetatomicSystem(device)


def test_upet_neighborlist_proxy_hooks_metatomic_ase_neighbors(
    monkeypatch: pytest.MonkeyPatch,
):
    neighbors = ModuleType("metatomic_ase._neighbors")
    events: list[tuple[object, ...]] = []

    def original_vesin(systems, calculators):
        events.append(("vesin", [system.device.type for system in systems], calculators))
        return systems

    class AllNeighborsCalculator:
        def compute(self, systems):
            events.append(("compute", [system.device.type for system in systems]))
            return neighbors._compute_requested_neighbors_vesin(systems, ["calc"])

    original_compute = AllNeighborsCalculator.compute
    neighbors.AllNeighborsCalculator = AllNeighborsCalculator
    neighbors._compute_requested_neighbors_vesin = original_vesin
    real_import_module = vpmdk.importlib.import_module

    def fake_import_module(name):
        if name == "metatomic_ase._neighbors":
            return neighbors
        if name == "metatomic.torch.ase_calculator":
            raise ImportError(name)
        return real_import_module(name)

    monkeypatch.setattr(vpmdk.importlib, "import_module", fake_import_module)

    result = backend_misc._run_with_upet_neighborlist_device(
        calculator=None,
        neighborlist_device="cpu",
        call=lambda: AllNeighborsCalculator().compute(
            [_FakeMetatomicSystem("cuda")]
        ),
    )

    assert events == [
        ("compute", ["cpu"]),
        ("vesin", ["cpu"], ["calc"]),
    ]
    assert [system.device.type for system in result] == ["cuda"]
    assert AllNeighborsCalculator.compute is original_compute
    assert neighbors._compute_requested_neighbors_vesin is original_vesin


def test_upet_neighborlist_proxy_keeps_legacy_metatomic_hook(
    monkeypatch: pytest.MonkeyPatch,
):
    legacy = ModuleType("metatomic.torch.ase_calculator")
    events: list[tuple[object, ...]] = []

    def original_vesin(systems, requested_options, check_consistency=False):
        events.append(
            (
                "legacy",
                [system.device.type for system in systems],
                requested_options,
                check_consistency,
            )
        )
        return systems

    legacy._compute_requested_neighbors_vesin = original_vesin
    real_import_module = vpmdk.importlib.import_module

    def fake_import_module(name):
        if name == "metatomic_ase._neighbors":
            raise ImportError(name)
        if name == "metatomic.torch.ase_calculator":
            return legacy
        return real_import_module(name)

    monkeypatch.setattr(vpmdk.importlib, "import_module", fake_import_module)

    result = backend_misc._run_with_upet_neighborlist_device(
        calculator=None,
        neighborlist_device="cpu",
        call=lambda: legacy._compute_requested_neighbors_vesin(
            [_FakeMetatomicSystem("cuda")],
            ["option"],
            check_consistency=True,
        ),
    )

    assert events == [("legacy", ["cpu"], ["option"], True)]
    assert [system.device.type for system in result] == ["cuda"]
    assert legacy._compute_requested_neighbors_vesin is original_vesin


def test_upet_missing_checkpoint_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(vpmdk, "UPETCalculator", lambda **kwargs: None)

    missing_path = tmp_path / "missing.ckpt"
    with pytest.raises(FileNotFoundError, match="not found"):
        vpmdk._build_upet_calculator({"MODEL": str(missing_path)})


def test_equflash_uses_flashtp_sevennet_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_path = tmp_path / "equflash.pth"
    model_path.write_text("dummy")
    seen: dict[str, object] = {}

    def fake_builder(tags, *, force_flash=False):
        seen["tags"] = tags
        seen["force_flash"] = force_flash
        return "equflash"

    monkeypatch.setattr(vpmdk, "SevenNetCalculator", object)
    monkeypatch.setattr(vpmdk, "_is_sevennet_flash_available", lambda: True)
    monkeypatch.setattr(vpmdk, "_build_sevennet_family_calculator", fake_builder)

    calc = vpmdk._build_equflash_calculator(
        {"MODEL": str(model_path), "DEVICE": "cuda:0"}
    )

    assert calc == "equflash"
    assert seen == {
        "tags": {
            "MODEL": str(model_path),
            "DEVICE": "cuda:0",
            "SEVENNET_FILE_TYPE": "checkpoint",
        },
        "force_flash": True,
    }


def test_equflash_named_checkpoint_is_unreleased(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(vpmdk, "SevenNetCalculator", object)
    monkeypatch.setattr(vpmdk, "_is_sevennet_flash_available", lambda: True)

    with pytest.raises(ValueError, match="no released checkpoint"):
        vpmdk._build_equflash_calculator({"MODEL": "equflash-29M-oam"})


def test_fairchem_default_uses_validated_uma_model_and_task(
    monkeypatch: pytest.MonkeyPatch,
):
    seen: dict[str, object] = {}

    class FakeFairChemCalculator:
        @classmethod
        def from_model_checkpoint(
            cls,
            name_or_path,
            *,
            task_name=None,
            inference_settings="default",
            device=None,
        ):
            seen.update(
                {
                    "model": name_or_path,
                    "task": task_name,
                    "settings": inference_settings,
                    "device": device,
                }
            )
            return "fairchem"

    monkeypatch.setattr(vpmdk, "FAIRChemCalculator", FakeFairChemCalculator)

    calc = vpmdk._build_fairchem_calculator({})

    assert calc == "fairchem"
    assert seen == {
        "model": "uma-s-1p1",
        "task": "omat",
        "settings": "default",
        "device": None,
    }


def test_fairchem_non_default_model_does_not_force_default_task(
    monkeypatch: pytest.MonkeyPatch,
):
    seen: dict[str, object] = {}

    class FakeFairChemCalculator:
        @classmethod
        def from_model_checkpoint(
            cls,
            name_or_path,
            *,
            task_name=None,
            inference_settings="default",
            device=None,
        ):
            seen.update({"model": name_or_path, "task": task_name})
            return "fairchem"

    monkeypatch.setattr(vpmdk, "FAIRChemCalculator", FakeFairChemCalculator)

    calc = vpmdk._build_fairchem_calculator({"MODEL": "esen-md-direct-all-omol"})

    assert calc == "fairchem"
    assert seen == {"model": "esen-md-direct-all-omol", "task": None}


@pytest.mark.parametrize("mlp", ["FAIRCHEM", "FAIRCHEM_V2", "ESEN"])
def test_fairchem_v2_delegates_path_shaped_model_selector(
    mlp: str, monkeypatch: pytest.MonkeyPatch
):
    seen: dict[str, object] = {}

    class FakeFairChemCalculator:
        @classmethod
        def from_model_checkpoint(cls, selector, **kwargs):
            seen.update({"selector": selector, "kwargs": kwargs})
            return "fairchem"

    monkeypatch.setattr(vpmdk, "FAIRChemCalculator", FakeFairChemCalculator)

    calculator = vpmdk._build_fairchem_calculator(
        {"MLP": mlp, "MODEL": "provider/runtime-model.ckpt", "DEVICE": "cuda"}
    )

    assert calculator == "fairchem"
    assert seen == {
        "selector": "provider/runtime-model.ckpt",
        "kwargs": {
            "task_name": None,
            "inference_settings": "default",
            "device": "cuda",
        },
    }


@pytest.mark.parametrize(
    "model_selector",
    ["ocp-registry-model", "checkpoints/runtime-resolved.pt"],
)
def test_fairchem_v1_delegates_unresolved_model_selectors(
    model_selector: str,
    monkeypatch: pytest.MonkeyPatch,
):
    seen: dict[str, object] = {}

    class Calculator:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(
        vpmdk,
        "_get_fairchem_v1_calculator_cls",
        lambda: Calculator,
    )
    monkeypatch.setattr(
        vpmdk,
        "_attach_fallback_calculator",
        lambda calculator, tags: calculator,
    )

    calculator = vpmdk._build_fairchem_v1_calculator(
        {
            "MODEL": model_selector,
            "FAIRCHEM_CONFIG": "runtime-config.yml",
            "DEVICE": "cpu",
        }
    )

    assert isinstance(calculator, Calculator)
    assert seen == {
        "checkpoint_path": model_selector,
        "cpu": True,
        "config_yml": "runtime-config.yml",
    }


def test_fairchem_v1_predictor_delegates_unresolved_model_selector(
    monkeypatch: pytest.MonkeyPatch,
):
    seen: dict[str, object] = {}

    class Predictor:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(
        vpmdk,
        "_get_fairchem_v1_predictor_cls",
        lambda: Predictor,
    )

    calculator = vpmdk._build_fairchem_v1_predictor(
        {
            "MODEL": "registry/checkpoint.pt",
            "DEVICE": "cuda",
        }
    )

    assert isinstance(calculator, vpmdk._FairChemV1PredictorCalculator)
    assert seen == {
        "checkpoint_path": "registry/checkpoint.pt",
        "cpu": False,
        "device": "cuda",
    }


def test_fairchem_prediction_without_energy_or_forces_raises_not_zero():
    import numpy as np
    from ase import Atoms

    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.75]])

    with pytest.raises(RuntimeError, match="no recognizable energy"):
        vpmdk._normalize_fairchem_prediction({"E_total": 1.0, "grad": []}, atoms)

    with pytest.raises(RuntimeError, match="no recognizable forces"):
        vpmdk._normalize_fairchem_prediction({"energy": -1.0}, atoms)

    # The error must name the keys the prediction actually carried, so the
    # user can see the rename instead of a bare refusal.
    with pytest.raises(RuntimeError, match="E_total"):
        vpmdk._normalize_fairchem_prediction({"E_total": 1.0}, atoms)

    # A complete prediction still normalizes.
    energy, forces, stress = vpmdk._normalize_fairchem_prediction(
        {"energy": -1.5, "forces": np.ones((2, 3)), "stress": np.arange(6.0)},
        atoms,
    )
    assert energy == -1.5
    assert forces.shape == (2, 3)
    assert stress.shape == (6,)


def test_fairchem_missing_stress_is_omitted_not_zero(monkeypatch: pytest.MonkeyPatch):
    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import PropertyNotImplementedError

    atoms = Atoms(
        "Cu",
        positions=[[0.0, 0.0, 0.0]],
        cell=np.eye(3) * 3.6,
        pbc=True,
    )

    class S2EFPredictor:
        def predict(self, target):
            return {"energy": -3.5, "forces": np.zeros((len(atoms), 3))}

    calculator = vpmdk._FairChemV1PredictorCalculator(S2EFPredictor())
    atoms.calc = calculator

    assert atoms.get_potential_energy() == pytest.approx(-3.5)
    assert "stress" not in calculator.results
    with pytest.raises(PropertyNotImplementedError):
        atoms.get_stress()


def test_equiformer_v3_imports_registration_module_and_uses_fairchem_v1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_path = tmp_path / "equiformer_v3.pt"
    model_path.write_text("dummy")
    seen: dict[str, object] = {}
    registered = {"ready": False}

    def fake_registered(model_name):
        seen.setdefault("registered", []).append(model_name)
        return registered["ready"]

    def fake_import_module(module_name):
        seen["module"] = module_name
        registered["ready"] = True
        return object()

    def fake_fairchem_v1_builder(tags, *, model_reference):
        seen["tags"] = dict(tags)
        seen["model_reference"] = model_reference
        return "equiformer-v3"

    monkeypatch.setattr(vpmdk, "_fairchem_model_registered", fake_registered)
    monkeypatch.setattr(vpmdk.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(
        vpmdk, "_build_fairchem_v1_calculator", fake_fairchem_v1_builder
    )

    calc = vpmdk._build_equiformer_v3_calculator(
        {
            "MODEL": str(model_path),
            "DEVICE": "cpu",
            "EQUIFORMER_V3_MODULE": "custom.equiformer_v3",
        }
    )

    assert calc == "equiformer-v3"
    assert seen["module"] == "custom.equiformer_v3"
    assert seen["tags"] == {
        "MODEL": str(model_path),
        "DEVICE": "cpu",
        "EQUIFORMER_V3_MODULE": "custom.equiformer_v3",
    }
    assert seen["model_reference"] == vpmdk.ModelReference(
        vpmdk.ModelReferenceKind.LOCAL_PATH,
        str(model_path),
        explicit=True,
        identity=str(model_path.resolve()),
    )
    assert seen["registered"] == ["equiformer_v3", "equiformer_v3"]


def test_equiformer_v3_resolves_model_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_path = tmp_path / "equiformer_v3.pt"
    model_path.write_text("dummy")
    resolver_calls: list[str] = []
    original_resolver = vpmdk._resolve_backend_model_reference

    def tracking_resolver(backend, model_value, **kwargs):
        resolver_calls.append(backend)
        return original_resolver(backend, model_value, **kwargs)

    class Calculator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        vpmdk, "_resolve_backend_model_reference", tracking_resolver
    )
    monkeypatch.setattr(vpmdk, "_import_equiformer_v3_model", lambda tags: None)
    monkeypatch.setattr(
        vpmdk, "_get_fairchem_v1_calculator_cls", lambda: Calculator
    )
    monkeypatch.setattr(
        vpmdk,
        "_attach_fallback_calculator",
        lambda calculator, tags: calculator,
    )

    calculator = vpmdk._build_equiformer_v3_calculator(
        {"MODEL": str(model_path), "DEVICE": "cpu"}
    )

    assert isinstance(calculator, Calculator)
    assert calculator.kwargs == {
        "checkpoint_path": str(model_path),
        "cpu": True,
    }
    assert resolver_calls == ["EQUIFORMER_V3"]


def test_equiformer_v3_requires_checkpoint_path():
    with pytest.raises(ValueError, match="requires MODEL"):
        vpmdk._build_equiformer_v3_calculator({})


def test_equiformer_v3_availability_requires_fairchem_v1_calculator(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(vpmdk, "_fairchem_model_registered", lambda _: True)
    monkeypatch.setattr(vpmdk, "_get_fairchem_v1_calculator_cls", lambda: None)

    assert vpmdk._is_equiformer_v3_available() is False


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


def test_matgl_stress_unit_is_pinned_through_one_forwarding_hop(
    monkeypatch: pytest.MonkeyPatch,
):
    class Declaring:
        def __init__(self, potential, *, device=None, stress_unit="GPa"):
            self.kwargs = {"device": device, "stress_unit": stress_unit}

    class Base:
        def __init__(self, potential, *, device=None, stress_unit="GPa"):
            self.kwargs = {"device": device, "stress_unit": stress_unit}

    class Forwarding(Base):
        # matgl's legacy shape: **kwargs landing on a declaring parent.
        def __init__(self, potential, **kwargs):
            super().__init__(potential, **kwargs)

    class Opaque:
        # **kwargs with nowhere to land: pinning would be silently swallowed.
        def __init__(self, potential, **kwargs):
            self.kwargs = dict(kwargs)

    class Legacy:
        def __init__(self, potential, *, device=None):
            self.kwargs = {"device": device}

    monkeypatch.setattr(vpmdk, "MatGLLoadModel", lambda identifier: "potential")
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)

    for calculator_cls, expected in (
        (Declaring, {"device": "cpu", "stress_unit": "eV/A3"}),
        (Forwarding, {"device": "cpu", "stress_unit": "eV/A3"}),
        (Opaque, {"device": "cpu"}),
        (Legacy, {"device": "cpu"}),
    ):
        monkeypatch.setattr(vpmdk, "M3GNetCalculator", calculator_cls)
        calculator = vpmdk._build_m3gnet_calculator({"DEVICE": "cpu"})
        assert calculator.kwargs == expected, calculator_cls.__name__


def test_matgl_fallback_keeps_the_stress_unit_pin(monkeypatch):
    import vpmdk
    from vpmdk_core.backends import m3gnet as m3gnet_module

    class DeclaresUnitRejectsDevice:
        def __init__(self, potential, *, stress_unit="GPa"):
            self.kwargs = {"stress_unit": stress_unit}

    monkeypatch.setattr(vpmdk, "M3GNetCalculator", DeclaresUnitRejectsDevice)
    calculator = m3gnet_module._construct_matgl_calculator(
        object(), {"device": "cpu"}
    )
    assert calculator.kwargs == {"stress_unit": "eV/A3"}

    # A signature that never declared stress_unit keeps the old bare
    # fallback (no pin was added, nothing physics-critical is dropped).
    class NoUnitNoDevice:
        def __init__(self, potential):
            self.kwargs = {}

    monkeypatch.setattr(vpmdk, "M3GNetCalculator", NoUnitNoDevice)
    calculator = m3gnet_module._construct_matgl_calculator(
        object(), {"device": "cpu"}
    )
    assert calculator.kwargs == {}

    # If the verified pin itself is rejected on retry, fail loudly instead
    # of silently computing GPa-scaled numbers.
    class LyingDeclaration:
        def __init__(self, potential, *, stress_unit="GPa", device=None):
            raise TypeError("rejects everything at call time")

    monkeypatch.setattr(vpmdk, "M3GNetCalculator", LyingDeclaration)
    with pytest.raises(TypeError):
        m3gnet_module._construct_matgl_calculator(object(), {"device": "cpu"})


def test_model_reference_rejects_a_fifo_path(tmp_path):
    import os

    fifo = tmp_path / "model.pkl"
    os.mkfifo(fifo)

    with pytest.raises(ValueError, match="FIFO"):
        vpmdk._resolve_backend_model_reference("BAM", str(fifo))
    with pytest.raises(ValueError, match="FIFO"):
        vpmdk._resolve_backend_model_reference("CHGNET", str(fifo))

    # A directory at MODEL still resolves as LOCAL_PATH (some backends load
    # directory-shaped checkpoints).
    model_dir = tmp_path / "model_dir"
    model_dir.mkdir()
    reference = vpmdk._resolve_backend_model_reference("CHGNET", str(model_dir))
    assert reference.kind is vpmdk.ModelReferenceKind.LOCAL_PATH

    # A regular file is untouched.
    regular = tmp_path / "model.pth"
    regular.write_text("dummy")
    reference = vpmdk._resolve_backend_model_reference("CHGNET", str(regular))
    assert reference.kind is vpmdk.ModelReferenceKind.LOCAL_PATH
