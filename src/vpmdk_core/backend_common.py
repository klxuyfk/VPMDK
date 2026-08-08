"""Helpers shared across backend integrations."""

from __future__ import annotations

import inspect
import os
import re
import shutil
import stat
import sys
import urllib.request
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Iterable, Mapping, TypeVar


# ``scheme://`` prefix marking a remote reference (http(s)://, s3://, gs://,
# hf://, ...). Such a value is unambiguously not a local-filesystem typo, so it
# is never treated as a missing-local-path error.
_REMOTE_URI_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+.\-]*://")


def _is_remote_model_uri(value: str) -> bool:
    """Return whether a MODEL value is a scheme-qualified remote reference."""

    return _REMOTE_URI_RE.match(str(value).strip()) is not None


def _has_path_separator_shape(text: str) -> bool:
    """Return whether a value is shaped like a filesystem path via separators.

    Shared by MODEL classification (``_resolve_model_reference``) and resident
    config-path detection (``server._normalize_path_or_name``) so both agree on
    what "looks like a path". An explicit ``./``/``../`` prefix is covered by the
    separator check.
    """

    return (
        os.path.isabs(text)
        or os.path.sep in text
        or (os.path.altsep is not None and os.path.altsep in text)
    )


# A Hugging Face repo id is exactly ``owner/model``: two identifier-like segments
# separated by a single forward slash, neither absolute nor a ``.``/``..``
# relative path. This is deliberately narrow so an absolute, multi-segment, or
# dot-relative filesystem path is never mistaken for a registry id.
_REGISTRY_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*")


def _is_registry_identifier_shape(text: str) -> bool:
    """Whether *text* is shaped like a Hugging Face ``owner/model`` registry id."""

    return _REGISTRY_IDENTIFIER_RE.fullmatch(text) is not None


_LoadedModelT = TypeVar("_LoadedModelT")


class ModelReferenceKind(str, Enum):
    """How a backend must interpret its effective model selection."""

    DEFAULT = "default"
    LOCAL_PATH = "local_path"
    NAMED_MODEL = "named_model"


@dataclass(frozen=True)
class ModelReference:
    """One model selection with separate loader and comparison identities."""

    kind: ModelReferenceKind
    value: str | None
    explicit: bool
    identity: str | None = None


@dataclass(frozen=True)
class BackendModelPolicy:
    """MODEL inputs accepted by one calculator builder.

    This table-facing description deliberately separates named identifiers
    resolved by VPMDK from opaque selectors delegated to an upstream loader.
    Keeping every built-in backend in one matrix prevents a new backend from
    silently inheriting the conservative local-file-only fallback.
    """

    default_attribute: str | None = None
    default_value: str | None = None
    required: bool = False
    local_only: bool = False
    allow_local: bool = True
    allow_named: bool = False
    delegate_unresolved: bool = False
    # Delegate slash-qualified registry identifiers (e.g. a Hugging Face
    # ``owner/model``) to the upstream loader while still rejecting a bare
    # suffix-shaped local-path typo. Narrower than ``delegate_unresolved``, which
    # forwards every path-shaped value.
    delegate_registry_ids: bool = False
    named_resolver: str | None = None
    resolver_supplies_default: bool = False
    known_names_attribute: str | None = None
    allow_remote_uri: bool = False


_BACKEND_MODEL_POLICIES: Mapping[str, BackendModelPolicy] = {
    # Local checkpoint only. An explicit missing value is always an error;
    # omission is allowed only where the upstream calculator has a default.
    "MACE": BackendModelPolicy(local_only=True),
    # ORB weights_path accepts a local checkpoint or a remote URI (the bundled
    # defaults are https/s3 URLs that orb-models downloads via cached_path). A
    # missing local path (no scheme) still errors early; a scheme-qualified URI
    # is delegated to the loader.
    "ORB": BackendModelPolicy(local_only=True, allow_remote_uri=True),
    "NEQUIP": BackendModelPolicy(required=True, local_only=True),
    "ALLEGRO": BackendModelPolicy(required=True, local_only=True),
    # BAM-torch ships no named-model downloader; checkpoints (e.g. the
    # published BAM-MP-core.pkl) are plain local files.
    "BAM": BackendModelPolicy(required=True, local_only=True),
    "EQUFLASH": BackendModelPolicy(required=True, local_only=True),
    "EQUIFORMER_V3": BackendModelPolicy(required=True, local_only=True),
    "DEEPMD": BackendModelPolicy(required=True, local_only=True),
    # MatterSim accepts existing paths and non-path preset names. A missing
    # path-shaped value remains an immediate error.
    "MATTERSIM": BackendModelPolicy(
        allow_named=True,
    ),
    # Upstream checkpoint selectors. Existing paths retain local-path
    # identity; unresolved values are passed to the upstream loader, which may
    # download or otherwise resolve even path-shaped provider identifiers.
    "FAIRCHEM": BackendModelPolicy(
        default_attribute="DEFAULT_FAIRCHEM_MODEL",
        allow_named=True,
        delegate_unresolved=True,
    ),
    "FAIRCHEM_V2": BackendModelPolicy(
        default_attribute="DEFAULT_FAIRCHEM_MODEL",
        allow_named=True,
        delegate_unresolved=True,
    ),
    "ESEN": BackendModelPolicy(
        default_attribute="DEFAULT_FAIRCHEM_MODEL",
        allow_named=True,
        delegate_unresolved=True,
    ),
    "FAIRCHEM_V1": BackendModelPolicy(
        required=True,
        allow_named=True,
        delegate_unresolved=True,
    ),
    # Named-model capable backends. Path-looking missing values remain errors
    # unless the upstream-selector group above explicitly delegates them.
    "CHGNET": BackendModelPolicy(allow_named=True),
    # matgl.load_model resolves its own registry and Hugging Face identifiers
    # (e.g. owner/model), so slash-qualified registry names are delegated to the
    # loader. A bare suffix-shaped local-path typo (weights.pt) is still rejected
    # early, preserving strict MODEL handling for real filesystem paths.
    "MATGL": BackendModelPolicy(
        default_attribute="DEFAULT_MATGL_MODEL",
        allow_named=True,
        delegate_registry_ids=True,
    ),
    "M3GNET": BackendModelPolicy(
        default_attribute="DEFAULT_MATGL_MODEL",
        allow_named=True,
        delegate_registry_ids=True,
    ),
    "EQNORM": BackendModelPolicy(
        default_attribute="DEFAULT_EQNORM_MODEL",
        named_resolver="eqnorm",
    ),
    "MATRIS": BackendModelPolicy(
        default_attribute="DEFAULT_MATRIS_MODEL",
        allow_named=True,
        known_names_attribute="_MATRIS_NAMED_MODEL_DOWNLOADS",
    ),
    "ALPHANET": BackendModelPolicy(
        default_attribute="DEFAULT_ALPHANET_MODEL",
        named_resolver="alphanet",
    ),
    "HIENET": BackendModelPolicy(
        default_attribute="DEFAULT_HIENET_MODEL",
        named_resolver="hienet",
    ),
    "NEQUIX": BackendModelPolicy(
        default_attribute="DEFAULT_NEQUIX_MODEL",
        named_resolver="nequix",
    ),
    "SEVENNET": BackendModelPolicy(
        default_attribute="DEFAULT_SEVENNET_MODEL",
        allow_named=True,
    ),
    "FLASHTP": BackendModelPolicy(
        default_attribute="DEFAULT_SEVENNET_MODEL",
        allow_named=True,
    ),
    "UPET": BackendModelPolicy(required=True, allow_named=True),
    "TACE": BackendModelPolicy(required=True, allow_named=True),
    "GRACE": BackendModelPolicy(
        named_resolver="grace",
        resolver_supplies_default=True,
    ),
    # Matlantis MODEL values are version strings, never filesystem paths.
    "MATLANTIS": BackendModelPolicy(
        default_value="v8.0.0",
        allow_local=False,
        allow_named=True,
    ),
}


_DEFAULT_SIMPLE_MODEL_POLICY = BackendModelPolicy(allow_named=True)


# Backends whose named-model resolver is a spec lookup returning a
# ``{"display_name": ...}`` mapping. They share one closure shape, keyed here by
# the ``named_resolver`` policy value to the root-module resolver attribute.
# Each entry stores the spec-resolver attribute, named-model registry attribute,
# and label used for an "Available: <names>" diagnostic.
_SPEC_NAMED_RESOLVERS = {
    "eqnorm": (
        "_resolve_eqnorm_named_model_spec",
        "_EQNORM_NAMED_MODELS",
        "Eqnorm",
    ),
    "alphanet": (
        "_resolve_alphanet_named_model_spec",
        "_ALPHANET_NAMED_MODELS",
        "AlphaNet",
    ),
    "hienet": (
        "_resolve_hienet_named_model_spec",
        "_HIENET_NAMED_MODELS",
        "HIENet",
    ),
}


# Checkpoint file extensions used to classify a non-existent MODEL as a
# mistyped local path (rather than a named model). Only unambiguous model-weight
# formats belong here: config formats (.yaml/.yml/.json) are deliberately absent
# so a named model for an ``allow_named`` backend is never mis-rejected as a
# missing path just because its identifier ends in a config-file extension.
_MODEL_PATH_SUFFIXES = (
    ".pt",
    ".pth",
    ".ckpt",
    ".model",
    ".tar",
    ".pb",
    ".nqx",
    ".jit",
    ".ts",
)

# Config-path detection (resident-server ``_normalize_path_or_name`` for
# ALPHANET_CONFIG / FAIRCHEM_CONFIG) additionally treats config-file extensions
# as path-shaped. It extends the checkpoint list so the shared part stays
# single-source and the two heuristics cannot drift apart on model formats.
_CONFIG_PATH_SUFFIXES = _MODEL_PATH_SUFFIXES + (".yaml", ".yml", ".json")


def _resolve_grace_policy_model(root, model_value: str | None) -> str:
    """Return the one GRACE identity used for defaults and explicit names."""

    installed_default = root._resolve_grace_foundation_model(None)
    fallback = installed_default or root.DEFAULT_GRACE_MODEL
    if not model_value:
        return str(fallback)

    selected = root._resolve_grace_foundation_model(model_value)
    if selected is not None:
        return str(selected)
    if installed_default is not None:
        # No warning here: this resolver is a pure identity function that the
        # resident server calls for every status/run request, so printing would
        # append a line to the daemon log per request. The GRACE builder reports
        # the substitution once, when a calculator is actually constructed.
        return str(installed_default)

    # Without registry metadata VPMDK cannot validate a foundation-model name.
    # Preserve it for the upstream loader rather than silently changing it.
    return str(model_value)


_FOUNDATION_UNSET = object()


def _grace_substitutes_unknown_model(
    root, reference, requested_model: str, *, foundation_model=_FOUNDATION_UNSET
) -> bool:
    """True when GRACE silently mapped an unknown foundation-model name to the default.

    Shared by the one-shot builder (which prints "using default …") and the
    resident server's request-warning helper (which prints "reusing resident
    default …") so the two cannot drift on *when* a substitution occurred and
    thereby skip or duplicate the warning. A local checkpoint path is a real
    model, not a foundation-name typo, so it never counts as a substitution.

    ``foundation_model`` lets a caller that has already resolved
    ``_resolve_grace_foundation_model(requested_model)`` pass it in so the
    registry is scanned once rather than twice.
    """

    requested = str(requested_model or "").strip()
    if not requested:
        return False
    if reference.kind is root.ModelReferenceKind.LOCAL_PATH:
        return False
    if foundation_model is _FOUNDATION_UNSET:
        foundation_model = root._resolve_grace_foundation_model(requested)
    return foundation_model is None and str(reference.value) != requested


def _resolve_model_reference(
    model_value: str | None,
    *,
    backend_name: str,
    default_model: str | None = None,
    required: bool = False,
    local_only: bool = False,
    allow_local: bool = True,
    allow_named: bool = False,
    allow_remote_uri: bool = False,
    delegate_unresolved: bool = False,
    delegate_registry_ids: bool = False,
    known_named_models: Mapping[str, str] | Iterable[str] = (),
    named_model_resolver: Callable[[str], str | None] | None = None,
    path_suffixes: Iterable[str] = (),
    base_dir: str | None = None,
) -> ModelReference:
    """Classify MODEL as default, an existing local path, or a named model.

    Explicit values never become defaults. Registry-capable backends may use a
    canonical-name resolver, a known-name collection, or delegate membership
    validation to their upstream loader with ``allow_named``.
    """

    raw_value = "" if model_value is None else str(model_value).strip()
    if not raw_value:
        if required and default_model is None:
            raise ValueError(f"{backend_name} requires MODEL.")
        return ModelReference(
            ModelReferenceKind.DEFAULT,
            default_model,
            explicit=False,
        )

    if allow_remote_uri and _is_remote_model_uri(raw_value):
        # A scheme-qualified remote reference (e.g. https://.../weights.ckpt) is
        # delegated to the loader, which downloads/caches it. It is never a local
        # filesystem typo, so it must not raise a missing-local-path error.
        return ModelReference(
            ModelReferenceKind.NAMED_MODEL,
            raw_value,
            explicit=True,
        )

    expanded = os.path.expanduser(raw_value)
    candidate = (
        expanded
        if os.path.isabs(expanded) or base_dir is None
        else os.path.join(base_dir, expanded)
    )
    if allow_local:
        candidate_exists = os.path.exists(candidate)
        if candidate_exists:
            # Reject FIFOs before a loader can block on open(). Directories stay
            # valid because some backends use directory-shaped checkpoints.
            try:
                candidate_stat = os.stat(candidate)
            except OSError:
                candidate_stat = None
            if candidate_stat is not None and stat.S_ISFIFO(candidate_stat.st_mode):
                raise ValueError(
                    f"{backend_name} MODEL path is a FIFO (named pipe): "
                    f"{candidate}. Opening it would block forever; point "
                    "MODEL at a regular checkpoint file or directory."
                )
            # Preserve the lexical path for backend loaders. Some builders
            # infer sibling files from the directory containing a symlink.
            # ``identity`` remains canonical for resident-server comparison.
            loader_path = (
                expanded if base_dir is None else os.path.abspath(candidate)
            )
            return ModelReference(
                ModelReferenceKind.LOCAL_PATH,
                loader_path,
                explicit=True,
                identity=os.path.realpath(candidate),
            )

        has_separator = _has_path_separator_shape(expanded)
        has_model_suffix = any(
            expanded.lower().endswith(suffix.lower()) for suffix in path_suffixes
        )
        looks_like_path = local_only or has_separator or has_model_suffix
        if looks_like_path and not delegate_unresolved:
            # A registry-delegating backend (matgl.load_model) accepts a Hugging
            # Face ``owner/model`` identifier. Only that exact shape (single
            # slash, identifier segments, not absolute/dot-relative, no
            # checkpoint suffix) is delegated to the loader; an absolute,
            # multi-segment, dot-relative, or suffix-shaped value stays a strict
            # missing-local-path error so a real path typo is still reported
            # early. (A bare single-slash relative pair like ``configs/model`` is
            # shape-identical to a repo id and so is delegated -- an unresolvable
            # one still errors, from the loader rather than as FileNotFoundError.)
            registry_id_shaped = (
                delegate_registry_ids
                and not has_model_suffix
                and not local_only
                and _is_registry_identifier_shape(expanded)
            )
            if not registry_id_shaped:
                raise FileNotFoundError(
                    f"{backend_name} MODEL path not found: "
                    f"{os.path.realpath(candidate)}"
                )

    canonical_name: str | None = None
    if named_model_resolver is not None:
        canonical_name = named_model_resolver(raw_value)
    else:
        if isinstance(known_named_models, Mapping):
            aliases = {
                str(alias).casefold(): str(canonical)
                for alias, canonical in known_named_models.items()
            }
        else:
            aliases = {str(name).casefold(): str(name) for name in known_named_models}
        canonical_name = aliases.get(raw_value.casefold())

    if canonical_name is not None and str(canonical_name).strip():
        return ModelReference(
            ModelReferenceKind.NAMED_MODEL,
            canonical_name,
            explicit=True,
        )
    if allow_named:
        return ModelReference(
            ModelReferenceKind.NAMED_MODEL,
            raw_value,
            explicit=True,
        )

    raise ValueError(
        f"Unsupported {backend_name} MODEL {raw_value!r}: expected an existing "
        "local path or a supported named model."
    )


def _resolve_backend_model_reference(
    backend: str,
    model_value: str | None,
    *,
    base_dir: str | None = None,
) -> ModelReference:
    """Apply the single MODEL policy table used by every backend builder."""

    root = sys.modules["vpmdk_core"]
    name = str(backend).strip().upper()

    policy = _BACKEND_MODEL_POLICIES.get(name)
    if policy is None:
        # Dynamic simple calculators accept an optional path or preset name.
        # Their builder always forwards explicit values, so this cannot become
        # a silent default-model substitution.
        policy = _DEFAULT_SIMPLE_MODEL_POLICY

    backend_name = "MatGL" if name in {"MATGL", "M3GNET"} else name
    if name in {"MATGL", "M3GNET"}:
        if getattr(root, "_USING_LEGACY_M3GNET", False):
            backend_name = "Legacy M3GNet"
            # Legacy M3GNet's loader resolves foundation-model preset names as
            # well as local checkpoints, so it must not be forced local-only.
            policy = BackendModelPolicy(allow_named=True)

    default_model = policy.default_value
    if policy.default_attribute is not None:
        default_model = getattr(root, policy.default_attribute)

    named_model_resolver: Callable[[str], str | None] | None = None
    spec_resolver_config = _SPEC_NAMED_RESOLVERS.get(policy.named_resolver)
    if spec_resolver_config is not None:
        spec_attribute, registry_attribute, label = spec_resolver_config
        spec_resolver = getattr(root, spec_attribute)

        def named_model_resolver(value: str) -> str | None:
            spec = spec_resolver(value)
            if spec is not None:
                return str(spec["display_name"])
            # The value failed local-path checks and is therefore a named-model
            # request; report the available names.
            available = ", ".join(
                sorted(
                    str(named["display_name"])
                    for named in getattr(root, registry_attribute).values()
                )
            )
            raise ValueError(
                f"Unsupported {label} model '{value}'. Available: {available}"
            )

    elif policy.named_resolver == "nequix":
        named_model_resolver = root._resolve_nequix_model_name
    elif policy.named_resolver == "grace":

        def named_model_resolver(value: str) -> str | None:
            return _resolve_grace_policy_model(root, value)

    model_omitted = not str(model_value or "").strip()
    if policy.resolver_supplies_default and model_omitted:
        if named_model_resolver is None:  # pragma: no cover - invalid policy
            raise RuntimeError(
                f"{name} MODEL policy requires a named-model resolver for its default."
            )
        default_model = named_model_resolver("")
    elif model_omitted and named_model_resolver is not None and default_model is not None:
        # Canonicalize the default identity the same way an explicit MODEL is
        # canonicalized, so the resident-default identity matches a request that
        # explicitly names the default (e.g. NEQUIX's URLS-cased key vs the raw
        # DEFAULT_* constant). Preserve the raw default if it is unresolvable so
        # a broken/mismatched registry never turns identity resolution into an
        # error.
        try:
            canonical_default = named_model_resolver(str(default_model))
        except (ValueError, FileNotFoundError):
            canonical_default = None
        if canonical_default is not None and str(canonical_default).strip():
            default_model = str(canonical_default)

    known_named_models: Mapping[str, str] | Iterable[str] = ()
    if policy.known_names_attribute is not None:
        known_named_models = getattr(root, policy.known_names_attribute).keys()

    return _resolve_model_reference(
        model_value,
        backend_name=backend_name,
        default_model=default_model,
        required=policy.required,
        local_only=policy.local_only,
        allow_local=policy.allow_local,
        allow_named=policy.allow_named,
        allow_remote_uri=policy.allow_remote_uri,
        delegate_unresolved=policy.delegate_unresolved,
        delegate_registry_ids=policy.delegate_registry_ids,
        known_named_models=known_named_models,
        named_model_resolver=named_model_resolver,
        path_suffixes=_MODEL_PATH_SUFFIXES,
        base_dir=base_dir,
    )


def _require_loaded_model(
    value: _LoadedModelT,
    *,
    backend_name: str,
    model: str | None,
) -> _LoadedModelT:
    """Reject a missing loader result before a calculator can silently continue.

    Only the ``None`` / ``False`` / empty-string sentinels that loaders use to
    signal "nothing was loaded" count as missing. A validly-loaded model may
    itself be falsy under ``bool()`` (e.g. a container-style potential whose
    ``__len__`` is 0 at construction time), so general truthiness is never used
    — such objects must not be rejected as load failures.
    """

    missing = (
        value is None
        or value is False
        or (isinstance(value, str) and not value.strip())
    )
    if missing:
        selected = "default model" if model is None else f"MODEL {model!r}"
        raise RuntimeError(f"Unable to load {backend_name} {selected}: loader returned no model")
    return value


def _coerce_int_tag(value: str, tag_name: str) -> int:
    """Parse integer BCAR tag values with a descriptive error message."""

    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        # OverflowError: int(float("inf"))/int(float("1e400")) raise it, unlike
        # int(float("nan")) which raises ValueError. Normalize both to ValueError
        # so every malformed integer value is classified identically -- e.g. the
        # server's exit-5 backend-mismatch guard (which catches ValueError) treats
        # TACE_FIDELITY_IDX=inf the same as =nan, instead of leaking OverflowError
        # out as a calculation_error (exit 2).
        raise ValueError(f"Invalid {tag_name} value: {value!r}") from None


def _coerce_bool_tag(value: str, tag_name: str) -> bool:
    """Parse boolean-like BCAR tags with descriptive errors."""

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid {tag_name} value: {value!r}")


def _resolve_device(device: str | None) -> str | None:
    """Return user-specified device or best-effort autodetection."""

    if device is not None:
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _parse_optional_bool_tag(
    bcar_tags: Dict[str, str], tag_name: str
) -> bool | None:
    """Return an optional boolean BCAR tag, preserving the unset state."""

    raw_value = bcar_tags.get(tag_name)
    if raw_value is None:
        return None
    return _coerce_bool_tag(raw_value, tag_name)


def _callable_declares_parameter(callable_obj: object, parameter_name: str) -> bool:
    """Return whether a callable explicitly declares a named parameter."""

    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return parameter_name in signature.parameters


def _callable_supports_parameter(callable_obj: object, parameter_name: str) -> bool:
    """Return whether a callable exposes a named parameter."""

    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    if parameter_name in signature.parameters:
        return True
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _download_file_to_path(url: str, destination_path: str) -> None:
    """Download a file to a local path atomically."""

    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    partial_path = f"{destination_path}.part"
    request = urllib.request.Request(url, headers={"User-Agent": "vpmdk"})
    try:
        with urllib.request.urlopen(request) as response, open(partial_path, "wb") as handle:
            shutil.copyfileobj(response, handle)
        os.replace(partial_path, destination_path)
    except Exception:
        if os.path.exists(partial_path):
            os.remove(partial_path)
        raise
