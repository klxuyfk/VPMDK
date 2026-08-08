"""AlphaNet backend builder."""

from __future__ import annotations

import sys
from typing import Any, Dict


def _root():
    return sys.modules["vpmdk_core"]


def _normalize_alphanet_precision(value: str | None) -> str:
    """Return AlphaNet precision in the calculator's expected form."""

    if value is None:
        return "32"
    normalized = str(value).strip().lower()
    if normalized in {"32", "float32", "fp32"}:
        return "32"
    if normalized in {"64", "float64", "fp64"}:
        return "64"
    raise ValueError(f"Invalid ALPHANET_PRECISION value: {value!r}")


def _resolve_alphanet_named_model_spec(model_name: str) -> Dict[str, Any] | None:
    """Return AlphaNet named-model metadata for a model key or alias."""

    root = _root()
    normalized = model_name.strip().casefold()
    direct = root._ALPHANET_NAMED_MODELS.get(normalized)
    if direct is not None:
        return direct

    for spec in root._ALPHANET_NAMED_MODELS.values():
        aliases = [spec["display_name"], *spec.get("aliases", [])]
        if normalized in {alias.casefold() for alias in aliases}:
            return spec
    return None


def _alphanet_named_model_cache_paths(model_name: str) -> tuple[str, str] | None:
    """Return a named model's (checkpoint, config) cache paths WITHOUT downloading.

    Single source of truth for the cache layout, shared by the downloading
    ``_ensure_alphanet_named_model_files`` and by the config inference the server
    uses to report its resident configuration (which must never hit the network).
    Returns None for an unknown model name.
    """

    root = _root()
    spec = _resolve_alphanet_named_model_spec(model_name)
    if spec is None:
        return None
    cache_dir = root.os.path.join(
        root.os.path.expanduser("~/.cache/alphanet"),
        spec["display_name"].replace("/", "_"),
    )
    return (
        root.os.path.join(cache_dir, spec["checkpoint_filename"]),
        root.os.path.join(cache_dir, spec["config_filename"]),
    )


def _infer_alphanet_config_path(
    bcar_tags: Dict[str, str], *, base_dir: str | None = None
) -> str | None:
    """Return the config path the builder would infer, without downloading.

    When ALPHANET_CONFIG is omitted the builder infers it (the single JSON beside
    the checkpoint, or the named model's cached config). The server must record
    that inferred value in the resident's effective configuration; otherwise a
    request that explicitly names the very same file the resident already uses is
    compared against ``server=None`` and rejected with exit 5, even though the
    one-shot builder constructs a byte-identical calculator (the server-mode backend-compatibility contract
    rejects only tags that DIFFER). Returns None when nothing can be inferred
    without side effects -- an unknown/ambiguous layout, or a named model whose
    cache is not populated yet -- leaving the tag simply unadvertised as before.
    """

    root = _root()
    if str(bcar_tags.get("ALPHANET_CONFIG") or "").strip():
        return None  # explicit: already canonicalized through the normal path
    raw_model = bcar_tags.get("MODEL")
    model_arg = str(raw_model) if raw_model is not None and str(raw_model).strip() else None
    try:
        if base_dir is None:
            reference = root._resolve_backend_model_reference("ALPHANET", model_arg)
        else:
            reference = root._resolve_backend_model_reference(
                "ALPHANET", model_arg, base_dir=base_dir
            )
        if reference.kind is root.ModelReferenceKind.LOCAL_PATH:
            # Pure filesystem inference -- reuse the builder's own resolver so the
            # advertised path cannot drift from the one actually loaded.
            return _resolve_alphanet_config_path(str(reference.value), bcar_tags)
        cache_paths = _alphanet_named_model_cache_paths(str(reference.value))
        if cache_paths is None:
            return None
        checkpoint_path, config_path = cache_paths
        if not root.os.path.exists(config_path):
            # Not fetched yet: reporting identity must never trigger a download.
            return None
        return _resolve_alphanet_config_path(
            checkpoint_path, bcar_tags, default_config_path=config_path
        )
    except Exception:
        # Inference is best effort: never let reporting the resident's identity
        # fail a server that already built its calculator successfully.
        return None


def _ensure_alphanet_named_model_files(model_name: str) -> tuple[str, str]:
    """Download a known AlphaNet named model and config when needed."""

    root = _root()
    cache_paths = _alphanet_named_model_cache_paths(model_name)
    if cache_paths is None:
        supported = ", ".join(
            sorted(named_spec["display_name"] for named_spec in root._ALPHANET_NAMED_MODELS.values())
        )
        raise ValueError(f"Unsupported AlphaNet model '{model_name}'. Available: {supported}")

    spec = _resolve_alphanet_named_model_spec(model_name)
    checkpoint_path, config_path = cache_paths
    root.os.makedirs(root.os.path.dirname(config_path), exist_ok=True)

    if not root.os.path.exists(config_path) or root.os.path.getsize(config_path) == 0:
        print(f"AlphaNet config not found, downloading to {config_path} ...")
        root._download_file_to_path(spec["config_url"], config_path)

    if not root.os.path.exists(checkpoint_path) or root.os.path.getsize(checkpoint_path) == 0:
        print(f"AlphaNet checkpoint not found, downloading to {checkpoint_path} ...")
        root._download_file_to_path(spec["checkpoint_url"], checkpoint_path)

    return checkpoint_path, config_path


def _resolve_alphanet_config_path(
    model_path: str,
    bcar_tags: Dict[str, str],
    *,
    default_config_path: str | None = None,
) -> str:
    """Resolve AlphaNet config JSON from BCAR or neighboring files."""

    root = _root()
    config_path = bcar_tags.get("ALPHANET_CONFIG") or default_config_path
    if config_path:
        if not root.os.path.exists(config_path):
            raise FileNotFoundError(f"AlphaNet config not found: {config_path}")
        return config_path

    parent_dir = root.os.path.dirname(model_path) or "."
    json_candidates = sorted(
        root.os.path.join(parent_dir, name)
        for name in root.os.listdir(parent_dir)
        if name.lower().endswith(".json")
    )
    if len(json_candidates) == 1:
        return json_candidates[0]

    raise ValueError(
        "AlphaNet requires ALPHANET_CONFIG pointing to a JSON config when it cannot "
        "be inferred from the checkpoint directory."
    )


def _load_alphanet_config(
    config_path: str,
    *,
    precision: str,
    use_pbc: bool,
    compute_stress: bool,
):
    """Load and normalize AlphaNet config for ASE inference."""

    root = _root()
    if root.AlphaNetAllConfig is None:
        raise RuntimeError("AlphaNet config loader not available. Install AlphaNet and dependencies.")

    config = root.AlphaNetAllConfig.from_json(config_path)
    model_config = getattr(config, "model", config)
    model_config.compute_forces = True
    model_config.compute_stress = compute_stress
    model_config.use_pbc = use_pbc
    model_config.dtype = precision
    return config


def _build_alphanet_calculator(bcar_tags: Dict[str, str], *, structure=None):
    """Create the AlphaNet ASE calculator configured from BCAR tags."""

    root = _root()
    if root.AlphaNetCalculator is None:
        raise RuntimeError("AlphaNet calculator not available. Install AlphaNet and dependencies.")

    model_reference = root._resolve_backend_model_reference(
        "ALPHANET", bcar_tags.get("MODEL")
    )
    model_value = str(model_reference.value)
    precision = _normalize_alphanet_precision(
        bcar_tags.get("ALPHANET_PRECISION") or bcar_tags.get("ALPHANET_DTYPE")
    )
    device = root._resolve_device(bcar_tags.get("DEVICE")) or "cpu"

    config_path = None
    checkpoint_path = model_value

    if model_reference.kind is root.ModelReferenceKind.LOCAL_PATH:
        config_path = _resolve_alphanet_config_path(model_value, bcar_tags)
    else:
        checkpoint_path, config_path = root._ensure_alphanet_named_model_files(model_value)
        config_path = _resolve_alphanet_config_path(
            checkpoint_path,
            bcar_tags,
            default_config_path=config_path,
        )

    use_pbc = True if structure is None else getattr(structure, "lattice", None) is not None
    config = root._load_alphanet_config(
        config_path,
        precision=precision,
        use_pbc=use_pbc,
        compute_stress=use_pbc,
    )

    return root.AlphaNetCalculator(
        ckpt_path=checkpoint_path,
        config=config,
        device=device,
        precision=precision,
    )
