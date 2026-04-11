"""AlphaNet backend builder."""

from __future__ import annotations

import os
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


def _ensure_alphanet_named_model_files(model_name: str) -> tuple[str, str]:
    """Download a known AlphaNet named model and config when needed."""

    root = _root()
    spec = _resolve_alphanet_named_model_spec(model_name)
    if spec is None:
        supported = ", ".join(
            sorted(named_spec["display_name"] for named_spec in root._ALPHANET_NAMED_MODELS.values())
        )
        raise ValueError(f"Unsupported AlphaNet model '{model_name}'. Available: {supported}")

    cache_dir = root.os.path.join(
        root.os.path.expanduser("~/.cache/alphanet"),
        spec["display_name"].replace("/", "_"),
    )
    root.os.makedirs(cache_dir, exist_ok=True)

    checkpoint_path = root.os.path.join(cache_dir, spec["checkpoint_filename"])
    config_path = root.os.path.join(cache_dir, spec["config_filename"])

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

    model_value = bcar_tags.get("MODEL") or root.DEFAULT_ALPHANET_MODEL
    precision = _normalize_alphanet_precision(
        bcar_tags.get("ALPHANET_PRECISION") or bcar_tags.get("ALPHANET_DTYPE")
    )
    device = root._resolve_device(bcar_tags.get("DEVICE")) or "cpu"

    config_path = None
    checkpoint_path = model_value

    if os.path.exists(model_value):
        config_path = _resolve_alphanet_config_path(model_value, bcar_tags)
    elif root._looks_like_filesystem_path(model_value, suffixes=(".ckpt", ".pt", ".pth")):
        raise FileNotFoundError(f"AlphaNet model not found: {model_value}")
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
