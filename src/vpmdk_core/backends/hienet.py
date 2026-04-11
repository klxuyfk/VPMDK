"""HIENet backend builder."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict


def _root():
    return sys.modules["vpmdk_core"]


def _normalize_hienet_file_type(value: str | None) -> str:
    """Return the normalized HIENet file type."""

    if value is None:
        return "checkpoint"
    normalized = str(value).strip().lower()
    if normalized in {"checkpoint", "torchscript"}:
        return normalized
    raise ValueError(f"Invalid HIENET_FILE_TYPE value: {value!r}")


def _resolve_hienet_named_model_spec(model_name: str) -> Dict[str, Any] | None:
    """Return HIENet named-model metadata for a model key or alias."""

    root = _root()
    normalized = model_name.strip().casefold()
    direct = root._HIENET_NAMED_MODELS.get(normalized)
    if direct is not None:
        return direct

    for spec in root._HIENET_NAMED_MODELS.values():
        aliases = spec.get("aliases", [])
        if any(normalized == alias.casefold() for alias in aliases):
            return spec
    return None


def _ensure_hienet_named_model_checkpoint(model_name: str) -> tuple[Dict[str, Any], str]:
    """Download a known HIENet named model into the standard cache when needed."""

    root = _root()
    spec = _resolve_hienet_named_model_spec(model_name)
    if spec is None:
        supported = ", ".join(
            sorted(named_spec["display_name"] for named_spec in root._HIENET_NAMED_MODELS.values())
        )
        raise ValueError(f"Unsupported HIENet model '{model_name}'. Available: {supported}")

    cache_dir = root.os.path.expanduser("~/.cache/hienet")
    root.os.makedirs(cache_dir, exist_ok=True)
    checkpoint_path = root.os.path.join(cache_dir, spec["checkpoint_filename"])
    if not root.os.path.exists(checkpoint_path) or root.os.path.getsize(checkpoint_path) == 0:
        print(f"HIENet checkpoint not found, downloading to {checkpoint_path} ...")
        root._download_file_to_path(spec["checkpoint_url"], checkpoint_path)
    return spec, checkpoint_path


def _build_hienet_calculator(bcar_tags: Dict[str, str]):
    """Create the HIENet ASE calculator configured from BCAR tags."""

    root = _root()
    if root.HIENetCalculator is None:
        raise RuntimeError("HIENet calculator not available. Install hienet and dependencies.")

    model_value = bcar_tags.get("MODEL") or root.DEFAULT_HIENET_MODEL
    device = root._resolve_device(bcar_tags.get("DEVICE")) or "cpu"
    file_type = _normalize_hienet_file_type(bcar_tags.get("HIENET_FILE_TYPE"))

    model_path = model_value
    if os.path.exists(model_value):
        pass
    elif root._looks_like_filesystem_path(
        model_value,
        suffixes=(".pth", ".pt", ".ckpt", ".jit", ".ts"),
    ):
        raise FileNotFoundError(f"HIENet model not found: {model_value}")
    else:
        if file_type != "checkpoint":
            raise ValueError(
                "HIENET_FILE_TYPE=torchscript requires MODEL pointing to a local TorchScript file."
            )
        _, model_path = root._ensure_hienet_named_model_checkpoint(model_value)

    return root.HIENetCalculator(model=model_path, file_type=file_type, device=device)
