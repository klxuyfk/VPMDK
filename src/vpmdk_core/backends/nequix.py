"""Nequix backend builder."""

from __future__ import annotations

import os
import sys
from typing import Dict, List


def _root():
    return sys.modules["vpmdk_core"]


def _normalize_nequix_backend(value: str | None) -> str:
    """Return the normalized Nequix backend name."""

    if value is None:
        return "jax"
    normalized = str(value).strip().lower()
    if normalized in {"jax", "torch"}:
        return normalized
    raise ValueError(f"Invalid NEQUIX_BACKEND value: {value!r}")


def _list_nequix_named_models() -> List[str]:
    """Return the named Nequix models exposed by the upstream calculator."""

    urls = getattr(_root().NequixCalculator, "URLS", None)
    if isinstance(urls, dict):
        return sorted(str(name) for name in urls)
    return []


def _resolve_nequix_model_name(model_name: str) -> str:
    """Resolve a named Nequix model case-insensitively when metadata is available."""

    normalized = model_name.strip().casefold()
    supported = _list_nequix_named_models()
    for candidate in supported:
        if normalized == candidate.casefold():
            return candidate
    if supported:
        supported_text = ", ".join(supported)
        raise ValueError(f"Unsupported Nequix model '{model_name}'. Available: {supported_text}")
    return model_name


def _build_nequix_calculator(bcar_tags: Dict[str, str], *, structure=None):
    """Create the Nequix ASE calculator configured from BCAR tags."""

    root = _root()
    if root.NequixCalculator is None:
        raise RuntimeError("Nequix calculator not available. Install nequix and dependencies.")

    model_reference = root._resolve_backend_model_reference(
        "NEQUIX", bcar_tags.get("MODEL")
    )
    model_value = str(model_reference.value)
    backend = _normalize_nequix_backend(bcar_tags.get("NEQUIX_BACKEND"))

    use_kernel_tag = bcar_tags.get("NEQUIX_USE_KERNEL")
    if use_kernel_tag is None:
        use_kernel_tag = bcar_tags.get("NEQUIX_KERNEL")
    use_kernel = (
        root._coerce_bool_tag(use_kernel_tag, "NEQUIX_USE_KERNEL")
        if use_kernel_tag is not None
        else False
    )

    use_compile_tag = bcar_tags.get("NEQUIX_USE_COMPILE")
    if use_compile_tag is None:
        use_compile_tag = bcar_tags.get("NEQUIX_COMPILE")
    use_compile = (
        root._coerce_bool_tag(use_compile_tag, "NEQUIX_USE_COMPILE")
        if use_compile_tag is not None
        else False
    )

    capacity_multiplier = 1.1
    capacity_tag = bcar_tags.get("NEQUIX_CAPACITY_MULTIPLIER")
    if capacity_tag is not None:
        try:
            capacity_multiplier = float(capacity_tag)
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid NEQUIX_CAPACITY_MULTIPLIER value: {capacity_tag!r}"
            ) from None

    kwargs: Dict[str, object] = {
        "backend": backend,
        "use_kernel": use_kernel,
        "use_compile": use_compile,
        "capacity_multiplier": capacity_multiplier,
    }

    if model_reference.kind is root.ModelReferenceKind.LOCAL_PATH:
        kwargs["model_path"] = model_value
        kwargs["model_name"] = os.path.splitext(os.path.basename(model_value))[0]
    else:
        # Validate/canonicalize the selected name against installed metadata for
        # both explicit and default (omitted-MODEL) selections. The explicit
        # path is already canonical so this is idempotent, while the default is
        # validated here so a registry/version mismatch raises a clear error
        # instead of a cryptic failure deep in the upstream loader.
        kwargs["model_name"] = _resolve_nequix_model_name(model_value)

    requested_device = bcar_tags.get("DEVICE")
    calculator = root.NequixCalculator(**kwargs)

    if backend == "torch" and requested_device:
        try:
            import torch

            torch_device = torch.device(requested_device)
            calculator.model = calculator.model.to(torch_device)
            calculator.device = torch_device
            calculator.model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"Unable to move Nequix torch backend to DEVICE={requested_device!r}."
            ) from exc

    return calculator
