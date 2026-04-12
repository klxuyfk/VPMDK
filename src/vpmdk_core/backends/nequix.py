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

    model_value = bcar_tags.get("MODEL") or root.DEFAULT_NEQUIX_MODEL
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

    if os.path.exists(model_value):
        kwargs["model_path"] = model_value
        kwargs["model_name"] = os.path.splitext(os.path.basename(model_value))[0]
    elif root._looks_like_filesystem_path(model_value, suffixes=(".nqx", ".pt", ".pth", ".ckpt")):
        raise FileNotFoundError(f"Nequix model not found: {model_value}")
    else:
        kwargs["model_name"] = _resolve_nequix_model_name(model_value)

    requested_device = bcar_tags.get("DEVICE")
    if backend == "torch":
        try:
            import torch

            nequix_module = root.importlib.import_module("nequix.calculator")
            nequix_data_module = root.importlib.import_module("nequix.data")

            torch_device = torch.device(
                root._resolve_device(requested_device)
                or ("cuda" if torch.cuda.is_available() else "cpu")
            )
            model, config = nequix_module.from_pretrained(
                model_name=kwargs.get("model_name"),
                model_path=kwargs.get("model_path"),
                backend="torch",
                use_kernel=use_kernel,
            )

            calculator = root.NequixCalculator.__new__(root.NequixCalculator)
            root.Calculator.__init__(calculator)
            calculator.model = model.to(torch_device)
            calculator.config = config
            calculator.device = torch_device
            calculator.model.eval()
            calculator.compile_state = (
                False if use_compile and torch_device.type == "cuda" else True
            )
            calculator.atom_indices = nequix_data_module.atomic_numbers_to_indices(
                config["atomic_numbers"]
            )
            calculator.cutoff = config["cutoff"]
            calculator._capacity = None
            calculator._capacity_multiplier = capacity_multiplier
            calculator.backend = "torch"
            return calculator
        except Exception as exc:
            raise RuntimeError(
                f"Unable to initialize Nequix torch backend for DEVICE={requested_device!r}."
            ) from exc

    return root.NequixCalculator(**kwargs)
