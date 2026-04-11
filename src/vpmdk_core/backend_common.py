"""Helpers shared across backend integrations."""

from __future__ import annotations

import inspect
import os
import shutil
import urllib.request
from typing import Dict, Iterable


def _coerce_int_tag(value: str, tag_name: str) -> int:
    """Parse integer BCAR tag values with a descriptive error message."""

    try:
        return int(float(value))
    except (TypeError, ValueError):
        raise ValueError(f"Invalid {tag_name} value: {value!r}") from None


def _coerce_bool_tag(value: str, tag_name: str) -> bool:
    """Parse boolean-like BCAR tags with descriptive errors."""

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid {tag_name} value: {value!r}")


def _looks_like_filesystem_path(value: str, *, suffixes: Iterable[str] = ()) -> bool:
    """Return whether a string likely denotes a local filesystem path."""

    altsep = os.path.altsep
    if os.path.sep in value or (altsep is not None and altsep in value):
        return True
    lowered = value.lower()
    return any(lowered.endswith(suffix.lower()) for suffix in suffixes)


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
