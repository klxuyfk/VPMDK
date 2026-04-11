"""SevenNet-family backend builders."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Dict


def _root():
    return sys.modules["vpmdk_core"]


def _normalize_sevennet_file_type(value: str | None) -> str:
    """Return a normalized SevenNet file type accepted from BCAR."""

    root = _root()
    file_type = (value or "checkpoint").strip().lower()
    if file_type == "model_instance":
        raise ValueError(
            "SEVENNET_FILE_TYPE=model_instance is not supported from BCAR. "
            "Use checkpoint or torchscript."
        )
    if file_type not in root._SEVENNET_FILE_TYPES:
        supported = ", ".join(sorted(root._SEVENNET_FILE_TYPES))
        raise ValueError(
            f"Invalid SEVENNET_FILE_TYPE value: {value!r}. Expected one of: {supported}."
        )
    return file_type


def _resolve_sevennet_accelerators(
    bcar_tags: Dict[str, str], *, force_flash: bool = False
) -> tuple[bool | None, bool | None, bool | None]:
    """Resolve SevenNet accelerator flags from BCAR with conflict checking."""

    root = _root()
    enable_cueq = root._parse_optional_bool_tag(bcar_tags, "SEVENNET_ENABLE_CUEQ")
    enable_flash = root._parse_optional_bool_tag(bcar_tags, "SEVENNET_ENABLE_FLASH")
    enable_oeq = root._parse_optional_bool_tag(bcar_tags, "SEVENNET_ENABLE_OEQ")

    if sum(flag is True for flag in (enable_cueq, enable_flash, enable_oeq)) > 1:
        raise ValueError(
            "Only one of SEVENNET_ENABLE_CUEQ, SEVENNET_ENABLE_FLASH, or "
            "SEVENNET_ENABLE_OEQ may be enabled at once."
        )

    if force_flash:
        if enable_cueq is True or enable_oeq is True or enable_flash is False:
            raise ValueError(
                "MLP=FLASHTP forces FlashTP acceleration and cannot be combined with "
                "SEVENNET_ENABLE_CUEQ=1, SEVENNET_ENABLE_OEQ=1, or "
                "SEVENNET_ENABLE_FLASH=0."
            )
        return False, True, False

    if any(flag is True for flag in (enable_cueq, enable_flash, enable_oeq)):
        return (
            False if enable_cueq is None else enable_cueq,
            False if enable_flash is None else enable_flash,
            False if enable_oeq is None else enable_oeq,
        )

    return enable_cueq, enable_flash, enable_oeq


def _sevennet_supported_accelerator_tags() -> Dict[str, str]:
    """Return supported SevenNet accelerator BCAR tags keyed by kwarg name."""

    root = _root()
    if root.SevenNetCalculator is None or root._SEVENNET_PACKAGE != "sevenn":
        return {}

    supported: Dict[str, str] = {}
    for kwarg_name, tag_name in (
        ("enable_cueq", "SEVENNET_ENABLE_CUEQ"),
        ("enable_flash", "SEVENNET_ENABLE_FLASH"),
        ("enable_oeq", "SEVENNET_ENABLE_OEQ"),
    ):
        if root._callable_declares_parameter(root.SevenNetCalculator, kwarg_name):
            supported[kwarg_name] = tag_name
    return supported


def _sevennet_supports_modal() -> bool:
    """Return whether the loaded SevenNet calculator supports modal selection."""

    root = _root()
    return root.SevenNetCalculator is not None and root._callable_declares_parameter(
        root.SevenNetCalculator, "modal"
    )


def _is_sevennet_flash_available() -> bool:
    """Return whether FlashTP is available through the current SevenNet install."""

    root = _root()
    if root._SEVENNET_PACKAGE != "sevenn":
        return False
    try:
        module = importlib.import_module("sevenn.nn.flash_helper")
    except Exception:
        return False
    checker = getattr(module, "is_flash_available", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            return False
    return False


def _build_sevennet_family_calculator(
    bcar_tags: Dict[str, str], *, force_flash: bool = False
):
    """Create a SevenNet-family calculator from BCAR tags."""

    root = _root()
    backend_name = "FlashTP" if force_flash else "SevenNet"
    if root.SevenNetCalculator is None:
        install_message = (
            "Install sevenn (preferred)"
            if not force_flash
            else "Install sevenn plus flashTP_e3nn"
        )
        raise RuntimeError(f"{backend_name} calculator not available. {install_message}.")

    file_type = _normalize_sevennet_file_type(bcar_tags.get("SEVENNET_FILE_TYPE"))
    modal = bcar_tags.get("SEVENNET_MODAL")
    model_value = bcar_tags.get("MODEL") or root.DEFAULT_SEVENNET_MODEL
    device = root._resolve_device(bcar_tags.get("DEVICE")) or "cpu"
    enable_cueq, enable_flash, enable_oeq = _resolve_sevennet_accelerators(
        bcar_tags,
        force_flash=force_flash,
    )
    supported_accelerators = _sevennet_supported_accelerator_tags()

    if file_type != "checkpoint" and (
        enable_flash is True or enable_cueq is True or enable_oeq is True
    ):
        raise ValueError(
            f"{backend_name} accelerator flags require SEVENNET_FILE_TYPE=checkpoint."
        )

    if os.path.exists(model_value):
        model_spec: str | Path = model_value
    elif root._looks_like_filesystem_path(
        model_value,
        suffixes=(".pt", ".pth", ".ckpt", ".jit", ".ts"),
    ):
        raise FileNotFoundError(f"{backend_name} model not found: {model_value}")
    else:
        model_spec = model_value

    unsupported_accelerators = [
        tag_name
        for kwarg_name, tag_name, flag_value in (
            ("enable_cueq", "SEVENNET_ENABLE_CUEQ", enable_cueq),
            ("enable_flash", "SEVENNET_ENABLE_FLASH", enable_flash),
            ("enable_oeq", "SEVENNET_ENABLE_OEQ", enable_oeq),
        )
        if flag_value is True and kwarg_name not in supported_accelerators
    ]
    if unsupported_accelerators:
        tags_text = ", ".join(unsupported_accelerators)
        raise RuntimeError(
            f"The installed SevenNet backend does not support {tags_text}."
        )

    if enable_flash is True and not root._is_sevennet_flash_available():
        raise RuntimeError(
            "FlashTP is not available. Install flashTP_e3nn and ensure CUDA is visible."
        )

    kwargs: Dict[str, object] = {"model": model_spec, "device": device}
    if root._callable_declares_parameter(root.SevenNetCalculator, "file_type"):
        kwargs["file_type"] = file_type
    elif file_type != "checkpoint":
        raise RuntimeError(
            "The installed SevenNet backend does not support SEVENNET_FILE_TYPE."
        )

    if modal is not None:
        if not _sevennet_supports_modal():
            raise RuntimeError(
                "The installed SevenNet backend does not support SEVENNET_MODAL."
            )
        kwargs["modal"] = modal

    if "enable_cueq" in supported_accelerators:
        kwargs["enable_cueq"] = enable_cueq
    if "enable_flash" in supported_accelerators:
        kwargs["enable_flash"] = enable_flash
    if "enable_oeq" in supported_accelerators:
        kwargs["enable_oeq"] = enable_oeq

    return root.SevenNetCalculator(**kwargs)


def _build_sevennet_calculator(bcar_tags: Dict[str, str]):
    """Create the SevenNet ASE calculator configured from BCAR tags."""

    return _build_sevennet_family_calculator(bcar_tags, force_flash=False)


def _build_flashtp_calculator(bcar_tags: Dict[str, str]):
    """Create a FlashTP-accelerated SevenNet ASE calculator."""

    return _build_sevennet_family_calculator(bcar_tags, force_flash=True)
