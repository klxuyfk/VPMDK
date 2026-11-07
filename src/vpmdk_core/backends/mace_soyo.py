"""MACE-SOYO CUDA checkpoint and AOTInductor runtimes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..backend_common import _resolve_backend_model_reference, _resolve_device


def _get_mace_soyo_calculator_cls() -> Any:
    """Keep the independent MACE-SOYO dependency optional."""
    try:
        from mace_soyo import MACESoyoCalculator
    except ImportError:
        return None
    return MACESoyoCalculator


def _build_mace_soyo_calculator(bcar_tags: dict[str, str], *, structure=None) -> Any:
    """Load a .pth checkpoint or a .pt2 package on CUDA."""
    reference = _resolve_backend_model_reference("MACE_SOYO", bcar_tags.get("MODEL"))
    path = Path(reference.value)
    if not path.is_file() or path.suffix.lower() not in {".pt2", ".pth"}:
        raise ValueError(
            "MACE_SOYO MODEL must be a .pth checkpoint or an AOTI .pt2 file."
        )
    device = _resolve_device(bcar_tags.get("DEVICE") or None)
    if str(device).split(":", 1)[0] != "cuda":
        raise ValueError("MACE_SOYO requires an NVIDIA CUDA device; CPU is not supported.")
    calculator_cls = _get_mace_soyo_calculator_cls()
    if calculator_cls is None:
        raise RuntimeError(
            "MACE-SOYO is not available. Install the upstream MACE-SOYO package "
            "and its matching PyTorch, cuEquivariance and nvalchemi dependencies."
        )
    method = bcar_tags.get("MACE_SOYO_CUEQ_METHOD", "native").strip().lower() or "native"
    if method not in {"native", "naive"}:
        raise ValueError("MACE_SOYO_CUEQ_METHOD must be native or naive.")
    head = bcar_tags.get("MACE_SOYO_HEAD", "").strip() or None
    if path.suffix.lower() == ".pth":
        from .mace_soyo_eager import MaceSoyoEagerCalculator

        return MaceSoyoEagerCalculator(str(path), device=device, head=head, cueq_method=method)
    if method != "native":
        raise ValueError("MACE_SOYO_CUEQ_METHOD applies to .pth loading; .pt2 fixes the method at export time.")
    return calculator_cls(
        package_path=str(path), device=device,
        head=head,
        use_d3=False,
    )
