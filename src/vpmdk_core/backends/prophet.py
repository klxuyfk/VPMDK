"""Prophet backend builder."""

from __future__ import annotations

import sys
from typing import Dict


def _root():
    return sys.modules["vpmdk_core"]


def _build_prophet_calculator(bcar_tags: Dict[str, str]):
    """Create the official Prophet ASE calculator from a local checkpoint."""

    root = _root()
    if root.ProphetCalculator is None:
        raise RuntimeError(
            "Prophet calculator not available. Install prophet-mlip and dependencies."
        )

    model_reference = root._resolve_backend_model_reference(
        "PROPHET", bcar_tags.get("MODEL")
    )
    # A blank DEVICE has the same meaning as an omitted one throughout VPMDK.
    # Avoid passing ``""`` to torch.device(), which rejects it.
    device = root._resolve_device(bcar_tags.get("DEVICE") or None)
    use_kernel_raw = bcar_tags.get("PROPHET_USE_KERNEL")
    use_kernel = (
        False
        if use_kernel_raw is None
        else root._coerce_bool_tag(use_kernel_raw, "PROPHET_USE_KERNEL")
    )
    use_compile_raw = bcar_tags.get("PROPHET_USE_COMPILE")
    use_compile = (
        False
        if use_compile_raw is None
        else root._coerce_bool_tag(use_compile_raw, "PROPHET_USE_COMPILE")
    )

    if use_kernel and not str(device or "").lower().startswith("cuda"):
        raise ValueError("PROPHET_USE_KERNEL requires a CUDA DEVICE.")

    return root.ProphetCalculator(
        model_path=str(model_reference.value),
        use_kernel=use_kernel,
        use_compile=use_compile,
        device=device,
    )
