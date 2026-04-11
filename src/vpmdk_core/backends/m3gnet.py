"""M3GNet and MACE backend builders."""

from __future__ import annotations

import os
import sys
from typing import Dict


def _root():
    return sys.modules["vpmdk_core"]


def _build_mace_calculator(bcar_tags: Dict[str, str], *, structure=None):
    """Create a MACE calculator with optional ``MODEL`` override."""

    root = _root()
    if root.MACECalculator is None:
        raise RuntimeError("MACECalculator not available. Install mace-torch and dependencies.")

    model_path = bcar_tags.get("MODEL")
    device = root._resolve_device(bcar_tags.get("DEVICE"))

    if model_path and os.path.exists(model_path):
        return root.MACECalculator(model_path, device=device)
    return root.MACECalculator(device=device)


def _build_m3gnet_calculator(bcar_tags: Dict[str, str]):
    """Create a MatGL or legacy M3GNet calculator based on availability."""

    root = _root()
    if root.M3GNetCalculator is None:
        raise RuntimeError("M3GNetCalculator not available. Install matgl or m3gnet.")

    model_path = bcar_tags.get("MODEL")
    device = root._resolve_device(bcar_tags.get("DEVICE"))

    if not root._USING_LEGACY_M3GNET:
        kwargs = {"device": device} if device is not None else {}
        if model_path and os.path.exists(model_path):
            if root.MatGLLoadModel is not None:
                try:
                    potential = root.MatGLLoadModel(model_path)
                    return root.M3GNetCalculator(potential, **kwargs)
                except Exception:
                    pass
            try:
                return root.M3GNetCalculator(model_path, **kwargs)
            except TypeError:
                return root.M3GNetCalculator(model_path)
        try:
            return root.M3GNetCalculator(**kwargs)
        except TypeError:
            return root.M3GNetCalculator()

    potential = None
    if model_path and os.path.exists(model_path) and root.LegacyM3GNetPotential is not None:
        try:
            potential = root.LegacyM3GNetPotential.from_checkpoint(model_path)
        except Exception:
            try:
                if root.LegacyM3GNet is not None:
                    potential = root.LegacyM3GNetPotential(
                        root.LegacyM3GNet.load(model_path)  # type: ignore[arg-type]
                    )
            except Exception:
                potential = None

    if (
        potential is None
        and root.LegacyM3GNetPotential is not None
        and root.LegacyM3GNet is not None
    ):
        potential = root.LegacyM3GNetPotential(root.LegacyM3GNet.load())

    if potential is None:
        raise RuntimeError("Legacy M3GNet calculator could not be initialized from available models.")

    if device is not None:
        try:
            return root.M3GNetCalculator(potential=potential, device=device)
        except TypeError:
            pass

    return root.M3GNetCalculator(potential=potential)
