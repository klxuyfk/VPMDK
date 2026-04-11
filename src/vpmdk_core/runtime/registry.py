"""Backend selection and calculator construction."""

from __future__ import annotations

import inspect
import sys
from typing import Dict


def _root():
    return sys.modules["vpmdk_core"]


_SIMPLE_CALCULATORS: Dict[str, tuple[str, str]] = {
    "MATTERSIM": (
        "MatterSimCalculator",
        "MatterSimCalculator not available. Install mattersim and dependencies.",
    ),
}


_CALCULATOR_BUILDERS: Dict[str, str] = {
    "CHGNET": "_build_chgnet_calculator",
    "MATGL": "_build_m3gnet_calculator",
    "M3GNET": "_build_m3gnet_calculator",
    "MACE": "_build_mace_calculator",
    "EQNORM": "_build_eqnorm_calculator",
    "MATRIS": "_build_matris_calculator",
    "ALPHANET": "_build_alphanet_calculator",
    "HIENET": "_build_hienet_calculator",
    "NEQUIX": "_build_nequix_calculator",
    "SEVENNET": "_build_sevennet_calculator",
    "FLASHTP": "_build_flashtp_calculator",
    "ALLEGRO": "_build_allegro_calculator",
    "NEQUIP": "_build_nequip_calculator",
    "MATLANTIS": "_build_matlantis_calculator",
    "ORB": "_build_orb_calculator",
    "UPET": "_build_upet_calculator",
    "EQUFLASH": "_build_equflash_calculator",
    "TACE": "_build_tace_calculator",
    "FAIRCHEM": "_build_fairchem_calculator",
    "FAIRCHEM_V2": "_build_fairchem_calculator",
    "ESEN": "_build_fairchem_calculator",
    "FAIRCHEM_V1": "_build_fairchem_v1_calculator",
    "GRACE": "_build_grace_calculator",
    "DEEPMD": "_build_deepmd_calculator",
}


def _build_simple_model_calculator(
    calculator_cls,
    bcar_tags: Dict[str, str],
    missing_message: str,
):
    """Return calculator initialized with optional ``MODEL`` path."""

    if calculator_cls is None:
        raise RuntimeError(missing_message)

    model_path = bcar_tags.get("MODEL")
    if model_path and _root().os.path.exists(model_path):
        return calculator_cls(model_path)
    return calculator_cls()


def _build_calculator_from_init_factory(calculator, bcar_tags: Dict[str, str]):
    init = getattr(calculator.__class__, "__init__", None)
    closure = getattr(init, "__closure__", None)
    if not closure:
        return None
    mlp = _root()._resolve_mlp_tag(bcar_tags, default="")
    for cell in closure:
        factory = cell.cell_contents
        if not callable(factory):
            continue
        try:
            candidate = factory(mlp)
        except TypeError:
            try:
                candidate = factory()
            except Exception:
                continue
        if hasattr(candidate, "get_potential_energy"):
            return candidate
    return None


def _attach_fallback_calculator(calculator, bcar_tags: Dict[str, str]):
    if hasattr(calculator, "get_potential_energy"):
        return calculator
    fallback = getattr(calculator, "calculator", None)
    if fallback is None or not hasattr(fallback, "get_potential_energy"):
        fallback = _build_calculator_from_init_factory(calculator, bcar_tags)
    if fallback is None:
        raise RuntimeError(
            "FAIRChem v1 calculator wrapper does not provide an inner ASE calculator."
        )
    setattr(calculator, "calculator", fallback)
    return calculator


def get_calculator(bcar_tags: Dict[str, str], *, structure=None):
    """Return ASE calculator based on BCAR tags."""

    root = _root()
    mlp = root._resolve_mlp_tag(bcar_tags)
    if mlp in root._SIMPLE_CALCULATORS:
        calculator_attr, message = root._SIMPLE_CALCULATORS[mlp]
        calculator_cls = getattr(root, calculator_attr, None)
        return _build_simple_model_calculator(calculator_cls, bcar_tags, message)

    builder_entry = root._CALCULATOR_BUILDERS.get(mlp)
    if builder_entry is None:
        raise ValueError(f"Unsupported MLP type: {mlp}")
    if callable(builder_entry):
        builder = builder_entry
        builder_name = getattr(builder_entry, "__name__", "")
    else:
        builder_name = builder_entry
        builder = getattr(root, builder_name, None)
    if builder is None:
        raise RuntimeError(f"Calculator builder not available: {builder_entry}")

    try:
        builder_signature = inspect.signature(builder)
    except (TypeError, ValueError):
        builder_signature = None

    accepts_structure = builder_name == "_build_deepmd_calculator"
    if builder_signature is not None:
        accepts_structure = accepts_structure or "structure" in builder_signature.parameters
        accepts_structure = accepts_structure or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in builder_signature.parameters.values()
        )

    if accepts_structure:
        return builder(bcar_tags, structure=structure)
    return builder(bcar_tags)
