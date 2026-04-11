"""Smaller or mixed backend builders."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Iterable, List


def _root():
    return sys.modules["vpmdk_core"]


def _list_matlantis_calc_modes() -> str:
    """Return comma-separated list of available Matlantis calc modes."""

    root = _root()
    if root.EstimatorCalcMode is None:
        return ""
    members = getattr(root.EstimatorCalcMode, "__members__", None)
    if isinstance(members, dict) and members:
        return ", ".join(sorted(members))
    candidates = [name for name in dir(root.EstimatorCalcMode) if name.isupper()]
    if candidates:
        return ", ".join(sorted(candidates))
    return ""


def _resolve_matlantis_calc_mode(name):
    """Return ``EstimatorCalcMode`` or passthrough string for Matlantis calc mode."""

    root = _root()
    if root.EstimatorCalcMode is None:
        raise RuntimeError(
            "Matlantis EstimatorCalcMode not available. Install pfp-api-client."
        )

    if isinstance(name, root.EstimatorCalcMode):
        return name
    if name is None:
        raise ValueError("MATLANTIS_CALC_MODE must not be None")

    text = str(name)
    normalized = text.upper()
    candidate = getattr(root.EstimatorCalcMode, normalized, None)
    if candidate is not None:
        return candidate
    members = getattr(root.EstimatorCalcMode, "__members__", None)
    if isinstance(members, dict) and normalized in members:
        return members[normalized]
    try:
        return root.EstimatorCalcMode[normalized]  # type: ignore[index]
    except Exception:
        pass
    try:
        return root.EstimatorCalcMode(normalized)  # type: ignore[call-arg]
    except Exception:
        pass
    return text


def _build_matlantis_calculator(bcar_tags: Dict[str, str]):
    """Create the Matlantis ASE calculator configured from BCAR tags."""

    root = _root()
    if root.MatlantisEstimator is None or root.MatlantisASECalculator is None or root.EstimatorCalcMode is None:
        raise RuntimeError(
            "Matlantis calculator not available. Install pfp-api-client and dependencies."
        )

    model_version = (
        bcar_tags.get("MATLANTIS_MODEL_VERSION")
        or bcar_tags.get("MODEL_VERSION")
        or bcar_tags.get("MODEL")
        or "v8.0.0"
    )
    priority_raw = bcar_tags.get("MATLANTIS_PRIORITY") or bcar_tags.get("PRIORITY")
    priority = 50 if priority_raw is None else root._coerce_int_tag(priority_raw, "MATLANTIS_PRIORITY")
    calc_mode_value = bcar_tags.get("MATLANTIS_CALC_MODE") or bcar_tags.get("CALC_MODE")
    calc_mode = _resolve_matlantis_calc_mode(calc_mode_value or "PBE")

    estimator_kwargs: Dict[str, Any] = {
        "model_version": model_version,
        "priority": priority,
        "calc_mode": calc_mode,
    }

    return root.MatlantisASECalculator(root.MatlantisEstimator(**estimator_kwargs))


def _build_orb_calculator(bcar_tags: Dict[str, str]):
    """Create the ORB ASE calculator configured from BCAR tags."""

    root = _root()
    if root.ORBCalculator is None or root.ORB_PRETRAINED_MODELS is None:
        raise RuntimeError("ORB calculator not available. Install orb-models and dependencies.")

    model_name = bcar_tags.get("ORB_MODEL") or root.DEFAULT_ORB_MODEL
    model_factory = root.ORB_PRETRAINED_MODELS.get(model_name)
    if model_factory is None:
        supported = ", ".join(sorted(root.ORB_PRETRAINED_MODELS))
        raise ValueError(f"Unsupported ORB model '{model_name}'. Available: {supported}")

    device = bcar_tags.get("DEVICE")
    precision = bcar_tags.get("ORB_PRECISION") or "float32-high"
    compile_value = bcar_tags.get("ORB_COMPILE")
    compile_flag = None if compile_value is None else root._coerce_bool_tag(compile_value, "ORB_COMPILE")
    weights_path = bcar_tags.get("MODEL")

    model = model_factory(
        weights_path=weights_path or None,
        device=device,
        precision=precision,
        compile=compile_flag,
        train=False,
    )

    return root.ORBCalculator(model, device=device)


def _build_upet_calculator(bcar_tags: Dict[str, str]):
    """Create the UPET ASE calculator configured from BCAR tags."""

    root = _root()
    if root.UPETCalculator is None:
        raise RuntimeError("UPET calculator not available. Install upet and dependencies.")

    model_value = bcar_tags.get("MODEL")
    if not model_value:
        raise ValueError(
            "UPET requires MODEL set to a checkpoint path or a named model such as pet-oam-xl."
        )

    device = root._resolve_device(bcar_tags.get("DEVICE"))
    kwargs: Dict[str, object] = {"device": device}

    version = bcar_tags.get("UPET_VERSION")
    if version:
        kwargs["version"] = version

    non_conservative_value = bcar_tags.get("UPET_NON_CONSERVATIVE")
    if non_conservative_value is not None:
        kwargs["non_conservative"] = root._coerce_bool_tag(
            non_conservative_value, "UPET_NON_CONSERVATIVE"
        )

    if os.path.exists(model_value):
        return root.UPETCalculator(checkpoint_path=model_value, **kwargs)

    if root._looks_like_filesystem_path(model_value, suffixes=(".ckpt", ".pt", ".pth")):
        raise FileNotFoundError(f"UPET model not found: {model_value}")

    return root.UPETCalculator(model=model_value, **kwargs)


def _get_equflash_calculator_cls():
    """Return the optional EquFlash ASE calculator class when installed."""

    root = _root()
    for module_name in ("GGNN.common.calculator", "ggnn.common.calculator"):
        try:
            module = root.importlib.import_module(module_name)
        except Exception:
            continue
        calculator_cls = getattr(module, "UCalculator", None)
        if calculator_cls is not None:
            return calculator_cls
    return None


def _build_equflash_calculator(bcar_tags: Dict[str, str]):
    """Create the EquFlash ASE calculator configured from BCAR tags."""

    root = _root()
    calculator_cls = root._get_equflash_calculator_cls()
    if calculator_cls is None:
        raise RuntimeError(
            "EquFlash calculator not available. Install the EquFlash/GGNN package "
            "that exposes GGNN.common.calculator.UCalculator and its dependencies."
        )

    model_value = bcar_tags.get("MODEL")
    if not model_value:
        raise ValueError(
            "EquFlash requires MODEL pointing to a local checkpoint file. "
            "Public checkpoints are not currently bundled."
        )
    if not os.path.exists(model_value):
        if root._looks_like_filesystem_path(
            model_value,
            suffixes=(".pt", ".pth", ".ckpt", ".tar"),
        ):
            raise FileNotFoundError(f"EquFlash model not found: {model_value}")
        raise ValueError(
            "EquFlash currently requires MODEL pointing to a local checkpoint file."
        )

    device = root._resolve_device(bcar_tags.get("DEVICE")) or "cpu"
    cpu_flag = str(device).strip().lower().startswith("cpu")
    kwargs: Dict[str, object] = {"checkpoint_path": model_value}
    if root._callable_supports_parameter(calculator_cls, "cpu"):
        kwargs["cpu"] = cpu_flag
    if root._callable_supports_parameter(calculator_cls, "device"):
        kwargs["device"] = device
    return calculator_cls(**kwargs)


def _build_tace_calculator(bcar_tags: Dict[str, str]):
    """Create the TACE ASE calculator configured from BCAR tags."""

    root = _root()
    if root.TACEAseCalc is None:
        raise RuntimeError("TACE calculator not available. Install TACE and dependencies.")

    model_value = bcar_tags.get("MODEL")
    if not model_value:
        raise ValueError(
            "TACE requires MODEL set to a checkpoint path or a named model such as TACE-v1-OMat24-M."
        )

    model_path = model_value
    if not os.path.exists(model_value):
        if root._looks_like_filesystem_path(model_value, suffixes=(".ckpt", ".pt", ".pth")):
            raise FileNotFoundError(f"TACE model not found: {model_value}")

        if root.tace_foundations is None:
            raise RuntimeError(
                "TACE named-model registry is not available. Install TACE with foundation-model "
                "support or provide MODEL as a local checkpoint path."
            )
        try:
            model_path = os.fspath(root.tace_foundations[model_value])
        except KeyError as exc:
            supported = (
                ", ".join(root.tace_foundations.list_models())
                if hasattr(root.tace_foundations, "list_models")
                else ""
            )
            if supported:
                raise ValueError(
                    f"Unsupported TACE model '{model_value}'. Available: {supported}"
                ) from exc
            raise ValueError(f"Unsupported TACE model '{model_value}'.") from exc

    kwargs: Dict[str, object] = {
        "model": model_path,
        "device": root._resolve_device(bcar_tags.get("DEVICE")),
    }

    dtype = bcar_tags.get("TACE_DTYPE")
    if dtype:
        kwargs["dtype"] = dtype

    spin_on_value = bcar_tags.get("TACE_SPIN_ON")
    if spin_on_value is not None:
        kwargs["spin_on"] = root._coerce_bool_tag(spin_on_value, "TACE_SPIN_ON")

    neighborlist_backend = bcar_tags.get("TACE_NEIGHBORLIST_BACKEND")
    if neighborlist_backend:
        kwargs["neighborlist_backend"] = neighborlist_backend

    level_tag = None
    if "TACE_FIDELITY_IDX" in bcar_tags:
        level_tag = "TACE_FIDELITY_IDX"
    elif "TACE_LEVEL" in bcar_tags:
        level_tag = "TACE_LEVEL"

    if level_tag is not None:
        level_value = root._coerce_int_tag(bcar_tags[level_tag], level_tag)
        if root._callable_supports_parameter(root.TACEAseCalc, "fidelity_idx"):
            kwargs["fidelity_idx"] = level_value
        elif root._callable_supports_parameter(root.TACEAseCalc, "level"):
            kwargs["level"] = level_value

    return root.TACEAseCalc(**kwargs)


def _build_grace_calculator(bcar_tags: Dict[str, str]):
    """Create a GRACE (TensorPotential) ASE calculator."""

    root = _root()
    if root.TPCalculator is None:
        raise RuntimeError(
            "TPCalculator not available. Install grace-tensorpotential and dependencies."
        )

    grace_kwargs: Dict[str, object] = {}

    pad_fraction = root._parse_optional_float(
        bcar_tags.get("GRACE_PAD_NEIGHBORS_FRACTION"), key="GRACE_PAD_NEIGHBORS_FRACTION"
    )
    if pad_fraction is not None:
        grace_kwargs["pad_neighbors_fraction"] = pad_fraction

    pad_atoms_raw = bcar_tags.get("GRACE_PAD_ATOMS_NUMBER")
    if pad_atoms_raw is not None:
        grace_kwargs["pad_atoms_number"] = root._coerce_int_tag(
            pad_atoms_raw, "GRACE_PAD_ATOMS_NUMBER"
        )

    recompilation_raw = bcar_tags.get("GRACE_MAX_RECOMPILATION")
    if recompilation_raw is not None:
        grace_kwargs["max_number_reduction_recompilation"] = root._coerce_int_tag(
            recompilation_raw, "GRACE_MAX_RECOMPILATION"
        )

    min_dist = root._parse_optional_float(bcar_tags.get("GRACE_MIN_DIST"), key="GRACE_MIN_DIST")
    if min_dist is not None:
        grace_kwargs["min_dist"] = min_dist

    float_dtype = bcar_tags.get("GRACE_FLOAT_DTYPE")
    if float_dtype:
        grace_kwargs["float_dtype"] = float_dtype

    model_value = bcar_tags.get("MODEL")
    if model_value and os.path.exists(model_value):
        return root.TPCalculator(model_value, **grace_kwargs)

    available_models = root.GRACE_MODEL_NAMES
    default_model = root.DEFAULT_GRACE_MODEL
    if available_models:
        default_model = default_model if default_model in available_models else available_models[0]

    if root.grace_fm is not None and available_models:
        selected = model_value or default_model
        if selected not in available_models:
            print(
                f"Warning: Unknown GRACE model '{selected}', using default {default_model} instead."
            )
            selected = default_model
        return root.grace_fm(selected, **grace_kwargs)

    if model_value:
        raise FileNotFoundError(f"GRACE model not found: {model_value}")

    raise RuntimeError(
        "GRACE calculator requires a MODEL path or available foundation models (grace_fm)."
    )


def _build_deepmd_calculator(bcar_tags: Dict[str, str], structure=None):
    """Create a DeePMD-kit calculator configured from BCAR tags."""

    root = _root()
    if root.DeePMDCalculator is None:
        raise RuntimeError(
            "DeePMD-kit calculator not available. Install deepmd-kit and dependencies."
        )

    model_path = bcar_tags.get("MODEL")
    if not model_path:
        raise ValueError("DeePMD-kit requires MODEL pointing to a frozen model file.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"DeePMD-kit model not found: {model_path}")

    type_map_value = bcar_tags.get("DEEPMD_TYPE_MAP")
    type_map: List[str] = []
    if type_map_value:
        type_map = [item for item in root.re.split(r"[\s,]+", type_map_value.strip()) if item]
    elif structure is not None:
        type_map = root._infer_type_map(structure)

    kwargs: Dict[str, object] = {}
    if type_map:
        kwargs["type_map"] = type_map

    head_value = bcar_tags.get("DEEPMD_HEAD")
    if head_value:
        kwargs["head"] = head_value

    return root.DeePMDCalculator(model=model_path, **kwargs)
