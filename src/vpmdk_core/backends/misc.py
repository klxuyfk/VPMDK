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


def _build_mattersim_calculator(bcar_tags: Dict[str, str]):
    """Create the MatterSim ASE calculator configured from BCAR tags."""

    root = _root()
    if root.MatterSimCalculator is None:
        raise RuntimeError(
            "MatterSimCalculator not available. Install mattersim and dependencies."
        )

    device = root._resolve_device(bcar_tags.get("DEVICE"))
    kwargs: Dict[str, object] = {}
    if device is not None and root._callable_supports_parameter(
        root.MatterSimCalculator, "device"
    ):
        kwargs["device"] = device

    compute_stress = root._parse_optional_bool_tag(
        bcar_tags, "MATTERSIM_COMPUTE_STRESS"
    )
    if compute_stress is not None and root._callable_supports_parameter(
        root.MatterSimCalculator, "compute_stress"
    ):
        kwargs["compute_stress"] = compute_stress

    stress_weight = root._parse_optional_float(
        bcar_tags.get("MATTERSIM_STRESS_WEIGHT"), key="MATTERSIM_STRESS_WEIGHT"
    )
    if stress_weight is not None and root._callable_supports_parameter(
        root.MatterSimCalculator, "stress_weight"
    ):
        kwargs["stress_weight"] = stress_weight

    model_value = bcar_tags.get("MODEL")
    if model_value and os.path.exists(model_value):
        from_checkpoint = getattr(root.MatterSimCalculator, "from_checkpoint", None)
        if callable(from_checkpoint):
            if device is not None and root._callable_supports_parameter(
                from_checkpoint, "device"
            ):
                kwargs.setdefault("device", device)
            return from_checkpoint(model_value, **kwargs)
        return root.MatterSimCalculator(model_value, **kwargs)
    if model_value and root._looks_like_filesystem_path(
        model_value,
        suffixes=(".pt", ".pth", ".ckpt"),
    ):
        raise FileNotFoundError(f"MatterSim model not found: {model_value}")

    return root.MatterSimCalculator(**kwargs)


def _normalize_upet_neighborlist_device(
    value: str | None, model_device: str | None
) -> str | None:
    """Return the UPET neighbor-list execution device policy."""

    requested_model_device = str(model_device or "").strip().lower()
    if value is None or str(value).strip().lower() == "auto":
        return "cpu" if requested_model_device.startswith("cuda") else None

    normalized = str(value).strip().lower()
    if normalized in {"cpu", "host"}:
        return "cpu"
    if normalized in {"cuda", "model", "device", "same"}:
        return None
    raise ValueError(f"Invalid UPET_NEIGHBORLIST_DEVICE value: {value!r}")


def _run_with_upet_neighborlist_device(
    calculator, neighborlist_device: str, call, *args, **kwargs
):
    """Run a UPET calculation with metatomic/vesin neighbor lists on a fixed device."""

    if neighborlist_device != "cpu":
        return call(*args, **kwargs)

    root = _root()
    patches: list[tuple[object, str, object]] = []

    def _devices(systems):
        return [system.device for system in systems]

    def _to_device(systems, device):
        return [system.to(device=device) for system in systems]

    def _restore_devices(systems, devices):
        return [
            system.to(device=device)
            for system, device in zip(systems, devices, strict=True)
        ]

    def _patch(target, attr: str, replacement) -> None:
        original = getattr(target, attr, None)
        if original is None:
            return
        setattr(target, attr, replacement(original))
        patches.append((target, attr, original))

    try:
        current_neighbors = root.importlib.import_module("metatomic_ase._neighbors")
    except Exception:
        current_neighbors = None

    if current_neighbors is not None:
        all_neighbors_calculator = getattr(
            current_neighbors, "AllNeighborsCalculator", None
        )
        if all_neighbors_calculator is not None:
            def _wrap_compute(original):
                def compute_with_cpu_neighbors(self, systems):
                    devices = _devices(systems)
                    cpu_systems = _to_device(systems, "cpu")
                    return _restore_devices(original(self, cpu_systems), devices)

                return compute_with_cpu_neighbors

            _patch(all_neighbors_calculator, "compute", _wrap_compute)

        def _wrap_current_vesin(original):
            def cpu_neighbor_lists(systems, calculators, *args, **kwargs):
                devices = _devices(systems)
                cpu_systems = _to_device(systems, "cpu")
                return _restore_devices(
                    original(cpu_systems, calculators, *args, **kwargs),
                    devices,
                )

            return cpu_neighbor_lists

        _patch(
            current_neighbors,
            "_compute_requested_neighbors_vesin",
            _wrap_current_vesin,
        )

    try:
        legacy_calculator = root.importlib.import_module("metatomic.torch.ase_calculator")
    except Exception:
        legacy_calculator = None

    if legacy_calculator is not None:
        def _wrap_legacy_vesin(original):
            def cpu_neighbor_lists(systems, requested_options, check_consistency=False):
                devices = _devices(systems)
                cpu_systems = _to_device(systems, "cpu")
                computed_systems = original(
                    cpu_systems,
                    requested_options,
                    check_consistency=check_consistency,
                )
                if computed_systems is None:
                    computed_systems = cpu_systems
                return _restore_devices(computed_systems, devices)

            return cpu_neighbor_lists

        _patch(
            legacy_calculator,
            "_compute_requested_neighbors_vesin",
            _wrap_legacy_vesin,
        )

    if not patches:
        return call(*args, **kwargs)

    try:
        return call(*args, **kwargs)
    finally:
        for target, attr, original in reversed(patches):
            setattr(target, attr, original)


class _UPETNeighborListDeviceProxy:
    """Proxy a UPET calculator while forcing metatomic neighbor lists to CPU."""

    def __init__(self, calculator, neighborlist_device: str):
        self.calculator = calculator
        self.neighborlist_device = neighborlist_device
        self.implemented_properties = getattr(calculator, "implemented_properties", [])

    def __getattr__(self, name):
        return getattr(self.calculator, name)

    @property
    def results(self):
        return getattr(self.calculator, "results", {})

    @results.setter
    def results(self, value):
        setattr(self.calculator, "results", value)

    @property
    def atoms(self):
        return getattr(self.calculator, "atoms", None)

    @atoms.setter
    def atoms(self, value):
        setattr(self.calculator, "atoms", value)

    def _call(self, method_name: str, *args, **kwargs):
        method = getattr(self.calculator, method_name)
        return _run_with_upet_neighborlist_device(
            self.calculator,
            self.neighborlist_device,
            method,
            *args,
            **kwargs,
        )

    def calculate(self, *args, **kwargs):
        return self._call("calculate", *args, **kwargs)

    def get_potential_energy(self, *args, **kwargs):
        return self._call("get_potential_energy", *args, **kwargs)

    def get_forces(self, *args, **kwargs):
        return self._call("get_forces", *args, **kwargs)

    def get_stress(self, *args, **kwargs):
        return self._call("get_stress", *args, **kwargs)


def _build_upet_calculator(bcar_tags: Dict[str, str]):
    """Create the UPET ASE calculator configured from BCAR tags."""

    root = _root()
    if root.UPETCalculator is None:
        raise RuntimeError(
            "UPET calculator not available. Install upet and dependencies."
        )

    model_value = bcar_tags.get("MODEL")
    if not model_value:
        raise ValueError(
            "UPET requires MODEL set to a checkpoint path or a named model such as "
            "pet-oam-xl."
        )

    device = root._resolve_device(bcar_tags.get("DEVICE"))
    kwargs: Dict[str, object] = {"device": device}
    neighborlist_device = _normalize_upet_neighborlist_device(
        bcar_tags.get("UPET_NEIGHBORLIST_DEVICE") or bcar_tags.get("UPET_NL_DEVICE"),
        device,
    )

    version = bcar_tags.get("UPET_VERSION")
    if version:
        kwargs["version"] = version

    non_conservative_value = bcar_tags.get("UPET_NON_CONSERVATIVE")
    if non_conservative_value is not None:
        kwargs["non_conservative"] = root._coerce_bool_tag(
            non_conservative_value, "UPET_NON_CONSERVATIVE"
        )

    if os.path.exists(model_value):
        calculator = root.UPETCalculator(checkpoint_path=model_value, **kwargs)
    elif root._looks_like_filesystem_path(
        model_value, suffixes=(".ckpt", ".pt", ".pth")
    ):
        raise FileNotFoundError(f"UPET model not found: {model_value}")
    else:
        calculator = root.UPETCalculator(model=model_value, **kwargs)

    if neighborlist_device is None or not hasattr(calculator, "get_potential_energy"):
        return calculator
    return _UPETNeighborListDeviceProxy(calculator, neighborlist_device)


def _is_equflash_unreleased_named_model(model_value: str | None) -> bool:
    if not model_value:
        return False
    normalized = model_value.strip().casefold().replace("_", "-")
    return normalized in {"equflash-29m-oam", "equflash"}


def _build_equflash_calculator(bcar_tags: Dict[str, str]):
    """Create the EquFlash ASE calculator configured from BCAR tags."""

    root = _root()
    if root.SevenNetCalculator is None or not root._is_sevennet_flash_available():
        raise RuntimeError(
            "EquFlash requires sevenn plus flashTP_e3nn support. Install FlashTP and "
            "ensure CUDA is visible."
        )

    model_value = bcar_tags.get("MODEL")
    if not model_value:
        raise ValueError(
            "EquFlash requires MODEL pointing to a local SevenNet/EquFlash checkpoint. "
            "The public matbench-discovery metadata for equflash-29M-oam lists "
            "checkpoint_url: missing."
        )
    if _is_equflash_unreleased_named_model(model_value):
        raise ValueError(
            "EquFlash named model 'equflash-29M-oam' has public metadata but no "
            "released checkpoint. Set MODEL to a local SevenNet/EquFlash checkpoint."
        )
    if not os.path.exists(model_value):
        if root._looks_like_filesystem_path(
            model_value,
            suffixes=(".pt", ".pth", ".ckpt", ".tar"),
        ):
            raise FileNotFoundError(f"EquFlash model not found: {model_value}")
        raise ValueError(
            "EquFlash currently requires MODEL pointing to a local SevenNet/EquFlash "
            "checkpoint file."
        )

    tags = dict(bcar_tags)
    tags.setdefault("DEVICE", "cuda")
    tags.setdefault("SEVENNET_FILE_TYPE", "checkpoint")
    return root._build_sevennet_family_calculator(tags, force_flash=True)


def _build_tace_calculator(bcar_tags: Dict[str, str]):
    """Create the TACE ASE calculator configured from BCAR tags."""

    root = _root()
    if root.TACEAseCalc is None:
        raise RuntimeError(
            "TACE calculator not available. Install TACE and dependencies."
        )

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
