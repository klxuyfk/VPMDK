"""Smaller or mixed backend builders."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List


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

    model_reference = root._resolve_backend_model_reference(
        "MATLANTIS",
        bcar_tags.get("MATLANTIS_MODEL_VERSION")
        or bcar_tags.get("MODEL_VERSION")
        or bcar_tags.get("MODEL"),
    )
    model_version = str(model_reference.value)
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
    weights_reference = root._resolve_backend_model_reference(
        "ORB", bcar_tags.get("MODEL")
    )
    # An explicit MODEL (a local checkpoint or a remote URI) overrides the
    # factory's bundled default weights; an omitted MODEL keeps that default.
    weights_path = (
        None
        if weights_reference.kind is root.ModelReferenceKind.DEFAULT
        else weights_reference.value
    )

    model = root._require_loaded_model(
        model_factory(
            weights_path=weights_path or None,
            device=device,
            precision=precision,
            compile=compile_flag,
            train=False,
        ),
        backend_name="ORB",
        model=str(model_name),
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
    compute_stress = root._parse_optional_bool_tag(
        bcar_tags, "MATTERSIM_COMPUTE_STRESS"
    )
    stress_weight = root._parse_optional_float(
        bcar_tags.get("MATTERSIM_STRESS_WEIGHT"), key="MATTERSIM_STRESS_WEIGHT"
    )

    def physics_kwarg_is_declared(callable_obj, key: str) -> bool:
        """Whether a physics kwarg verifiably reaches a DECLARED parameter.

        Requiring ``callable_obj`` itself to declare the parameter is too strict
        for the real upstream shape: ``MatterSimCalculator.from_checkpoint(
        load_path, *, device=..., **kwargs)`` FORWARDS ``**kwargs`` to
        ``__init__``, which is where compute_stress/stress_weight are declared.
        Rejecting that hard-failed a documented, previously-working configuration.

        So accept exactly one forwarding hop -- and only when the target really
        declares the parameter. A bare ``**kwargs`` with nowhere to land is still
        rejected, because there the value would be silently swallowed and the run
        would quietly compute different physics (the case this gate exists for).
        """

        if root._callable_declares_parameter(callable_obj, key):
            return True
        if callable_obj is root.MatterSimCalculator:
            # The constructor itself: no further hop to inspect, so a **kwargs-only
            # signature genuinely cannot accept the tag.
            return False
        return root._callable_supports_parameter(
            callable_obj, key
        ) and root._callable_declares_parameter(root.MatterSimCalculator, key)

    def optional_kwargs(callable_obj, description: str) -> Dict[str, object]:
        """Return supported optional kwargs for whichever loader is used.

        A physics tag comes only from an explicit BCAR entry, so dropping one
        silently changes what the run computes and must fail instead. DEVICE is
        resolved automatically and only affects placement, so a loader that
        picks its own device may still ignore it.
        """

        selected: Dict[str, object] = {}
        unsupported: List[str] = []
        for key, tag, value, physics in (
            ("device", "DEVICE", device, False),
            ("compute_stress", "MATTERSIM_COMPUTE_STRESS", compute_stress, True),
            ("stress_weight", "MATTERSIM_STRESS_WEIGHT", stress_weight, True),
        ):
            if value is None:
                continue
            # Physics tags (compute_stress/stress_weight) MUST reach an EXPLICITLY
            # declared parameter -- directly or through one verified forwarding
            # hop (see physics_kwarg_is_declared). A bare ``**kwargs`` with no
            # declaring target would silently swallow them, computing without
            # stress and giving no error, so that still raises. DEVICE is fine to
            # forward through ``**kwargs``, so it keeps the looser check.
            accepts = (
                physics_kwarg_is_declared(callable_obj, key)
                if physics
                else root._callable_supports_parameter(callable_obj, key)
            )
            if accepts:
                selected[key] = value
            elif physics:
                unsupported.append(tag)
        if unsupported:
            raise RuntimeError(
                f"The installed {description} does not accept "
                f"{', '.join(unsupported)}; remove the tag(s) or install a "
                "MatterSim release that supports them."
            )
        return selected

    model_reference = root._resolve_backend_model_reference(
        "MATTERSIM", bcar_tags.get("MODEL")
    )
    if model_reference.kind is not root.ModelReferenceKind.DEFAULT:
        model_value = str(model_reference.value)
        from_checkpoint = getattr(root.MatterSimCalculator, "from_checkpoint", None)
        if callable(from_checkpoint):
            checkpoint_kwargs = optional_kwargs(
                from_checkpoint, "MatterSimCalculator.from_checkpoint"
            )
            calculator = from_checkpoint(model_value, **checkpoint_kwargs)
            return root._require_loaded_model(
                calculator, backend_name="MatterSim", model=model_value
            )

        if root._callable_declares_parameter(
            root.MatterSimCalculator, "load_path"
        ):
            # Require an explicit ``load_path`` parameter. A ``**kwargs``-only
            # signature would silently absorb and ignore ``load_path``, loading
            # the default model instead of the requested checkpoint.
            calculator = root.MatterSimCalculator(
                load_path=model_value,
                **optional_kwargs(root.MatterSimCalculator, "MatterSimCalculator"),
            )
            return root._require_loaded_model(
                calculator, backend_name="MatterSim", model=model_value
            )

        if model_reference.kind is root.ModelReferenceKind.LOCAL_PATH:
            calculator = root.MatterSimCalculator(
                model_value,
                **optional_kwargs(root.MatterSimCalculator, "MatterSimCalculator"),
            )
            return root._require_loaded_model(
                calculator, backend_name="MatterSim", model=model_value
            )

        raise RuntimeError(
            "The installed MatterSimCalculator cannot load named MODEL "
            f"{model_value!r}: from_checkpoint and load_path are unavailable."
        )

    return root.MatterSimCalculator(
        **optional_kwargs(root.MatterSimCalculator, "MatterSimCalculator")
    )


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

    model_reference = root._resolve_backend_model_reference(
        "UPET", bcar_tags.get("MODEL")
    )
    model_value = str(model_reference.value)

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

    if model_reference.kind is root.ModelReferenceKind.LOCAL_PATH:
        calculator = root.UPETCalculator(checkpoint_path=model_value, **kwargs)
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
    if _is_equflash_unreleased_named_model(model_value):
        raise ValueError(
            "EquFlash named model 'equflash-29M-oam' has public metadata but no "
            "released checkpoint. Set MODEL to a local SevenNet/EquFlash checkpoint."
        )
    model_reference = root._resolve_backend_model_reference("EQUFLASH", model_value)

    tags = dict(bcar_tags)
    tags["MODEL"] = str(model_reference.value)
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

    model_reference = root._resolve_backend_model_reference(
        "TACE", bcar_tags.get("MODEL")
    )
    model_value = str(model_reference.value)

    model_path = model_value
    if model_reference.kind is root.ModelReferenceKind.NAMED_MODEL:
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
        # The fidelity selector picks WHICH DFT level the model predicts, so
        # dropping it silently changes the physics of the run. It must therefore
        # reach an EXPLICITLY declared parameter: _callable_supports_parameter is
        # True for a bare ``**kwargs`` signature, which would absorb and ignore
        # the value and quietly compute with fidelity head 0. Use
        # _callable_declares_parameter (False for ``**kwargs``) and raise when
        # neither spelling exists -- the previous if/elif had no ``else``, so an
        # unsupported build discarded the tag outright. Mirrors the MatterSim
        # physics-tag guard in this module.
        if root._callable_declares_parameter(root.TACEAseCalc, "fidelity_idx"):
            kwargs["fidelity_idx"] = level_value
        elif root._callable_declares_parameter(root.TACEAseCalc, "level"):
            kwargs["level"] = level_value
        else:
            raise RuntimeError(
                f"The installed TACE calculator does not accept {level_tag}"
                " (no fidelity_idx/level parameter); remove the tag or install a"
                " TACE release that supports selecting the fidelity level."
            )

    return root.TACEAseCalc(**kwargs)


def _resolve_grace_foundation_model(model_value: str | None = None) -> str | None:
    """Return the foundation model GRACE would select for an optional name."""

    root = _root()
    available_models = list(root.GRACE_MODEL_NAMES)
    if not available_models:
        return None
    fallback = (
        root.DEFAULT_GRACE_MODEL
        if root.DEFAULT_GRACE_MODEL in available_models
        else available_models[0]
    )
    if model_value:
        normalized = str(model_value).casefold()
        return next(
            (candidate for candidate in available_models if candidate.casefold() == normalized),
            None,
        )
    return fallback


# One constant shared with the resident server's request path: the resident
# builder never re-runs per request, so the server synthesizes this exact
# warning for a DEVICE-carrying GRACE request to preserve one-shot output
# equivalence.
_GRACE_DEVICE_IGNORED_WARNING = (
    "Warning: GRACE ignores the DEVICE tag; device placement is "
    "decided entirely by the installed TensorFlow build. GPU "
    "execution requires a CUDA-enabled tensorflow compatible with "
    "the local driver (tensorpotential requires tensorflow<2.20)."
)


def _build_grace_calculator(bcar_tags: Dict[str, str]):
    """Create a GRACE (TensorPotential) ASE calculator."""

    root = _root()
    if root.TPCalculator is None:
        raise RuntimeError(
            "TPCalculator not available. Install grace-tensorpotential and dependencies."
        )

    if str(bcar_tags.get("DEVICE") or "").strip():
        # TPCalculator/grace_fm take no device argument at all: placement is
        # decided by the installed TensorFlow build, so a user writing
        # DEVICE=cuda got silence and, on a TF build without working CUDA
        # support, CPU execution they believed was GPU.
        print(_GRACE_DEVICE_IGNORED_WARNING)

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

    model_reference = root._resolve_backend_model_reference(
        "GRACE", bcar_tags.get("MODEL")
    )
    requested_model = str(bcar_tags.get("MODEL") or "").strip()
    if root._grace_substitutes_unknown_model(root, model_reference, requested_model):
        # Report the substitution once, at construction time. The shared
        # resolver stays side-effect free because the server calls it per
        # request for identity comparison. The same predicate gates the resident
        # server's request warning so the two cannot drift.
        print(
            f"Warning: Unknown GRACE model '{requested_model}', using default "
            f"{model_reference.value} instead."
        )
    if model_reference.kind is root.ModelReferenceKind.LOCAL_PATH:
        calculator = root.TPCalculator(model_reference.value, **grace_kwargs)
        return root._require_loaded_model(
            calculator, backend_name="GRACE", model=str(model_reference.value)
        )

    selected = model_reference.value
    if root.grace_fm is not None and selected is not None:
        if root._resolve_grace_foundation_model(str(selected)) is None:
            # The selected model could not be validated against the foundation
            # registry (e.g. an empty MODELS_NAME_LIST via version skew). Do not
            # forward an unverified name to grace_fm, which may silently load a
            # substituted/default model; fail clearly per the strict
            # no-silent-wrong-model contract. A local checkpoint path still works.
            if model_reference.explicit:
                raise FileNotFoundError(f"GRACE model not found: {selected}")
            # An omitted MODEL fell back to the default constant, which cannot be
            # validated against the empty registry either.
            raise RuntimeError(
                "GRACE has no enumerable foundation models; set MODEL to a local "
                "checkpoint path or install a TensorPotential release that lists "
                "its foundation models."
            )
        calculator = root.grace_fm(str(selected), **grace_kwargs)
        return root._require_loaded_model(
            calculator, backend_name="GRACE", model=str(selected)
        )

    if model_reference.explicit:
        # An explicitly named foundation model cannot be loaded without the
        # TensorPotential foundation loader. Preserve the FileNotFoundError
        # contract so callers distinguishing a missing model from a generic
        # failure (and exception-to-exit-code mappings) keep working.
        raise FileNotFoundError(f"GRACE model not found: {selected}")

    # No MODEL was supplied and the foundation loader is unavailable, so there
    # is no default to fall back on. This is an environment problem, not a
    # missing named model, hence a RuntimeError rather than FileNotFoundError.
    raise RuntimeError(
        "GRACE calculator requires a MODEL path or an available TensorPotential "
        "foundation loader (grace_fm)."
    )


def _build_deepmd_calculator(bcar_tags: Dict[str, str], structure=None):
    """Create a DeePMD-kit calculator configured from BCAR tags."""

    root = _root()
    if root.DeePMDCalculator is None:
        raise RuntimeError(
            "DeePMD-kit calculator not available. Install deepmd-kit and dependencies."
        )

    model_reference = root._resolve_backend_model_reference(
        "DEEPMD", bcar_tags.get("MODEL")
    )
    model_path = str(model_reference.value)

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
