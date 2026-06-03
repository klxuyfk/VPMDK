"""Public library API built on top of the legacy backend integrations."""

from __future__ import annotations

import sys
from typing import Any

from .compat.vasp import VaspCompatConfig
from .charge_density import predict_charge_density
from .execution import execute_md, execute_relaxation, execute_single_point
from .models import (
    BackendCapabilities,
    BackendConfig,
    BackendSpec,
    ChargeDensityResult,
    MDConfig,
    MDResult,
    RelaxConfig,
    RelaxResult,
    RunContext,
    SinglePointConfig,
    SinglePointResult,
    coerce_backend_config,
)


def _root():
    return sys.modules["vpmdk_core"]


def _derive_structure_from_atoms(atoms, structure=None):
    """Return an explicit structure or derive one from ASE atoms when possible."""

    if structure is not None:
        return structure
    adaptor = getattr(_root(), "AseAtomsAdaptor", None)
    if adaptor is None:
        return None
    get_structure = getattr(adaptor, "get_structure", None)
    if callable(get_structure):
        try:
            return get_structure(atoms)
        except Exception:
            return None
    return None


def _require_backend_config(backend: object) -> BackendConfig:
    """Return a validated public backend config."""

    if not isinstance(backend, BackendConfig):
        raise TypeError(
            "Public Python API backend arguments must be BackendConfig instances."
        )
    return backend


_BASE_CAPABILITIES: dict[str, BackendCapabilities] = {
    "CHGNET": BackendCapabilities(spin=True),
    "MATGL": BackendCapabilities(spin=False),
    "M3GNET": BackendCapabilities(spin=False),
    "MACE": BackendCapabilities(spin=True, fine_tune=True),
    "MATTERSIM": BackendCapabilities(),
    "MATLANTIS": BackendCapabilities(uncertainty=False),
    "EQNORM": BackendCapabilities(fine_tune=True),
    "MATRIS": BackendCapabilities(),
    "ALPHANET": BackendCapabilities(),
    "HIENET": BackendCapabilities(),
    "NEQUIX": BackendCapabilities(fine_tune=True),
    "SEVENNET": BackendCapabilities(spin=False),
    "FLASHTP": BackendCapabilities(spin=False),
    "ALLEGRO": BackendCapabilities(fine_tune=True),
    "NEQUIP": BackendCapabilities(fine_tune=True),
    "ORB": BackendCapabilities(),
    "UPET": BackendCapabilities(fine_tune=True),
    "TACE": BackendCapabilities(spin=True, fine_tune=True),
    "EQUFLASH": BackendCapabilities(),
    "EQUIFORMER_V3": BackendCapabilities(fine_tune=True),
    "FAIRCHEM": BackendCapabilities(fine_tune=True),
    "FAIRCHEM_V2": BackendCapabilities(fine_tune=True),
    "ESEN": BackendCapabilities(fine_tune=True),
    "FAIRCHEM_V1": BackendCapabilities(fine_tune=True),
    "GRACE": BackendCapabilities(fine_tune=True),
    "DEEPMD": BackendCapabilities(fine_tune=True),
}

_DEFAULT_MODEL_ATTRS: dict[str, str] = {
    "EQNORM": "DEFAULT_EQNORM_MODEL",
    "MATRIS": "DEFAULT_MATRIS_MODEL",
    "ALPHANET": "DEFAULT_ALPHANET_MODEL",
    "HIENET": "DEFAULT_HIENET_MODEL",
    "NEQUIX": "DEFAULT_NEQUIX_MODEL",
    "SEVENNET": "DEFAULT_SEVENNET_MODEL",
    "FLASHTP": "DEFAULT_SEVENNET_MODEL",
    "ORB": "DEFAULT_ORB_MODEL",
    "FAIRCHEM": "DEFAULT_FAIRCHEM_MODEL",
    "FAIRCHEM_V2": "DEFAULT_FAIRCHEM_MODEL",
    "ESEN": "DEFAULT_FAIRCHEM_MODEL",
    "GRACE": "DEFAULT_GRACE_MODEL",
}

_STRUCTURE_INPUT_BACKENDS = frozenset({"ALPHANET", "DEEPMD"})


def _backend_available(name: str) -> bool:
    """Return whether the named backend appears usable in the current environment."""

    root = _root()
    checks = {
        "CHGNET": lambda: root.CHGNetCalculator is not None,
        "MATGL": lambda: root.M3GNetCalculator is not None,
        "M3GNET": lambda: root.M3GNetCalculator is not None,
        "MACE": lambda: root.MACECalculator is not None,
        "MATTERSIM": lambda: root.MatterSimCalculator is not None,
        "MATLANTIS": lambda: (
            root.MatlantisEstimator is not None
            and root.MatlantisASECalculator is not None
            and root.EstimatorCalcMode is not None
        ),
        "EQNORM": lambda: root.EqnormCalculator is not None,
        "MATRIS": lambda: root.MatRISCalculator is not None,
        "ALPHANET": lambda: root.AlphaNetCalculator is not None,
        "HIENET": lambda: root.HIENetCalculator is not None,
        "NEQUIX": lambda: root.NequixCalculator is not None,
        "SEVENNET": lambda: root.SevenNetCalculator is not None,
        "FLASHTP": lambda: root._is_sevennet_flash_available(),
        "ALLEGRO": lambda: root.NequIPCalculator is not None,
        "NEQUIP": lambda: root.NequIPCalculator is not None,
        "ORB": lambda: (
            root.ORBCalculator is not None and root.ORB_PRETRAINED_MODELS is not None
        ),
        "UPET": lambda: root.UPETCalculator is not None,
        "TACE": lambda: root.TACEAseCalc is not None,
        "EQUFLASH": lambda: (
            root.SevenNetCalculator is not None and root._is_sevennet_flash_available()
        ),
        "EQUIFORMER_V3": lambda: root._is_equiformer_v3_available(),
        "FAIRCHEM": lambda: root.FAIRChemCalculator is not None,
        "FAIRCHEM_V2": lambda: root.FAIRChemCalculator is not None,
        "ESEN": lambda: root.FAIRChemCalculator is not None,
        "FAIRCHEM_V1": lambda: (
            root._get_fairchem_v1_calculator_cls() is not None
            or root._get_fairchem_v1_predictor_cls() is not None
        ),
        "GRACE": lambda: root.TPCalculator is not None and root.grace_fm is not None,
        "DEEPMD": lambda: root.DeePMDCalculator is not None,
    }
    checker = checks.get(name, lambda: True)
    try:
        return bool(checker())
    except Exception:
        return False


def _resolve_backend_capabilities(config: BackendConfig) -> BackendCapabilities:
    """Return capabilities for one backend config after applying config-specific hints."""

    base = _BASE_CAPABILITIES.get(config.mlp, BackendCapabilities())
    metadata = dict(base.metadata)
    if config.mlp == "MATRIS":
        task = str(config.options.get("MATRIS_TASK", "efs")).strip().lower()
        if task == "e":
            return BackendCapabilities(
                energy=True,
                forces=False,
                stress=False,
                spin=base.spin,
                fine_tune=base.fine_tune,
                uncertainty=base.uncertainty,
                metadata={**metadata, "matris_task": task},
            )
        if task == "ef":
            return BackendCapabilities(
                energy=True,
                forces=True,
                stress=False,
                spin=base.spin,
                fine_tune=base.fine_tune,
                uncertainty=base.uncertainty,
                metadata={**metadata, "matris_task": task},
            )
        metadata["matris_task"] = task
    return BackendCapabilities(
        energy=base.energy,
        forces=base.forces,
        stress=base.stress,
        spin=base.spin,
        fine_tune=base.fine_tune,
        uncertainty=base.uncertainty,
        metadata=metadata,
    )


def list_backends() -> list[BackendSpec]:
    """Return all known backend entrypoints and their capability metadata."""

    root = _root()
    names = sorted(
        {
            *getattr(root, "_CALCULATOR_BUILDERS", {}).keys(),
            *getattr(root, "_SIMPLE_CALCULATORS", {}).keys(),
        }
    )
    specs: list[BackendSpec] = []
    for name in names:
        default_model_attr = _DEFAULT_MODEL_ATTRS.get(name)
        default_model = (
            getattr(root, default_model_attr, None) if default_model_attr is not None else None
        )
        config = BackendConfig(mlp=name)
        specs.append(
            BackendSpec(
                name=name,
                default_model=default_model,
                supports_structure_input=name in _STRUCTURE_INPUT_BACKENDS,
                capabilities=_resolve_backend_capabilities(config),
                available=_backend_available(name),
            )
        )
    return specs


def get_backend_capabilities(
    config_or_name: BackendConfig | str,
    **backend_kwargs: Any,
) -> BackendCapabilities:
    """Return resolved capability metadata for one backend selection."""

    if isinstance(config_or_name, BackendConfig):
        config = config_or_name.with_options(**backend_kwargs) if backend_kwargs else config_or_name
    else:
        config = BackendConfig(mlp=config_or_name, options=backend_kwargs)
    return _resolve_backend_capabilities(config)


def build_calculator(
    backend: BackendConfig,
    *,
    structure=None,
):
    """Build an ASE calculator from ``BackendConfig``."""

    config = coerce_backend_config(_require_backend_config(backend))
    return _root()._build_calculator_from_tags(config.to_legacy_tags(), structure=structure)


def get_calculator(
    backend: BackendConfig,
    *,
    structure=None,
):
    """Return an ASE calculator from ``BackendConfig``."""

    return build_calculator(backend, structure=structure)


def single_point(
    atoms,
    backend: BackendConfig | None = None,
    *,
    calculator=None,
    structure=None,
    config: SinglePointConfig | None = None,
    observer=None,
    compatibility: VaspCompatConfig | None = None,
) -> SinglePointResult:
    """Run a single-point evaluation using either a supplied or constructed calculator."""

    if calculator is None:
        if backend is None:
            raise ValueError("single_point() requires either calculator=... or backend=...")
        structure = _derive_structure_from_atoms(atoms, structure)
        calculator = build_calculator(backend, structure=structure)
    config = config or SinglePointConfig()
    context = RunContext(
        mode="single_point",
        ibrion=config.effective_ibrion,
        isif=config.effective_isif,
        vasp_compat=compatibility,
    )
    return execute_single_point(
        atoms,
        calculator,
        config=config,
        observer=observer,
        context=context,
    )


def relax(
    atoms,
    backend: BackendConfig | None = None,
    *,
    calculator=None,
    structure=None,
    config: RelaxConfig | None = None,
    steps: int = 200,
    fmax: float = 0.02,
    relax_cell: bool = False,
    pressure_kbar: float | None = None,
    energy_tolerance: float | None = None,
    observer=None,
    compatibility: VaspCompatConfig | None = None,
) -> RelaxResult:
    """Run a relaxation using the stable library API."""

    if calculator is None:
        if backend is None:
            raise ValueError("relax() requires either calculator=... or backend=...")
        structure = _derive_structure_from_atoms(atoms, structure)
        calculator = build_calculator(backend, structure=structure)
    config = config or RelaxConfig(
        steps=steps,
        fmax=fmax,
        relax_cell=relax_cell,
        pressure_kbar=pressure_kbar,
        energy_tolerance=energy_tolerance,
    )
    context = RunContext(
        mode="relax",
        ibrion=config.effective_ibrion,
        isif=config.effective_stress_isif,
        vasp_compat=compatibility,
    )
    return execute_relaxation(
        atoms,
        calculator,
        config=config,
        observer=observer,
        context=context,
    )


def md(
    atoms,
    backend: BackendConfig | None = None,
    *,
    calculator=None,
    structure=None,
    config: MDConfig | None = None,
    temperature: float = 300.0,
    steps: int = 1000,
    timestep: float = 1.0,
    thermostat: str = "nve",
    temperature_end: float | None = None,
    thermostat_kwargs: dict[str, float] | None = None,
    smass: float | None = None,
    observer=None,
    compatibility: VaspCompatConfig | None = None,
) -> MDResult:
    """Run molecular dynamics using the stable library API."""

    if calculator is None:
        if backend is None:
            raise ValueError("md() requires either calculator=... or backend=...")
        structure = _derive_structure_from_atoms(atoms, structure)
        calculator = build_calculator(backend, structure=structure)
    config = config or MDConfig(
        steps=steps,
        temperature=temperature,
        timestep_fs=timestep,
        thermostat=thermostat,
        temperature_end=temperature_end,
        thermostat_kwargs=dict(thermostat_kwargs or {}),
        smass=smass,
    )
    context = RunContext(
        mode="md",
        ibrion=0,
        isif=config.effective_isif,
        potim=config.timestep_fs,
        mdalgo=config.effective_mdalgo,
        vasp_compat=compatibility,
    )
    return execute_md(
        atoms,
        calculator,
        config=config,
        observer=observer,
        context=context,
    )
