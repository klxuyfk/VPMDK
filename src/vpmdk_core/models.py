"""Public-facing configuration and result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


def _normalize_option_key(key: object) -> str:
    """Return a canonical BCAR-style option key."""

    text = str(key).strip()
    if not text:
        raise ValueError("Backend option keys must not be empty.")
    return text.replace("-", "_").upper()


def _stringify_legacy_value(value: object) -> str:
    """Return a legacy tag value accepted by the existing backend builders."""

    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _normalize_backend_options(
    values: Mapping[str, Any] | None = None,
    **extra_values: Any,
) -> dict[str, Any]:
    """Return backend options with normalized keys."""

    normalized: dict[str, Any] = {}
    for source in (values or {}, extra_values):
        for key, value in source.items():
            normalized[_normalize_option_key(key)] = value
    return normalized


def normalize_thermostat_name(value: str | None) -> str:
    """Return a normalized public thermostat name."""

    if value is None:
        return "nve"
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "verlet": "nve",
        "velocity_verlet": "nve",
        "none": "nve",
        "nose_hoover": "nose_hoover",
        "nosehoover": "nose_hoover",
        "nose_hoover_chain": "nose_hoover_chain",
        "nosehooverchain": "nose_hoover_chain",
        "csvr": "bussi",
    }
    return aliases.get(normalized, normalized)


def thermostat_to_mdalgo(value: str | None) -> int:
    """Return the VASP-style ``MDALGO`` value for a public thermostat name."""

    normalized = normalize_thermostat_name(value)
    mapping = {
        "nve": 0,
        "andersen": 1,
        "nose_hoover": 2,
        "langevin": 3,
        "nose_hoover_chain": 4,
        "bussi": 5,
    }
    if normalized not in mapping:
        supported = ", ".join(sorted(mapping))
        raise ValueError(
            f"Unsupported thermostat {value!r}. Expected one of: {supported}."
        )
    return mapping[normalized]


@dataclass(frozen=True)
class BackendCapabilities:
    """Feature metadata exposed for one backend."""

    energy: bool = True
    forces: bool | None = True
    stress: bool | None = True
    spin: bool | None = None
    fine_tune: bool | None = None
    uncertainty: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendSpec:
    """Describes one accepted backend entrypoint."""

    name: str
    default_model: str | None = None
    supports_structure_input: bool = False
    capabilities: BackendCapabilities = field(default_factory=BackendCapabilities)
    available: bool = True


@dataclass(frozen=True)
class BackendConfig:
    """Filesystem-independent backend selection."""

    mlp: str = "CHGNET"
    model: str | None = None
    device: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mlp = str(self.mlp).strip().upper()
        if not mlp:
            raise ValueError("Backend MLP must not be empty.")
        object.__setattr__(self, "mlp", mlp)
        object.__setattr__(self, "options", _normalize_backend_options(self.options))

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        default_mlp: str = "CHGNET",
    ) -> "BackendConfig":
        """Build backend config from a BCAR-like mapping."""

        normalized = _normalize_backend_options(values)
        mlp_value = normalized.get("MLP", normalized.get("NNP", default_mlp))
        model = normalized.get("MODEL")
        device = normalized.get("DEVICE")
        options = {
            key: value
            for key, value in normalized.items()
            if key not in {"MLP", "NNP", "MODEL", "DEVICE"}
        }
        return cls(mlp=str(mlp_value), model=None if model is None else str(model), device=None if device is None else str(device), options=options)

    @classmethod
    def from_bcar(
        cls,
        values: Mapping[str, Any],
        *,
        default_mlp: str = "CHGNET",
    ) -> "BackendConfig":
        """Backward-compatible alias of :meth:`from_mapping`."""

        return cls.from_mapping(values, default_mlp=default_mlp)

    def with_options(self, **options: Any) -> "BackendConfig":
        """Return a copy with updated backend options."""

        merged = dict(self.options)
        merged.update(_normalize_backend_options(options))
        return BackendConfig(
            mlp=self.mlp,
            model=self.model,
            device=self.device,
            options=merged,
        )

    def to_legacy_tags(self) -> dict[str, str]:
        """Return BCAR-like tags for the existing internal builders."""

        tags: dict[str, str] = {"MLP": self.mlp}
        if self.model is not None:
            tags["MODEL"] = _stringify_legacy_value(self.model)
        if self.device is not None:
            tags["DEVICE"] = _stringify_legacy_value(self.device)
        for key, value in self.options.items():
            tags[key] = _stringify_legacy_value(value)
        return tags


def coerce_backend_config(
    config_or_tags: BackendConfig | Mapping[str, Any] | None = None,
    *,
    mlp: str | None = None,
    model: str | None = None,
    device: str | None = None,
    options: Mapping[str, Any] | None = None,
    **backend_kwargs: Any,
) -> BackendConfig:
    """Return :class:`BackendConfig` from config, BCAR tags, or public kwargs."""

    if isinstance(config_or_tags, BackendConfig):
        config = config_or_tags
    elif config_or_tags is not None:
        config = BackendConfig.from_mapping(config_or_tags)
    else:
        merged_options = _normalize_backend_options(options, **backend_kwargs)
        config = BackendConfig(
            mlp=mlp or "CHGNET",
            model=model,
            device=device,
            options=merged_options,
        )

    if any(value is not None for value in (mlp, model, device)) or options or backend_kwargs:
        merged = dict(config.options)
        merged.update(_normalize_backend_options(options, **backend_kwargs))
        config = BackendConfig(
            mlp=mlp or config.mlp,
            model=model if model is not None else config.model,
            device=device if device is not None else config.device,
            options=merged,
        )
    return config


@dataclass(frozen=True)
class SinglePointConfig:
    """Configuration for a single-point evaluation."""

    isif: int | None = None


@dataclass(frozen=True)
class RelaxConfig:
    """Configuration for a geometry optimization."""

    steps: int = 200
    fmax: float = 0.02
    relax_cell: bool = False
    pressure_kbar: float | None = None
    energy_tolerance: float | None = None
    isif: int = 2
    stress_isif: int | None = None
    ibrion: int = 2


@dataclass(frozen=True)
class MDConfig:
    """Configuration for molecular dynamics."""

    steps: int = 1000
    temperature: float = 300.0
    timestep_fs: float = 1.0
    thermostat: str = "nve"
    temperature_end: float | None = None
    thermostat_kwargs: dict[str, float] = field(default_factory=dict)
    smass: float | None = None
    isif: int | None = 0
    mdalgo: int | None = None

    @property
    def effective_mdalgo(self) -> int:
        """Return the MD algorithm after resolving the public thermostat."""

        if self.mdalgo is not None:
            return int(self.mdalgo)
        return thermostat_to_mdalgo(self.thermostat)


@dataclass(frozen=True)
class VaspCompatConfig:
    """Compatibility-output settings used by the legacy wrappers and CLI."""

    enabled: bool = True
    write_pseudo_scf: bool = False
    write_contcar: bool = True
    write_xdatcar: bool = False
    write_lammps_traj: bool = False
    lammps_traj_interval: int = 1
    lammps_traj_path: str = "lammps.lammpstrj"
    neb_mode: bool = False
    neb_prev_positions: Any = None
    neb_next_positions: Any = None


@dataclass(frozen=True)
class RunContext:
    """Execution metadata shared with observers."""

    mode: str
    ibrion: int
    isif: int | None = None
    potim: float | None = None
    mdalgo: int | None = None
    vasp_compat: VaspCompatConfig | None = None


@dataclass(frozen=True)
class RunStep:
    """One recorded ionic or MD step."""

    index: int
    potential_energy: float
    total_energy: float
    kinetic_energy: float = 0.0
    thermostat_potential: float = 0.0
    thermostat_kinetic: float = 0.0
    temperature: float = 0.0
    sc_time: float = 0.0


@dataclass
class CalculationResult:
    """Common result fields for public execution APIs."""

    atoms: Any
    calculator: Any
    potential_energy: float
    forces: Any | None = None
    stress: Any | None = None


@dataclass
class ChargeDensityResult:
    """Result returned by charge-density prediction APIs."""

    atoms: Any
    density: Any
    grid_shape: tuple[int, int, int]
    backend: str
    spin_density: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SinglePointResult(CalculationResult):
    """Result returned by :func:`vpmdk.single_point`."""


@dataclass
class RelaxResult(CalculationResult):
    """Result returned by :func:`vpmdk.relax`."""

    steps: list[RunStep] = field(default_factory=list)
    converged: bool | None = None


@dataclass
class MDResult(CalculationResult):
    """Result returned by :func:`vpmdk.md`."""

    steps: list[RunStep] = field(default_factory=list)


def run_steps_to_energy_rows(steps: Iterable[RunStep]) -> list[list[float]]:
    """Return CSV rows compatible with the legacy ``energy.csv`` output."""

    return [[float(step.potential_energy)] for step in steps]
