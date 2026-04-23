# API Reference

## Module-Level Entry Points

All public symbols are re-exported through `vpmdk`.

## Calculator Construction

### `get_calculator(...)`

Builds and returns an ASE calculator from:

- a `BackendConfig`
- a BCAR-like mapping
- keyword arguments such as `mlp=`, `model=`, `device=`

Accepted high-level arguments:

| Argument | Meaning | Default |
|----------|---------|---------|
| `config_or_tags` | `BackendConfig` or BCAR-like mapping | `None` |
| `structure` | optional pymatgen structure for backends that benefit from it | `None` |
| `mlp` | backend name | `CHGNET` when building from keyword args |
| `model` | checkpoint path or named model | `None` |
| `device` | backend device hint | `None` |
| `options` | mapping of backend-specific options | `None` |
| `**backend_kwargs` | backend-specific options normalized to uppercase keys | none |

### `build_calculator(...)`

Same purpose as `get_calculator(...)`, but intended as the explicit new API
entry point built on top of `BackendConfig`.

## Execution Functions

### `single_point(atoms, ...) -> SinglePointResult`

Runs one energy/force/stress evaluation.

Key parameters:

| Parameter | Meaning | Default |
|-----------|---------|---------|
| `atoms` | ASE atoms object | required |
| `calculator` | prebuilt calculator or calculator wrapper | `None` |
| `backend` | `BackendConfig` or BCAR-like mapping | `None` |
| `mlp`, `model`, `device` | backend-selection convenience args | `None` |
| `structure` | optional pymatgen structure | `None` |
| `config` | `SinglePointConfig` | `SinglePointConfig()` |
| `observer` | one observer or iterable of observers | `None` |
| `vasp_compat` | `VaspCompatConfig` | `None` |

### `relax(atoms, ...) -> RelaxResult`

Runs a BFGS-based geometry optimization.

Convenience arguments when `config` is omitted:

| Parameter | Default |
|-----------|---------|
| `steps` | `200` |
| `fmax` | `0.02` |
| `relax_cell` | `False` |
| `pressure_kbar` | `None` |
| `energy_tolerance` | `None` |

Derived defaults:

- `isif=2`, `stress_isif=2` when `relax_cell=False`
- `isif=3`, `stress_isif=3` when `relax_cell=True`

Special semantics:

- `steps=0` is valid and returns one fallback step with `converged=False`

### `md(atoms, ...) -> MDResult`

Runs molecular dynamics through ASE.

Convenience arguments when `config` is omitted:

| Parameter | Default |
|-----------|---------|
| `temperature` | `300.0` |
| `steps` | `1000` |
| `timestep` | `1.0` fs |
| `thermostat` | `nve` |
| `temperature_end` | `None` |
| `thermostat_kwargs` | `{}` |
| `smass` | `None` |

Special semantics:

- `steps=0` is valid and behaves like a single-point evaluation of the current
  structure without advancing dynamics
- `advanced=False` marks that fallback step

## Charge-Density Functions

### `predict_charge_density(...) -> ChargeDensityResult`

Parameters are grouped into:

- grid selection: `grid_shape`, `incar`, `reference`
- backend selection: `backend`, `model_path`, `device`, `source_dir`, `python_executable`
- ChargE3Net options: `cutoff`, `num_interactions`, `num_neighbors`, `mul`,
  `lmax`, `basis`, `num_basis`, `spin`
- generic batching: `max_probes_per_batch`
- DeepCDP options: `metadata_path`, `charge_species`, `soap_rcut`,
  `soap_nmax`, `soap_lmax`, `soap_sigma`, `soap_periodic`, `activation`,
  `weighting`

### `charge_density(...)`

Backward-compatible alias of `predict_charge_density(...)`.

### `determine_vasp_fft_grid(reference, incar) -> tuple[int, int, int]`

Returns the fine FFT grid derived from VASP-like `INCAR` tags.

### `write_chgcar(path, atoms, density, spin_density=None) -> None`

Writes a VASP-like `CHGCAR` from one or two 3D arrays.

## Capability Helpers

### `list_backends() -> list[BackendSpec]`

Returns known backend entry points, their default models when declared, whether
they support explicit structure input, their capability metadata, and a
best-effort `available` flag based on import/runtime checks.

### `get_backend_capabilities(config_or_name, **backend_kwargs) -> BackendCapabilities`

Returns capability metadata, including configuration-sensitive values such as
the reduced force/stress capability of `MATRIS_TASK=e`.

## Config Objects

### `BackendConfig`

Fields:

| Field | Type | Default |
|-------|------|---------|
| `mlp` | `str` | `CHGNET` |
| `model` | `str | None` | `None` |
| `device` | `str | None` | `None` |
| `options` | `dict[str, Any]` | `{}` |

Behavior:

- `mlp` is uppercased and must not be empty
- option keys are normalized to uppercase BCAR-style names
- `to_legacy_tags()` converts values into the existing string-based internal tag format

Class helpers:

- `from_mapping(...)`
- `from_bcar(...)`
- `with_options(...)`

### `SinglePointConfig`

| Field | Meaning | Default |
|-------|---------|---------|
| `isif` | stress/output mode hint | `None` |

### `RelaxConfig`

| Field | Default | Notes |
|-------|---------|-------|
| `steps` | `200` | non-negative integer |
| `fmax` | `0.02` | force criterion in eV/Ang |
| `relax_cell` | `False` | upgrades default `isif` to `3` |
| `pressure_kbar` | `None` | mapped to ASE scalar pressure |
| `energy_tolerance` | `None` | ionic `delta E` stop criterion |
| `isif` | `2` | upgraded to `3` when `relax_cell=True` and still default |
| `stress_isif` | `None` | if unset and `relax_cell=True`, becomes `isif` |
| `ibrion` | `2` | compatibility metadata only |

### `MDConfig`

| Field | Default |
|-------|---------|
| `steps` | `1000` |
| `temperature` | `300.0` |
| `timestep_fs` | `1.0` |
| `thermostat` | `nve` |
| `temperature_end` | `None` |
| `thermostat_kwargs` | `{}` |
| `smass` | `None` |
| `isif` | `0` |
| `mdalgo` | `None` |

Computed property:

- `effective_mdalgo`: explicit `mdalgo` when set, otherwise the value derived
  from `thermostat`

### `VaspCompatConfig`

| Field | Default |
|-------|---------|
| `enabled` | `True` |
| `write_pseudo_scf` | `False` |
| `write_contcar` | `True` |
| `write_xdatcar` | `False` |
| `write_lammps_traj` | `False` |
| `lammps_traj_interval` | `1` |
| `lammps_traj_path` | `lammps.lammpstrj` |
| `neb_mode` | `False` |
| `neb_prev_positions` | `None` |
| `neb_next_positions` | `None` |

### `RunContext`

Observer-facing execution metadata:

- `mode`
- `ibrion`
- `isif`
- `potim`
- `mdalgo`
- `vasp_compat`

## Result Objects

### `SinglePointResult`

Extends `CalculationResult`:

- `atoms`
- `calculator`
- `potential_energy`
- `forces`
- `stress`

### `RelaxResult`

Adds:

- `steps: list[RunStep]`
- `converged: bool | None`

### `MDResult`

Adds:

- `steps: list[RunStep]`

### `ChargeDensityResult`

Fields:

- `atoms`
- `density`
- `grid_shape`
- `backend`
- `spin_density`
- `metadata`

## Metadata Models

### `BackendCapabilities`

Fields:

- `energy`
- `forces`
- `stress`
- `spin`
- `fine_tune`
- `uncertainty`
- `metadata`

### `BackendSpec`

Fields:

- `name`
- `default_model`
- `supports_structure_input`
- `capabilities`
- `available`

## Utility Functions

### `normalize_thermostat_name(value) -> str`

Normalizes aliases such as:

- `velocity_verlet` -> `nve`
- `nosehoover` -> `nose_hoover`
- `nosehooverchain` -> `nose_hoover_chain`
- `csvr` -> `bussi`

### `thermostat_to_mdalgo(value) -> int`

Maps public thermostat names to VASP-style `MDALGO` integers.
