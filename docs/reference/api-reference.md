# API Reference

## Module-Level Entry Points

All public symbols are re-exported through `vpmdk`.

## Calculator Construction

### `get_calculator(...)`

Builds and returns an ASE calculator from:

- a `BackendConfig`

Accepted high-level arguments:

| Argument | Meaning | Default |
|----------|---------|---------|
| `backend` | `BackendConfig` | required |
| `structure` | optional pymatgen structure for backends that benefit from it | `None` |

### `build_calculator(...)`

Same purpose as `get_calculator(...)`, but intended as the explicit new API
entry point built on top of `BackendConfig`.

## Execution Functions

### `single_point(atoms, ...) -> SinglePointResult`

Runs one energy/forces evaluation, plus stress when exposed by the calculator.

Key parameters:

| Parameter | Meaning | Default |
|-----------|---------|---------|
| `atoms` | ASE atoms object | required |
| `backend` | `BackendConfig` | `None` |
| `calculator` | prebuilt calculator or calculator wrapper | `None` |
| `structure` | optional pymatgen structure | `None` |
| `config` | `SinglePointConfig` | `SinglePointConfig()` |
| `observer` | one observer or iterable of observers | `None` |
| `compatibility` | `vpmdk.compat.vasp.VaspCompatConfig` | `None` |

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

- compatibility metadata uses `ISIF=2` when `relax_cell=False`
- compatibility metadata uses `ISIF=3` when `relax_cell=True`

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
- `nose_hoover` and `nose_hoover_chain` require positive `temperature` and
  `temperature_end` values
- `thermostat_kwargs` accepts thermostat-specific keys such as
  `ANDERSEN_PROB`, `LANGEVIN_GAMMA`, `CSVR_PERIOD`, `NHC_NCHAINS`, and
  `NHC_PERIOD`

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

Moved under `vpmdk.compat.vasp.determine_vasp_fft_grid(...)`.

Returns the fine FFT grid derived from VASP-like `INCAR` tags.

### `write_chgcar(path, atoms, density, spin_density=None) -> None`

Moved under `vpmdk.compat.vasp.write_chgcar(...)`.

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
- `with_options(...)`

### `SinglePointConfig`

| Field | Meaning | Default |
|-------|---------|---------|
| `compat` | `vpmdk.compat.vasp.VaspSinglePointConfig \| None` | `None` |

### `RelaxConfig`

| Field | Default | Notes |
|-------|---------|-------|
| `steps` | `200` | non-negative integer |
| `fmax` | `0.02` | force criterion in eV/Ang |
| `relax_cell` | `False` | upgrades default `isif` to `3` |
| `pressure_kbar` | `None` | mapped to ASE scalar pressure |
| `energy_tolerance` | `None` | ionic `delta E` stop criterion |
| `compat` | `vpmdk.compat.vasp.VaspRelaxConfig \| None` | advanced compatibility metadata |

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
| `compat` | `vpmdk.compat.vasp.VaspMDConfig \| None` |

Computed property:

- `effective_mdalgo`: explicit `mdalgo` when set, otherwise the value derived
  from `thermostat`

Nose-Hoover chain notes:

- `NHC_PERIOD` is interpreted in MD steps and multiplied by `timestep_fs`
  before ASE receives the damping time
- `NHC_NCHAINS=0` is VASP's NVE switch-off mode; use `thermostat="nve"` or
  `MDALGO=0` instead

## Resident Server Client

### `VPMDKClient(socket_path=None, connect_timeout=2.0)`

Synchronous POSIX Unix-socket client used by the server-mode CLI. `socket_path`
uses the same resolution order as the CLI: explicit argument,
`VPMDK_SOCKET`, then the per-user default. Positive method timeouts bound the
entire protocol exchange, including connection and request transmission; they
do not begin only after the request has been sent.

Use `from vpmdk_client import VPMDKClient` in orchestration-only processes. That
module depends only on the Python standard library and does not import
`vpmdk_core` or ML packages. The historical
`from vpmdk import VPMDKClient` spelling remains available but loads the full
calculation API.

Methods:

#### `run(workdir=".", *, timeout=0.0, log_callback=None, event_callback=None)`

Converts `workdir` to an absolute path, submits one calculation, and blocks for
its terminal event. A positive `timeout` is a single deadline covering connect,
send, and receive; `timeout=0.0` means no request deadline after the independent
connection timeout. `log_callback` receives streamed stdout lines.
`event_callback` receives every accepted, log, heartbeat, and terminal protocol
object. The request includes the caller's current directory for one-shot
equivalence of relative environment-provided charge paths. Returns the
successful `done` object.

A single stdout line larger than the 1 MiB protocol event limit is transported
as multiple continuation events, but is reassembled before one
`log_callback` call. `event_callback` still receives each raw continuation
event. Oversized remote exception text is marked and truncated, but still
raises `RemoteCalculationError` or `RemoteBackendMismatch` rather than
`ProtocolError`.

Missing or invalid VASP-style inputs raise `RemoteInputError`; the CLI maps
that exception to exit code 1. Backend execution failures remain
`RemoteCalculationError` and exit code 2.

A client timeout or disconnect does not cancel a job already accepted by the
server.

#### `status(*, timeout=2.0)`

Returns the status object, including state, backend identity, PID, uptime,
completed/failed counts, queue length, protocol version, and current work
directory when busy.

#### `stop(*, force=False, timeout=60.0)`

Requests shutdown and waits for socket removal. Returns the server's accepted
shutdown object. `force=True` rejects queued work and disconnects the active
client, but server teardown still waits for an active executor to return because
Python threads and GPU kernels cannot be cancelled safely. `timeout=0.0` does
not wait for socket removal. Both graceful and force shutdown become observable
only after the acknowledgement has been sent. `force` must be a Python `bool`;
other types raise `TypeError` before a connection is attempted.

Exception hierarchy:

- `VPMDKClientError`: base class
  - `ServerConnectionError`: unavailable server or lost connection
    - `ProtocolError`: malformed or incompatible peer response
  - `ClientTimeoutError`: client-side deadline expired
  - `RemoteCalculationError`: calculation failed in the server; exposes
    `.traceback`
    - `RemoteInputError`: required workdir input is missing or invalid
    - `RemoteBackendMismatch`: request BCAR conflicts with the resident backend

All methods are synchronous. A client instance does not own or start the
server, and no method falls back to one-shot execution.

See [Server Mode](../user-guide/server-mode.md) for lifecycle, queueing,
configuration authority, protocol, and security behavior.

## Server Embedding Primitives

The CLI is the supported default for server lifecycle management. Advanced
integrations can use the following re-exported primitives.

### `VPMDKServer(...)`

```python
VPMDKServer(
    socket_path,
    calculator,
    backend_tags,
    *,
    backend_base_dir,
    idle_timeout=0.0,
    heartbeat_interval=30.0,
    pidfile=None,
    log_file=None,
    executor=None,
)
```

Owns one prebuilt calculator and one FIFO worker. `serve_forever()` binds the
socket, invokes an optional `ready_callback`, and blocks until shutdown.
`status()` returns the in-process status object, while `request_stop(force=False)`
changes lifecycle state. Signal handler installation is explicit through
`install_signal_handlers()` and `restore_signal_handlers()` and only operates
from the main thread.

`backend_tags` and `backend_base_dir` define the authoritative resident
identity used to reject conflicting request BCAR settings. Applications using
this class directly are responsible for calculator construction, process
supervision, POSIX availability, and calling `serve_forever()`.
For `MLP=DEEPMD`, `backend_tags` must include a non-empty
`DEEPMD_TYPE_MAP`; the server rejects structure-derived resident type maps even
when the calculator was constructed by the embedding application.

Related helpers:

- `default_socket_path()`
- `resolve_socket_path(explicit=None)`
- `backend_identity(tags, *, base_dir)`
- `validate_request_backend(resident, request_tags, *, request_base_dir)`
- `PROTOCOL_VERSION`

### `vpmdk.compat.vasp.VaspCompatConfig`

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
