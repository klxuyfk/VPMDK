# Architecture

## Package Layout

The project has four layers:

1. Compatibility shims
   - `vpmdk.py`
   - `src/vpmdk.py`
2. Lightweight server client
   - `src/vpmdk_entry.py`
   - `src/vpmdk_client.py`
   - `src/vpmdk_protocol.py`
3. Public package
   - `src/vpmdk_core/`
4. Tests and runnable examples
   - `tests/`
   - `examples/`

The root and `src/` shims both re-export `vpmdk_core`, so user code can import
`vpmdk` while the package implementation remains under `vpmdk_core`.

## Main Subpackages

| Path | Responsibility |
|------|----------------|
| `vpmdk_entry.py` | import-light console dispatch before the ML runtime loads |
| `vpmdk_client.py` | standard-library-only NDJSON client and client CLI |
| `vpmdk_protocol.py` | protocol constants and socket-path resolution shared by both peers |
| `vpmdk_core/api.py` | stable public object-based API |
| `vpmdk_core/backend_common.py` | shared MODEL classification, loader guards, device/tag helpers |
| `vpmdk_core/models.py` | config/result dataclasses and thermostat helpers |
| `vpmdk_core/compat/vasp.py` | VASP-only compatibility config and helpers |
| `vpmdk_core/runtime/registry.py` | backend registry and calculator construction |
| `vpmdk_core/backends/` | backend-specific builders |
| `vpmdk_core/settings/incar.py` | `INCAR` parsing and execution settings |
| `vpmdk_core/io/inputs.py` | `BCAR`, `POSCAR`, `POTCAR`, and `MAGMOM` helpers |
| `vpmdk_core/execution.py` | pure execution layer for single-point, relax, and MD |
| `vpmdk_core/observers.py` | observer interfaces and compatibility bridge |
| `vpmdk_core/io/vasp_compat.py` | `OUTCAR`, `OSZICAR`, and `vasprun.xml` writers |
| `vpmdk_core/io/trajectories.py` | `XDATCAR` and LAMMPS trajectory writers |
| `vpmdk_core/charge_density.py` | FFT-grid logic and charge-density subprocess runners |
| `vpmdk_core/cli.py` | VASP-style CLI orchestration |
| `vpmdk_core/server.py` | resident calculator, Unix socket, FIFO worker, and lifecycle |
| `vpmdk_core/client.py` | compatibility exports for `vpmdk_client` |

## Execution Flow

### CLI

The installed console script enters `vpmdk_entry.main()` first. `run`,
`status`, and `stop` load only `vpmdk_client` and `vpmdk_protocol`; they never
import `vpmdk_core`, ASE, pymatgen, or an ML backend. Other arguments, including
legacy one-shot execution and `serve`, load `vpmdk_core.main()`. Direct source
wrappers apply the same executable dispatch while retaining their historical
`import vpmdk` compatibility behavior.

`vpmdk_core.main()` preserves the legacy parser unless the first argument is a
known server subcommand. Legacy execution delegates to `run_workdir()`:

1. parses `--dir`
2. reads `INCAR` and `BCAR`
3. warns about unsupported or ignored inputs
4. selects NEB, single-point, relaxation, or MD mode
5. builds the backend calculator, or resets and reuses an injected calculator
6. routes into `run_single_point`, `run_force_constants`, `run_relaxation`,
   `run_md`, or `run_neb_images`
7. optionally runs charge-density prediction and writes `CHGCAR`

The CLI always opts into compatibility observers.

Server mode constructs one calculator before binding its socket, then calls the
same `run_workdir(workdir, calculator=resident_calculator)` function from a
single FIFO worker. Socket handler threads only parse requests, report status,
and enqueue work. This keeps calculation-related process globals confined to
the one execution thread while allowing `status` and `stop` to respond during a
long calculation.

### Resident Server

Server mode constructs one calculator at startup and passes it to the same
`run_workdir()` execution path used by the one-shot CLI. `VPMDKServer` owns one
FIFO worker; lightweight socket handlers enqueue calculations and keep
`status` and `stop` responsive while work is running.

The main invariants are:

- one server evaluates only one calculation at a time
- startup environment and backend-construction settings remain authoritative
- request settings and output state are isolated between calculations
- explicit request construction settings must match the resident calculator
- client timeout or disconnect does not cancel accepted work
- graceful shutdown drains accepted work; force shutdown rejects queued work
- socket and pidfile cleanup verifies ownership before removing paths

`vpmdk_client` provides the standard-library-only client path so orchestration
processes do not import the calculation runtime. Backend identity and model
resolution are shared with one-shot builders; new backend construction options
must be added to both the builder and resident-configuration comparison.

Resident NEB calculations reuse the loaded calculator serially across images.
When ASE does not support `allow_shared_calculator`, distinct delegate objects
preserve ASE compatibility while forwarding to the same resident calculator.

User-facing lifecycle, configuration, security, and timeout behavior belongs in
[Server Mode](../user-guide/server-mode.md). Keep this architecture page focused
on component ownership and invariants.

### Public API

`vpmdk.single_point`, `vpmdk.relax`, and `vpmdk.md`:

1. coerce backend config
2. derive a pymatgen structure when useful
3. build or accept a calculator
4. create a `RunContext`
5. call the pure execution layer in `execution.py`

The public API is deliberately free of implicit filesystem side effects.

## Compatibility Output Model

VASP-like outputs are implemented as observers:

- `VaspCompatObserver`
- `PrintProgressObserver`

That split is important:

- execution logic stays pure and reusable
- file writing is optional and attached from the CLI or explicit Python code

Compatibility state is stored in `_VaspCompatRecorder`.

The `IBRION=5`/`6`/`7`/`8` force-constants compatibility path is documented separately
in [VASP Force-Constants Compatibility](force-constants.md), including the
finite-difference formula and VASP `dynmat` Hessian convention.

## Backend Registry

`runtime/registry.py` contains:

- `_SIMPLE_CALCULATORS`
- `_CALCULATOR_BUILDERS`

Backends are selected by `MLP` / `NNP`, normalized to uppercase, then routed to
the corresponding builder in `backends/`.

Most builders consume a legacy string-based BCAR mapping because the project
grew from the CLI first. `BackendConfig` is now the primary public object; the
BCAR mapping path remains an internal/CLI compatibility mechanism that still
feeds those builders.

## Relaxation and MD Semantics

Relaxation and MD are implemented twice conceptually:

- pure execution in `execution.py`
- compatibility wrappers in `runtime/relax.py` and `runtime/md.py`

The wrappers mostly translate VASP-like settings into the public API and attach
observers.

## Charge-Density Design

Charge-density inference is intentionally isolated:

- the main process prepares geometry and grid metadata
- a backend-specific subprocess runner performs the actual inference
- the result is loaded back into the main process and optionally written as `CHGCAR`

This keeps optional heavy dependencies out of the main runtime environment.

## Testing Layout

Tests are organized by behavior:

- `test_api.py`: public API guarantees
- `test_main.py`: CLI and compatibility behavior
- `test_relaxation.py`, `test_md.py`: execution semantics and file writers
- `test_backends.py`: backend builder behavior
- `test_charge_density.py`: FFT-grid and charge-density logic
- `test_server.py`: server protocol, concurrency, isolation, and CLI lifecycle
- `tests/integration/`: backend-dependent end-to-end checks

The documentation in `docs/` should track those behavioral contracts rather than
older one-off notes.
