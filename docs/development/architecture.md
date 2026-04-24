# Architecture

## Package Layout

The project has three layers:

1. Compatibility shims
   - `vpmdk.py`
   - `src/vpmdk.py`
2. Public package
   - `src/vpmdk_core/`
3. Tests and runnable examples
   - `tests/`
   - `examples/`

The root and `src/` shims both re-export `vpmdk_core`, so user code can import
`vpmdk` while the package implementation remains under `vpmdk_core`.

## Main Subpackages

| Path | Responsibility |
|------|----------------|
| `vpmdk_core/api.py` | stable public object-based API |
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

## Execution Flow

### CLI

`vpmdk_core.main()`:

1. parses `--dir`
2. reads `INCAR` and `BCAR`
3. warns about unsupported or ignored inputs
4. selects NEB, single-point, relaxation, or MD mode
5. builds the backend calculator
6. routes into `run_single_point`, `run_relaxation`, `run_md`, or `run_neb_images`
7. optionally runs charge-density prediction and writes `CHGCAR`

The CLI always opts into compatibility observers.

`run_neb_images` handles VTST-style numbered image directories. For
`NSW > 0`, `IBRION != 0`, and `ICHAIN=0` or unset, it builds one ASE `NEB`
object across all images and optimizes the moving images with spring-coupled
band forces. Single-point and MD NEB layouts remain independent per-image
compatibility workflows.

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
- `tests/integration/`: backend-dependent end-to-end checks

The documentation in `docs/` should track those behavioral contracts rather than
older one-off notes.
