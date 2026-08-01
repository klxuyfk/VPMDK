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

The server is deliberately process-local and serial:

1. `serve_cli()` resolves and protects the socket path.
2. The startup BCAR is parsed and the calculator is fully constructed.
3. Foreground mode enters `serve_forever()` directly; daemon mode retains a
   readiness pipe so the original process observes actual model-load success.
4. `VPMDKServer` binds the Unix socket, starts one worker, then accepts control
   connections in lightweight handler threads.
5. Handler reads may proceed concurrently, but an accept-sequence gate makes
   operations observable in connection-accept order. A run handler validates
   protocol shape and absolute workdir, publishes queue position, and hands
   connection ownership to the FIFO worker.
6. The worker parses request BCAR once, checks that snapshot's explicit
   construction tags against the resident identity, resets calculator results,
   and passes the same snapshot to `run_workdir()`.
7. Request stdout becomes streamed `log` events; a heartbeat thread emits
   liveness events during long calculations.
8. Terminal success or failure updates counters and closes that request
   connection without replacing the model.

`_EventSender` enforces the 1 MiB limit on encoded NDJSON before every send.
Oversized log lines become multiple lossless log events carrying `continued`
metadata; the client rejoins them into one logical line before invoking its
log callback or printing CLI output. Oversized terminal errors retain their
event type, error code, and useful prefixes while marking truncated
error/traceback text. This keeps client exit-code classification independent
of exception or output size.

Key invariants are covered in `test_server.py`:

- the socket is unavailable until model construction completes
- one server has one calculator and never overlaps evaluations
- status remains responsive while the worker is busy
- request failures and backend mismatches do not terminate later work
- a client timeout does not cancel an accepted calculation
- graceful stop drains work, while force stop rejects queued work and
  disconnects the active client but joins its executor before teardown
- socket cleanup is inode guarded; daemon pidfile writes and cleanup require
  matching PID/socket ownership metadata
- stale cleanup treats any socket accepting a connection as owned, without
  depending on response timing or protocol compatibility
- stop acknowledgement is sent while holding the enqueue lock, before either
  graceful or force shutdown is exposed to the serve loop

`test_client_entry.py` runs each client subcommand in an isolated interpreter
and asserts that `vpmdk_core`, torch, e3nn, MACE, CHGNet, ASE, pymatgen, and
NumPy are absent from `sys.modules`. Performance is checked manually rather
than with a scheduler-sensitive wall-clock assertion in pytest.

The worker claims a queued job and marks it busy while holding the same
enqueue lock used by acceptance and shutdown checks. This atomic transition is
required: an accepted job must never appear as neither queued nor active to a
concurrent graceful-stop or idle-timeout decision.

The startup process environment is authoritative because calculator builders
may consult CUDA visibility, caches, credentials, and backend-specific
variables. Clients do not transfer their environment. A run request does carry
its absolute workdir and caller cwd; the latter is used only as the base for
relative environment-provided charge-density paths.

When `DEVICE` is omitted, server construction performs calculator-device
detection before constructing `backend_identity`. Status metadata, request
validation, and device-dependent defaults such as UPET neighbor-list placement
therefore remain one consistent snapshot.

`BACKEND_CONFIGURATION_TAGS` separates calculator-construction settings from
request settings. Explicit request construction tags must normalize to the
resident value. Omitted tags inherit. Output, charge-density, and numerical
finite-difference controls stay request-scoped. This check prevents a request
BCAR from appearing to select a model option that cannot take effect after the
calculator has already been constructed.

All backend builders and server identity checks call
`_resolve_backend_model_reference`. It is the sole policy boundary that
classifies an omitted MODEL, an existing local path, or a named model. Add new
backend model semantics there rather than branching on `os.path.exists()` in a
builder. `_BACKEND_MODEL_POLICIES` is the exhaustive capability matrix for all
built-in backends: required/optional MODEL, local-only handling, named-model
support, upstream delegation, and default source live together there. Its test
matrix must contain exactly the same backend set. Explicit selectors must
always be forwarded to the chosen loader or constructor and must never fall
through to a no-argument/default construction, except for GRACE's intentional
warning plus effective-default behavior.
Model-returning loader APIs must pass through `_require_loaded_model` so a
`None`, false, or empty result cannot reach a calculator.

`ModelReference.value` is the loader-facing spelling. For a local symlink it
retains the symlink path so sibling config inference remains compatible with
one-shot builders. `ModelReference.identity` stores the canonical real path for
status and resident request comparison. Do not replace `value` with `identity`
before calculator construction. Dynamic selector delegation is an explicit
backend policy: both FAIRChem generations allow path-shaped upstream selectors,
MatterSim delegates only non-path preset names, Nequix delegates names only
through its resolver when registry metadata is unavailable, and Matlantis
treats every version as opaque. MACE and other local-only backends reject
every explicit missing value rather than risking a silent default-model
substitution. GRACE's resolver supplies both its installed default and explicit
name normalization, so no separate default override may be added outside the
policy table.

Backends registered through `_SIMPLE_CALCULATORS` use the generic optional
path-or-name policy. Their builder forwards every explicit MODEL positionally
and calls the no-argument constructor only when MODEL is omitted. This supports
plugin preset names without restoring silent default substitution; missing
path-shaped values still fail before construction.

`_canonical_configuration` folds documented names for the same construction
option into one comparison key, using the same precedence as the backend
builder. Add new builder aliases there as well as to the parser/reference and
cover both startup-name/request-name directions with server regression tests.
`backend_identity` retains both the explicitly supplied configuration (for
status reporting) and an effective configuration overlaid with defaults that
VPMDK itself supplies before calculator construction. Request validation uses
the effective form and canonical Python values, including parsed booleans and
numbers, normalized enums, device-dependent UPET policy, and list-like options.
Do not invent defaults owned by third-party constructor implementations.
Startup tags are canonicalized once and the resulting mapping is reused as the
explicit overlay while constructing the effective configuration; keep this
single-pass property when extending identity resolution.

Calculator reset and server device detection share the same wrapper/resolved
calculator candidate helper. Backend-specific nested calculator/model scanning
belongs after that common first step so wrappers are treated consistently.

DeepMD is deliberately stricter in resident mode than in one-shot mode. The
one-shot builder may infer `DEEPMD_TYPE_MAP` from its current structure, but a
server must receive the model-ordered map explicitly and never passes the
startup POSCAR into DeepMD construction. This prevents the calculator from
retaining a request-inappropriate atom-type mapping.

The user-facing lifecycle, security model, and wire format are documented in
[Server Mode](../user-guide/server-mode.md). Keep that contract and the
protocol tests synchronized when changing server behavior.

`run_neb_images` handles VTST-style numbered image directories. For
`NSW > 0`, `IBRION > 0`, and `ICHAIN=0` or unset, it builds one ASE `NEB`
object across all images and optimizes the moving images with spring-coupled
band forces. Single-point and MD NEB layouts remain independent per-image
compatibility workflows.

NEB construction inspects the installed ASE signature. One-shot calculations
do not pass `allow_shared_calculator`, preserving compatibility with older ASE
releases allowed by package metadata. Resident NEB passes the option when
supported. Otherwise each image receives a distinct proxy object that delegates
to the same calculator; ASE's identity guard is satisfied while VPMDK's serial
worker guarantees that the resident model is never evaluated concurrently.

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
