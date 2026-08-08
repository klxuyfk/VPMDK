# Server Mode

Server mode is an optional optimization for directory-based CLI workflows. It
loads one calculator, keeps its model resident in CPU or GPU memory, and runs
many normal VPMDK calculation directories through that calculator. It is most
useful when repeatedly loading a large model costs a meaningful fraction of
each calculation.

The regular one-shot command remains the recommended default:

```bash
vpmdk --dir ./calc-001
```

Client commands (`run`, `status`, and `stop`) start through a
standard-library-only path. They do not initialize ASE, pymatgen, PyTorch, or
the installed model packages in the client process. Model/runtime imports occur
in `serve`, where they can be amortized across requests.

Use server mode deliberately when the workload benefits from model reuse:

| Prefer one-shot mode | Consider server mode |
| --- | --- |
| one or a few calculations | many calculation directories |
| different models or backend options per calculation | the same model, device, and construction options |
| model loading is small relative to calculation time | model loading or VRAM placement is a bottleneck |
| process isolation is more important than startup cost | a long-lived local process fits the workflow |

Server mode is POSIX-only and requires Unix-domain sockets. It is suitable for
Linux, macOS, WSL, local batch scripts, and POSIX job nodes. It does not open a
TCP port and is not a remote execution service.

## Operational Model

One server owns exactly one resident calculator and one FIFO calculation
worker:

```text
calc directories -> Unix socket -> FIFO queue -> resident calculator -> outputs
                                      |
                                      +-> status remains responsive
```

Several clients may submit work concurrently, but a single server evaluates
only one calculation at a time. This prevents concurrent mutation of an ASE
calculator and keeps request-scoped working-directory and output state
isolated. To calculate in parallel, start independent servers with different
socket paths, normally one per GPU or model.

FIFO order is the Unix-socket connection acceptance order. Request bodies may
be read by concurrent handlers, but a later accepted run cannot enter the
calculation queue before an earlier accepted run.

The calculation directory is otherwise the same as a one-shot CLI directory.
It may contain `POSCAR`, `INCAR`, `POTCAR`, `KPOINTS`, and `BCAR`, and the
server writes the usual outputs into that directory. Single points,
relaxations, MD, force-constant modes, and supported NEB layouts all enter the
same `run_workdir()` path used by one-shot execution.

Coupled NEB reuses the resident calculator without loading one model per image.
On current ASE releases VPMDK enables `allow_shared_calculator`. On older ASE
releases, VPMDK attaches a distinct delegating calculator identity to each image
to satisfy ASE's guard while evaluations remain serial and use the same loaded
resident model. One-shot NEB continues to construct independent calculators.

### Randomness is replayed, not continued

Every request restores the process-global NumPy RNG to the state the server
started with, so a repeated request reproduces its earlier result exactly. That
is what makes an `A -> B -> A` sequence deterministic, but it has a consequence
worth knowing before you use the server for sampling:

**Stochastic runs submitted to one server replay the same random stream.** MD
velocity initialisation (`MaxwellBoltzmannDistribution`) and the Langevin and
Andersen thermostats draw from that RNG, so submitting N identical MD
directories to a single server yields N *identical* trajectories — not N
samples. Running the same N directories as N one-shot `vpmdk --dir` processes
gives N independent samples, because each process seeds itself from OS entropy.

To collect an ensemble through server mode, make the runs differ by input
rather than by luck — for example give each replica its own `TEBEG` or its own
starting geometry. Note that a velocity block in `POSCAR`/`CONTCAR` does **not**
work for this and does not carry over at all: velocities are always re-drawn
from a Maxwell-Boltzmann distribution at `TEBEG`, in one-shot mode as well, so
two directories differing only in their velocity blocks produce identical
trajectories. Alternatively, start one server per replica (or fall back to
one-shot runs) when independent sampling is the point.

Deterministic (single-point, relaxation, force constants) workloads are
unaffected.

## Quick Start

Create a startup directory containing the authoritative backend configuration:

```text
model_config/
└── BCAR
```

For example, `model_config/BCAR`:

```text
MLP=MACE
MODEL=/models/mace.model
DEVICE=cuda
```

Start the server in the foreground:

```bash
vpmdk serve --dir ./model_config
```

After the model has loaded, use another shell to inspect and submit work:

```bash
vpmdk status
vpmdk run --dir ./calc-001
vpmdk run --dir ./calc-002
vpmdk stop
```

The final socket is created only after calculator construction succeeds. A
successful `vpmdk run` streams the calculation's stdout and ends with the same
`Calculation completed.` marker as one-shot mode. The calculation's stderr
(third-party warnings) is relayed too and written to the client's stderr, so
both streams match what the one-shot CLI would print; each job re-emits
warnings like a fresh one-shot process.

The submitted directories may omit backend tags from `BCAR` and inherit the
resident calculator. They may still select per-run output options:

```text
# calc-001/BCAR
WRITE_PSEUDO_SCF=1
WRITE_ENERGY_CSV=1
```

See the runnable [server batch example](../../examples/server_batch/README.md)
for readiness polling, a directory loop, shell cleanup, and graceful stop.

## Socket Selection and Readiness

The default socket is:

```text
${XDG_RUNTIME_DIR:-/tmp}/vpmdk-<uid>/default.sock
```

Socket selection has the following priority:

1. `--socket PATH`
2. `VPMDK_SOCKET`
3. the default path

The default per-user directory is mode `0700` and the socket is mode `0600`.
For a custom path, use a directory that is private to the current user.

Both the server and the client key their hardening of that predictable location
on the *directory*, not on the file name: any socket inside
`${XDG_RUNTIME_DIR:-/tmp}/vpmdk-<uid>/` — including a per-GPU `gpu0.sock` beside
the default one — is refused when the directory is a symlink, owned by another
user, or group/world-writable. A socket you place elsewhere is your own choice
and is not gated.

For automation, treat a successful status request as the readiness test:

```bash
until vpmdk status --socket "$socket" >/dev/null 2>&1; do
    sleep 0.1
done
```

Do not treat process creation alone as readiness: model loading happens before
the socket becomes usable. In daemon mode, the original `serve` process waits
for this readiness point and returns a nonzero status if startup fails. That
wait is bounded by `VPMDK_DAEMON_START_TIMEOUT` (seconds, default `600`); raise
it when a very large checkpoint needs longer to load.

## CLI Reference

### `vpmdk serve`

```text
vpmdk serve [--dir DIR] [--bcar PATH] [--socket PATH]
            [--idle-timeout SEC] [--daemon] [--log-file PATH]
```

| Option | Meaning |
| --- | --- |
| `--dir DIR` | Startup directory; defaults to `.` and supplies `BCAR` and an optional representative `POSCAR`/`POTCAR`. |
| `--bcar PATH` | Use this backend configuration instead of `DIR/BCAR`. |
| `--socket PATH` | Override `VPMDK_SOCKET` and the default socket. |
| `--idle-timeout SEC` | Stop after this many idle seconds; `0` disables automatic stop. |
| `--daemon` | Double-fork after initial path checks, re-exec a fresh interpreter, and return only when startup succeeds or fails. The daemon runs from `/`, so relative paths are resolved before it detaches. |
| `--log-file PATH` | Write server lifecycle messages and failures to this file. In daemon mode the default is `<socket>.log`. |

If `--bcar` names a missing file, startup fails. If the default `DIR/BCAR` is
absent, the server warns and builds the default `MLP=CHGNET` calculator.
Explicit relative model and configuration paths are resolved from the selected
startup BCAR's directory. A startup `MODEL` that its backend treats as a local
checkpoint must exist before the calculator is constructed; this includes
extensionless values for local-only backends such as `MACE` and `ORB`. Startup
fails instead of allowing those backends to fall back to a different default
model. MatterSim selectors are instead forwarded to `from_checkpoint`, which
can resolve non-path packaged or downloadable names as well as existing local
checkpoints. A path-shaped MatterSim value must exist locally and is never
reinterpreted as a downloadable name.
Backend model identifiers retain their named-model behavior: in particular, a
FAIRChem identifier such as `org/model` remains the same identifier when a
request is submitted from another directory. MatGL's documented default
`M3GNet-MP-2021.2.8-PES` is likewise compared as an upstream model name, not a
path relative to the startup or request directory. When VPMDK falls back to
legacy `m3gnet`, its bundled default is distinct and status does not label it
with the MatGL model name.
For an existing local MODEL, the calculator receives the lexical path selected
by the user, so a symlink can infer configuration files from its own directory.
Status and request compatibility use the target's canonical real path, allowing
different symlink spellings of the same checkpoint to compare equal.
For GRACE foundation models, status reports the model actually selected from
the installed registry. If the configured VPMDK default is unavailable, this
is the registry's first model, matching the GRACE calculator builder.
An explicitly supplied unknown non-path GRACE name emits a warning and selects
that same effective fallback; status and compatibility checks use the fallback
identity rather than the misspelling. Other static registries reject unknown
names during MODEL resolution; dynamic upstream registries are called with the
exact requested identifier and their failure (or empty loader result) aborts
startup. Request-side MODEL resolution failures are reported as backend
incompatibility (exit code 5), not as a model calculation failure.
Nequix names are validated when the installed calculator exposes `URLS`; older
or alternate builds without that metadata receive the exact requested name.
Matlantis versions are opaque even if a same-named filesystem entry exists.
Both FAIRChem generations forward unresolved checkpoint selectors to their
upstream loader while still recognizing existing local checkpoints. MatterSim
forwards non-path preset names but rejects missing path-shaped values.

The server inherits its environment and launch working directory at `serve`
time. A later `run` client does not transfer its environment. Set
`CUDA_VISIBLE_DEVICES`, model-cache variables, credentials, backend-specific
environment settings, and charge-backend settings before starting the server.
For inherited relative `VPMDK_CHARGE_*` values, each request uses the submitting
client's transmitted cwd as its base; prefer absolute paths when clients launch
from different directories.

Some backends inspect a structure while building species maps or related model
state. If `DIR/POSCAR` exists, VPMDK supplies it during server construction.
Use a representative startup structure for such a backend. A later structure
that is incompatible with the resident model fails only that request; the
server does not rebuild the calculator.

Foreground mode is usually easiest to supervise with a batch scheduler,
systemd, tmux, or a shell trap. Both modes append `.pid` to the complete
socket path—for example, `model.cpu` uses `model.cpu.pid` and `model.gpu` uses
`model.gpu.pid`—and remove it on normal shutdown. Its first line is the PID,
its second line records the owning socket, and a third line records the
process's kernel start time so a restart can tell the recorded process apart
from an unrelated one that recycled the PID. VPMDK refuses to overwrite or remove
a pidfile whose ownership does not match. The pidfile is lifecycle metadata,
not a substitute for `vpmdk status`; it is also the liveness evidence a
restart consults, so a server that has stopped answering while it drains an
uninterruptible job after `stop --force` is refused rather than replaced (a
replacement would load a second resident model beside the draining one).

`--idle-timeout` is a safeguard against an orphaned process retaining VRAM.
The timer begins after startup or the most recent job completion. A status
request counts as activity. Busy and queued time never count as idle.

Startup refuses to replace any socket that accepts a connection, even when the
peer responds slowly or does not speak the VPMDK protocol. Only connection
refusal or a vanished pathname can classify a socket as stale after the short
startup race-safety check. VPMDK never removes a non-socket filesystem entry at
the configured path.

### `vpmdk run`

```text
vpmdk run [--dir DIR] [--socket PATH] [--timeout SEC]
```

`--dir` defaults to the current directory. The client converts it to an
absolute path, submits it, streams log lines, and blocks until the terminal
result arrives. It also sends the client's current directory so relative
`VPMDK_CHARGE_*` environment paths retain one-shot path semantics. Parsing,
model evaluation, and output writing happen in the server process.

`--timeout 0` waits indefinitely after applying the client's connection
timeout. A positive timeout is one end-to-end deadline covering connection,
request transmission, queue wait, execution, and response receipt. If it
expires after the server accepted the job, only the client stops waiting: the
calculation continues and its outputs may appear later. Use `status` to observe
the server. VPMDK intentionally does not attempt unsafe thread or GPU-kernel
cancellation.

There is no implicit fallback to `vpmdk --dir DIR` when the server is absent.
This makes a broken batch setup visible instead of unexpectedly reloading the
model for every directory.

Before each request, VPMDK clears the ASE calculator result cache while keeping
the loaded model weights. Atoms, output recorders, working-directory state, and
calculation settings remain request-local.

### `vpmdk status`

```text
vpmdk status [--socket PATH] [--json]
```

Human-readable status includes:

- state: `idle`, `busy`, or `stopping`
- backend, model, device, and explicit construction options
- PID, uptime, protocol version, and VPMDK version
- completed, failed, and queued job counts
- current absolute work directory while busy

Example:

```text
VPMDK server: idle
Backend: MLP=MACE MODEL=/models/mace.model DEVICE=cuda
PID: 12345  Uptime: 42.1 s
Jobs: completed=8 failed=0 queued=0
Protocol: 1  VPMDK: <version>
```

`--json` emits the status protocol object for scripts. Field names are stable
within protocol version 1; scripts should tolerate additional fields.

### `vpmdk stop`

```text
vpmdk stop [--socket PATH] [--force] [--timeout SEC]
```

Normal stop rejects new jobs, drains the existing FIFO queue, waits for the
active calculation, then removes the socket and daemon pidfile. The default
client wait is 60 seconds. Unlike `run --timeout 0`, `stop --timeout 0` does not
wait for socket removal. The server sends the stop acknowledgement before it
publishes the stopping state, so even an idle server cannot exit between
accepting the command and replying to the client.

SIGINT and SIGTERM request the same graceful shutdown. A second signal and
`--force` stop further queue processing, reject queued jobs, and disconnect the
active client. A job removed from the queue but not yet started is also rejected
and is never passed to the calculator. Python threads and in-flight GPU kernels
cannot be cancelled safely, so the server keeps its socket and does not report
teardown complete until the active executor returns. If that takes longer than
the client's positive timeout, `stop` exits with code 4 while server teardown
remains pending; `stop --timeout 0 --force` returns after acknowledgement
without waiting for teardown.

## Startup BCAR and Request BCAR

A server cannot change models between requests. Its startup BCAR is
authoritative:

| Request setting | Behavior |
| --- | --- |
| No request `BCAR` | Inherit the resident calculator; use normal defaults for request-scoped options. |
| Backend tags omitted | Inherit the resident values. |
| Matching backend tags repeated | Accepted. |
| Different `MLP`/`NNP`, `MODEL`, or `DEVICE` | Reject only that request with exit code 5. |
| Different backend construction option | Reject only that request with exit code 5. |
| Output, charge-density, or finite-difference option | Apply it to that request. |

Construction options include backend-specific model variants, dtype and
precision, compilation, graph conversion, type maps, inference configuration,
and similar settings. The complete comparison set follows the backend tags in
the implementation; consult the [BCAR reference](../reference/bcar-tags.md)
when configuring a model.

Relative checkpoint paths are canonicalized against the directory containing
their respective BCAR before comparison. A mismatch response lists every
differing explicit tag and leaves the server available for later requests.
Omitting a construction tag means inheritance; it does not request that tag's
normal one-shot default.

Documented aliases that feed the same backend option are compared by their
effective canonical setting, so startup `MATLANTIS_MODEL_VERSION=v1` matches a
request containing `MODEL_VERSION=v1`, and `UPET_NEIGHBORLIST_DEVICE` matches
`UPET_NL_DEVICE` when their values agree. The same rule covers the documented
Matlantis, AlphaNet, Nequix, TACE, and CHGNet/MatRIS graph-converter aliases.
If multiple names for one option appear in a single BCAR, the normal backend
builder precedence applies.

For `EQUIFORMER_V3`, registration modules from `EQUIFORMER_V3_MODULE` and
`EQUIFORMER_V3_IMPORT_MODULE` are combined in builder order and deduplicated
without reordering. Repeating a module through both aliases therefore does not
create a different resident identity.

Backend names that are documented routes to the same builder also share one
resident identity. In particular, `MATGL` matches `M3GNET`, and
`FAIRCHEM`, `FAIRCHEM_V2`, and `ESEN` match one another when their model,
device, and construction options agree. Status continues to display the name
used to start the server.

Named-model aliases are likewise compared by the checkpoint identity selected
by the backend builder. For example, an EQNORM server using the default
`eqnorm-mptrj` model accepts `MODEL=eqnorm`, and an AlphaNet server using
`AlphaNet-MATPES-r2scan` accepts `MODEL=matpes`. This canonicalization also
covers documented HIENet aliases and case-insensitive MatRIS/Nequix named
models. Local model paths remain canonicalized as paths rather than aliases.
When the legacy `m3gnet` package supplies the resident calculator, its bundled
default is reported as an unspecified backend default rather than as MatGL's
`M3GNet-MP-2021.2.8-PES`; explicitly requesting that MatGL model is therefore
rejected.

Comparison also uses the value semantics and defaults applied by VPMDK's
backend builders. For example, an ORB server started without `ORB_PRECISION`
accepts a request that explicitly repeats `ORB_PRECISION=float32-high`, while
`ORB_COMPILE=true` and `ORB_COMPILE=1` compare as the same boolean value.
Integer, floating-point, case-insensitive enum, AlphaNet precision, UPET
neighbor-list policy, and list-like type-map/module values are normalized in
the same way as their builders. Defaults owned only by an external calculator
library are not guessed; omit such a request tag to inherit the resident
calculator unambiguously.

DeepMD has one additional startup safety requirement:
`DEEPMD_TYPE_MAP` must be explicit in the server BCAR and ordered exactly as
the model's type indices. One-shot mode can infer a map from each calculation
structure, but a resident calculator would otherwise retain the map inferred
from the startup POSCAR and could misinterpret a later request with different
species or ordering. A DeepMD server therefore rejects startup before loading
the calculator when this tag is absent or empty.

One-shot and server runs call the same execution function and produce the same
calculation results and optional files. The `OUTCAR` timing/accounting footer
contains live process time, memory, page-fault, and context-switch values, so
that diagnostic footer is not expected to be byte-identical across runs.

## GPU and Parallel-Server Patterns

Set `DEVICE` explicitly in the startup BCAR. Use `CUDA_VISIBLE_DEVICES` to give
each process a stable one-GPU view, and assign a unique socket:

Place the sockets in a private, user-owned directory rather than directly under
shared `/tmp`, where a predictable name (`/tmp/vpmdk-gpu0.sock`) lets another
user pre-bind it — failing your `serve` while later clients connect to their
listener. `mktemp -d` creates a `0700` directory owned by you, without following
a pre-planted symlink:

```bash
rundir="$(mktemp -d "${XDG_RUNTIME_DIR:-/tmp}/vpmdk-gpu.XXXXXX")"

CUDA_VISIBLE_DEVICES=0 vpmdk serve \
    --dir ./mace-config --socket "$rundir/gpu0.sock" --idle-timeout 3600 &

CUDA_VISIBLE_DEVICES=1 vpmdk serve \
    --dir ./orb-config --socket "$rundir/gpu1.sock" --idle-timeout 3600 &

VPMDK_SOCKET="$rundir/gpu0.sock" vpmdk run --dir ./calc-a
VPMDK_SOCKET="$rundir/gpu1.sock" vpmdk run --dir ./calc-b
```

With one visible GPU per process, `DEVICE=cuda` is normally preferable to
embedding host-wide CUDA indices in BCAR. The model intentionally remains in
VRAM while the server is idle. It is released only when the server exits.

Do not point two server processes at the same socket. Do not use one server as
a concurrent multi-GPU worker: its queue is serial. Start more servers when
you need calculation parallelism and have enough VRAM.

For a scheduler allocation, foreground mode plus a cleanup trap is robust:

```bash
# A private directory, not a predictable name under shared /tmp: another
# local user who guesses /tmp/vpmdk-<jobid>.sock (job ids are visible in
# squeue) could pre-bind it and make serve refuse to start for the whole
# allocation. A custom socket parent gets none of the default parent's
# hardening, so it must be private by construction.
socket_dir="$(mktemp -d)"
socket="$socket_dir/vpmdk.sock"
cleanup() {
    vpmdk stop --socket "$socket" --timeout 60 >/dev/null 2>&1 || true
    rm -rf "$socket_dir"
}
trap cleanup EXIT INT TERM

vpmdk serve --dir ./model_config --socket "$socket" --idle-timeout 3600 &
server_pid=$!

until vpmdk status --socket "$socket" >/dev/null 2>&1; do
    kill -0 "$server_pid" 2>/dev/null || exit 1
    sleep 0.1
done

for directory in calculations/*; do
    vpmdk run --socket "$socket" --dir "$directory"
done

vpmdk stop --socket "$socket" --timeout 60
wait "$server_pid"
trap - EXIT INT TERM
```

## Failures, Timeouts, and Exit Codes

| Code | Meaning |
| ---: | --- |
| `0` | success |
| `1` | local argument/startup error, or invalid/missing calculation input |
| `2` | calculation failed in the server |
| `3` | unavailable server, connection failure, or unexpected disconnect |
| `4` | client timeout |
| `5` | request BCAR does not match the resident backend |

An invalid input such as a missing `POSCAR` returns code 1, while a backend
exception or CUDA out-of-memory condition returns code 2. A workdir that
cannot be written — an `OUTCAR` directory in the way, or a read-only tree —
also returns code 1: retrying reproduces it byte-for-byte, so it must not be
advertised as retryable. Other I/O failures that can genuinely clear up
between attempts (disk full, a network filesystem flapping) stay code 2. Each fails only its
request. The traceback is returned to the `run` client and recorded in the
server log. The worker then accepts the next queued job. After an exception
whose message identifies CUDA out of memory, the server makes a best-effort
`torch.cuda.empty_cache()` call; this is recovery assistance, not a guarantee
that a damaged upstream calculator can continue safely.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `Cannot connect to VPMDK server` | Confirm the same `--socket`/`VPMDK_SOCKET` is used, then run `status`; the server may still be loading or may have exited. |
| Socket never appears | Run `serve` in the foreground or inspect `<socket>.log`; model construction occurs before readiness. |
| Exit code 5 / backend mismatch | Remove request construction tags to inherit, or start another socket with the requested model/device/options. |
| `run --timeout` returned but GPU remains busy | The accepted job continues by design. Check `status` and output files. |
| VRAM remains allocated while nothing runs | This is the purpose of resident mode. Stop the server or configure `--idle-timeout`. |
| `status` reports `busy` for an unexpected directory | Inspect `current_workdir`; another client may have submitted a job. |
| Jobs do not run in parallel | One server has one FIFO worker. Start independent servers and sockets for parallelism. |
| Relative model/cache path behaves differently | Remember that startup paths and environment belong to the `serve` process, not the later client shell. Prefer absolute checkpoint paths. |
| A stale path prevents startup | VPMDK removes stale sockets only. Move or remove a regular file/directory at that path yourself after verifying it is safe. |
| `socket directory is not writable` | Creating a Unix socket requires write permission on its parent directory. `serve` checks this before loading the model; point `--socket` at a directory the server user can write. |
| `unable to open the log file` | An explicit `--log-file` is probed before the model load in both foreground and daemon modes. Point it at a writable regular file path (a FIFO without a reader is refused instead of blocking startup). |
| Interrupted directory has partial files | Treat it as incomplete; clean or replace its generated outputs before rerunning. |

## Python Client

The synchronous client exposes the same operations. Import it directly for the
same lightweight dependency boundary used by the CLI:

```python
from vpmdk_client import VPMDKClient

# Point at the socket the server created (e.g. "$rundir/gpu0.sock" from the
# private directory above); with no argument the private default socket is used.
client = VPMDKClient("/run/user/1000/vpmdk-1000/default.sock")
print(client.status())
client.run("./calc-001", log_callback=print)
client.stop()
```

`from vpmdk import VPMDKClient` remains supported for compatibility, but
importing the main `vpmdk` API also loads the calculation runtime. Downstream
orchestration tools that only submit and monitor jobs should prefer
`vpmdk_client`.

`run()` and `stop()` return their terminal protocol dictionaries; `status()`
returns the status dictionary. `event_callback` on `run()` receives accepted,
log, heartbeat, and terminal events. See the
[API reference](../reference/api-reference.md#resident-server-client) for
exceptions and signatures.

The client API does not start a server and does not fall back to one-shot mode.

## Protocol Reference

Most users should use the CLI or `VPMDKClient`. For integrations that need the
wire contract, protocol version 1 uses one request per Unix socket connection
and newline-delimited JSON objects. A request or event JSON body is limited to
1 MiB; the NDJSON newline is not counted.
The server splits a stdout line across multiple `log` events when necessary;
concatenating their `line` values reconstructs the original logical line.
Lines from the calculation's stderr are `log` events carrying
`"stream": "stderr"`; VPMDK clients write them to their own stderr, and
clients that do not know the key print them to stdout, so the line is never
lost.
Oversized calculation error messages and tracebacks are truncated with an
explicit marker while retaining the `done`, `ok=false`, and failure-code
contract, so they remain calculation failures rather than protocol failures.

Requests:

```json
{"op":"run","version":1,"workdir":"/absolute/calc-dir","caller_cwd":"/absolute/client-cwd"}
{"op":"status","version":1}
{"op":"stop","version":1,"force":false}
```

`stop.force` may be omitted (equivalent to `false`) or supplied as a JSON
boolean. Strings, numbers, `null`, and other JSON types are protocol errors;
they are never coerced into a force shutdown.

`workdir` and `caller_cwd` must be absolute. A relative value is a malformed
protocol request, not a calculation failure. `caller_cwd` is emitted by VPMDK
clients; older clients may omit it, in which case the server falls back to
`workdir`. It affects only relative environment-provided charge-density paths
and does not transfer the client's environment.

A run response is a stream ending in `done`:

```json
{"event":"accepted","queue_position":0}
{"event":"log","line":"Calculation completed."}
{"event":"heartbeat","elapsed_s":30.0}
{"event":"done","ok":true,"elapsed_s":31.2}
```

If a single logical log line exceeds the protocol event limit, the server sends
multiple `log` events with `continued=true` until the final chunk, which has
`continued=false`. VPMDK clients reassemble these chunks before delivering or
printing the line, so CLI stdout retains its original line boundaries.

Heartbeats are emitted every 30 seconds during calculation. Failures use
`done` with `ok=false`, a code, message, traceback, and elapsed time. Unknown
operations, malformed requests, and unsupported protocol versions receive an
`error` event without stopping the server.

## Security Model

There is no application-level authentication. Access control is the filesystem
permission on the local socket. Anyone who can connect can ask the server
process to read a calculation directory and write outputs there with the
server user's permissions.

- Keep custom sockets in a directory not writable by untrusted users.
- Output artifacts are created with the **submitting client's** umask (sent
  with each `run` request), so `umask 077; vpmdk run` produces the same 0600
  files as the one-shot CLI. A pre-R152 client that does not send its umask
  falls back to the server's launch umask.
- Do not expose the socket through a TCP bridge.
- Run separate servers for separate users or trust boundaries.
- Do not run a shared server with broader filesystem permissions than its
  clients should exercise.
