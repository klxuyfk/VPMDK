# Server Mode

Server mode keeps one calculator resident in memory and reuses it for many
VASP-style calculation directories. Use it when model loading is a meaningful
part of the runtime. For isolated calculations, different models, or maximum
process isolation, continue to use:

```bash
vpmdk --dir ./calc
```

Server mode is available on POSIX systems with Unix-domain sockets. One server
has one calculator and processes calculations serially.

## Quick Start

Create a startup directory whose `BCAR` selects the resident backend:

```text
model_config/
└── BCAR
```

```text
MLP=MACE
MODEL=/models/mace.model
DEVICE=cuda
```

Start the server and submit work:

```bash
vpmdk serve --dir ./model_config --daemon --idle-timeout 3600
vpmdk status
vpmdk run --dir ./calc-001
vpmdk run --dir ./calc-002
vpmdk stop
```

The socket becomes ready only after the calculator has loaded. In automation,
poll `vpmdk status` rather than treating process creation as readiness.

## Operational Model

- The startup process owns the model, device, environment, caches, and
  credentials.
- Accepted calculations enter a FIFO queue and execute one at a time.
- `status` and `stop` remain responsive while a calculation is running.
- Each request uses the same work-directory execution path as
  `vpmdk --dir DIR` and writes outputs into that directory.
- A failed request does not normally terminate the server or later queued work.
- A client timeout stops waiting; it does not cancel an accepted calculation.

To run calculations in parallel, start independent servers with different
sockets and enough CPU/GPU memory.

### Randomness

Each request starts from the server's saved NumPy random state. Identical
stochastic inputs submitted to one server therefore replay the same random
stream. Use different starting inputs, separate servers, or one-shot processes
when independent MD replicas are required.

## Configuration Authority

The startup `BCAR` defines calculator-construction settings. A request may
omit those settings and inherit the resident calculator.

| Request setting | Result |
| --- | --- |
| No request `BCAR` | Inherit the resident calculator. |
| Backend settings omitted | Inherit resident values. |
| Equivalent settings repeated | Accept the request. |
| Different model, device, or construction option | Reject the request with exit code 5. |
| Output or charge-density option | Apply it only to that request. |

Relative model paths are resolved from the directory containing their
respective `BCAR`. Prefer absolute paths in batch and scheduler workflows.
Backend-specific construction tags are listed in the
[BCAR reference](../reference/bcar-tags.md).

The environment belongs to the `serve` process and is not copied from later
clients. Set CUDA visibility, cache locations, credentials, and backend
environment variables before starting the server.

DeepMD servers require an explicit model-ordered `DEEPMD_TYPE_MAP`; a
structure-derived map cannot safely be reused across arbitrary requests.

## Socket and Security

Socket selection uses:

1. `--socket PATH`
2. `VPMDK_SOCKET`
3. `${XDG_RUNTIME_DIR:-/tmp}/vpmdk-<uid>/default.sock`

The default directory is private to the current user. Put custom sockets in a
user-owned directory with mode `0700`; do not place predictable socket names
directly in a shared `/tmp`.

There is no application-level authentication. Anyone who can connect can ask
the server process to read and write calculation directories using the
server's filesystem permissions. Do not expose the socket through a TCP bridge
or share it across trust boundaries.

Output files use the submitting client's umask when the client supports that
protocol field.

## Commands

Use `vpmdk <command> --help` for the complete option list.

### `serve`

```text
vpmdk serve [--dir DIR] [--bcar PATH] [--socket PATH]
            [--idle-timeout SEC] [--daemon] [--log-file PATH]
```

`--dir` supplies the startup `BCAR`; `--bcar` selects another file.
Without either file, the server warns and uses the normal CHGNet default.
`--idle-timeout` releases an abandoned resident model after an idle period.
Daemon mode writes lifecycle output to `<socket>.log` unless `--log-file`
is supplied.

### `run`

```text
vpmdk run [--dir DIR] [--socket PATH] [--timeout SEC]
```

The client submits an absolute work directory, streams calculation output, and
waits for the terminal result. `--timeout 0` waits indefinitely. There is no
automatic fallback to one-shot execution when the server is unavailable.

### `status`

```text
vpmdk status [--socket PATH] [--json]
```

Status reports lifecycle state, backend identity, PID, uptime, queue counts,
and the current work directory. Use `--json` for scripts.

### `stop`

```text
vpmdk stop [--socket PATH] [--force] [--timeout SEC]
```

Normal stop rejects new work and drains accepted work before removing the
socket. `--force` rejects queued work and disconnects the active client, but
Python threads and in-flight GPU kernels cannot be cancelled safely; teardown
still waits for the active executor to return.

## GPU and Batch Use

Give each server a private socket and a stable device view:

```bash
socket_dir="$(mktemp -d)"
socket="$socket_dir/vpmdk.sock"
cleanup() {
    vpmdk stop --socket "$socket" --timeout 60 >/dev/null 2>&1 || true
    rm -rf "$socket_dir"
}
trap cleanup EXIT

CUDA_VISIBLE_DEVICES=0 vpmdk serve \
    --dir ./model_config --socket "$socket" --idle-timeout 3600 &

until vpmdk status --socket "$socket" >/dev/null 2>&1; do
    sleep 0.1
done

for directory in calculations/*; do
    vpmdk run --socket "$socket" --dir "$directory"
done

vpmdk stop --socket "$socket"
```

The bundled [server batch example](../../examples/server_batch/README.md)
provides a reusable shell script.

## Exit Codes

| Code | Meaning |
| ---: | --- |
| `0` | Success. |
| `1` | Invalid arguments, startup configuration, or calculation input. |
| `2` | Calculation failed in the server. |
| `3` | Server unavailable or connection lost. |
| `4` | Client timeout. |
| `5` | Request backend settings do not match the resident calculator. |

## Python Client

Use the import-light client in orchestration processes:

```python
from vpmdk_client import VPMDKClient

client = VPMDKClient()
print(client.status())
client.run("./calc-001", log_callback=print)
client.stop()
```

The client is synchronous, does not start a server, and never falls back to a
one-shot calculation. Signatures and exceptions are documented in the
[API reference](../reference/api-reference.md#resident-server-client).

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Cannot connect | Verify the socket selection and inspect the server log. The model may still be loading. |
| Exit code 5 | Remove request construction tags to inherit, or start a server with the requested configuration. |
| Timeout but GPU remains busy | The accepted job continues. Check `status` and its output directory. |
| Idle server retains VRAM | Stop it or configure `--idle-timeout`. |
| Jobs are serial | Start additional servers with separate sockets. |
| Relative paths differ | Use absolute model and cache paths; the server owns the startup environment. |
| Stale path blocks startup | Verify ownership, then remove the stale non-socket entry manually. |

The wire protocol is versioned but is not the recommended integration surface.
Use the CLI or `VPMDKClient` rather than constructing protocol messages
directly.
