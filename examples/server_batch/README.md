# Resident Server Batch

This example is a generic shell pattern for processing many VASP-style input
directories with one resident model. The bundled layout is:

```text
server_batch/
├── model_config/BCAR
└── calculations/
    ├── 0001/{POSCAR,INCAR,BCAR}
    └── 0002/{POSCAR,INCAR,BCAR}
```

The startup `model_config/BCAR` selects the calculator. The bundled inputs use
CHGNet on CPU and require the optional `chgnet` package. A calculation BCAR may
repeat the same backend settings or omit them and contain only per-run output
options.

By default the script creates a private `0700` directory (via `mktemp -d`) and
places the socket inside it, so a predictable path under a shared `/tmp` cannot
be pre-bound by another user. Export `VPMDK_SOCKET` to pin an explicit, *unused*
socket path when running several batches, and `VPMDK_READY_TIMEOUT` (seconds,
default 600) to widen the startup budget for large checkpoints.

The script starts and stops its own server only. If a server already answers at
the target socket it exits rather than adopting it, because stopping a
pre-existing resident model on exit would disrupt its other clients.

Run from the repository root or from this directory:

```bash
./examples/server_batch/run.sh
```

The script starts a foreground server as a shell background job, waits for
`status` readiness, submits every immediate child of `calculations/` in lexical
order, and gracefully stops the server on normal or abnormal shell exit. It
sets a one-hour idle timeout as a second cleanup layer.

To exercise GPU residency, change `model_config/BCAR` to a GPU-capable backend
and set `DEVICE=cuda`. For example:

```text
MLP=MACE
MODEL=/absolute/path/to/mace.model
DEVICE=cuda
```

Use absolute checkpoint paths for portable batch behavior. While the server is
idle, `vpmdk status` (or `vpmdk status --socket PATH`) should report
`DEVICE=cuda`, and the model remains in VRAM until `stop` or the idle timeout
exits the process.

This example intentionally uses sequential submission. For parallel execution,
start separate server processes with separate sockets and enough GPU memory.
See the [Server Mode guide](../../docs/user-guide/server-mode.md) for the full
operational and security contract.
