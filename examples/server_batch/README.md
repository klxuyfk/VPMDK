# Resident Server Batch

This example reuses one resident calculator for two VASP-style calculation
directories:

```text
server_batch/
├── model_config/BCAR
└── calculations/
    ├── 0001/{POSCAR,INCAR,BCAR}
    └── 0002/{POSCAR,INCAR,BCAR}
```

The bundled configuration uses CHGNet on CPU. Run from the repository root:

```bash
./examples/server_batch/run.sh
```

The script creates a private socket directory, starts a foreground server,
waits for readiness, submits calculation directories in lexical order, and
stops the server on exit. It refuses to adopt an already-running server.

To use another backend or GPU, edit `model_config/BCAR` and use absolute model
paths. Set `VPMDK_SOCKET` for an explicit unused socket and
`VPMDK_READY_TIMEOUT` to change the startup timeout.

See the [Server Mode guide](../../docs/user-guide/server-mode.md) for
configuration inheritance, security, parallel-server patterns, and exit codes.
