# VPMDK Documentation

This directory is the canonical home for user and developer documentation.
The intent is to keep long-form material here and leave the repository root
focused on package metadata, the main README, and runnable examples.

## Document Map

- [Getting Started](getting-started/quickstart.md)
  First successful runs with the CLI and Python API.
- [Installation](getting-started/installation.md)
  Core installation, backend packages, and environment setup patterns.
- [CLI Workflows](user-guide/cli-workflows.md)
  VASP-style inputs, execution modes, outputs, NEB handling, and compatibility behavior.
- [Python API](user-guide/python-api.md)
  Stable library usage without implicit filesystem side effects.
- [Charge Density](user-guide/charge-density.md)
  `CHGCAR` generation, FFT-grid rules, and charge-backend configuration.
- [API Reference](reference/api-reference.md)
  Public functions, config objects, result objects, and helper models.
- [INCAR Reference](reference/incar-tags.md)
  Supported `INCAR` tags, defaults, and mode-selection semantics.
- [BCAR Reference](reference/bcar-tags.md)
  Backend-selection tags, output flags, charge tags, and backend-specific knobs.
- [Backend Reference](reference/backends.md)
  Per-backend package requirements, `MODEL` semantics, defaults, and caveats.
- [Architecture](development/architecture.md)
  Internal package layout and how the CLI/API layers fit together.
- [Backend Environment Notes](development/backend-environments.md)
  Environment and dependency tips collected from real integration work.
- [Validation Notes](development/validation.md)
  Current validation scope and known manually verified backend paths.

## Recommended Reading Order

If you are new to the project:

1. Read [Installation](getting-started/installation.md).
2. Follow [Quick Start](getting-started/quickstart.md).
3. Use either [CLI Workflows](user-guide/cli-workflows.md) or
   [Python API](user-guide/python-api.md) depending on your entry point.
4. Keep [INCAR Reference](reference/incar-tags.md),
   [BCAR Reference](reference/bcar-tags.md), and
   [Backend Reference](reference/backends.md) open while configuring runs.

If you are maintaining or extending the project:

1. Read [Architecture](development/architecture.md).
2. Review [Backend Environment Notes](development/backend-environments.md).
3. Check [Validation Notes](development/validation.md) before changing backend integrations.

## Scope

The documentation in `docs/` is written against the implementation in
`src/vpmdk_core/` and the regression tests in `tests/`. When behavior here
differs from older free-form notes in the repository, treat the implementation
and these `docs/` pages as authoritative.
