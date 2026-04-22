# Validation Notes

## Scope

This page records the validation status currently documented in the repository.
It is based on the maintained validation notes and test suite, not on an
automatic rerun performed every time docs are edited.

There are two validation layers:

- unit/regression tests under `tests/`
- longer backend-dependent integration or manual runs under `tests/integration/`
  and historical maintainer notes

## Regression Coverage

The test suite exercises:

- public API side-effect guarantees
- `INCAR` and `BCAR` parsing
- backend builder argument forwarding
- VASP-compatible output formatting
- NEB-like directory handling
- MD driver selection
- charge-density grid logic and subprocess argument construction

Useful command:

```bash
pytest -m "not integration"
```

## Integration Coverage

`tests/integration/test_integration_md.py` covers short MD workflows for a
backend matrix and skips cleanly when required packages or checkpoints are not
available.

Those tests are intended to verify runnable backend paths rather than numerical
validation against a reference dataset.

## Historically Recorded Manual CUDA MD Runs

The repository's maintained notes record successful manual CUDA MD runs for:

- CHGNet
- SevenNet
- MatterSim
- MACE
- MatGL
- ORB
- GRACE
- DeePMD (PyTorch backend)
- NequIP
- Allegro
- FAIRChem v1
- FAIRChem v2 (UMA)

Recorded exclusions:

- Matlantis was explicitly excluded from that manual validation round

The same notes also record recurring environment fixes:

- GRACE: missing `ptxas`
- MatGL: DGL backend mismatch
- DeePMD: MPI library issues
- FAIRChem v2: gated checkpoint access
- NequIP/Allegro: `.nequip` archive compilation

## Model Storage

Historical validation notes reference external checkpoint storage such as:

```text
/mnt/d/lin_temp/codex
```

This is only a storage convention used during manual validation and is not a
hard-coded repository requirement.

## Interpreting "Validated"

In this project, "validated" should usually be read as:

- the backend path executed successfully through the VPMDK wrapper
- expected outputs were produced
- major integration issues were addressed

It does not automatically imply:

- benchmark-quality force/energy agreement
- exhaustive testing across all model variants
- support for every upstream package version

## Recommended Validation Workflow for Changes

When you change backend logic or compatibility output:

1. Run `pytest -m "not integration"`.
2. Run the smallest relevant example under `examples/`.
3. Re-run at least the directly affected backend integration path if that
   backend is available locally.
4. Update these docs if the supported tags, defaults, or environment caveats changed.
