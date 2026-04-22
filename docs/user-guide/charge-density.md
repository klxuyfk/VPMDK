# Charge Density and CHGCAR

## Overview

VPMDK separates force-field execution from charge-density prediction. The main
run can use one environment and backend, while `CHGCAR` generation can use a
different interpreter and model stack.

Supported charge backends:

- `CHARGE3NET`
- `DEEPDFT`
- `DEEPCDP`

The CLI enables this path through:

```text
WRITE_CHGCAR=1
```

and then forwards charge options from `BCAR` into `predict_charge_density()`.

## Grid Selection

When `grid_shape` is not passed explicitly, VPMDK derives a VASP-like fine FFT
grid from `INCAR`.

Priority order:

1. `NGXF`, `NGYF`, `NGZF`
2. `NGX`, `NGY`, `NGZ` plus `PREC`
3. `ENCUT` plus `PREC`

Recognized `PREC` aliases:

- `LOW`, `L`
- `MEDIUM`, `M`
- `NORMAL`, `N`
- `HIGH`, `H`
- `ACCURATE`, `A`
- `SINGLE`, `S`
- `SINGLEN`, `SN`

Current multipliers:

- `LOW`, `MEDIUM`, `NORMAL`: coarse factor `1.5`, fine multiplier `2.0`
- `HIGH`, `ACCURATE`: coarse factor `2.0`, fine multiplier `2.0`
- `SINGLE`, `SINGLEN`: coarse factor `1.5`, fine multiplier `1.0`

The implementation rounds up to even smooth FFT sizes whose largest prime
factor does not exceed 7.

## BCAR Tags

Core charge tags:

| Tag | Meaning |
|-----|---------|
| `WRITE_CHGCAR` | Enable final `CHGCAR` writing |
| `CHARGE_MLP` | Primary charge-backend selector |
| `CHARGE_BACKEND` | Legacy alias of `CHARGE_MLP` |
| `CHARGE_MODEL` | Checkpoint path or model directory |
| `CHARGE_DEVICE` | Charge inference device |
| `CHARGE_SOURCE_DIR` | Source checkout used by the runner |
| `CHARGE_PYTHON` | Python executable used by the runner |
| `CHARGE_CUTOFF` | ChargE3Net cutoff override |
| `CHARGE_MAX_PROBES_PER_BATCH` | Probe batch size |

ChargE3Net model-config tags:

- `CHARGE_NUM_INTERACTIONS`
- `CHARGE_NUM_NEIGHBORS`
- `CHARGE_MUL`
- `CHARGE_LMAX`
- `CHARGE_BASIS`
- `CHARGE_NUM_BASIS`
- `CHARGE_SPIN`

DeepCDP tags:

- `CHARGE_DEEPCDP_METADATA`
- `CHARGE_DEEPCDP_SPECIES`
- `CHARGE_DEEPCDP_RCUT`
- `CHARGE_DEEPCDP_NMAX`
- `CHARGE_DEEPCDP_LMAX`
- `CHARGE_DEEPCDP_SIGMA`
- `CHARGE_DEEPCDP_PERIODIC`
- `CHARGE_DEEPCDP_ACTIVATION`
- `CHARGE_DEEPCDP_WEIGHTING_FUNCTION`
- `CHARGE_DEEPCDP_WEIGHTING_R0`
- `CHARGE_DEEPCDP_WEIGHTING_C`
- `CHARGE_DEEPCDP_WEIGHTING_M`
- `CHARGE_DEEPCDP_WEIGHTING_D`

## Environment Variable Precedence

The runtime resolves charge paths in this order:

1. explicit function arguments / BCAR tags
2. backend-specific environment variables
3. generic environment variables
4. backend-specific fallbacks

Generic variables:

- `VPMDK_CHARGE_PYTHON`
- `VPMDK_CHARGE_SOURCE_DIR`
- `VPMDK_CHARGE_MODEL`

Backend-specific variables:

- ChargE3Net:
  `VPMDK_CHARGE3NET_PYTHON`,
  `VPMDK_CHARGE3NET_SOURCE_DIR`,
  `VPMDK_CHARGE3NET_MODEL`
- DeepDFT:
  `VPMDK_DEEPDFT_PYTHON`,
  `VPMDK_DEEPDFT_SOURCE_DIR`,
  `VPMDK_DEEPDFT_MODEL`
- DeepCDP:
  `VPMDK_DEEPCDP_PYTHON`,
  `VPMDK_DEEPCDP_SOURCE_DIR`,
  `VPMDK_DEEPCDP_MODEL`

Path resolution detail:

- if a path is absolute, it is used directly
- explicit `CHARGE_PYTHON`, `CHARGE_SOURCE_DIR`, and `CHARGE_MODEL` values are
  used as written; in CLI runs they therefore behave like paths relative to the
  active calculation directory when `vpmdk --dir ...` is used
- environment-variable fallbacks are resolved against the original caller
  working directory through `VPMDK_CHARGE_ENV_BASE_DIR`

## ChargE3Net

Minimal example:

```text
WRITE_CHGCAR=1
CHARGE_MLP=CHARGE3NET
CHARGE_PYTHON=/path/to/env/bin/python
CHARGE_SOURCE_DIR=/path/to/charge3net
CHARGE_MODEL=/path/to/charge3net_mp.pt
```

Behavior:

- if `CHARGE_MODEL` is missing and `CHARGE_SOURCE_DIR/models/charge3net_mp.pt`
  exists, that checkpoint is used automatically
- `CHARGE_DEVICE` defaults to the runner's auto-detected device
- `CHARGE_SPIN=1` requests spin-density prediction when the checkpoint supports it
- result metadata reports the resolved model path and effective model config

## DeepDFT

Minimal example:

```text
WRITE_CHGCAR=1
CHARGE_MLP=DEEPDFT
CHARGE_PYTHON=/path/to/env/bin/python
CHARGE_SOURCE_DIR=/path/to/DeepDFT
CHARGE_MODEL=/path/to/deepdft-model-dir
```

`CHARGE_MODEL` should resolve to a directory containing DeepDFT artifacts such
as `arguments.json` and `best_model.pth`. Passing `best_model.pth` directly is
also accepted; VPMDK normalizes that to the parent directory.

## DeepCDP

Minimal example:

```text
WRITE_CHGCAR=1
CHARGE_MLP=DEEPCDP
CHARGE_PYTHON=/path/to/env/bin/python
CHARGE_MODEL=/path/to/model.pt
CHARGE_DEEPCDP_SPECIES=O,H
CHARGE_DEEPCDP_ACTIVATION=relu
```

Metadata handling:

- if `CHARGE_DEEPCDP_METADATA` is set, that file is used
- otherwise VPMDK looks next to the checkpoint for:
  `deepcdp_config.json`, `metadata.json`, or `config.json`
- `VPMDK_DEEPCDP_METADATA` is also supported

If species or activation cannot be inferred from metadata, they must be supplied
explicitly.

## Python API

`predict_charge_density()` accepts either:

- `grid_shape=(nx, ny, nz)`, or
- `incar=...` plus `reference=...`

If both `grid_shape` and `incar` are omitted, the call fails.

Returned object:

- `density`: 3D charge-density array
- `spin_density`: optional 3D spin-density array
- `grid_shape`
- `backend`
- `metadata`

`charge_density()` is a backwards-compatible alias.

## Writing CHGCAR

`write_chgcar(path, atoms, density, spin_density=None)` writes a VASP-like
volumetric file using ASE's `VaspChargeDensity`.

Constraints:

- `density` must be 3D
- `spin_density`, if provided, must match the same shape

Current limitation:

- PAW augmentation occupancies are not reconstructed
- the file is suitable for visualization and analysis, not as a full-fidelity
  DFT restart artifact
