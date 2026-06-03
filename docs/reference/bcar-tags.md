# BCAR Reference

## Format

`BCAR` is a simple `key=value` file.

Rules:

- keys are normalized to uppercase
- `#` and `!` start comments
- blank lines are ignored
- `NNP` is accepted as a legacy alias for `MLP`

Boolean parsing for tags that expect booleans:

- true: `1`, `true`, `yes`, `on`
- false: `0`, `false`, `no`, `off`

Unknown tags are preserved in the parsed mapping but have no effect unless a
backend or helper explicitly consumes them.

## Core Selection Tags

| Tag | Meaning | Default |
|-----|---------|---------|
| `MLP` | backend name | `CHGNET` |
| `NNP` | legacy alias of `MLP` | none |
| `MODEL` | checkpoint path or named model | backend-dependent |
| `DEVICE` | device hint such as `cpu`, `cuda`, `cuda:0` | auto-detected or backend default |

## Output and Compatibility Tags

| Tag | Meaning | Default |
|-----|---------|---------|
| `WRITE_ENERGY_CSV` | write `energy.csv` during relaxation | `0` |
| `WRITE_LAMMPS_TRAJ` | write `lammps.lammpstrj` during MD | `0` |
| `LAMMPS_TRAJ_INTERVAL` | frame interval for the LAMMPS trajectory | `1` |
| `WRITE_PSEUDO_SCF` | echo pseudo electronic-step blocks into compatibility files | `0` |
| `WRITE_OSZICAR_PSEUDO_SCF` | legacy alias of `WRITE_PSEUDO_SCF` | none |
| `WRITE_CHGCAR` | run charge-density prediction after the main run | `0` |
| `FORCE_CONSTANTS_DISPLACEMENT` | VPMDK finite-difference displacement in Angstrom for `IBRION=7`/`8`; `IBRION=5`/`6` use `POTIM` instead | `0.01` |

See
[VASP Force-Constants Compatibility](../development/force-constants.md) for the
finite-difference formula and `vasprun.xml` Hessian convention.

## Shared Backend Tuning Tags

These are interpreted by more than one backend:

| Tag | Applies to | Values |
|-----|------------|--------|
| `GRAPH_CONVERTER` | CHGNet, MatRIS | `fast`, `legacy` |
| `GRAPH_CONVERTER_ALGORITHM` | CHGNet, MatRIS | `fast`, `legacy` |
| `CHGNET_GRAPH_CONVERTER` | CHGNet | `fast`, `legacy` |
| `CHGNET_GRAPH_CONVERTER_ALGORITHM` | CHGNet | `fast`, `legacy` |
| `MATRIS_GRAPH_CONVERTER` | MatRIS | `fast`, `legacy` |
| `MATRIS_GRAPH_CONVERTER_ALGORITHM` | MatRIS | `fast`, `legacy` |

Backend-specific overrides win over the shared graph-converter tags.

## Charge-Density Tags

### Shared Charge Tags

| Tag | Meaning |
|-----|---------|
| `CHARGE_MLP` | charge backend name |
| `CHARGE_BACKEND` | legacy alias of `CHARGE_MLP` |
| `CHARGE_MODEL` | charge checkpoint or model directory |
| `CHARGE_DEVICE` | charge backend device |
| `CHARGE_SOURCE_DIR` | source checkout used by the subprocess runner |
| `CHARGE_PYTHON` | Python interpreter used by the subprocess runner |
| `CHARGE_CUTOFF` | ChargE3Net cutoff override |
| `CHARGE_MAX_PROBES_PER_BATCH` | probe batch size |

### ChargE3Net Model-Config Tags

| Tag | Meaning |
|-----|---------|
| `CHARGE_NUM_INTERACTIONS` | number of message-passing interactions |
| `CHARGE_NUM_NEIGHBORS` | neighbor count / cutoff helper |
| `CHARGE_MUL` | multiplicity parameter |
| `CHARGE_LMAX` | maximum angular momentum |
| `CHARGE_BASIS` | basis family |
| `CHARGE_NUM_BASIS` | basis count |
| `CHARGE_SPIN` | request spin-density output |

### DeepCDP Tags

| Tag | Meaning |
|-----|---------|
| `CHARGE_DEEPCDP_METADATA` | explicit metadata JSON path |
| `CHARGE_DEEPCDP_SPECIES` | comma-separated species list |
| `CHARGE_DEEPCDP_RCUT` | SOAP cutoff |
| `CHARGE_DEEPCDP_NMAX` | SOAP radial basis size |
| `CHARGE_DEEPCDP_LMAX` | SOAP angular basis size |
| `CHARGE_DEEPCDP_SIGMA` | SOAP Gaussian width |
| `CHARGE_DEEPCDP_PERIODIC` | SOAP periodic flag |
| `CHARGE_DEEPCDP_ACTIVATION` | network activation name |
| `CHARGE_DEEPCDP_WEIGHTING_FUNCTION` | weighting function name |
| `CHARGE_DEEPCDP_WEIGHTING_R0` | weighting parameter |
| `CHARGE_DEEPCDP_WEIGHTING_C` | weighting parameter |
| `CHARGE_DEEPCDP_WEIGHTING_M` | weighting parameter |
| `CHARGE_DEEPCDP_WEIGHTING_D` | weighting parameter |

## Force-Field Backend Tags

### Matlantis

- `MATLANTIS_MODEL_VERSION`
- `MODEL_VERSION`
- `MATLANTIS_PRIORITY`
- `PRIORITY`
- `MATLANTIS_CALC_MODE`
- `CALC_MODE`

### ORB

- `ORB_MODEL`
- `ORB_PRECISION`
- `ORB_COMPILE`

### Eqnorm

- `EQNORM_VARIANT`
- `EQNORM_COMPILE`

### MatRIS

- `MATRIS_TASK`

### MatterSim

- `MATTERSIM_COMPUTE_STRESS`
- `MATTERSIM_STRESS_WEIGHT`

### AlphaNet

- `ALPHANET_CONFIG`
- `ALPHANET_PRECISION`
- `ALPHANET_DTYPE`

### HIENet

- `HIENET_FILE_TYPE`

### Nequix

- `NEQUIX_BACKEND`
- `NEQUIX_USE_KERNEL`
- `NEQUIX_KERNEL`
- `NEQUIX_USE_COMPILE`
- `NEQUIX_COMPILE`
- `NEQUIX_CAPACITY_MULTIPLIER`

### SevenNet / FlashTP

- `SEVENNET_FILE_TYPE`
- `SEVENNET_MODAL`
- `SEVENNET_ENABLE_CUEQ`
- `SEVENNET_ENABLE_FLASH`
- `SEVENNET_ENABLE_OEQ`

### UPET

- `UPET_VERSION`
- `UPET_NON_CONSERVATIVE`
- `UPET_NEIGHBORLIST_DEVICE`
- `UPET_NL_DEVICE`

### TACE

- `TACE_DTYPE`
- `TACE_SPIN_ON`
- `TACE_NEIGHBORLIST_BACKEND`
- `TACE_FIDELITY_IDX`
- `TACE_LEVEL`

### FAIRChem

- `FAIRCHEM_TASK`
- `FAIRCHEM_INFERENCE_SETTINGS`
- `FAIRCHEM_CONFIG`
- `FAIRCHEM_V1_PREDICTOR`

### EquiformerV3

- `EQUIFORMER_V3_MODULE`
- `EQUIFORMER_V3_IMPORT_MODULE`
- `FAIRCHEM_CONFIG`

### GRACE

- `GRACE_PAD_NEIGHBORS_FRACTION`
- `GRACE_PAD_ATOMS_NUMBER`
- `GRACE_MAX_RECOMPILATION`
- `GRACE_MIN_DIST`
- `GRACE_FLOAT_DTYPE`

### DeePMD

- `DEEPMD_TYPE_MAP`
- `DEEPMD_HEAD`

## Notes on Relative Paths

`MODEL` and most backend-local paths are resolved relative to the active run
directory because the CLI changes into the selected calculation directory
before constructing the calculator.

Charge-environment paths are a special case: relative `CHARGE_*` paths are
handled differently depending on how they are provided:

- explicit `CHARGE_PYTHON`, `CHARGE_SOURCE_DIR`, and `CHARGE_MODEL` values in
  `BCAR` are used as written, so when you use `--dir` they are interpreted
  relative to the selected calculation directory
- environment-variable fallbacks are resolved against the caller's original
  shell working directory

This means that:

```bash
vpmdk --dir /other/location
```

does not make this `BCAR` entry relative to the shell that launched `vpmdk`:

```text
CHARGE_PYTHON=./env/bin/python
```

For that use case, prefer an absolute path or an environment-variable fallback.
