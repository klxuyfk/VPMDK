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
| `DEVICE` | device hint such as `cpu`, `cuda`, `cuda:0` | auto-detected or backend default; the value is case-folded at parse (`CPU` ≡ `cpu`), matching the server's case-insensitive comparison |

### Interpretation in Server Mode

For `vpmdk serve`, the startup BCAR constructs the resident calculator. A
submitted calculation may omit `MLP`/`NNP`, `MODEL`, `DEVICE`, shared backend
tuning tags, and the force-field backend tags listed below; omission means
inheritance from the server. If a request states one of those construction
tags explicitly, its normalized value must match the resident configuration or
that request is rejected.

Output/compatibility tags, all `CHARGE_*` tags, and
`FORCE_CONSTANTS_DISPLACEMENT` remain per-request. They do not rebuild or alter
the resident force-field calculator. See
[Server Mode](../user-guide/server-mode.md#startup-bcar-and-request-bcar) for
the complete authority and mismatch behavior.

## Output and Compatibility Tags

| Tag | Meaning | Default |
|-----|---------|---------|
| `WRITE_ENERGY_CSV` | write `energy.csv` during relaxation | `0` |
| `WRITE_LAMMPS_TRAJ` | write `lammps.lammpstrj` during MD (LAMMPS `metal` units: velocities in Å/ps) | `0` |
| `LAMMPS_TRAJ_INTERVAL` | frame interval for the LAMMPS trajectory | `1` |
| `WRITE_PSEUDO_SCF` | echo pseudo electronic-step blocks into compatibility files | `0` |
| `WRITE_OSZICAR_PSEUDO_SCF` | legacy alias of `WRITE_PSEUDO_SCF` | none |
| `WRITE_CHGCAR` | run charge-density prediction after the main run | `0` |
| `FORCE_CONSTANTS_DISPLACEMENT` | VPMDK finite-difference displacement in Angstrom for `IBRION=7`/`8`; `IBRION=5`/`6` use `POTIM` instead | `0.01`; must be at least `1e-6` Angstrom and a plain number (a corrupted token such as `1D-2` is rejected, not read as its leading digits) |

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

A charge configuration that can never work — no checkpoint resolvable through
`CHARGE_MODEL`/environment variables, or a `CHARGE_PYTHON` interpreter that
does not exist — is reported as an input error (exit 1 in both one-shot and
server mode), not as a retryable calculation failure.

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

`MODEL` may be an existing local checkpoint or a selector understood by
`MatterSimCalculator.from_checkpoint`, such as `mattersim-v1.0.0-5M`. An
explicit non-path selector is always forwarded; it cannot silently fall back
to the calculator default. Absolute/relative paths, values containing a path
separator, and checkpoint-suffixed values must exist locally. Older MatterSim
builds without `from_checkpoint` may use a declared `load_path` constructor;
otherwise named selectors fail with an explicit compatibility error.

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

### SevenNet / FlashTP / EquFlash

- `SEVENNET_FILE_TYPE`
- `SEVENNET_MODAL`
- `SEVENNET_ENABLE_CUEQ`
- `SEVENNET_ENABLE_FLASH`
- `SEVENNET_ENABLE_OEQ`

EquFlash uses checkpoint mode with FlashTP forced on and CUEQ/OEQ forced off;
explicit repetitions of those effective values are accepted in server mode.

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

EquiformerV2 / eqV2 checkpoints use the FAIRChem v1/OCP path:

```text
MLP=FAIRCHEM_V1
MODEL=/path/to/eqV2_checkpoint.pt
DEVICE=cuda
```

There is no `MLP=EQUIFORMER_V2` tag. Original Equiformer V1
`graph_attention_transformer` checkpoints are not part of the documented
support surface because they depend on older OCP trainer conventions.

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

When `MODEL` is omitted, GRACE selects `GRACE-2L-MP-r6` if it is present in
the installed foundation-model registry, or the registry's first model
otherwise. Server status and `list_backends()` report that effective choice.
An unknown non-path foundation-model name produces a warning and selects the
same effective default. A missing path-shaped value remains an error.

### DeePMD

- `DEEPMD_TYPE_MAP`
- `DEEPMD_HEAD`

For one-shot execution, omitting `DEEPMD_TYPE_MAP` lets VPMDK infer a list from
that calculation's structure. Resident server mode instead requires an
explicit `DEEPMD_TYPE_MAP` in the startup BCAR because one calculator cannot
safely retain an ordering inferred from a startup POSCAR and then process
requests with different species subsets or orderings. List elements must be in
the type-index order expected by the DeepMD model, for example:

```text
DEEPMD_TYPE_MAP=Si,O
```

DPA-family checkpoints use the DeePMD path:

```text
MLP=DEEPMD
MODEL=/path/to/dpa_checkpoint.pt
```

For DPA-4 / DPA4 / SeZM checkpoints, use a `deepmd-kit` environment that
registers the `dpa4` / `SeZM` model type. Some DeepMD environments also need
`LD_LIBRARY_PATH` to include the environment's `lib` directory so MPI-related
shared libraries are resolved consistently.

## Notes on Relative Paths

`MODEL` and most backend-local paths are resolved relative to the active run
directory because the CLI changes into the selected calculation directory
before constructing the calculator.

Server startup is the exception to that one-shot wording: an explicit relative
`MODEL` or configuration path in the startup BCAR is resolved from the
directory containing that BCAR. A request that explicitly repeats a relative
model path resolves it from the request calculation directory before VPMDK
compares it with the resident path. Prefer absolute checkpoint paths when the
startup configuration and calculations live in different directory trees.
Server startup rejects a local-path `MODEL` that does not exist, preventing a
backend's default-model fallback from disagreeing with the resident identity
reported by `vpmdk status`. For local-only backends, extensionless values are
also paths. Non-path named-model identifiers remain supported, including
slash-containing FAIRChem identifiers such as `org/model`.
The same MODEL classifier is used by one-shot builders, server startup, status,
and request compatibility checks. Except for GRACE's documented warning plus
fallback, an explicit MODEL is never changed to the default. Unknown
static-registry names, rejected upstream registry names, missing checkpoints
for local-only selectors, and empty loader results fail explicitly. Backends
whose upstream runtime resolves opaque selectors (MatterSim, all FAIRChem v2
aliases, FAIRChem v1/OCP, and Nequix when `URLS` metadata is absent) receive the
exact explicit name and are responsible for accepting or rejecting it.
MatGL's `M3GNet-MP-2021.2.8-PES` default is also a named upstream model and is
not resolved relative to the BCAR directory. Other MatGL registry names are
also preserved as opaque model identities and loaded verbatim. Values that are
absolute, explicitly relative, have a checkpoint suffix, contain a path
separator, or already exist are normally treated as filesystem paths. Existing
local paths preserve their lexical spelling for the loader, including symlinks,
while server comparison uses their canonical real path. Matlantis model versions
are always opaque strings. Both FAIRChem generations may delegate unresolved
path-like selectors to their upstream loaders; MatterSim delegates only
non-path preset names. The legacy
`m3gnet` fallback uses its own bundled default only when `MODEL` is omitted;
explicit legacy values must resolve to readable local files and are never
replaced with that default after a load failure.

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
