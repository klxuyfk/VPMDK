# Backend Reference

## Reading This Table

`MODEL` semantics differ by backend. Some integrations accept named models,
others require local checkpoints, and some can do both. The table below records
the behavior implemented in `src/vpmdk_core/backends/`.

## Force-Field Backends

| Backend | Package / calculator | `MODEL` expectation | No-`MODEL` default | Key tags / notes |
|---------|----------------------|---------------------|--------------------|------------------|
| `CHGNET` | `chgnet` / `CHGNetCalculator` | local checkpoint path or upstream named/default model | upstream default loader | `DEVICE`, CHGNet graph-converter tags |
| `MATGL` / `M3GNET` | `matgl` or legacy `m3gnet` / `M3GNetCalculator` | model directory or checkpoint path when supported | upstream default model | `DEVICE` |
| `MACE` | `mace-torch` / `MACECalculator` | local model path | upstream default calculator behavior | `DEVICE` |
| `MATTERSIM` | `mattersim` / `MatterSimCalculator` | optional local model path | calculator default | simple wrapper |
| `MATLANTIS` | `pfp-api-client` / estimator service | model version string or optional model name | `v8.0.0` | `MATLANTIS_MODEL_VERSION`, `MATLANTIS_PRIORITY`, `MATLANTIS_CALC_MODE` |
| `EQNORM` | `eqnorm` / `EqnormCalculator` | local checkpoint or named model | `eqnorm-mptrj` | `EQNORM_VARIANT`, `EQNORM_COMPILE`; named models cached in `~/.cache/eqnorm` |
| `MATRIS` | `matris` / `MatRISCalculator` | local checkpoint or named model | `matris_10m_oam` | `MATRIS_TASK`, graph-converter tags; named models cached in `~/.cache/matris` |
| `ALPHANET` | `alphanet` / `AlphaNetCalculator` | local checkpoint plus JSON config, or named model | `AlphaNet-MATPES-r2scan` | `ALPHANET_CONFIG`, `ALPHANET_PRECISION`; named models cached in `~/.cache/alphanet` |
| `HIENET` | `hienet` / `HIENetCalculator` | local checkpoint / torchscript file, or named model | `HIENet-0` | `HIENET_FILE_TYPE`; named models cached in `~/.cache/hienet` |
| `NEQUIX` | `nequix` / `NequixCalculator` | local model file or named model | `nequix-mp-1` | `NEQUIX_BACKEND`, kernel/compile tags |
| `SEVENNET` | `sevenn` / `SevenNetCalculator` | local checkpoint / torchscript or named model | `7net-0` | `SEVENNET_FILE_TYPE`, `SEVENNET_MODAL`, accelerator tags |
| `FLASHTP` | `sevenn` + `flashTP_e3nn` | same as SevenNet, but checkpoint mode for flash acceleration | `7net-0` | forces flash acceleration and rejects conflicting accelerator flags |
| `NEQUIP` | `nequip` / `NequIPCalculator` | required local deployed or compiled model file | none | `MODEL` required |
| `ALLEGRO` | `allegro` + `nequip` / `NequIPCalculator` | required local deployed or compiled model file | none | `MODEL` required |
| `ORB` | `orb-models` / `ORBCalculator` | optional local weights path plus optional ORB model key | `orb-v3-conservative-20-omat` | `ORB_MODEL`, `ORB_PRECISION`, `ORB_COMPILE` |
| `UPET` | `upet` / `UPETCalculator` | required local checkpoint or named model | none | `UPET_VERSION`, `UPET_NON_CONSERVATIVE` |
| `TACE` | `TACE` / `TACEAseCalc` | required local checkpoint or named foundation model | none | `TACE_DTYPE`, `TACE_SPIN_ON`, `TACE_NEIGHBORLIST_BACKEND`, `TACE_FIDELITY_IDX` / `TACE_LEVEL` |
| `EQUFLASH` | EquFlash / `GGNN.common.calculator.UCalculator` | required local checkpoint file | none | currently no bundled named models |
| `FAIRCHEM` / `FAIRCHEM_V2` / `ESEN` | `fairchem-core` 2.x / `FAIRChemCalculator` | named checkpoint/model identifier | `esen-sm-direct-all-oc25` | `FAIRCHEM_TASK`, `FAIRCHEM_INFERENCE_SETTINGS`, `DEVICE` |
| `FAIRCHEM_V1` | `fairchem-core==1.10.0` baseline or compatible OCP/FAIRChem v1 install / `OCPCalculator` or predictor | required local checkpoint; config usually required | none | `FAIRCHEM_CONFIG`, `FAIRCHEM_V1_PREDICTOR`, `DEVICE` |
| `GRACE` | TensorPotential / `TPCalculator` or `grace_fm` | local model path or foundation-model name | `GRACE-2L-MP-r6` when available | GRACE padding/dtype tags |
| `DEEPMD` | `deepmd-kit` / `DP` | required local frozen model or supported checkpoint | none | `DEEPMD_TYPE_MAP`, `DEEPMD_HEAD` |

## Capability Metadata

`vpmdk.list_backends()` and `vpmdk.get_backend_capabilities()` expose a compact
capability model. Highlights:

- `CHGNET`, `MACE`, and `TACE` declare `spin=True`
- `EQNORM`, `MACE`, `NEQUIX`, `ALLEGRO`, `NEQUIP`, `UPET`, `FAIRCHEM`, `GRACE`,
  and `DEEPMD` declare `fine_tune=True`
- `ALPHANET` and `DEEPMD` are marked as structure-aware backends
- `MATRIS_TASK=e` downgrades capabilities to energy-only

These metadata are descriptive and are not a full runtime guarantee.

## Device Handling

Device handling is backend-specific:

- most Torch-style backends use `DEVICE` directly
- if `DEVICE` is omitted, VPMDK's helper prefers `cuda` when `torch.cuda.is_available()`
- `NEQUIX_BACKEND=torch` supports explicit post-construction device transfer
- `NEQUIX_BACKEND=jax` follows the active JAX runtime rather than VPMDK moving the model

## Named-Model Downloads Implemented by VPMDK

These integrations perform their own download/caching:

- Eqnorm
- MatRIS
- AlphaNet
- HIENet

For other backends, named-model behavior is delegated to the upstream package if
it supports that concept.

## Charge-Density Backends

`CHGCAR` generation is configured separately from `MLP`.

| Charge backend | `CHARGE_MODEL` expectation | Notes |
|----------------|----------------------------|-------|
| `CHARGE3NET` | checkpoint path | falls back to `<CHARGE_SOURCE_DIR>/models/charge3net_mp.pt` when present |
| `DEEPDFT` | model directory or a file inside one | normalizes `best_model.pth` to the parent directory |
| `DEEPCDP` | checkpoint path or directory containing exactly one `.pt` file | can also discover metadata JSON next to the checkpoint |

## Known Backend-Specific Caveats

- `FAIRCHEM_V1` and `FAIRCHEM_V2` are not environment-compatible in practice;
  use separate environments and pin `fairchem-core` versions intentionally.
- `FLASHTP` requires `sevenn` plus FlashTP support visible to the installed
  `SevenNetCalculator`.
- `SEVENNET_ENABLE_CUEQ`, `SEVENNET_ENABLE_FLASH`, and `SEVENNET_ENABLE_OEQ`
  are mutually exclusive when explicitly enabled.
- `HIENET_FILE_TYPE=torchscript` requires `MODEL` to point to a local TorchScript file.
- `ALPHANET` local checkpoints need a matching JSON config, either explicit or
  inferred from the checkpoint directory.
- `DEEPMD` local checkpoints are mandatory; there is no built-in named default.
