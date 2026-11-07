# Backend Reference

## Reading This Table

`MODEL` semantics differ by backend. Some integrations accept named models,
others require local checkpoints, and some can do both. The table below records
the behavior implemented in `src/vpmdk_core/backends/`.

All builders use the same three-way MODEL resolver: an omitted value selects
the documented default, an existing filesystem value selects that local model,
and a supported registry/name value selects that named model. An explicit value
is never silently replaced with the backend default. Missing local paths for
VPMDK-owned local-only selectors and unsupported static-registry names fail
before calculator construction. For registries and checkpoint selectors whose
contents are resolved by the optional upstream package, config, or service,
VPMDK forwards the exact value and propagates rejection; an empty loader result
is also an error.

## Force-Field Backends

| Backend | Package / calculator | `MODEL` expectation | No-`MODEL` default | Key tags / notes |
|---------|----------------------|---------------------|--------------------|------------------|
| `ALIGNN` | `alignn` / `AlignnAtomwiseCalculator` | named force field, local model directory, or checkpoint file with sibling `config.json` | `alignn_ff_mptrj` | `DEVICE`; energy, forces, stress via ALIGNN-FF; pure PyTorch for current models, compatible DGL needed for legacy models |
| `CHGNET` | `chgnet` / `CHGNetCalculator` | local checkpoint path or upstream named/default model | upstream default loader | `DEVICE`, CHGNet graph-converter tags |
| `MATGL` / `M3GNET` | `matgl` or legacy `m3gnet` / `M3GNetCalculator` or `PESCalculator` | modern MatGL registry name, model directory, or checkpoint path; legacy explicit models must be local files | MatGL `M3GNet-MP-2021.2.8-PES`, or the legacy calculator default | `DEVICE`; explicit MatGL registry names are loaded verbatim; explicit legacy files never fall back to the default when loading fails |
| `MACE` | `mace-torch` / `MACECalculator` | local model path | upstream default calculator behavior when `MODEL` is omitted | `DEVICE`; an explicit missing/non-path `MODEL` is rejected rather than replaced with the default |
| `MACE_SOYO` | `MACE-SOYO` / `MACESoyoCalculator` | required local `.pth` checkpoint or exported `.pt2` package | none | CUDA only; `MACE_SOYO_HEAD`; independent of ordinary MACE; `MACE_SOYO_CUEQ_METHOD` for direct checkpoints |
| `PROPHET` | `prophet-mlip` / `KairosCalculator` | required local checkpoint, such as `prophet-oame-mbd.pt` | none | `DEVICE`, `PROPHET_USE_KERNEL`, `PROPHET_USE_COMPILE`; VPMDK defaults both options to false, selecting Prophet's portable e3nn path; kernels require a CUDA device and the upstream `kernel` extra |
| `MATTERSIM` | `mattersim` / `MatterSimCalculator` | optional existing local checkpoint or non-path `from_checkpoint` preset such as `mattersim-v1.0.0-5M` | calculator default | `DEVICE`, `MATTERSIM_COMPUTE_STRESS`, `MATTERSIM_STRESS_WEIGHT`; missing path-shaped values fail immediately; explicit presets require `from_checkpoint` or the legacy `load_path` API and are never replaced by the default |
| `MATLANTIS` | `pfp-api-client` / estimator service | opaque model version string or optional model name | `v9.0.0` with `R2SCAN` | `MATLANTIS_MODEL_VERSION`, `MATLANTIS_PRIORITY`, `MATLANTIS_CALC_MODE`, `MATLANTIS_MAX_RETRIES`; nondefault model versions use their own upstream calc-mode default unless a mode is explicit; same-named local files do not change version resolution |
| `EQNORM` | `eqnorm` / `EqnormCalculator` | local checkpoint or named model | `eqnorm-mptrj` | `EQNORM_VARIANT`, `EQNORM_COMPILE`; named models cached in `~/.cache/eqnorm` |
| `MATRIS` | `matris` / `MatRISCalculator` | local checkpoint or named model | `matris_10m_oam` | `MATRIS_TASK`, graph-converter tags; named models cached in `~/.cache/matris` |
| `ALPHANET` | `alphanet` / `AlphaNetCalculator` | local checkpoint plus JSON config, or named model | `AlphaNet-MATPES-r2scan` | `ALPHANET_CONFIG`, `ALPHANET_PRECISION`; named models cached in `~/.cache/alphanet` |
| `HIENET` | `hienet` / `HIENetCalculator` | local checkpoint / torchscript file, or named model | `HIENet-0` | `HIENET_FILE_TYPE`; named models cached in `~/.cache/hienet` |
| `NEQUIX` | `nequix` / `NequixCalculator` | local model file or named model | `nequix-mp-1` | `NEQUIX_BACKEND`, kernel/compile tags; names are checked against `URLS` when exposed, otherwise forwarded to the upstream runtime |
| `SEVENNET` | `sevenn` / `SevenNetCalculator` | local checkpoint / torchscript or named model | `7net-0` | `SEVENNET_FILE_TYPE`, `SEVENNET_MODAL`, accelerator tags |
| `FLASHTP` | `sevenn` + `flashTP_e3nn` | same as SevenNet, but checkpoint mode for flash acceleration | `7net-0` | forces flash acceleration and rejects conflicting accelerator flags |
| `BAM` | `bam-torch` / `RACECalculator` | required local checkpoint file: `BAM-MP-core.pkl` or `BAM-MP-DeNS-model.pkl` | none | `MODEL` required, `DEVICE`; DeNS requires the upstream `BAM_DeNS` version (see below) |
| `NEQUIP` | `nequip` / `NequIPCalculator` | required local deployed or compiled model file | none | `MODEL` required |
| `ALLEGRO` | `allegro` + `nequip` / `NequIPCalculator` | required local deployed or compiled model file | none | `MODEL` required |
| `ORB` | `orb-models` / `ORBCalculator` | optional local weights path plus optional ORB model key | `orb-v3-conservative-20-omat` | `ORB_MODEL`, `ORB_PRECISION`, `ORB_COMPILE` |
| `UPET` | `upet` / `UPETCalculator` | required local checkpoint or named model | none | `UPET_VERSION`, `UPET_NON_CONSERVATIVE`, `UPET_NEIGHBORLIST_DEVICE` / `UPET_NL_DEVICE` |
| `TACE` | `TACE` / `TACEAseCalc` | required local checkpoint or named foundation model | none | `TACE_DTYPE`, `TACE_SPIN_ON`, `TACE_NEIGHBORLIST_BACKEND`, `TACE_FIDELITY_IDX` / `TACE_LEVEL` |
| `EQUFLASH` | official `equflash` / `GGNN.common.calculator.UCalculator` | required local EquFlash or EquFlashV2 checkpoint file | none | checkpoint architecture is selected by GGNN metadata; `DEVICE=cpu` requests GGNN CPU mode, but checkpoints that depend on CUDA-only operators (including the validated V1 OAM runtime) still require CUDA |
| `EQUIFORMER_V3` | official `atomicarchitects/equiformer_v3` FAIRChem v1/OCP runtime / `OCPCalculator` | required local EquiformerV3 checkpoint | none | `EQUIFORMER_V3_MODULE`, `FAIRCHEM_CONFIG`, `DEVICE`; imports the EquiformerV3 registration module before using the FAIRChem v1 builder |
| `FAIRCHEM` / `FAIRCHEM_V2` / `ESEN` | `fairchem-core` 2.x / `FAIRChemCalculator` | upstream checkpoint/model selector, including names, paths, and provider-resolved path-shaped identifiers | `uma-s-1p1` with `FAIRCHEM_TASK=omat` | `FAIRCHEM_TASK`, `FAIRCHEM_INFERENCE_SETTINGS`, `DEVICE`; unresolved selectors are forwarded exactly to `from_model_checkpoint` |
| `FAIRCHEM_V1` | `fairchem-core==1.10.0` baseline or compatible OCP/FAIRChem v1 install / `OCPCalculator` or predictor | required checkpoint selector: existing local path or value resolved by OCP/config | none | `FAIRCHEM_CONFIG`, `FAIRCHEM_V1_PREDICTOR`, `DEVICE`; unresolved selectors are forwarded exactly and never replaced by a default |
| `GRACE` | TensorPotential / `TPCalculator` or `grace_fm` | local model path or foundation-model name | `GRACE-2L-MP-r6` when present in the installed registry; otherwise that registry's first model | Unknown non-path names emit a warning and use that effective default, matching TensorPotential-era behavior; status reports the model actually loaded; GRACE padding/dtype tags |
| `DEEPMD` | `deepmd-kit` / `DP` | required local frozen model or supported checkpoint | none | `DEEPMD_TYPE_MAP`, `DEEPMD_HEAD`; server mode requires an explicit model-ordered type map |

## Non-Obvious Entrypoints and Aliases

Some model-family names are not accepted as `MLP` tags. Use the actual VPMDK
entrypoint below.

| Model family / user-facing name | Use this `MLP` | Runtime path | Notes |
|---------------------------------|----------------|--------------|-------|
| EquiformerV2 / eqV2 checkpoints | `FAIRCHEM_V1` | FAIRChem v1/OCP `OCPCalculator` | Set `MODEL` to the local eqV2 checkpoint. There is no `MLP=EQUIFORMER_V2` alias. The original Equiformer V1 `graph_attention_transformer` checkpoints are not part of the documented support surface because they require older OCP trainer conventions. |
| EquiformerV3 checkpoints | `EQUIFORMER_V3` | FAIRChem v1/OCP `OCPCalculator` plus the official EquiformerV3 registration module | Dedicated VPMDK tag; do not use `FAIRCHEM_V2` for this path. |
| DPA / DPA-2 / DPA-3 / DPA-4 / DPA4 / SeZM checkpoints | `DEEPMD` | DeePMD-kit `DP` ASE calculator | Set `MODEL` to the local DeepMD checkpoint. DPA-4 / SeZM checkpoints require a DeePMD-kit release that registers the `dpa4` / `SeZM` model type. |
| UMA / FAIRChem v2 named models | `FAIRCHEM`, `FAIRCHEM_V2`, or `ESEN` | `fairchem-core` 2.x `FAIRChemCalculator` | `ESEN` is routed through the same builder as `FAIRCHEM_V2`; set `FAIRCHEM_TASK` when upstream requires it. |
| Flash-accelerated SevenNet | `FLASHTP` | SevenNet calculator with FlashTP acceleration forced | Same model semantics as SevenNet, but FlashTP support must be visible to `sevenn`. |
| EquFlash / EquFlashV2 checkpoints | `EQUFLASH` | official GGNN `UCalculator` | Set `MODEL` to a downloaded upstream `.pt` checkpoint. This is not the SevenNet `FLASHTP` runtime path. |
| Allegro deployed/compiled models | `ALLEGRO` | NequIP `NequIPCalculator` | The runtime calculator class is shared with NequIP, but use `MLP=ALLEGRO` for Allegro artifacts. |
| MatGL / M3GNet checkpoints | `MATGL` or `M3GNET` | MatGL or legacy M3GNet calculator | Both names route to the same VPMDK builder. |

## BAM-MP-core and BAM-MP-DeNS

Both models use `MLP = BAM` and
`bam_torch.tase.base_calculator.RACECalculator(model=..., device=...)`.
Existing BAM installations and BCAR files remain supported. There is no new
MLP tag, filename-based architecture detection, or automatic checkpoint download.

For BAM-MP-DeNS, install the DeNS-capable upstream implementation in your
VPMDK environment. The [Matbench model specification](https://github.com/janosh/matbench-discovery/blob/main/models/bam/bam-mp-dens.yml)
pins the following commit on the `BAM_DeNS` branch:

```bash
python -m pip install "bam-torch @ git+https://github.com/myung-group/BAM-torch@24ed0f042660a1e11ad5d91466b4275955df98c6"
curl -L --fail https://ndownloader.figshare.com/files/70285206 -o BAM-MP-DeNS-model.pkl
```

This installs the same `bam-torch` package, replacing its installed version;
use a separate environment if you need to retain a particular legacy BAM runtime.
The DeNS commit above was also tested with the original BAM-MP-core checkpoint
on GPU; see the [validation record](../development/validation.md#2026-10-26-bam-gpu-validation)
for the exact checkpoints, environments, and results.
The [upstream model notes](https://github.com/janosh/matbench-discovery/blob/main/models/bam/readme-bam-mp-dens.md)
explain why the DeNS checkpoint requires this implementation. A successful import
of a standard BAM calculator alone does not establish DeNS compatibility.

Set BCAR to:

```ini
MLP = BAM
MODEL = /absolute/path/to/BAM-MP-DeNS-model.pkl
DEVICE = cpu
```

Use `DEVICE = cuda` for GPU execution. For the original model, continue using
your existing BAM environment and `MODEL = /absolute/path/to/BAM-MP-core.pkl`.
Checkpoint loading and architecture selection are handled by BAM-torch; VPMDK
does not rewrite checkpoint settings or retry with another model.

Optional real-checkpoint tests accept `VPMDK_BAM_MODEL` (core) and
`VPMDK_BAM_DENS_MODEL` (DeNS), with `VPMDK_BAM_DEVICE` defaulting to `cpu`:

```bash
VPMDK_BAM_DENS_MODEL=/absolute/path/to/BAM-MP-DeNS-model.pkl pytest -m integration tests/integration/test_bam.py
```

## Capability Metadata

`vpmdk.list_backends()` and `vpmdk.get_backend_capabilities()` expose a compact
capability model. Highlights:

- `CHGNET`, `MACE`, and `TACE` declare `spin=True`
- `EQNORM`, `MACE`, `NEQUIX`, `ALLEGRO`, `NEQUIP`, `UPET`, `EQUIFORMER_V3`,
  `FAIRCHEM`, `GRACE`, and `DEEPMD` declare `fine_tune=True`
- `ALPHANET` and `DEEPMD` are marked as structure-aware backends
- `MATRIS_TASK=e` downgrades capabilities to energy-only

These metadata are descriptive and are not a full runtime guarantee.

A workdir run (`vpmdk --dir ...` and server mode) always reports per-atom
forces, so a backend configuration whose capability model declares `forces=False`
-- currently only `MATRIS_TASK=e` -- is refused up front with exit 1 instead of
computing anything. In server mode the decision uses the RESIDENT's effective
configuration, so a request that inherits the backend tags instead of restating
them gets the same exit code and the same warnings. Use `MATRIS_TASK=ef` or `efs` there; the energy-only
configuration remains available through the Python API for energy work. A
configuration without stress (`MATRIS_TASK=ef`) still runs: the OUTCAR/vasprun
stress block is omitted and a warning says so.

## Device Handling

Device handling is backend-specific:

- most Torch-style backends use `DEVICE` directly
- backends that call VPMDK's `_resolve_device()` helper prefer `cuda` when
  `torch.cuda.is_available()` and `DEVICE` is omitted
- other backends, such as FAIRChem and ORB, pass the raw `DEVICE` value through
  and otherwise stay on the backend's own default device behavior
- `NEQUIX_BACKEND=torch` supports explicit post-construction device transfer
- `NEQUIX_BACKEND=jax` follows the active JAX runtime rather than VPMDK moving the model
- `GRACE` ignores `DEVICE` entirely (a warning is printed when it is set):
  TPCalculator takes no device argument, so placement follows the installed
  TensorFlow build. GPU execution requires a CUDA-enabled `tensorflow`
  (tensorpotential pins `tensorflow<2.20`) that supports the local GPU and
  driver; on stacks where that combination is unavailable, GRACE-2L and the
  other TensorFlow GRACE models effectively run CPU-only. The upstream
  GRACE/FS C++ path is a separate CPU-oriented implementation.

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

- `BAM` imports `e3nn`. On installs with `e3nn < 0.5` and `torch >= 2.6`,
  `e3nn` itself can only be imported when `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`
  is set; VPMDK's own MACE import sets it when `mace-torch` is installed, but
  in an environment with neither `mace-torch` nor `e3nn >= 0.5.6` (BAM's
  declared requirement) the `BAM` backend reports itself unavailable.
- `FAIRCHEM_V1` and `FAIRCHEM_V2` are not environment-compatible in practice;
  use separate environments and pin `fairchem-core` versions intentionally.
- EquiformerV2 / eqV2 checkpoints use `MLP=FAIRCHEM_V1`. VPMDK intentionally
  does not expose an `EQUIFORMER_V2` backend tag, because the supported route is
  the FAIRChem v1/OCP checkpoint loader. Original Equiformer V1
  `graph_attention_transformer` checkpoints require older OCP trainer APIs and
  are not a documented compatibility target.
- `EQUIFORMER_V3` uses the FAIRChem v1/OCP calculator path, not the FAIRChem
  v2 `FAIRChemCalculator`. The official EquiformerV3 source must be importable
  so the `equiformer_v3` model is registered. Put the repository's `src`
  directory on `PYTHONPATH`, or set `EQUIFORMER_V3_MODULE` to a custom
  registration module.
- DPA-family checkpoints use `MLP=DEEPMD`. Newer DPA architectures such as
  DPA-4 / DPA4 / SeZM require a `deepmd-kit` build that includes the matching
  PyTorch model registration and compiled operator support.
- `FAIRCHEM` / `FAIRCHEM_V2` / `ESEN` default to the `uma-s-1p1` checkpoint with
  `FAIRCHEM_TASK=omat`. This registry name is present in both the
  fairchem-core 2.13.0 validation environment and current 2.x releases. For
  other named checkpoints, set `FAIRCHEM_TASK` explicitly when the upstream
  calculator requires it.
- `FLASHTP` requires `sevenn` plus FlashTP support visible to the installed
  `SevenNetCalculator`.
- `EQUFLASH` requires the official package exposing
  `GGNN.common.calculator.UCalculator`. Both EquFlash and EquFlashV2 use this
  entrypoint; GGNN selects the architecture from checkpoint metadata. Download
  an upstream checkpoint such as `equflash_oam.pt` or `equflashv2_oam.pt` and
  set `MODEL` to its local path. The validated EquFlash V1 OAM checkpoint uses
  CUDA-only `cuequivariance` operators; CPU support therefore depends on the
  checkpoint and upstream runtime, while both V1 and V2 are validated on CUDA.
  EquFlash 0.0.2 cannot select a nonzero logical CUDA index, so VPMDK rejects
  `DEVICE=cuda:N` for `N != 0`; use `CUDA_VISIBLE_DEVICES=N` with `DEVICE=cuda`
  to map the desired physical GPU to logical GPU 0.
- `UPET` defaults to building neighbor lists on CPU when the model is on CUDA.
  Set `UPET_NEIGHBORLIST_DEVICE=model` to run neighbor-list construction on the
  model device when the local `metatomic`/`vesin` stack supports it.
- `SEVENNET_ENABLE_CUEQ`, `SEVENNET_ENABLE_FLASH`, and `SEVENNET_ENABLE_OEQ`
  are mutually exclusive when explicitly enabled.
- `HIENET_FILE_TYPE=torchscript` requires `MODEL` to point to a local TorchScript file.
- `ALPHANET` local checkpoints need a matching JSON config, either explicit or
  inferred from the checkpoint directory.
- `DEEPMD` local checkpoints are mandatory; there is no built-in named default.
- `PROPHET` requires a local checkpoint. `PROPHET_USE_KERNEL=0` is the portable
  default and works on CPU or CUDA. Set `PROPHET_USE_KERNEL=1` only with a CUDA
  `DEVICE` and a `prophet-mlip[kernel]` installation.

## ALIGNN-FF

`MLP=ALIGNN` uses the [official ALIGNN-FF ASE calculator](https://github.com/atomgptlab/alignn/blob/main/alignn/ff/calculators.py).
This integration accepts scalar force-field regression configurations named
`alignn_atomwise_pure` or `alignn_atomwise`, with `calculate_gradient=true`,
`classification=false`, and `output_features=1`. Property predictors (band
gaps, formation energies, etc.) and other atomwise architectures are outside
this backend's supported E/F/S profile. `ealignn_atomwise` is explicitly
unsupported, including configurations with torque postprocessing disabled;
its E/F/S inference paths have not been validated as a supported backend.

ASE inputs must be fully periodic: all three `atoms.pbc` axes must be enabled.
Any disabled axis raises `ValueError` before upstream structure conversion or
inference, including molecular, slab and wire boundary conditions. This check
applies at each evaluation of calculators returned by `build_calculator()`
and to `single_point()`. Input PBC flags are preserved. The current upstream
ASE-to-JARVIS conversion drops these flags and its graph path always creates
fully periodic neighbors, so nonperiodic and partially periodic inputs are
outside the supported profile.

```ini
MLP = ALIGNN
MODEL = alignn_ff_mptrj
DEVICE = cpu
```

Omitting `MODEL` selects `alignn_ff_mptrj` explicitly, independently of changes
to the upstream default. Named models are checked against the installed
ALIGNN force-field registry and downloaded by its loader. Unknown names and
missing paths raise errors. The current public models include
`alignn_ff_matpes_pbe`, `alignn_ff_matpes_r2scan`, and `alignn_ff_omat24_1m`;
registry contents depend on the installed upstream version.

For local use, specify a directory containing `best_model.pt` and `config.json`,
or a checkpoint file with a sibling `config.json`. No download is attempted
for local models. BCAR-relative paths follow the usual VPMDK work-directory
rules. For a local model, resident identity includes the resolved checkpoint,
the sibling `config.json` path selected before resolving weight symlinks, and
a SHA-256 snapshot of that configuration. Sharing weight bytes does not permit
a different sibling config. A directory and its `best_model.pt` are equivalent
when the config also matches. Omitted `MODEL` still inherits the loaded
resident. The inferred `ALIGNN_CONFIG` and `ALIGNN_CONFIG_SHA256` status fields
are diagnostic metadata, not BCAR overrides.
Models using `alignn_atomwise_pure` and `pure_torch` need no DGL;
legacy DGL graph configurations need compatible DGL/PyTorch versions.
`v12.2.2024_dft_3d_307k` was validated on CPU and CUDA with DGL 2.1.0+cu121
and torch 2.2.1+cu121. The adapter validates the effective atom-count scaling
of the actual model branch before loading weights. Its calculator always
returns total energy (`intensive=true`), so the force calculation must apply
exactly one factor of the atom count to the learned per-atom energy gradient:

| Model branch | Compatible `(energy_mult_natoms, force_mult_natoms)` |
|--------------|---------------------------------------------------|
| `alignn_atomwise_pure`, either `include_pos_deriv` value | `(true, false)` or `(false, true)` |
| DGL `alignn_atomwise`, `include_pos_deriv=false` | `(true, false)` or `(false, true)` |
| DGL `alignn_atomwise`, `include_pos_deriv=true` | Rejected: upstream has no usable stress |
| `ealignn_atomwise` | Rejected: unsupported architecture |

Pure PyTorch ignores `include_pos_deriv`. DGL Cartesian configurations
(`alignn_atomwise` with `include_pos_deriv=true`) are rejected even when their
force normalization is compatible: upstream leaves stress as a placeholder
and its ASE wrapper silently replaces it with zeros. This backend always
enables stress, so that branch cannot provide the required observables.
Omitted atom-count flags default to `energy_mult_natoms=true` and
`force_mult_natoms=false`; normalization flags must be JSON booleans.
For pure PyTorch, the alternate `(false, true)` pair also requires
`use_zbl=false`: upstream otherwise multiplies ZBL forces by the atom count
while still reporting its unscaled total energy. ZBL is enabled automatically
when `use_zbl` is omitted or null in a force-field configuration.
Incompatible configurations, including `v5.27.2024`, raise an explicit error.

The accepted derivative/output profile also requires:

- `lg_on_fly=true` whenever `alignn_layers>0`, so bond-angle derivatives remain
  in the autograd graph. Omitted values default to `true` and `2`, respectively.
  VPMDK inserts the layer-count default into the private runtime config because
  the upstream calculator directly indexes that key; weights must match the
  resulting two-layer architecture. Explicit layer counts are retained.
  Pure PyTorch's ASE path requires `alignn_layers>0`; DGL's zero-angle-layer
  branch can use `lg_on_fly=false`.
- `grad_multiplier=-1`, `stress_multiplier=1`, and `link=identity`, including
  their upstream defaults when omitted. Numeric settings must be finite JSON
  numbers. Sign changes, additional gradient scales and nonlinear output links
  are rejected rather than changing a pretrained force convention.

`stresswise_weight` gates inference stress without multiplying its value.
When this setting is omitted or zero, VPMDK enables it with `0.1` in a private
runtime config. Positive values are retained. The checkpoint and sibling
`config.json` are unchanged. The ASE wrapper's extra force and stress scales
are fixed at one, its additional atom-count/batch-size force multipliers are
disabled, and per-graph batch stress is enabled explicitly.

Models that omit outgoing-edge contributions (`add_reverse_forces=false`), including
`alignnff_fmult`, are also rejected, even with `force_mult_natoms=true`.
This check applies to the DGL `alignn_atomwise` edge-derivative branch and to
`alignn_atomwise_pure`, which always uses edge derivatives.
VPMDK does not silently change a
pretrained force field's gradient scale. See the
[GPU/DGL validation](../development/validation.md#gpu-and-legacy-dgl-follow-up).

The upstream calculator multiplies per-atom energies by the atom count and
converts stress from GPa to ASE eV/Å³. VPMDK retains these conversions and uses
the default force scale. For models with `use_penalty=true`, the adapter also
includes the short-range penalty differentiated by upstream in the reported
energy. It sums `penalty_factor * max(penalty_threshold - r, 0)` over the actual
directed inference edges, including periodic images, with the model's energy
normalization. For DGL `alignn_atomwise` with `use_cutoff_function=true` and
`multiply_cutoff=false`, `r` is the upstream cutoff-transformed length, using
the configured `inner_cutoff` and `exponent`. Other accepted branches use the
physical bond length for the penalty, including pure PyTorch's cutoff branch.
Legacy configurations that already report this contribution
are not corrected twice. Forces and stress retain their pretrained scales.
The adapter exposes the standard energy, forces and
stress workflows, including resident reuse. Training and spin prediction are
not exposed. This backend requires fully periodic structures.

See the [runnable example](../../examples/alignn_si/README.md),
[environment recipe](../development/backend-environments.md#alignn-ff), and
[real-model checks](../development/validation.md#2026-11-06-alignn-ff-validation).

## MACE-SOYO

[MACE-SOYO](https://github.com/PrincessHakuryu/MACE-SOYO) has a separate model
architecture, checkpoint format and ASE runtime. Its public `.pth` was tested
with VPMDK's `MLP=MACE` and failed (`dict` has no `.to` method); the generated
AOTI `.pt2` also failed the ordinary MACE loader. Use the dedicated backend:

```ini
MLP = MACE_SOYO
MODEL = /absolute/path/to/mace_soyo_mptraj.pth
DEVICE = cuda:0
MACE_SOYO_HEAD = mptraj
MACE_SOYO_CUEQ_METHOD = naive
```

`.pth` selects direct inference using the upstream checkpoint loader and
`TensorEFS` energy/force/stress implementation. `MACE_SOYO_CUEQ_METHOD=native`
(the default) retains upstream kernels; `naive` explicitly selects
cuEquivariance's PyTorch evaluator on CUDA. The latter was validated on
TITAN V. It keeps the learned weights and model equations and does not use
the distributed accelerated kernels, whose SM 7.0 support is missing.
Direct inference supports unconditioned checkpoints, including the public
MPtraj model, and does not perform AOTI export at calculation startup.

`.pt2` selects upstream `MACESoyoCalculator` AOTInductor inference. Export
with `vpmdk-mace-soyo-export` on compatible target hardware. The command
checks native kernels and publishes the package and a JSON provenance report
only after the upstream numerical verification succeeds.
The package fixes its cutoff, precision, ABI and method at export time;
`MACE_SOYO_CUEQ_METHOD=naive` is rejected for `.pt2` because runtime selection
cannot change compiled operations. The corresponding `BCAR.aoti` example and compiled-versus-eager integration
checks are included. An isolated Python 3.12/cuEquivariance 0.6.1 configuration
also passed native AOTI inference on TITAN V (SM 7.0). This is an experimental
override of upstream's 0.12 pins, with a version-scoped fix for duplicate Linear
evaluator registrations; it does not change learned parameters. See the
validation notes for the exact environment and package hashes.

`MODEL` is required and local; VPMDK does not download weights implicitly.
CPU requests and unsupported file formats fail explicitly. Head names are
case sensitive. `MACE_SOYO_HEAD` may be omitted for a single-head model;
a multi-head checkpoint requires it. Server mode records the loaded head
and method and rejects requests that change them. Dispersion corrections,
training and spin prediction are not exposed. For periodic stress, use fully
periodic structures; nonperiodic stress follows the upstream placeholder
convention.

See the [example](../../examples/mace_soyo_si/README.md),
[environment recipe](../development/backend-environments.md#mace-soyo), and
[validation record](../development/validation.md#2026-11-06-mace-soyo-validation).
