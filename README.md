# VPMDK

VPMDK (*Vasp-Protocol Machine-learning Dynamics Kit*, aka “VasP-MoDoKi”) is a lightweight engine that reads and writes VASP-style inputs and outputs while running molecular dynamics, relaxations, and single-point calculations with ASE-compatible machine-learning interatomic potentials.

It keeps familiar VASP artifacts such as `POSCAR`, `INCAR`, `OUTCAR`, `OSZICAR`, and `vasprun.xml`, while the actual calculator is an ML potential.

Supported calculators currently include **CHGNet**, **SevenNet**, **FlashTP** (via SevenNet), **EquFlash** (with a local checkpoint), **MatterSim**, **MACE**, **Matlantis**, **Eqnorm**, **MatRIS**, **AlphaNet**, **HIENet**, **Nequix**, **NequIP**, **Allegro**, **ORB**, **UPET**, **TACE**, **MatGL** (via M3GNet), **FAIRChem**, **GRACE**, and **DeePMD-kit**. Availability depends on which Python packages are installed.

*Not affiliated with, endorsed by, or a replacement for VASP; “VASP” is a trademark of its respective owner. VPMDK only mimics VASP I/O conventions for compatibility.*

## Quick Start

### 1. Install VPMDK and one backend

The fastest way to start is CHGNet:

```bash
pip install vpmdk chgnet
```

If you want another backend, install `vpmdk` plus the corresponding package such as `mace-torch`, `matgl`, `matris`, `sevenn`, `eqnorm`, or `orb-models`.

### 2. Prepare the current working directory

You always need `POSCAR`.

- `INCAR` is optional, but usually present
- `BCAR` is optional, but usually the easiest way to choose a backend and model
- `POTCAR` is accepted for compatibility and species alignment, but is not used as a real pseudopotential input
- `KPOINTS`, `WAVECAR`, and `CHGCAR` are ignored

The typical workflow is to enter the directory that already contains your VASP-style inputs and run VPMDK there:

```text
.
├── POSCAR
├── INCAR
└── BCAR
```

### 3. Start from one of these minimal inputs

Single-point calculation:

`INCAR`

```text
IBRION = -1
NSW = 0
```

`BCAR`

```text
MLP=CHGNET
DEVICE=cuda
```

Geometry relaxation:

`INCAR`

```text
IBRION = 2
NSW = 200
EDIFFG = -0.02
ISIF = 3
```

`BCAR`

```text
MLP=CHGNET
DEVICE=cuda
```

Molecular dynamics:

`INCAR`

```text
IBRION = 0
NSW = 1000
POTIM = 1.0
TEBEG = 300
TEEND = 300
MDALGO = 3
```

`BCAR`

```text
MLP=MATRIS
MODEL=matris_10m_oam
DEVICE=cuda
```

### 4. Run

```bash
vpmdk
```

### 5. Check the outputs

Typical outputs are:

- `CONTCAR`
- `OUTCAR`
- `OSZICAR`
- `vasprun.xml`
- `XDATCAR` for MD runs

## Input Overview

### What Each File Does

| File | Required | Purpose |
|------|----------|---------|
| `POSCAR` | Yes | Structure, lattice, species, and coordinates |
| `INCAR` | No | Run mode and control parameters such as MD vs relaxation |
| `BCAR` | No | Backend selection, model selection, device, and backend-specific knobs |
| `POTCAR` | No | Species-name compatibility only |

If `BCAR` is omitted, VPMDK defaults to `MLP=CHGNET`.

### Most-Used INCAR Tags

These are the tags most users actually need first:

| Tag | What it controls | Typical values |
|-----|------------------|----------------|
| `IBRION` | Run mode | `-1` single-point, `0` MD, `2` relaxation |
| `NSW` | Number of steps | `0`, `100`, `1000`, ... |
| `EDIFFG` | Relaxation stopping criterion | `-0.02` is a common force threshold |
| `ISIF` | Whether the cell relaxes | `2` fixed cell, `3` relax ions + cell |
| `POTIM` | MD time step in fs | `1.0` to `2.0` |
| `TEBEG`, `TEEND` | MD temperature range | `300`, `1000`, ... |
| `MDALGO` | MD thermostat/integrator | `0` NVE, `3` Langevin, `5` Bussi |
| `MAGMOM` | Initial magnetic moments | VASP-style syntax such as `2*1.0 4*0.0` |

### Most-Used BCAR Tags

These are the tags most users need to understand immediately:

| Tag | What it controls | Typical values |
|-----|------------------|----------------|
| `MLP` | Backend name | `CHGNET`, `MACE`, `MATGL`, `MATRIS`, ... |
| `MODEL` | Checkpoint path or named model | `/path/to/model.pt`, `matris_10m_oam`, `7net-0`, ... |
| `DEVICE` | Device hint | `cpu`, `cuda`, `cuda:0` |
| `MATRIS_TASK` | MatRIS task | `e`, `ef`, `efs`, `efsm` |
| `GRAPH_CONVERTER` / `GRAPH_CONVERTER_ALGORITHM` | CHGNet / MatRIS graph converter | `fast`, `legacy` |
| `WRITE_PSEUDO_SCF` | Emit pseudo-SCF compatibility blocks | `0`, `1` |
| `WRITE_LAMMPS_TRAJ` | Write `lammps.lammpstrj` during MD | `0`, `1` |
| `LAMMPS_TRAJ_INTERVAL` | Trajectory write interval | `1`, `10`, ... |

Backend-specific graph converter overrides take precedence over the shared tags:

- `CHGNET_GRAPH_CONVERTER`
- `CHGNET_GRAPH_CONVERTER_ALGORITHM`
- `MATRIS_GRAPH_CONVERTER`
- `MATRIS_GRAPH_CONVERTER_ALGORITHM`

## Installation

Install from PyPI:

```bash
pip install vpmdk
```

Or from a checkout:

```bash
pip install -e .
```

You will also need:

- `ase`
- `pymatgen`
- one or more backend packages such as `chgnet`, `mace-torch`, `matgl`, `matris`, `sevenn`, `eqnorm`, `alphanet`, `hienet`, `nequix`, `upet`, `tace`, `orb-models`, `nequip`, or `deepmd-kit`

Install the GPU-enabled build of PyTorch or JAX if you want GPU execution.

## Optional Compatibility Modes

The standard workflow is to run `vpmdk` in the current working directory.

If needed, VPMDK also supports:

- `vpmdk --dir calc_dir`
  Run against a different input directory without changing the current directory.
- `python vpmdk.py`
  Legacy wrapper from a repository checkout.

## Detailed INCAR Reference

### Supported INCAR Tags

VPMDK reads a subset of common VASP `INCAR` settings. Other tags are ignored with a warning.

| Tag | Meaning | Default / Notes |
|-----|---------|-----------------|
| `NSW` | Number of ionic steps. | `0` (single-point calculation). |
| `IBRION` | Ionic movement algorithm. | `<0` performs a single-point calculation without moving ions, `0` runs molecular dynamics, positive values trigger a BFGS geometry optimisation with a fixed cell. Defaults to `-1`. |
| `ISIF` | Controls whether the cell changes during relaxations. | `2` keeps the cell fixed (default). `3` relaxes ions and the full cell, `4` keeps the volume constant while optimising ions and the cell shape, `5` optimises the cell shape at constant volume with fixed ions, `6` changes only the cell, `7` enables isotropic cell changes with fixed ions, and `8` couples ionic relaxations to isotropic volume changes. Stress output follows VASP-style semantics: `ISIF<=0` omits stress blocks, `ISIF=1` writes trace-only pressure information, and `ISIF>=2` writes the full stress tensor block. Unsupported values fall back to `2` behavior with a warning. |
| `EDIFFG` | Convergence criterion for relaxations. | `<0`: force criterion using `abs(EDIFFG)` in eV/Å (default `-0.02`). `>0`: energy criterion using `EDIFFG` in eV (`|ΔE|` between ionic steps). |
| `TEBEG` | Initial temperature in kelvin for molecular dynamics (`IBRION=0`). | `300`. |
| `TEEND` | Final temperature in kelvin when ramping MD runs. | Same as `TEBEG`. Temperature is linearly ramped between `TEBEG` and `TEEND` over the MD steps. |
| `POTIM` | Time step in femtoseconds for molecular dynamics (`IBRION=0`). | `2`. |
| `MDALGO` | Selects the MD integrator / thermostat. | `0` (NVE). When left at `0`, `SMASS>0` falls back to Nose–Hoover (`MDALGO=2`) and `SMASS<0` falls back to Langevin (`MDALGO=3`). See [MD algorithms](#md-algorithms) for details. |
| `SMASS` | Thermostat-specific mass parameter. | Used for Nose–Hoover time constant (`abs(SMASS)` fs) or as a fallback to set `LANGEVIN_GAMMA` when negative. |
| `ANDERSEN_PROB` | Collision probability for the Andersen thermostat. | `0.1`. Only used with `MDALGO=1`. |
| `LANGEVIN_GAMMA` | Friction coefficient (1/ps) for Langevin dynamics. | `1.0`. Only used with `MDALGO=3`; falls back to `abs(SMASS)` when `SMASS<0`. |
| `CSVR_PERIOD` | Relaxation time (fs) for the canonical sampling velocity rescaling thermostat. | `max(100×POTIM, POTIM)`. Only used with `MDALGO=5`. |
| `NHC_NCHAINS` | Nose–Hoover chain length. | `1` for `MDALGO=2`, `3` for `MDALGO=4`. |
| `PSTRESS` | External pressure in kBar applied during relaxations. | Converts to scalar pressure in the ASE optimiser when `ISIF` allows cell changes. |
| `MAGMOM` | Initial magnetic moments. | Parsed like VASP; supports shorthand such as `2*1.0`. |
| `IMAGES` | Number of NEB images. | Enables NEB mode detection. When numbered image directories (`00`, `01`, ...) exist, VPMDK runs them sequentially and emits NEB-style projection lines in each image `OUTCAR`. |
| `LCLIMB` | Climbing-image switch for NEB. | Used as a compatibility hint for VTST post-processing outputs. |
| `SPRING` | Spring constant for NEB. | Used as a compatibility hint for VTST post-processing outputs. |

### MD Algorithms

`MDALGO` selects between different ASE molecular dynamics drivers. Some options require optional ASE modules; if they are missing VPMDK falls back to plain velocity-Verlet (NVE) integration and prints a warning.

| `MDALGO` | Integrator | Notes |
|---------:|-----------|-------|
| `0` | Velocity-Verlet (NVE) | No thermostat. |
| `1` | Andersen thermostat | Controlled by `ANDERSEN_PROB`. |
| `2` | Nose–Hoover chain (single thermostat) | Uses `SMASS`/`NHC_NCHAINS` to configure the chain. |
| `3` | Langevin thermostat | Uses `LANGEVIN_GAMMA` (or `abs(SMASS)` if negative). |
| `4` | Nose–Hoover chain (three thermostats) | Chain length defaults to 3 unless overridden. |
| `5` | Bussi (canonical sampling velocity rescaling) thermostat | Uses `CSVR_PERIOD`. |

## Detailed BCAR Reference

`BCAR` is a `key=value` file for backend selection, model selection, and backend-specific options.

Minimal example:

```text
MLP=CHGNET
MODEL=/path/to/model
DEVICE=cuda
```

### Core Selection

| Tag | Meaning | Default |
|-----|---------|---------|
| `MLP` | Backend name (`CHGNET`, `MACE`, `MATGL`, `MATLANTIS`, `MATTERSIM`, `EQNORM`, `MATRIS`, `ALPHANET`, `HIENET`, `NEQUIX`, `SEVENNET`, `FLASHTP`, `EQUFLASH`, `NEQUIP`, `ALLEGRO`, `ORB`, `UPET`, `TACE`, `FAIRCHEM`, `FAIRCHEM_V2`, `FAIRCHEM_V1`, `GRACE`, `DEEPMD`) | `CHGNET` |
| `MODEL` | Path to a trained parameter set or a backend-defined named model | Backend default or bundled weights |
| `DEVICE` | Device hint for backends that support it (`cpu`, `cuda`, `cuda:N`) | Auto-detects GPU when available |

`NNP` is accepted as a backward-compatible alias of `MLP`.

### Output And Workflow Aids

| Tag | Meaning | Default |
|-----|---------|---------|
| `WRITE_ENERGY_CSV` | Write `energy.csv` during relaxation (`1` to enable) | `0` |
| `WRITE_LAMMPS_TRAJ` | Write a LAMMPS trajectory during MD (`1` to enable) | `0` |
| `LAMMPS_TRAJ_INTERVAL` | MD steps between trajectory frames (only when `WRITE_LAMMPS_TRAJ=1`) | `1` |
| `WRITE_PSEUDO_SCF` | Add pseudo electronic-step compatibility blocks to `OSZICAR`, `OUTCAR`, and `vasprun.xml` (`1` to enable) | `0` |
| `DEEPMD_TYPE_MAP` | Comma/space-separated species list mapped to the DeePMD graph | Inferred from `POSCAR` order |
| `DEEPMD_HEAD` | Select a DeePMD model head by name (when supported by the checkpoint) | Unset |

`WRITE_OSZICAR_PSEUDO_SCF` is accepted as a backward-compatible alias of `WRITE_PSEUDO_SCF`.

### Backend-Specific Knobs

| Tag | Applies to | Meaning | Default |
|-----|-----------|---------|---------|
| `GRAPH_CONVERTER` / `GRAPH_CONVERTER_ALGORITHM` | CHGNet, MatRIS | Shared graph converter selector (`fast`, `legacy`) | Upstream default |
| `CHGNET_GRAPH_CONVERTER` / `CHGNET_GRAPH_CONVERTER_ALGORITHM` | CHGNet | CHGNet-specific graph converter selector | Shared/default behavior |
| `MATRIS_GRAPH_CONVERTER` / `MATRIS_GRAPH_CONVERTER_ALGORITHM` | MatRIS | MatRIS-specific graph converter selector | Shared/default behavior |
| `MATLANTIS_MODEL_VERSION` | Matlantis | Estimator version identifier | `v8.0.0` |
| `MATLANTIS_PRIORITY` | Matlantis | Job priority forwarded to the estimator | `50` |
| `MATLANTIS_CALC_MODE` | Matlantis | Calculation mode (`CRYSTAL`, `MOLECULE`, …) | `PBE` |
| `ORB_MODEL` | ORB | Pretrained architecture key recognised by `orb_models` | `orb-v3-conservative-20-omat` |
| `ORB_PRECISION` | ORB | Floating-point precision string forwarded to orb-model loaders | `float32-high` |
| `ORB_COMPILE` | ORB | Whether to `torch.compile` the ORB model | Library default |
| `EQNORM_VARIANT` | Eqnorm | Eqnorm architecture variant used with a local checkpoint (`eqnorm-mptrj`, `eqnorm-omat`, `eqnorm-max-mptrj`) | Inferred from `MODEL` filename or named-model default |
| `EQNORM_COMPILE` | Eqnorm | Whether to `torch.compile` the Eqnorm model | `0` |
| `MATRIS_TASK` | MatRIS | Prediction task forwarded to `MatRISCalculator` (`e`, `ef`, `efs`, `efsm`) | `efs` |
| `ALPHANET_CONFIG` | AlphaNet | Path to the AlphaNet JSON config when `MODEL` is a local checkpoint and the config cannot be inferred | Paired config for named models or inferred sibling JSON |
| `ALPHANET_PRECISION` | AlphaNet | Floating-point precision forwarded to the AlphaNet ASE calculator | `32` |
| `HIENET_FILE_TYPE` | HIENet | Model serialization type accepted by `HIENetCalculator` (`checkpoint`, `torchscript`) | `checkpoint` |
| `NEQUIX_BACKEND` | Nequix | Upstream backend (`jax` or `torch`) | `jax` |
| `NEQUIX_USE_KERNEL` | Nequix | Enable OpenEquivariance kernels; `NEQUIX_KERNEL` is accepted as an alias | `0` |
| `NEQUIX_USE_COMPILE` | Nequix | Enable `torch.compile` on the torch backend; `NEQUIX_COMPILE` is accepted as an alias | `0` |
| `NEQUIX_CAPACITY_MULTIPLIER` | Nequix | JAX graph padding factor forwarded to `NequixCalculator` | `1.1` |
| `SEVENNET_MODAL` | SevenNet / FlashTP | Multi-fidelity modal/task name forwarded to `SevenNetCalculator` | Unset |
| `SEVENNET_FILE_TYPE` | SevenNet / FlashTP | Model serialization type (`checkpoint`, `torchscript`) | `checkpoint` |
| `SEVENNET_ENABLE_CUEQ` | SevenNet | Enable cuEquivariance acceleration | Checkpoint default |
| `SEVENNET_ENABLE_FLASH` | SevenNet | Enable FlashTP acceleration | Checkpoint default |
| `SEVENNET_ENABLE_OEQ` | SevenNet | Enable OpenEquivariance acceleration when supported | Checkpoint default |
| `UPET_VERSION` | UPET | Version string used when `MODEL` is a named model rather than a local checkpoint | Latest stable model version |
| `UPET_NON_CONSERVATIVE` | UPET | Enable UPET direct-force/direct-stress inference | `0` |
| `TACE_DTYPE` | TACE | Floating-point dtype forwarded to the TACE ASE calculator | Model default |
| `TACE_FIDELITY_IDX` | TACE | Fidelity index / level for multi-fidelity models (`TACE_LEVEL` alias accepted) | Model default |
| `TACE_SPIN_ON` | TACE | Enable spin-polarized inference when the model supports it | Model default |
| `TACE_NEIGHBORLIST_BACKEND` | TACE | Neighbor-list backend (`matscipy`, `ase`, `vesin`) | `matscipy` |
| `FAIRCHEM_TASK` | FAIRChem v2 | Task head to use | Auto-detected when possible |
| `FAIRCHEM_INFERENCE_SETTINGS` | FAIRChem v2 | Inference profile forwarded to FAIRChem | `default` |
| `FAIRCHEM_CONFIG` | FAIRChem v1 | Path to the YAML config used with the checkpoint | Required for most checkpoints |
| `FAIRCHEM_V1_PREDICTOR` | FAIRChem v1 | Use the predictor directly instead of the OCPCalculator | `0` |
| `GRACE_PAD_NEIGHBORS_FRACTION` | GRACE | Fake-neighbour padding fraction forwarded to TensorPotential | Library default |
| `GRACE_PAD_ATOMS_NUMBER` | GRACE | Number of fake atoms for padding | Library default |
| `GRACE_MAX_RECOMPILATION` | GRACE | Max recompilations triggered by padding reduction | Library default |
| `GRACE_MIN_DIST` | GRACE | Minimum allowed interatomic distance | Unset |
| `GRACE_FLOAT_DTYPE` | GRACE | Floating-point dtype passed to TensorPotential | `float64` |

## Backend Packages And Model Conventions

### Required Python Modules

`ase` and `pymatgen` are always required. Additional modules depend on the selected potential or thermostat.

| Feature | Module to install | Notes |
|---------|-------------------|-------|
| CHGNet potential | `chgnet` | Bundled with a default model; `MODEL` may also be a named CHGNet release or a local checkpoint |
| SevenNet potential | `sevenn` | Bundled with a default model; specify `MODEL`, `SEVENNET_MODAL`, or `SEVENNET_FILE_TYPE` to override |
| FlashTP backend | `sevenn` + `flashTP_e3nn` | Uses the SevenNet ASE calculator with `enable_flash=True` |
| EquFlash backend | EquFlash / GGNN package exposing `GGNN.common.calculator.UCalculator` | Requires a local checkpoint via `MODEL=/path/to/equflash.ckpt` |
| NequIP potential | `nequip` | `MODEL` should point to a deployed or compiled model file |
| Allegro potential | `allegro` plus `nequip` | `MODEL` should point to a deployed or compiled model file |
| MatGL (M3GNet) potential | `matgl` | Bundled with a default model; `MODEL` may be another model directory |
| MACE potential | `mace-torch` | `MODEL` should point to a trained `.model` file |
| DeePMD-kit potential | `deepmd-kit` | `MODEL` should point to a frozen graph or supported checkpoint |
| Matlantis potential | `pfp-api-client` plus `matlantis-features` | Uses the Matlantis estimator service |
| Eqnorm potential | `eqnorm` | Uses named models or local checkpoints |
| MatRIS potential | `matris` | Uses named models such as `matris_10m_oam` or local `.pth.tar` checkpoints |
| AlphaNet potential | `alphanet` | Uses named models or local checkpoints |
| HIENet potential | `hienet` | Uses `HIENet-0` or local checkpoints |
| Nequix potential | `nequix` | Uses named models or local `.nqx` / `.pt` checkpoints |
| ORB potential | `orb-models` | Downloads pretrained weights unless `MODEL` points to a checkpoint |
| UPET potential | `upet` | Accepts a local checkpoint or a named model such as `pet-oam-xl` |
| TACE potential | `TACE==0.1.0` | Accepts a local checkpoint or a named foundation model |
| MatterSim potential | `mattersim` | Set `MODEL` to the trained parameters when needed |
| GRACE potential | `grace-tensorpotential` | Uses TensorPotential checkpoints or foundation models |
| Andersen thermostat | `ase.md.andersen` | Optional ASE MD module |
| Langevin thermostat | `ase.md.langevin` | Included with ASE |
| Bussi thermostat | `ase.md.bussi` | Included in ASE >= 3.22 |
| Nose–Hoover chain thermostat | `ase.md.nose_hoover_chain` | Included in ASE >= 3.22 |

### Backend Notes

Matlantis uses the [Matlantis API](https://matlantis.com) via `pfp-api-client`; configure credentials in the environment before running `MLP=MATLANTIS`.

SevenNet now prefers the current `sevenn` package and `sevenn.calculator.SevenNetCalculator`. Omitting `MODEL` uses the default pretrained model `7net-0`. Use `SEVENNET_MODAL` for multi-fidelity checkpoints such as `7net-omni` or `7net-mf-ompa`.

FlashTP is exposed as `MLP=FLASHTP`, which is effectively SevenNet with `enable_flash=True`. Install FlashTP separately and set `CUDA_ARCH_LIST` to match your GPU architecture.

EquFlash is exposed as `MLP=EQUFLASH` for environments that provide the ASE-compatible `GGNN.common.calculator.UCalculator` entry point. Public named checkpoints are not bundled, so this path currently requires a local checkpoint.

Eqnorm omits `MODEL` by default and uses `eqnorm-mptrj`, downloading the official checkpoint into `~/.cache/eqnorm` when necessary.

MatRIS omits `MODEL` by default and uses `matris_10m_oam`. Named models such as `matris_10m_mp` are downloaded into `~/.cache/matris`.

AlphaNet omits `MODEL` by default and uses `AlphaNet-MATPES-r2scan`. Named models and their configs are downloaded into `~/.cache/alphanet`.

HIENet omits `MODEL` by default and uses `HIENet-0`, downloading the model into `~/.cache/hienet`.

Nequix omits `MODEL` by default and uses `nequix-mp-1`, resolving named models through `~/.cache/nequix/models`.

UPET accepts either a local checkpoint or a named model such as `pet-oam-xl`.

TACE accepts either a local checkpoint or a named foundation model such as `TACE-v1-OMat24-M`.

FAIRChem 2.x and 1.x are incompatible. Use `MLP=FAIRCHEM` / `FAIRCHEM_V2` for v2 checkpoints and `MLP=FAIRCHEM_V1` for legacy OCP / FAIRChem v1 checkpoints.

### Where To Put Model Files

If `MODEL` is a filesystem path, VPMDK loads that exact file. It can be:

- inside the calculation directory
- an absolute path elsewhere on the machine
- a relative path from the run directory

If `MODEL` is not a filesystem path, behavior depends on the backend:

- some backends treat it as a named model and download / resolve it automatically
- others expect a local checkpoint path
- if omitted entirely, many backends fall back to a bundled or default named model

## Output Files

Depending on the calculation type, VPMDK produces the following files in VASP format:

| File | When produced | Contents |
|------|---------------|----------|
| `CONTCAR` | Always | Final atomic positions and cell |
| `OUTCAR` | Always | VASP-like step blocks plus a simplified timing / memory footer |
| `OSZICAR` | Always | Ionic-step energy summary and MD thermostat terms |
| `vasprun.xml` | Always | Minimal VASP-like XML with structures, energies, and forces |
| `XDATCAR` | MD only (`IBRION=0`) | Trajectory snapshots |
| `lammps.lammpstrj` | MD with `WRITE_LAMMPS_TRAJ=1` | LAMMPS text trajectory |
| `energy.csv` | Relaxations with `WRITE_ENERGY_CSV=1` | Per-step potential energy |

When `WRITE_PSEUDO_SCF=1`, VPMDK also adds pseudo electronic-step compatibility blocks to `OSZICAR`, `OUTCAR`, and `vasprun.xml`.

Relaxation convergence follows VASP-like `EDIFFG` sign semantics:

- `EDIFFG < 0`: converged when the maximum force is below `abs(EDIFFG)` (eV/Å)
- `EDIFFG > 0`: converged when `|ΔE|` between ionic steps is below `EDIFFG` (eV)

When `INCAR` contains NEB-style tags such as `IMAGES`, `LCLIMB`, or `SPRING`, and numbered image directories are present, VPMDK runs the images sequentially and emits VTST-style compatibility lines in each image `OUTCAR`. These runs are independent per-image calculations; spring-coupled NEB forces are not applied.

Initial magnetic moments from `MAGMOM` are propagated to ASE when they can be matched to the atom count or species counts in the POSCAR.

Final energies are also printed to the console for single-point calculations.

## GPU Usage

VPMDK itself does not manage GPU scheduling directly. Device selection is mostly delegated to the backend.

These backends honor `DEVICE` directly in `BCAR`:

- CHGNet
- MatGL / M3GNet
- MACE
- Eqnorm
- MatRIS
- AlphaNet
- SevenNet
- FlashTP
- ORB
- HIENet
- UPET
- TACE
- FAIRChem

Nequix supports `DEVICE` on the torch backend. On the JAX backend, placement follows the active JAX runtime and environment variables such as `JAX_PLATFORMS` and `CUDA_VISIBLE_DEVICES`.

Use `CUDA_VISIBLE_DEVICES` if you want to pin a specific GPU. A GPU with at least 8 GB of memory is a practical starting point for many models, but CPU execution also works.

## License

VPMDK is distributed under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.
