# VPMDK

VPMDK (*Vasp-Protocol Machine-learning Dynamics Kit*, aka “VasP-MoDoKi”) is a lightweight engine that **reads and writes VASP-style inputs/outputs** and performs **molecular dynamics and structure relaxations** using **machine-learning interatomic potentials**. Keep familiar VASP workflows and artifacts while computations run through ASE-compatible ML calculators. The `vpmdk` command (and legacy `vpmdk.py` wrapper) are provided.

**Supported calculators (via ASE):** **CHGNet**, **SevenNet**, **MatterSim**, **MACE**, **Matlantis**, **Eqnorm**, **MatRIS**, **AlphaNet**, **HIENet**, **Nequix**, **NequIP**, **Allegro**, **ORB**, **UPET**, **TACE**, **MatGL** (via the M3GNet model), **FAIRChem** (including eSEN checkpoints), **GRACE** (TensorPotential foundation models or checkpoints), and **DeePMD-kit**. Availability depends on the corresponding Python packages being installed.

*Not affiliated with, endorsed by, or a replacement for VASP; “VASP” is a trademark of its respective owner. VPMDK only mimics VASP I/O conventions for compatibility.*

## Installation

Install the package from PyPI (or from a checkout):

```bash
pip install vpmdk
```

## Usage

1. Prepare a directory containing at least `POSCAR`. Optional files are
   `INCAR`, `POTCAR`, and `BCAR`. `KPOINTS`, `WAVECAR`, and `CHGCAR` are
   recognised but ignored (a note is printed if they are present).
2. Install requirements: `ase`, `pymatgen` and, depending on the potential you
   wish to use, `chgnet`, `mattersim`, `mace-torch`, `matgl`, `eqnorm`,
   `matris`, `alphanet`, `hienet`, `nequix`, `upet`, or `tace`.
3. Run:

   ```bash
   vpmdk [--dir PATH_TO_INPUT]
   ```

If `--dir` is omitted, the current directory (`.`) is used.

When running directly from a repository checkout, the legacy wrapper still works:

```bash
python vpmdk.py [--dir PATH_TO_INPUT]
```

## Input files

Calculation directories may contain the following files:

- `POSCAR` *(required)* – atomic positions and cell.
- `INCAR` – VASP-style run parameters; only a subset of tags is supported.
- `BCAR` – simple `key=value` file selecting the machine-learning potential.
- `POTCAR` – accepted for compatibility but ignored except for aligning species
  names.

### Supported INCAR tags

The script reads a subset of common VASP `INCAR` settings. Other tags are ignored with a warning.

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
| `SMASS` | Thermostat-specific mass parameter. | Used for Nose–Hoover time constant (`abs(SMASS)` fs) or as a fallback to set `LANGEVIN_GAMMA` when negative. |
| `ANDERSEN_PROB` | Collision probability for the Andersen thermostat. | `0.1`. Only used with `MDALGO=1`. |
| `LANGEVIN_GAMMA` | Friction coefficient (1/ps) for Langevin dynamics. | `1.0`. Only used with `MDALGO=3`; falls back to `abs(SMASS)` when `SMASS<0`. |
| `CSVR_PERIOD` | Relaxation time (fs) for the canonical sampling velocity rescaling thermostat. | `max(100×POTIM, POTIM)`. Only used with `MDALGO=5`. |
| `NHC_NCHAINS` | Nose–Hoover chain length. | `1` for `MDALGO=2`, `3` for `MDALGO=4`. |
| `PSTRESS` | External pressure in kBar applied during relaxations. | Converts to scalar pressure in the ASE optimiser when `ISIF` allows cell changes. |
| `MAGMOM` | Initial magnetic moments. | Parsed like VASP; supports shorthand such as `2*1.0`. |
| `IMAGES` | Number of NEB images. | Enables NEB mode detection. When numbered image directories (`00`, `01`, ...) exist, VPMDK runs them sequentially and emits NEB-style projection lines in each image `OUTCAR`. |
| `LCLIMB` | Climbing-image switch for NEB. | Used as a compatibility hint for VTST post-processing outputs. |
| `SPRING` | Spring constant for NEB. | Used as a compatibility hint for VTST post-processing outputs. |

### MD algorithms

`MDALGO` selects between different ASE molecular dynamics drivers. Some options require optional ASE modules; if they are missing VPMDK falls back to plain velocity-Verlet (NVE) integration and prints a warning.

| `MDALGO` | Integrator | Notes |
|---------:|-----------|-------|
| `0` | Velocity-Verlet (NVE) | No thermostat. |
| `1` | Andersen thermostat | Controlled by `ANDERSEN_PROB`. |
| `2` | Nose–Hoover chain (single thermostat) | Uses `SMASS`/`NHC_NCHAINS` to configure the chain. |
| `3` | Langevin thermostat | Uses `LANGEVIN_GAMMA` (or `abs(SMASS)` if negative). |
| `4` | Nose–Hoover chain (three thermostats) | Chain length defaults to 3 unless overridden. |
| `5` | Bussi (canonical sampling velocity rescaling) thermostat | Uses `CSVR_PERIOD`. |

### BCAR tags

`BCAR` is a concise `key=value` file that selects the backend, its weights, and a handful of quality-of-life options. A minimal
file looks like:

```
MLP=CHGNET            # Potential backend
MODEL=/path/to/model  # Optional path to weights (varies by backend)
DEVICE=cuda           # Optional device override when the backend supports it
```

**Core selection.**

| Tag | Meaning | Default |
|-----|---------|---------|
| `MLP` | Backend name (`CHGNET`, `MACE`, `MATGL`, `MATLANTIS`, `MATTERSIM`, `EQNORM`, `MATRIS`, `ALPHANET`, `HIENET`, `NEQUIX`, `NEQUIP`, `ALLEGRO`, `ORB`, `UPET`, `TACE`, `FAIRCHEM`, `FAIRCHEM_V2`, `FAIRCHEM_V1`, `GRACE`, `DEEPMD`, `SEVENNET`) | `CHGNET` |
| `MODEL` | Path to a trained parameter set (Eqnorm accepts local `.pt` / `.pth` checkpoints or named models such as `eqnorm-mptrj`; MatRIS accepts local `.pth.tar` checkpoints or named models such as `matris_10m_oam`; AlphaNet accepts local `.ckpt` / `.pt` checkpoints or named models such as `AlphaNet-MATPES-r2scan`; HIENet accepts local `.pth` / `.pt` / `.ckpt` checkpoints or the named model `HIENet-0`; Nequix accepts local `.nqx` / `.pt` checkpoints or named models such as `nequix-mp-1`; ORB accepts checkpoints; UPET also accepts model names such as `pet-oam-xl`; TACE also accepts foundation-model names such as `TACE-v1-OMat24-M`; FAIRChem also accepts model names such as `esen-sm-direct-all-oc25`) | Backend default or bundled weights |
| `DEVICE` | Device hint for backends that support it (`cpu`, `cuda`, `cuda:N`) | Auto-detects GPU when available |

`NNP` is accepted as a backward-compatible alias of `MLP`.

**Output and workflow aids.**

| Tag | Meaning | Default |
|-----|---------|---------|
| `WRITE_ENERGY_CSV` | Write `energy.csv` during relaxation (`1` to enable) | `0` |
| `WRITE_LAMMPS_TRAJ` | Write a LAMMPS trajectory during MD (`1` to enable) | `0` |
| `LAMMPS_TRAJ_INTERVAL` | MD steps between trajectory frames (only when `WRITE_LAMMPS_TRAJ=1`) | `1` |
| `WRITE_PSEUDO_SCF` | Add pseudo electronic-step compatibility blocks to `OSZICAR`, `OUTCAR`, and `vasprun.xml` (`1` to enable) | `0` (OFF) |
| `DEEPMD_TYPE_MAP` | Comma/space-separated species list mapped to the DeePMD graph | Inferred from `POSCAR` order |
| `DEEPMD_HEAD` | Select a DeePMD model head by name (when supported by the checkpoint) | Unset |

`WRITE_OSZICAR_PSEUDO_SCF` is accepted as a backward-compatible alias of
`WRITE_PSEUDO_SCF`.

**Backend-specific knobs.** Only relevant when the corresponding backend is chosen.

| Tag | Applies to | Meaning | Default |
|-----|-----------|---------|---------|
| `MATLANTIS_MODEL_VERSION` | Matlantis | Estimator version identifier | `v8.0.0` |
| `MATLANTIS_PRIORITY` | Matlantis | Job priority forwarded to the estimator | `50` |
| `MATLANTIS_CALC_MODE` | Matlantis | Calculation mode (`CRYSTAL`, `MOLECULE`, …) | `PBE` |
| `ORB_MODEL` | ORB | Pretrained architecture key recognised by `orb_models` | `orb-v3-conservative-20-omat` |
| `ORB_PRECISION` | ORB | Floating-point precision string forwarded to orb-models loaders | `float32-high` |
| `ORB_COMPILE` | ORB | Whether to `torch.compile` the ORB model (`0/1`, `true/false`, …) | Library default |
| `EQNORM_VARIANT` | Eqnorm | Eqnorm architecture variant used with a local checkpoint (`eqnorm-mptrj`, `eqnorm-omat`, `eqnorm-max-mptrj`) | Inferred from `MODEL` filename or `eqnorm-mptrj` for named models |
| `EQNORM_COMPILE` | Eqnorm | Whether to `torch.compile` the Eqnorm model (`0/1`, `true/false`, …) | `0` |
| `MATRIS_TASK` | MatRIS | Prediction task forwarded to `MatRISCalculator` (`e`, `ef`, `efs`, `efsm`) | `efs` |
| `ALPHANET_CONFIG` | AlphaNet | Path to the AlphaNet JSON config when `MODEL` is a local checkpoint and the config cannot be inferred | Paired config for named models or inferred sibling JSON |
| `ALPHANET_PRECISION` | AlphaNet | Floating-point precision forwarded to the AlphaNet ASE calculator (`32`, `64`, `float32`, `float64`) | `32` |
| `HIENET_FILE_TYPE` | HIENet | Model serialization type accepted by `HIENetCalculator` (`checkpoint`, `torchscript`) | `checkpoint` |
| `NEQUIX_BACKEND` | Nequix | Upstream backend (`jax` or `torch`) | `jax` |
| `NEQUIX_USE_KERNEL` | Nequix | Enable OpenEquivariance kernels (`0/1`, `true/false`, …); `NEQUIX_KERNEL` is accepted as an alias | `0` |
| `NEQUIX_USE_COMPILE` | Nequix | Enable `torch.compile` on the torch backend; `NEQUIX_COMPILE` is accepted as an alias | `0` |
| `NEQUIX_CAPACITY_MULTIPLIER` | Nequix | JAX graph padding factor forwarded to `NequixCalculator` | `1.1` |
| `UPET_VERSION` | UPET | Version string used when `MODEL` is a named UPET model rather than a local checkpoint | Latest stable model version |
| `UPET_NON_CONSERVATIVE` | UPET | Enable UPET direct-force/direct-stress inference (`1` to enable) | `0` |
| `TACE_DTYPE` | TACE | Floating-point dtype forwarded to the TACE ASE calculator | Model default |
| `TACE_FIDELITY_IDX` | TACE | Fidelity index / level for multi-fidelity models (`TACE_LEVEL` is accepted as an alias) | Model default |
| `TACE_SPIN_ON` | TACE | Enable spin-polarized inference when the model supports it (`1` to enable) | Model default |
| `TACE_NEIGHBORLIST_BACKEND` | TACE | Neighbor-list backend (`matscipy`, `ase`, `vesin`) | `matscipy` |
| `FAIRCHEM_TASK` | FAIRChem v2 (`FAIRCHEM`/`FAIRCHEM_V2`) | Task head to use (e.g. `omol`) | Auto-detected when possible |
| `FAIRCHEM_INFERENCE_SETTINGS` | FAIRChem v2 (`FAIRCHEM`/`FAIRCHEM_V2`) | Inference profile forwarded to FAIRChem | `default` |
| `FAIRCHEM_CONFIG` | FAIRChem v1 (`FAIRCHEM_V1`) | Path to the YAML config used with the checkpoint | Required for most checkpoints |
| `FAIRCHEM_V1_PREDICTOR` | FAIRChem v1 (`FAIRCHEM_V1`) | Use the predictor directly instead of the OCPCalculator (`1` to enable) | `0` |
| `GRACE_PAD_NEIGHBORS_FRACTION` | GRACE | Fake-neighbour padding fraction forwarded to TensorPotential | Library default (typically `0.05`) |
| `GRACE_PAD_ATOMS_NUMBER` | GRACE | Number of fake atoms for padding | Library default (typically `10`) |
| `GRACE_MAX_RECOMPILATION` | GRACE | Max recompilations triggered by padding reduction | Library default (typically `2`) |
| `GRACE_MIN_DIST` | GRACE | Minimum allowed interatomic distance | Unset (no extra validation) |
| `GRACE_FLOAT_DTYPE` | GRACE | Floating-point dtype passed to TensorPotential | `float64` |

Matlantis calculations rely on the [Matlantis API](https://matlantis.com) via
`pfp-api-client`; ensure your environment is configured with the required API
credentials before running VPMDK with `MLP=MATLANTIS`.

Eqnorm calculations rely on the [eqnorm](https://github.com/yzchen08/eqnorm)
package. Omitting `MODEL` uses the default named model `eqnorm-mptrj`, which
VPMDK downloads into `~/.cache/eqnorm` using the official Figshare artifact. To
use a local checkpoint, set `MODEL=/path/to/model.pt`; add
`EQNORM_VARIANT=eqnorm-mptrj` / `eqnorm-omat` / `eqnorm-max-mptrj` when the
variant cannot be inferred from the filename.

ORB calculations rely on the [orb-models](https://github.com/orbital-materials/orb-models)
package. When `MODEL` is omitted, VPMDK downloads the pretrained weights specified by
`ORB_MODEL` using orb-models; set `MODEL=/path/to/checkpoint.ckpt` to run with local weights.

MatRIS calculations rely on the [MatRIS](https://github.com/HPC-AI-Team/MatRIS) package.
Omitting `MODEL` uses the default named model (`matris_10m_oam`). Set
`MODEL=matris_10m_mp` to auto-download an official named model into
`~/.cache/matris`, or `MODEL=/path/to/MatRIS_10M_OAM.pth.tar` to load a local
checkpoint directly.

AlphaNet calculations rely on the [AlphaNet](https://github.com/zmyybc/AlphaNet)
package. Omitting `MODEL` uses the default named model
`AlphaNet-MATPES-r2scan`. VPMDK can auto-download official named models such as
`AlphaNet-MATPES-r2scan`, `AlphaNet-AQCAT25`, `AlphaNet-MPtrj-v1`, and
`AlphaNet-oma-v1` into `~/.cache/alphanet`. When `MODEL` points to a local
checkpoint, set `ALPHANET_CONFIG=/path/to/config.json` unless the config can be
inferred from a sibling JSON file.

HIENet calculations rely on the
[HIENet implementation in AIRS](https://github.com/divelab/AIRS/tree/main/OpenMat/HIENet).
Omitting `MODEL` uses the named model `HIENet-0`, which VPMDK downloads into
`~/.cache/hienet` from the official AIRS repository. Set
`MODEL=/path/to/HIENet-V3.pth` (or another local `.pt` / `.ckpt` checkpoint) to
use a local model file directly. If you have a TorchScript export instead, set
`HIENET_FILE_TYPE=torchscript`.

Nequix calculations rely on the [nequix](https://github.com/atomicarchitects/nequix)
package. Omitting `MODEL` uses the default named model `nequix-mp-1`. VPMDK
can also resolve official named models such as `nequix-omat-1`, `nequix-oam-1`,
and `nequix-oam-1-pft` through the upstream cache at `~/.cache/nequix/models`.
Set `MODEL=/path/to/model.nqx` (or `.pt`) to use a local checkpoint directly.
`NEQUIX_BACKEND` selects `jax` or `torch`; `NEQUIX_USE_KERNEL=1` requires the
optional OpenEquivariance extras provided by the upstream project.

UPET calculations rely on the [upet](https://github.com/lab-cosmo/upet) package.
Set `MODEL=/path/to/model.ckpt` to use a local checkpoint, or `MODEL=pet-oam-xl`
with optional `UPET_VERSION=1.0.0` to fetch a named UPET model through the library.

TACE calculations rely on the [TACE](https://github.com/xvzemin/tace) package.
Set `MODEL=/path/to/model.pt` (or `.pth` / `.ckpt`) to use a local checkpoint, or
`MODEL=TACE-v1-OMat24-M` / `MODEL=TACE-v1-OAM-M` to auto-download a named
foundation model through `tace_foundations`. For multi-fidelity models, set
`TACE_FIDELITY_IDX` (or the compatibility alias `TACE_LEVEL`).

FAIRChem 2.x and 1.x are incompatible. Select `MLP=FAIRCHEM` (or `MLP=FAIRCHEM_V2`) to
use FAIRChem v2 checkpoints via `FAIRChemCalculator.from_model_checkpoint`, and
`MLP=FAIRCHEM_V1` when running legacy OCP/FAIRChem v1 checkpoints with
`OCPCalculator`. Switching conda environments per checkpoint version is supported by
selecting the appropriate tag.

## Output files

Depending on the calculation type, VPMDK produces the following files in VASP format:

| File | When produced | Contents |
|------|---------------|----------|
| `CONTCAR` | Always | Final atomic positions and cell. |
| `OUTCAR` | Always | VASP-like step blocks plus a simplified timing/memory footer at the end of each run. When `WRITE_PSEUDO_SCF=1`, pseudo electronic-step metadata such as `NELM` and `Iteration ... (   1)` is added for compatibility. |
| `OSZICAR` | Always | VASP-like ionic-step energy summary (`F`, `E0`, `dE`; and MD thermostat terms) with VASP-like aligned scientific notation. Optional pseudo electronic-step (`DAV:`) lines are added only when `WRITE_PSEUDO_SCF=1`; these are compatibility placeholders, not real electronic SCF iterations. |
| `vasprun.xml` | Always | Minimal VASP-like XML containing initial/final structures, per-step energies, and forces. When `WRITE_PSEUDO_SCF=1`, dummy `NELM` metadata and one `<scstep>` block per ionic step are also emitted for compatibility. |
| `XDATCAR` | MD only (`IBRION=0`) | Atomic positions at each MD step (trajectory). |
| `lammps.lammpstrj` | MD with `WRITE_LAMMPS_TRAJ=1` | LAMMPS text dump of atomic positions at the requested interval. |
| `energy.csv` | Relaxations with `WRITE_ENERGY_CSV=1` | Potential energy at each relaxation step. |

Relaxation convergence follows VASP-like `EDIFFG` sign semantics:

- `EDIFFG < 0`: converged when the maximum force is below `abs(EDIFFG)` (eV/Å).
- `EDIFFG > 0`: converged when `|ΔE|` between ionic steps is below `EDIFFG` (eV).

When `INCAR` contains NEB-style tags (for example `IMAGES`, `LCLIMB`, or
`SPRING`) and numbered image directories are present, VPMDK iterates over those
directories (`00`, `01`, ...) and runs each image. For compatibility with VTST
post-processing scripts, each image `OUTCAR` includes a
`NEB: projections ...` line in every ionic block. This runner performs
independent per-image calculations and does not apply spring-coupled NEB forces
between images.

The NEB `CHAIN` block values are approximate: VPMDK estimates the local tangent
from neighboring image position differences (`next-prev`, or one-sided at
endpoints), then projects the per-atom force onto that tangent to populate
`tangential force`, `TANGENT/CHAIN-FORCE`, and `CHAIN + TOTAL`.

For NEB-style directory runs, VPMDK also writes parent-level aggregate
`OUTCAR`, `OSZICAR`, and `vasprun.xml` files in the top NEB directory using the
final state from each numbered image.

Initial magnetic moments from `MAGMOM` are propagated to ASE when the value can
be matched with the number of atoms or species counts in the POSCAR.

Final energies are also printed to the console for single-point calculations.

### Required Python modules

`ase` and `pymatgen` are always required. Additional modules depend on the
selected potential or thermostat:

| Feature | Module to install | Notes |
|---------|-------------------|-------|
| CHGNet potential | `chgnet` (uses PyTorch) | Bundled with a default model; specify `MODEL` to use another |
| SevenNet potential | `sevennet` (uses PyTorch) | Bundled with a default model; specify `MODEL` to use another |
| NequIP potential | `nequip` (uses PyTorch) | `MODEL` should point to a deployed model file; compiled/TorchScript models are also accepted when supported by the NequIP version (`from_compiled_model`) |
| Allegro potential | `allegro` (uses PyTorch and depends on `nequip`) | `MODEL` should point to a deployed model file; compiled/TorchScript models are also accepted when supported by the NequIP version (`from_compiled_model`) |
| MatGL (M3GNet) potential | `matgl` (uses PyTorch + DGL or JAX, depending on install) | Bundled with a default model; specify `MODEL` to use another. MatGL 1.x commonly expects a model directory passed through `matgl.load_model`. |
| MACE potential | `mace-torch` (PyTorch) | Set `MODEL` to a trained `.model` file |
| DeePMD-kit potential | `deepmd-kit` | Set `MODEL` to the frozen graph (`.pb`) or a PyTorch checkpoint (`.pt`), depending on the DeePMD backend, and optionally `DEEPMD_TYPE_MAP`/`DEEPMD_HEAD` |
| Matlantis potential | `pfp-api-client` (plus `matlantis-features`) | Uses the Matlantis estimator service; configure with `MATLANTIS_*` BCAR tags |
| Eqnorm potential | `eqnorm` (PyTorch) | Uses the named model `eqnorm-mptrj` or local checkpoints; optionally set `EQNORM_VARIANT` / `EQNORM_COMPILE` |
| MatRIS potential | `matris` (PyTorch) | Uses named models such as `matris_10m_oam` / `matris_10m_mp` or local `.pth.tar` checkpoints; optionally set `MATRIS_TASK` |
| AlphaNet potential | `alphanet` (PyTorch) | Uses local `.ckpt` / `.pt` checkpoints or named models such as `AlphaNet-MATPES-r2scan`; optionally set `ALPHANET_CONFIG` / `ALPHANET_PRECISION` |
| HIENet potential | `hienet` (PyTorch) | Uses the named model `HIENet-0` or local `.pth` / `.pt` / `.ckpt` checkpoints; optionally set `HIENET_FILE_TYPE` |
| Nequix potential | `nequix` (JAX by default, optional PyTorch backend) | Uses named models such as `nequix-mp-1` or local `.nqx` / `.pt` checkpoints; optionally set `NEQUIX_BACKEND` / `NEQUIX_USE_KERNEL` / `NEQUIX_USE_COMPILE` |
| ORB potential | `orb-models` (PyTorch) | Downloads pretrained weights unless `MODEL` points to a checkpoint |
| UPET potential | `upet` (PyTorch) | Set `MODEL` to a local `.ckpt` checkpoint or a named model such as `pet-oam-xl`; optionally set `UPET_VERSION`/`UPET_NON_CONSERVATIVE` |
| TACE potential | `TACE==0.1.0` (PyTorch) | Set `MODEL` to a local checkpoint or a named foundation model such as `TACE-v1-OMat24-M`; optionally set `TACE_DTYPE` / `TACE_FIDELITY_IDX` / `TACE_SPIN_ON` |
| MatterSim potential | `mattersim` (PyTorch) | Set `MODEL` to the trained parameters |
| GRACE potential | `grace-tensorpotential` (TensorFlow) | Uses TensorPotential checkpoints (`MODEL=/path/to/model`) or foundation models when available |
| Andersen thermostat | `ase.md.andersen` (part of ASE extras) | Install ASE with MD extras to enable |
| Langevin thermostat | `ase.md.langevin` | Ships with ASE; ensure ASE is up to date |
| Bussi thermostat | `ase.md.bussi` | Included in ASE >= 3.22 |
| Nose–Hoover chain thermostat | `ase.md.nose_hoover_chain` | Included in ASE >= 3.22 |

Install each module using `pip install MODULE_NAME`. Install the GPU-enabled
version of PyTorch or JAX if you want to use GPUs.

### Where to place parameter files

The model file is loaded from the path given by `MODEL` in `BCAR`. Typically the
file is located within the calculation directory or specified via an absolute
path. CHGNet, MatGL, Eqnorm, MatRIS, AlphaNet, HIENet, and Nequix ship with default
models; omitting `MODEL` uses those defaults automatically. Eqnorm can resolve
`eqnorm-mptrj` by downloading the official checkpoint into `~/.cache/eqnorm`
when `MODEL` is not a filesystem path. MatRIS can also resolve named models such
as `matris_10m_oam` and `matris_10m_mp` by downloading them into
`~/.cache/matris` when `MODEL` is not a filesystem path. AlphaNet can resolve
named models such as `AlphaNet-MATPES-r2scan` and `AlphaNet-oma-v1` by
downloading a checkpoint plus JSON config into `~/.cache/alphanet` when `MODEL`
is not a filesystem path. HIENet can resolve `HIENet-0` into `~/.cache/hienet`
when `MODEL` is not a filesystem path. Nequix can resolve named models such as
`nequix-mp-1` and `nequix-oam-1` into `~/.cache/nequix/models` when `MODEL` is
not a filesystem path. UPET can also resolve named models such as
`pet-oam-xl` through the `upet` package when `MODEL` is not a filesystem path.
TACE can resolve named foundation models such as `TACE-v1-OMat24-M` through the
`tace_foundations` registry.

### GPU usage

This script does not directly manage GPU settings. Each potential selects a
device on its own. CHGNet, MatGL/M3GNet, MACE, Eqnorm, MatRIS, AlphaNet, ORB,
HIENet, UPET, TACE, and FAIRChem honour `DEVICE` in `BCAR` (e.g. `DEVICE=cpu` to force
a CPU run). Nequix supports `DEVICE` when `NEQUIX_BACKEND=torch`; on the JAX
backend, placement follows the active JAX runtime (`JAX_PLATFORMS`,
`CUDA_VISIBLE_DEVICES`, etc.). With CUDA devices you can choose which GPU to use
with `CUDA_VISIBLE_DEVICES`. MatGL GPU tuning is backend-dependent
(PyTorch+DGL vs JAX), so environment variables differ between installations. A
GPU with at least 8 GB of memory is recommended, though running on a CPU also
works.

### Example directory layout

```
calc_dir/
├── POSCAR      # required
├── INCAR       # optional
├── POTCAR      # optional
└── BCAR        # optional, specify potential and model path
```

Example command:

```bash
vpmdk --dir calc_dir
```

## License

VPMDK is distributed under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.
