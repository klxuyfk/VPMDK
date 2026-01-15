# VPMDK

VPMDK (*Vasp-Protocol Machine-learning Dynamics Kit*, aka “VasP-MoDoKi”) is a lightweight engine that **reads and writes VASP-style inputs/outputs** and performs **molecular dynamics and structure relaxations** using **machine-learning interatomic potentials**. Keep familiar VASP workflows and artifacts while computations run through ASE-compatible ML calculators. The `vpmdk` command (and legacy `vpmdk.py` wrapper) are provided.

**Supported calculators (via ASE):** **CHGNet**, **SevenNet**, **MatterSim**, **MACE**, **Matlantis**, **NequIP**, **Allegro**, **ORB**, **MatGL** (via the M3GNet model), **FAIRChem** (including eSEN checkpoints), **GRACE** (TensorPotential foundation models or checkpoints), and **DeePMD-kit**. Availability depends on the corresponding Python packages being installed.

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
   wish to use, `chgnet`, `mattersim`, `mace-torch` or `matgl`.
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
| `ISIF` | Controls whether the cell changes during relaxations. | `2` keeps the cell fixed (default). `3` relaxes ions and the full cell, `4` keeps the volume constant while optimising ions and the cell shape, `5` optimises the cell shape at constant volume with fixed ions, `6` changes only the cell, `7` enables isotropic cell changes with fixed ions, and `8` couples ionic relaxations to isotropic volume changes. Unsupported values fall back to `2` with a warning. |
| `EDIFFG` | Convergence threshold for relaxations in eV/Å. | `-0.02`. |
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
NNP=CHGNET            # Potential backend
MODEL=/path/to/model  # Optional path to weights (varies by backend)
DEVICE=cuda           # Optional device override when the backend supports it
```

**Core selection.**

| Tag | Meaning | Default |
|-----|---------|---------|
| `NNP` | Backend name (`CHGNET`, `MACE`, `MATGL`, `MATLANTIS`, `MATTERSIM`, `NEQUIP`, `ALLEGRO`, `ORB`, `FAIRCHEM`, `FAIRCHEM_V2`, `FAIRCHEM_V1`, `GRACE`, `DEEPMD`, `SEVENNET`) | `CHGNET` |
| `MODEL` | Path to a trained parameter set (ORB accepts checkpoints; FAIRChem also accepts model names such as `esen-sm-direct-all-oc25`) | Backend default or bundled weights |
| `DEVICE` | Device hint for backends that support it (`cpu`, `cuda`, `cuda:N`) | Auto-detects GPU when available |

**Output and workflow aids.**

| Tag | Meaning | Default |
|-----|---------|---------|
| `WRITE_ENERGY_CSV` | Write `energy.csv` during relaxation (`1` to enable) | `0` |
| `WRITE_LAMMPS_TRAJ` | Write a LAMMPS trajectory during MD (`1` to enable) | `0` |
| `LAMMPS_TRAJ_INTERVAL` | MD steps between trajectory frames (only when `WRITE_LAMMPS_TRAJ=1`) | `1` |
| `DEEPMD_TYPE_MAP` | Comma/space-separated species list mapped to the DeePMD graph | Inferred from `POSCAR` order |

**Backend-specific knobs.** Only relevant when the corresponding backend is chosen.

| Tag | Applies to | Meaning | Default |
|-----|-----------|---------|---------|
| `MATLANTIS_MODEL_VERSION` | Matlantis | Estimator version identifier | `v8.0.0` |
| `MATLANTIS_PRIORITY` | Matlantis | Job priority forwarded to the estimator | `50` |
| `MATLANTIS_CALC_MODE` | Matlantis | Calculation mode (`CRYSTAL`, `MOLECULE`, …) | `PBE` |
| `ORB_MODEL` | ORB | Pretrained architecture key recognised by `orb_models` | `orb-v3-conservative-20-omat` |
| `ORB_PRECISION` | ORB | Floating-point precision string forwarded to orb-models loaders | `float32-high` |
| `ORB_COMPILE` | ORB | Whether to `torch.compile` the ORB model (`0/1`, `true/false`, …) | Library default |
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
credentials before running VPMDK with `NNP=MATLANTIS`.

ORB calculations rely on the [orb-models](https://github.com/orbital-materials/orb-models)
package. When `MODEL` is omitted, VPMDK downloads the pretrained weights specified by
`ORB_MODEL` using orb-models; set `MODEL=/path/to/checkpoint.ckpt` to run with local weights.

FAIRChem 2.x and 1.x are incompatible. Select `NNP=FAIRCHEM` (or `FAIRCHEM_V2`) to
use FAIRChem v2 checkpoints via `FAIRChemCalculator.from_model_checkpoint`, and
`NNP=FAIRCHEM_V1` when running legacy OCP/FAIRChem v1 checkpoints with
`OCPCalculator`. Switching conda environments per checkpoint version is supported by
selecting the appropriate tag.

## Output files

Depending on the calculation type, VPMDK produces the following files in VASP format:

| File | When produced | Contents |
|------|---------------|----------|
| `CONTCAR` | Always | Final atomic positions and cell. |
| `OUTCAR` | Relaxations and MD | Step-by-step potential, kinetic, and total energies along with temperature. |
| `XDATCAR` | MD only (`IBRION=0`) | Atomic positions at each MD step (trajectory). |
| `lammps.lammpstrj` | MD with `WRITE_LAMMPS_TRAJ=1` | LAMMPS text dump of atomic positions at the requested interval. |
| `energy.csv` | Relaxations with `WRITE_ENERGY_CSV=1` | Potential energy at each relaxation step. |

Relaxations terminate when either the maximum force drops below the value set
by `EDIFFG` (default `0.02` eV/Å) or, when `EDIFFG` is positive, when the
change in energy between ionic steps falls below the specified threshold.

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
| NequIP potential | `nequip` (uses PyTorch) | Requires `MODEL` pointing to a deployed model file |
| Allegro potential | `allegro` (uses PyTorch and depends on `nequip`) | Requires `MODEL` pointing to a deployed model file |
| MatGL (M3GNet) potential | `matgl` (uses JAX) | Bundled with a default model; specify `MODEL` to use another |
| MACE potential | `mace-torch` (PyTorch) | Set `MODEL` to a trained `.model` file |
| DeePMD-kit potential | `deepmd-kit` | Set `MODEL` to the frozen graph (`.pb`) and optionally `DEEPMD_TYPE_MAP` |
| Matlantis potential | `pfp-api-client` (plus `matlantis-features`) | Uses the Matlantis estimator service; configure with `MATLANTIS_*` BCAR tags |
| ORB potential | `orb-models` (PyTorch) | Downloads pretrained weights unless `MODEL` points to a checkpoint |
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
path. CHGNet and MatGL ship with default models; omitting `MODEL` uses those
defaults automatically.

### GPU usage

This script does not directly manage GPU settings. Each potential selects a
device on its own. CHGNet, MatGL/M3GNet, MACE, ORB, and FAIRChem honour
`DEVICE` in `BCAR` (e.g. `DEVICE=cpu` to force a CPU run). With CUDA devices you
can choose which GPU to use with `CUDA_VISIBLE_DEVICES`. When running MatGL you
may also set `XLA_PYTHON_CLIENT_PREALLOCATE=false`. A GPU with at least 8 GB of
memory is recommended, though running on a CPU also works.

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
