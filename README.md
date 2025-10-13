# VPMDK

VPMDK (*Vasp-Protocol Machine-learning Dynamics Kit*, aka “VasP-MoDoKi”) is a lightweight engine that **reads and writes VASP-style inputs/outputs** and performs **molecular dynamics and structure relaxations** using **machine-learning interatomic potentials**. Keep familiar VASP workflows and artifacts while computations run through ASE-compatible ML calculators. A simple driver script, `vpmdk.py`, is provided.

**Supported calculators (via ASE):** **CHGNet**, **MatterSim**, **MACE**, and **MatGL** (via the M3GNet model). Availability depends on the corresponding Python packages being installed.

*Not affiliated with, endorsed by, or a replacement for VASP; “VASP” is a trademark of its respective owner. VPMDK only mimics VASP I/O conventions for compatibility.*

## Usage

1. Prepare a directory containing at least `POSCAR`. Optional files are
   `INCAR`, `POTCAR`, and `BCAR`. `KPOINTS`, `WAVECAR`, and `CHGCAR` are
   recognised but ignored (a note is printed if they are present).
2. Install requirements: `ase`, `pymatgen` and, depending on the potential you
   wish to use, `chgnet`, `mattersim`, `mace-torch` or `matgl`.
3. Run:

   ```bash
   python vpmdk.py [--dir PATH_TO_INPUT]
   ```

If `--dir` is omitted, the current directory (`.`) is used.

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
| `MDALGO` | Selects the MD integrator / thermostat. | `0` (NVE). See [MD algorithms](#md-algorithms) for details. |
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

`BCAR` configures the machine-learning potential and optional outputs:

```
NNP=CHGNET            # Name of the potential
MODEL=/path/to/model  # Optional path to a trained parameter set
DEVICE=cuda           # Optional for MACE: 'cuda' or 'cpu'
WRITE_ENERGY_CSV=1    # Optional: write energy.csv during relaxation
```

Available `BCAR` tags and defaults:

| Tag | Meaning | Default |
|-----|---------|---------|
| `NNP` | Name of the potential | `CHGNET` |
| `MODEL` | Path to a trained parameter set | potential's built-in model |
| `DEVICE` | Device for MACE (`cuda` or `cpu`) | auto-detect (`cuda` if available, else `cpu`) |
| `WRITE_ENERGY_CSV` | Write `energy.csv` during relaxation (`1` to enable) | `0` (disabled) |

## Output files

Depending on the calculation type, VPMDK produces the following files in VASP format:

| File | When produced | Contents |
|------|---------------|----------|
| `CONTCAR` | Always | Final atomic positions and cell. |
| `OUTCAR` | Relaxations and MD | Step-by-step potential, kinetic, and total energies along with temperature. |
| `XDATCAR` | MD only (`IBRION=0`) | Atomic positions at each MD step (trajectory). |
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
| MatGL (M3GNet) potential | `matgl` (uses JAX) | Bundled with a default model; specify `MODEL` to use another |
| MACE potential | `mace-torch` (PyTorch) | Set `MODEL` to a trained `.model` file |
| MatterSim potential | `mattersim` (PyTorch) | Set `MODEL` to the trained parameters |
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
device on its own. MACE checks for a GPU and uses it when available unless
`DEVICE=cpu` is specified in `BCAR`. With CUDA devices you can choose which GPU
to use with `CUDA_VISIBLE_DEVICES`. When running MatGL you may also set
`XLA_PYTHON_CLIENT_PREALLOCATE=false`. A GPU with at least 8 GB of memory is
recommended, though running on a CPU also works.

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
python vpmdk.py --dir calc_dir
```

## License

VPMDK is distributed under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.
