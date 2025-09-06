# VPMDK

VPMDK (*Vasp-Protocol Machine-learning Dynamics Kit*, aka “VasP-MoDoKi”) is a lightweight engine that **reads and writes VASP-style inputs/outputs** and performs **molecular dynamics and structure relaxations** using **machine-learning interatomic potentials**. Keep familiar VASP workflows and artifacts while computations run through ASE-compatible ML calculators. A simple driver script, `vpmdk.py`, is provided.

**Supported calculators (via ASE):** **CHGNet**, **MatterSim**, **MACE**, and **MatGL** (via the M3GNet model). Availability depends on the corresponding Python packages being installed.

*Not affiliated with, endorsed by, or a replacement for VASP; “VASP” is a trademark of its respective owner. VPMDK only mimics VASP I/O conventions for compatibility.*


## Usage

1. Prepare a directory containing at least `POSCAR`. Optional files are
   `INCAR`, `POTCAR`, and `BCAR`.
2. Install requirements: `ase`, `pymatgen` and, depending on the potential you
   wish to use, `chgnet`, `mattersim`, `mace-torch` or `matgl`.
3. Run:

```bash
python vpmdk.py --dir PATH_TO_INPUT
```

If `--dir` is omitted, the current directory (`.`) is used.

The script writes `CONTCAR` for the final structure and prints energies similar
to VASP output. Unsupported VASP tags are ignored with a warning.


## Supported INCAR tags

The script reads a subset of common VASP `INCAR` settings. Other tags are ignored with a warning.

| Tag | Meaning |
|-----|--------|
| `NSW` | Number of ionic steps. Defaults to `0` (single-point calculation). |
| `IBRION` | Ionic movement algorithm. `0` runs molecular dynamics, any other value triggers a BFGS geometry optimisation with a fixed cell. Defaults to `-1`. |
| `ISIF` | Accepted for compatibility but ignored; the cell shape and volume remain fixed regardless of the value (default `None`). |
| `EDIFFG` | Convergence threshold for relaxations in eV/Å. Defaults to `-0.02`. |
| `TEBEG` | Initial temperature in kelvin for molecular dynamics (`IBRION=0`). Defaults to `300`. |
| `POTIM` | Time step in femtoseconds for molecular dynamics (`IBRION=0`). Defaults to `2`. |

## Detailed setup instructions

In addition to `POSCAR`, optionally place `INCAR`, `POTCAR` and `BCAR` in the
same directory. `BCAR` is a simple `key=value` text file used to specify the
machine-learning potential:

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

### Required Python modules

`ase` and `pymatgen` are always required. Additional modules depend on the
selected potential:

| Potential        | Module to install        | Notes |
|------------------|--------------------------|-------|
| CHGNet           | `chgnet` (uses PyTorch)  | Bundled with a default model; specify `MODEL` to use another |
| MatGL (M3GNet)   | `matgl` (uses JAX)       | Bundled with a default model; specify `MODEL` to use another |
| MACE             | `mace-torch` (PyTorch)   | Set `MODEL` to a trained `.model` file |
| MatterSim        | `mattersim` (PyTorch)    | Set `MODEL` to the trained parameters |

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
