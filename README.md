# vp_modoki

This repository provides a simple script `vp_modoki.py` to run machine-learning
potentials using VASP style input files. The script supports several
potentials through their ASE calculators: **CHGNet**, **MatterSim**, **MACE**
and **MatGL** (via the M3GNet model). Availability of these calculators
depends on the corresponding Python packages being installed.

## Usage

1. Prepare a directory containing at least `POSCAR`. Optional files are
   `INCAR`, `POTCAR`, and `BCAR`.
2. Install requirements: `ase`, `pymatgen` and, depending on the potential you
   wish to use, `chgnet`, `mattersim`, `mace-torch` or `matgl`.
3. Run:

```bash
python vp_modoki.py --dir PATH_TO_INPUT
```

The script writes `CONTCAR` for the final structure and prints energies similar
to VASP output. Unsupported VASP tags are ignored with a warning.

## Detailed setup instructions

In addition to `POSCAR`, optionally place `INCAR`, `POTCAR` and `BCAR` in the
same directory. `BCAR` is a simple `key=value` text file used to specify the
machine-learning potential:

```
NNP=CHGNET            # Name of the potential
MODEL=/path/to/model  # Optional path to a trained parameter set
```

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

This script does not manage GPU settings, but the potentials themselves use a
GPU if available. With CUDA devices you can choose which GPU to use with
`CUDA_VISIBLE_DEVICES`. When running MatGL you may also set
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
python vp_modoki.py --dir calc_dir
```
