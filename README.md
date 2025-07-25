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
