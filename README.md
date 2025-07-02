# vp_modoki

This repository provides a simple script `vp_modoki.py` to run machine-learning
potentials using VASP style input files. The script currently supports the
CHGNet potential through its ASE calculator.

## Usage

1. Prepare a directory containing at least `POSCAR`. Optional files are
   `INCAR`, `POTCAR`, and `BCAR`.
2. Install requirements: `ase`, `pymatgen`, and `chgnet`.
3. Run:

```bash
python vp_modoki.py --dir PATH_TO_INPUT
```

The script writes `CONTCAR` for the final structure and prints energies similar
to VASP output. Unsupported VASP tags are ignored with a warning.
