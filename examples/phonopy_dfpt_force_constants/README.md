# phonopy DFPT-Style Force Constants

This example demonstrates the VPMDK compatibility path for phonopy's VASP
`--fc` parser.

The calculation is started with normal VASP-style input files:

- `INCAR`: sets `IBRION=7`
- `BCAR`: selects MACE, CUDA, and the finite-difference displacement
- `POSCAR`: primitive two-atom Si cell

VPMDK does not run electronic DFPT. It computes force constants from central
finite differences of MLP forces, writes a VASP-like `dynmat/hessian` block into
`vasprun.xml`, and lets phonopy create `FORCE_CONSTANTS`:

```bash
phonopy --fc vasprun.xml
```

Run from the repository root:

```bash
./examples/phonopy_dfpt_force_constants/run.sh
```

Results are copied to `output/`:

- `vasprun.xml`
- `FORCE_CONSTANTS`
- `OUTCAR`
- `OSZICAR`
- `CONTCAR`
- `vpmdk.log`
- `phonopy_fc.log`

Prerequisites:

- VPMDK importable from this checkout or installed in the active environment
- `phonopy` on `PATH`
- MACE backend dependencies available in the active Python environment
- `MACE_MODEL` pointing to a MACE checkpoint. On this workstation the script
  defaults to `/mnt/d/lin_temp/codex/mace/mace_mp_small.model`.
- `PYTHON` may be set explicitly. If omitted, the script uses
  `/home/nei/miniconda3/envs/codex_pt/bin/python` when available.
