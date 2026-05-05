# phonopy VASP Finite-Difference Force Constants

This example demonstrates the VASP finite-difference phonon mapping in VPMDK.

The `INCAR` uses:

```text
IBRION = 5
POTIM = 0.015
NFREE = 2
```

VPMDK interprets this as a VASP-style finite-difference phonon calculation:
`POTIM` is the displacement in Angstrom, and `NFREE=2` selects central
differences. VPMDK writes a VASP-like `dynmat/hessian` block into `vasprun.xml`;
phonopy then creates `FORCE_CONSTANTS` with:

```bash
phonopy --fc vasprun.xml
```

Run from the repository root:

```bash
./examples/phonopy_vasp_finite_difference/run.sh
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
- SevenNet backend dependencies available in the active Python environment
- The sample `BCAR` uses `MLP=SEVENNET`, `MODEL=7net-0`, and `DEVICE=cuda`.
  Set `SEVENNET_MODEL` to use another SevenNet model.
- `PYTHON` may be set explicitly. If omitted, the script uses
  `/home/nei/miniconda3/envs/codex_sevenn/bin/python` when available.
