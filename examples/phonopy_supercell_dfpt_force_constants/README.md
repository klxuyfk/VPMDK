# phonopy Supercell DFPT-Style FORCE_CONSTANTS

This example mirrors phonopy's VASP-DFPT workflow shape:

1. `phonopy -d --dim "2 2 2"` creates `SPOSCAR`.
2. `SPOSCAR` is used as the VPMDK `POSCAR`.
3. VPMDK runs `IBRION=8` and writes a VASP-like `dynmat/hessian` block into
   `vasprun.xml`.
4. `phonopy --fc vasprun.xml` creates `FORCE_CONSTANTS`.

VPMDK does not run electronic DFPT. In this compatibility path, `IBRION=8`
uses symmetry-reduced finite differences of ORB-backed MLP forces and writes the
mass-normalized Hessian in the location read by phonopy's VASP parser. This
example is useful when you want the same file-level workflow as the phonopy
VASP-DFPT recipe: primitive input, phonopy supercell generation, a complete
supercell calculation, and `FORCE_CONSTANTS` extraction from `vasprun.xml`.

Run from the repository root:

```bash
./examples/phonopy_supercell_dfpt_force_constants/run.sh
```

You can override the supercell size:

```bash
PHONOPY_DIM="3 3 3" ./examples/phonopy_supercell_dfpt_force_constants/run.sh
```

Results are copied to `output/`:

- `SPOSCAR`
- `phonopy_disp.yaml`
- `vasprun.xml`
- `FORCE_CONSTANTS`
- `OUTCAR`
- `OSZICAR`
- `CONTCAR`
- `vpmdk.log`
- `phonopy_supercell.log`
- `phonopy_fc.log`

Prerequisites:

- VPMDK importable from this checkout or installed in the active environment
- `phonopy` on `PATH`
- ORB backend dependencies available in the active Python environment
- `ORB_MODEL_PATH` pointing to an ORB checkpoint
- `PYTHON` may be set explicitly. If omitted, the script uses `python`.
