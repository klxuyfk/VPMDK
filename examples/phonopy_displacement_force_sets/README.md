# phonopy Displacement FORCE_SETS

This example uses phonopy as the displacement generator and force-set parser,
while VPMDK is used only as the NequIP-backed VASP-compatible force calculator.

The workflow is:

1. `phonopy -d --dim "2 2 2"` creates displaced supercells.
2. Each `POSCAR-###` is run as a static VPMDK calculation.
3. `phonopy -f .../vasprun.xml` reads the VASP-like force blocks and creates
   `FORCE_SETS`.

This is independent of the `IBRION=7`/`8` force-constants path. It exercises the
older phonopy finite-displacement workflow, where phonopy owns the displacement
pattern and VPMDK only has to write VASP-compatible forces in `vasprun.xml`.

Run from the repository root:

```bash
./examples/phonopy_displacement_force_sets/run.sh
```

You can override the supercell size:

```bash
PHONOPY_DIM="3 3 3" ./examples/phonopy_displacement_force_sets/run.sh
```

Results are copied to `output/`:

- `SPOSCAR`
- `phonopy_disp.yaml`
- `FORCE_SETS`
- `displacements/disp-###/vasprun.xml`
- per-displacement VPMDK logs

Prerequisites:

- VPMDK importable from this checkout or installed in the active environment
- `phonopy` on `PATH`
- NequIP backend dependencies available in the active Python environment
- `NEQUIP_MODEL` pointing to a deployed or compiled NequIP model
- `PYTHON` may be set explicitly. If omitted, the script uses `python`.
