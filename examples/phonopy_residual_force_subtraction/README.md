# phonopy Residual-Force Subtraction FORCE_SETS

This example demonstrates phonopy's residual-force subtraction workflow using
VPMDK as the Allegro-backed VASP-compatible force provider.

The workflow is:

1. `phonopy -d --dim "2 2 2"` creates `SPOSCAR` and displaced `POSCAR-###`
   files.
2. `SPOSCAR` is run once as a static VPMDK calculation to obtain residual
   forces on the undisplaced supercell.
3. Each displaced `POSCAR-###` is run as a static VPMDK calculation.
4. `phonopy --fz perfect/vasprun.xml displacements/disp-*/vasprun.xml`
   subtracts the residual forces from every displaced-force set and creates
   `FORCE_SETS`.

This is useful for MLP-backed workflows because relaxed structures can still
carry small residual forces under a given model. VPMDK only has to write
VASP-like force arrays in each `vasprun.xml`; phonopy owns displacement
generation and force-set assembly.

Run from the repository root:

```bash
./examples/phonopy_residual_force_subtraction/run.sh
```

You can override the supercell size:

```bash
PHONOPY_DIM="3 3 3" ./examples/phonopy_residual_force_subtraction/run.sh
```

Results are copied to `output/`:

- `SPOSCAR`
- `phonopy_disp.yaml`
- `FORCE_SETS`
- `perfect/vasprun.xml`
- `displacements/disp-###/vasprun.xml`
- per-calculation VPMDK logs
- `phonopy_displacements.log`
- `phonopy_fz.log`

Prerequisites:

- VPMDK importable from this checkout or installed in the active environment
- `phonopy` on `PATH`
- Allegro and NequIP backend dependencies available in the active Python
  environment
- `ALLEGRO_MODEL` pointing to a deployed or compiled Allegro model. On this
  workstation the script defaults to
  `/mnt/d/lin_temp/codex/allegro/Allegro-OAM-L-0.1.nequip.pth`.
- `PYTHON` may be set explicitly. If omitted, the script uses
  `/home/nei/miniconda3/envs/codex_nequip/bin/python` when available.
