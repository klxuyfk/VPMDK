# VPMDK Examples

CLI-oriented examples plus Python API examples. Some directories include
`reference/` outputs generated from bundled inputs. Generated `output/`
directories are ignored; charge-density references are representative only,
because numerical values depend on the local charge-density backend and
checkpoint you provide. The USPEX input deck is a workflow-integration example
and does not include calculation output.

## Examples

- `relax_chgnet`: ionic relaxation with CHGNet
- `md_mace`: short MD run with MACE
- `neb_nequip_vtst`: NEB-style run with NequIP + VTST `nebresults.pl`
- `api_chgnet`: Python library examples using `single_point`, `relax`, and backend discovery
- `chgcar_charge3net`: CLI and API examples for optional `CHGCAR` output. The sample files use ChargE3Net, but the same `WRITE_CHGCAR` / `CHARGE_MLP` flow also supports DeepDFT and DeepCDP.
- `bader_chgcar_charge3net`: optional `CHGCAR` output followed by Henkelman's `bader CHGCAR`
- `phonopy_dfpt_force_constants`: MACE-backed `IBRION=7` VASP `dynmat` compatibility output followed by `phonopy --fc vasprun.xml`
- `phonopy_vasp_finite_difference`: SevenNet-backed `IBRION=5`, `POTIM`, and `NFREE=2` VASP finite-difference phonon mapping followed by `phonopy --fc vasprun.xml`
- `phonopy_supercell_dfpt_force_constants`: ORB-backed phonopy `SPOSCAR` generation followed by `IBRION=8` VASP-DFPT-style `FORCE_CONSTANTS` extraction
- `phonopy_displacement_force_sets`: NequIP-backed phonopy-generated displaced supercells followed by static VPMDK force calculations and `phonopy -f`
- `phonopy_residual_force_subtraction`: Allegro-backed phonopy-generated displaced supercells plus perfect-supercell residual-force subtraction with `phonopy --fz`
- `uspex_9_4_4_si`: USPEX 9.4.4 input deck showing `vpmdk` as a drop-in executable in a VASP-oriented structure-search workflow

## Run

From repository root:

```bash
./examples/relax_chgnet/run.sh
./examples/md_mace/run.sh
./examples/neb_nequip_vtst/run.sh
python ./examples/api_chgnet/single_point.py
python ./examples/api_chgnet/relax.py
python ./examples/api_chgnet/list_backends.py
./examples/chgcar_charge3net/run.sh
python ./examples/chgcar_charge3net/predict_api.py
./examples/bader_chgcar_charge3net/run.sh
./examples/phonopy_dfpt_force_constants/run.sh
./examples/phonopy_vasp_finite_difference/run.sh
./examples/phonopy_supercell_dfpt_force_constants/run.sh
./examples/phonopy_displacement_force_sets/run.sh
./examples/phonopy_residual_force_subtraction/run.sh
```

## Notes

- `md_mace/BCAR` and `neb_nequip_vtst/BCAR` use placeholders.
- Set `MODEL=...` in each `BCAR` to a checkpoint path before running.
- `relax_chgnet/run.sh` and `md_mace/run.sh` intentionally run only `vpmdk`.
- `api_chgnet` does not use `BCAR` or `INCAR`; it demonstrates the stable Python API directly.
- `chgcar_charge3net` requires a working charge-density backend environment. For the bundled example values that means ChargE3Net plus `VPMDK_CHARGE_SOURCE_DIR`, `VPMDK_CHARGE_PYTHON`, and optionally `VPMDK_CHARGE_MODEL`. To try DeepDFT or DeepCDP instead, change `CHARGE_MLP` and backend-specific model settings in `BCAR`.
- `bader_chgcar_charge3net` additionally requires Henkelman's `bader` executable on `PATH` or `BADER_BIN=/path/to/bader`.
- `bader_chgcar_charge3net/reference` contains representative VPMDK and Bader outputs with local paths and timing text sanitized.
- `chgcar_charge3net/INCAR` uses an explicit small fine FFT grid so the example stays runnable; for production-style inputs you would usually let `PREC`/`ENCUT` decide the grid.
- The phonopy examples require `phonopy` on `PATH`. They intentionally use
  different backend families: MACE, SevenNet, ORB, NequIP, and Allegro.
- The phonopy scripts default to local `codex_*` Python environments and
  checkpoints under `/mnt/d/lin_temp/codex` when those paths exist. Override
  `PYTHON`, `MACE_MODEL`, `SEVENNET_MODEL`, `ORB_MODEL_PATH`, `NEQUIP_MODEL`,
  or `ALLEGRO_MODEL` for another machine.
- `phonopy_dfpt_force_constants` demonstrates VPMDK's VASP `dynmat` compatibility layer. It does not run electronic DFPT.
- `phonopy_vasp_finite_difference` demonstrates the physically direct finite-difference mapping for `IBRION=5`, `POTIM`, and `NFREE=2`.
- `phonopy_supercell_dfpt_force_constants` demonstrates the full phonopy VASP-DFPT-style file workflow: `phonopy -d --dim`, `SPOSCAR`, `IBRION=8`, `vasprun.xml`, and `phonopy --fc`.
- `phonopy_displacement_force_sets` demonstrates the standard phonopy finite-displacement flow where phonopy owns displacement generation and VPMDK supplies VASP-like force outputs.
- `phonopy_residual_force_subtraction` demonstrates the same finite-displacement flow with residual forces from the perfect supercell subtracted by `phonopy --fz`.
- Band structures, meshes, DOS, thermal properties, irreducible representations, and animation-style post-processing are not VPMDK-specific once `FORCE_CONSTANTS` or `FORCE_SETS` exists, so they are intentionally left to normal phonopy commands rather than duplicated as separate VPMDK examples.
- `neb_nequip_vtst/run.sh` optionally accepts `NEQUIP_SOURCE` env var when NequIP is not installed in the current Python environment.
- VTST scripts are downloaded to a temporary directory and removed automatically.
- No checkpoint or VTST source code is stored in this repository.
- `reference/` is only an easy-to-read example of expected output style.
- Numerical values can change with checkpoint/model selection, so these examples are not intended as V&V baselines.
- `uspex_9_4_4_si` is a workflow-integration example rather than a standalone runnable calculation. It intentionally omits proprietary VASP pseudopotential content.
