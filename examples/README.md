# VPMDK Examples

CLI-oriented examples plus Python API examples. Runnable calculation examples include `reference/` outputs generated from the bundled inputs. The USPEX input deck is a workflow-integration example and does not include calculation output.

## Examples

- `relax_chgnet`: ionic relaxation with CHGNet
- `md_mace`: short MD run with MACE
- `neb_nequip_vtst`: NEB-style run with NequIP + VTST `nebresults.pl`
- `api_chgnet`: Python library examples using `single_point`, `relax`, and backend discovery
- `chgcar_charge3net`: CLI and API examples for optional `CHGCAR` output. The sample files use the DeepCDP charge-density backend so the reference can be generated from the available Si checkpoint.
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
```

## Notes

- `md_mace/BCAR` and `neb_nequip_vtst/BCAR` use placeholders.
- Set `MODEL=...` in each `BCAR` to a checkpoint path before running.
- `relax_chgnet/run.sh` and `md_mace/run.sh` intentionally run only `vpmdk`.
- `api_chgnet` does not use `BCAR` or `INCAR`; it demonstrates the stable Python API directly.
- `chgcar_charge3net` requires a working charge-density backend environment and a model provided through `CHARGE_MODEL`, `VPMDK_CHARGE_MODEL`, or a backend-specific model environment variable.
- `chgcar_charge3net/INCAR` uses an explicit small fine FFT grid so the example stays runnable; for production-style inputs you would usually let `PREC`/`ENCUT` decide the grid.
- `neb_nequip_vtst/run.sh` optionally accepts `NEQUIP_SOURCE` env var when NequIP is not installed in the current Python environment.
- VTST scripts are downloaded to a temporary directory and removed automatically.
- No checkpoint or VTST source code is stored in this repository.
- `reference/` is only an easy-to-read example of expected output style.
- Numerical values can change with checkpoint/model selection, so these examples are not intended as V&V baselines.
- `uspex_9_4_4_si` is a workflow-integration example rather than a standalone runnable calculation. It intentionally omits proprietary VASP pseudopotential content.
