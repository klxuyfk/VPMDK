# VPMDK Examples

Three runnable examples with pre-generated outputs in each `reference/` directory.

## Examples

- `relax_chgnet`: ionic relaxation with CHGNet
- `md_mace`: short MD run with MACE
- `neb_nequip_vtst`: NEB-style run with NequIP + VTST `nebresults.pl`
- `uspex_9_4_4_si`: USPEX 9.4.4 input deck showing `vpmdk` as a drop-in executable in a VASP-oriented structure-search workflow

## Run

From repository root:

```bash
./examples/relax_chgnet/run.sh
./examples/md_mace/run.sh
./examples/neb_nequip_vtst/run.sh
```

## Notes

- `md_mace/BCAR` and `neb_nequip_vtst/BCAR` use placeholders.
- Set `MODEL=...` in each `BCAR` to your local checkpoint path before running.
- `relax_chgnet/run.sh` and `md_mace/run.sh` intentionally run only `vpmdk`.
- `neb_nequip_vtst/run.sh` optionally accepts `NEQUIP_SOURCE` env var when NequIP is not installed in the current Python environment.
- VTST scripts are downloaded to a temporary directory and removed automatically.
- No checkpoint or VTST source code is stored in this repository.
- `reference/` is only an easy-to-read example of expected output style.
- Numerical values can change with checkpoint/model selection, so these examples are not intended as V&V baselines.
- `uspex_9_4_4_si` is a workflow-integration example rather than a standalone runnable calculation. It intentionally omits proprietary VASP pseudopotential content.
