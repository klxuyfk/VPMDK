# CHGCAR With Charge-Density Backends

This directory demonstrates the new optional `CHGCAR` flow in both CLI and API usage.

It uses the final atomic structure from the run and predicts a charge-density grid.
The bundled input uses `CHARGE_MLP=DEEPCDP` with explicit SOAP metadata for Si so the reference output can be regenerated from an external checkpoint. The same `WRITE_CHGCAR` flow also supports `CHARGE_MLP=CHARGE3NET` and `CHARGE_MLP=DEEPDFT`.

## Files

- `POSCAR`: small Si test structure
- `INCAR`: requests a single-point style run and defines a small explicit fine FFT grid for a quick example run
- `BCAR`: enables `WRITE_CHGCAR` and selects the charge-density backend
- `run.sh`: CLI example using `vpmdk`
- `predict_api.py`: Python API example using `vpmdk.predict_charge_density(...)`
- `reference/`: generated CLI and API reference outputs

## Required Environment

The selected charge-density backend needs its model checkpoint. For the bundled DeepCDP settings, provide the checkpoint through either:

```bash
export VPMDK_DEEPCDP_MODEL=/path/to/deepcdp-model.pt
```

or:

```bash
export VPMDK_CHARGE_MODEL=/path/to/deepcdp-model.pt
```

For ChargE3Net, switch `CHARGE_MLP` and set the corresponding source, Python, and model values, for example:

```bash
export VPMDK_CHARGE_SOURCE_DIR=/path/to/charge3net
export VPMDK_CHARGE_PYTHON=/path/to/charge3net-env/bin/python
export VPMDK_CHARGE_MODEL=/path/to/charge3net/models/charge3net_mp.pt
```

To run a charge-density backend on GPU, set `CHARGE_DEVICE=cuda` in `BCAR` and use a compatible backend environment.

## Run

CLI:

```bash
./examples/chgcar_charge3net/run.sh
```

API:

```bash
python ./examples/chgcar_charge3net/predict_api.py
```

## Notes

- The CLI example updates `examples/chgcar_charge3net/reference/`.
- The API example writes `api_CHGCAR` in this directory; the reference copy is stored as `reference/api_CHGCAR`.
- This example pins `NGXF/NGYF/NGZF=24` so it finishes quickly. In real calculations you can omit them and let VPMDK derive the fine grid from `PREC`, `ENCUT`, and optional `NGX/NGY/NGZ`.
- The generated `CHGCAR` is VASP-like for the volumetric density block, but does not include PAW augmentation occupancies reconstructed from DFT.
- For DeepDFT, point `CHARGE_MODEL` at a directory with `arguments.json` and `best_model.pth`, and set `CHARGE_SOURCE_DIR` to a DeepDFT checkout if needed.
- For DeepCDP, point `CHARGE_MODEL` at a `.pt` checkpoint and provide SOAP metadata either in JSON or through `CHARGE_DEEPCDP_*` tags.
