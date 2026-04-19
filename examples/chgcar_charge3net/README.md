# CHGCAR With ChargE3Net

This directory demonstrates the new optional `CHGCAR` flow in both CLI and API usage.

It uses the final atomic structure from the run and predicts a charge-density grid with ChargE3Net.

## Files

- `POSCAR`: small Si test structure
- `INCAR`: requests a single-point style run and defines a small explicit fine FFT grid for a quick example run
- `BCAR`: enables `WRITE_CHGCAR`
- `run.sh`: CLI example using `vpmdk`
- `predict_api.py`: Python API example using `vpmdk.predict_charge_density(...)`

## Required Environment

The current ChargE3Net backend is executed in a separate Python environment or source checkout.

Set at least:

```bash
export VPMDK_CHARGE_SOURCE_DIR=/path/to/charge3net
export VPMDK_CHARGE_PYTHON=/path/to/charge3net-env/bin/python
```

Optional:

```bash
export VPMDK_CHARGE_MODEL=/path/to/charge3net/models/charge3net_mp.pt
```

If `VPMDK_CHARGE_MODEL` is not set, VPMDK looks for `models/charge3net_mp.pt` under `VPMDK_CHARGE_SOURCE_DIR`.

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

- The CLI example writes outputs under `examples/chgcar_charge3net/output/`.
- The API example writes `api_CHGCAR` in this directory.
- This example pins `NGXF/NGYF/NGZF=24` so it finishes quickly. In real calculations you can omit them and let VPMDK derive the fine grid from `PREC`, `ENCUT`, and optional `NGX/NGY/NGZ`.
- The generated `CHGCAR` is VASP-like for the volumetric density block, but does not include PAW augmentation occupancies reconstructed from DFT.
