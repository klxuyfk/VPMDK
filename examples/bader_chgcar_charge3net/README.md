# Bader Charges From VPMDK CHGCAR

This example runs the optional VPMDK `CHGCAR` path and then passes the generated
file to the Henkelman Bader executable.

The run sequence is:

```bash
vpmdk
bader CHGCAR
```

## Requirements

- A working force backend for `MLP=CHGNET`
- A working charge-density backend environment
- Henkelman's `bader` executable on `PATH`, or `BADER_BIN=/path/to/bader`

Set at least:

```bash
export VPMDK_CHARGE_SOURCE_DIR=/path/to/charge3net
export VPMDK_CHARGE_PYTHON=/path/to/charge3net-env/bin/python
```

Optional:

```bash
export VPMDK_CHARGE_MODEL=/path/to/charge3net/models/charge3net_mp.pt
export BADER_BIN=/path/to/bader
```

Alternatively, edit the commented `CHARGE_*` paths in `BCAR` if you prefer to
keep the charge-backend paths inside the example input deck.

## Run

```bash
./examples/bader_chgcar_charge3net/run.sh
```

Outputs are copied to `examples/bader_chgcar_charge3net/output/`, including
`CHGCAR`, `ACF.dat`, `BCF.dat`, and `bader.log`.

`reference/` contains a representative run with local paths and timing text
sanitized. Treat the charge-density and Bader numbers as model/checkpoint
specific, not as validation baselines.

## Notes

- The generated `CHGCAR` contains the ML-predicted volumetric density. It does
  not include PAW all-electron augmentation data like `AECCAR0 + AECCAR2`.
- If your analysis requires a VASP-style all-electron reference density, VPMDK
  cannot synthesize that reference from an MLP charge model.
- `NGXF/NGYF/NGZF=24` keeps the example small. Use production grid settings for
  quantitative work.
