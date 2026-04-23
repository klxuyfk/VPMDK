# Quick Start

## 1. Prepare a Calculation Directory

At minimum, VPMDK needs a `POSCAR`. Typical runs also include `INCAR` and
`BCAR`.

```text
calc_dir/
├── POSCAR
├── INCAR
└── BCAR
```

Compatibility inputs:

- `POTCAR` is optional and can affect species reconciliation and some
  compatibility-output metadata.
- `KPOINTS`, `WAVECAR`, and existing `CHGCAR` files are detected but ignored by
  the force-field run itself.

If `BCAR` is absent, VPMDK defaults to `MLP=CHGNET`.

## 2. Install One Backend

For a first run, CHGNet is the simplest path:

```bash
pip install vpmdk chgnet
```

## 3. Run a Minimal Relaxation

`INCAR`

```text
IBRION = 2
NSW = 200
EDIFFG = -0.02
ISIF = 3
```

`BCAR`

```text
MLP=CHGNET
DEVICE=cpu
```

Run:

```bash
vpmdk --dir ./calc_dir
```

Outputs written into `calc_dir/`:

- `CONTCAR`
- `OUTCAR`
- `OSZICAR`
- `vasprun.xml`

## 4. Run a Single-Point Calculation

Single-point mode is selected when either:

- `IBRION < 0`, or
- `NSW <= 0`

Example:

```text
IBRION = -1
NSW = 0
ISIF = 2
```

This still writes the compatibility outputs, but no ionic motion occurs.

## 5. Run Molecular Dynamics

`INCAR`

```text
IBRION = 0
NSW = 100
POTIM = 1.0
TEBEG = 300
TEEND = 300
MDALGO = 3
LANGEVIN_GAMMA = 1.0
```

`BCAR`

```text
MLP=MACE
MODEL=/path/to/mace.model
DEVICE=cuda
```

MD writes the usual compatibility files plus `XDATCAR`. If you also set
`WRITE_LAMMPS_TRAJ=1`, VPMDK writes `lammps.lammpstrj`.

## 6. Write CHGCAR

`WRITE_CHGCAR=1` triggers a separate charge-density prediction after the final
structure has been obtained.

```text
MLP=CHGNET
DEVICE=cpu
WRITE_CHGCAR=1
CHARGE_MLP=CHARGE3NET
CHARGE_PYTHON=/path/to/charge-env/bin/python
CHARGE_SOURCE_DIR=/path/to/charge3net
CHARGE_MODEL=/path/to/charge3net_mp.pt
```

The `CHGCAR` grid is derived from `INCAR` using `NGXF/NGYF/NGZF`, then
`NGX/NGY/NGZ`, then `ENCUT`. If `PREC` is omitted, VPMDK falls back to
`PREC=NORMAL`.

## 7. Try the Python API

The library API does not write `OUTCAR` or similar files unless you explicitly
attach compatibility observers.

```python
from ase.io import read
import vpmdk

atoms = read("POSCAR")

sp = vpmdk.single_point(atoms, mlp="CHGNET", device="cpu")
relaxed = vpmdk.relax(atoms, mlp="CHGNET", steps=100, fmax=0.02)
traj = vpmdk.md(atoms, mlp="MACE", model="/path/to/model", steps=20, temperature=300)
```

See [Python API](../user-guide/python-api.md) for the side-effect model and
compatibility options.

## 8. Use the Bundled Examples

Runnable examples live under [`examples/`](../../examples/README.md):

- `relax_chgnet`
- `md_mace`
- `neb_nequip_vtst`
- `api_chgnet`
- `chgcar_charge3net`
- `uspex_9_4_4_si`

Those examples are the best place to see complete directory layouts and
backend-specific `BCAR` snippets.
