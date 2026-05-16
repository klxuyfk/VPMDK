# VPMDK

VPMDK (*Vasp-Protocol Machine-learning Dynamics Kit*, aka “VasP-MoDoKi”) is an ASE-oriented layer
for machine-learning interatomic potentials. Different MLP packages expose
different calculator constructors, model-loading conventions, and optional
features; VPMDK provides one place to absorb those differences and present a
more uniform workflow around `ase.Atoms`.

On top of that core API, VPMDK also provides a VASP-compatible CLI for
directory-based workflows. In practice it provides:

- a stable Python API for calculator construction, single-point runs,
  relaxations, MD, and charge-density prediction
- a compatibility CLI that reads `POSCAR` / `INCAR` / `BCAR` and writes
  VASP-like outputs such as `OUTCAR`, `OSZICAR`, `CONTCAR`, and `vasprun.xml`

Supported integrations include CHGNet, MACE, MatGL/M3GNet, SevenNet, FlashTP,
Eqnorm, MatRIS, AlphaNet, HIENet, Nequix, NequIP, Allegro, ORB, UPET, TACE,
EquFlash, FAIRChem, GRACE, DeePMD, MatterSim, and Matlantis, plus optional
charge-density backends such as ChargE3Net, DeepDFT, and DeepCDP. Actual
availability depends on which backend packages are installed in your
environment.

## Installation

Install the package itself:

```bash
pip install vpmdk
```

Or from a checkout:

```bash
pip install -e .
```

You also need at least one backend package for real calculations, for example:

```bash
pip install chgnet
```

More setup details:

- docs index: [docs/README.md](docs/README.md)
- installation guide: [docs/getting-started/installation.md](docs/getting-started/installation.md)
- backend reference: [docs/reference/backends.md](docs/reference/backends.md)

## Choose Your Entry Point

- Use the CLI if you want to run from VASP-style input directories and keep
  compatibility outputs.
- Use the Python API if you want filesystem-independent workflows around
  `ase.Atoms`.

CLI entry point:

```bash
vpmdk
```

Use `--dir PATH` only when you want to run against a calculation directory
other than the current one.

Python API entry points:

- `vpmdk.BackendConfig(...)`
- `vpmdk.get_calculator(...)`
- `vpmdk.single_point(...)`
- `vpmdk.relax(...)`
- `vpmdk.md(...)`
- `vpmdk.predict_charge_density(...)`

## Quick Start

### CLI

Work in a calculation directory containing:

```text
./
├── POSCAR
├── INCAR
└── BCAR
```

Minimal relaxation example:

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
vpmdk
```

If you prefer launching from outside that directory, use `vpmdk --dir ./calc_dir`.

### Python API

```python
from ase.io import read
import vpmdk
import vpmdk.compat.vasp as vasp_compat

atoms = read("POSCAR")
backend = vpmdk.BackendConfig(mlp="CHGNET", device="cpu")

sp = vpmdk.single_point(atoms, backend)
relaxed = vpmdk.relax(atoms, backend, steps=200, fmax=0.02, relax_cell=True)
traj = vpmdk.md(
    atoms,
    backend,
    temperature=300,
    steps=100,
    timestep=1.0,
    thermostat="langevin",
)

charge = vpmdk.predict_charge_density(atoms, incar={"ENCUT": 520})
vasp_compat.write_chgcar("CHGCAR", atoms, charge.density, spin_density=charge.spin_density)
```

The public Python API does not write `OUTCAR`, `OSZICAR`, or `vasprun.xml` by
default.

## Documentation Map

- docs index: [docs/README.md](docs/README.md)
- quick start: [docs/getting-started/quickstart.md](docs/getting-started/quickstart.md)
- CLI workflows: [docs/user-guide/cli-workflows.md](docs/user-guide/cli-workflows.md)
- Python API guide: [docs/user-guide/python-api.md](docs/user-guide/python-api.md)
- charge density and `CHGCAR`: [docs/user-guide/charge-density.md](docs/user-guide/charge-density.md)
- API reference: [docs/reference/api-reference.md](docs/reference/api-reference.md)
- `INCAR` reference: [docs/reference/incar-tags.md](docs/reference/incar-tags.md)
- `BCAR` reference: [docs/reference/bcar-tags.md](docs/reference/bcar-tags.md)
- backend reference: [docs/reference/backends.md](docs/reference/backends.md)
- architecture: [docs/development/architecture.md](docs/development/architecture.md)
- backend environment notes: [docs/development/backend-environments.md](docs/development/backend-environments.md)
- validation notes: [docs/development/validation.md](docs/development/validation.md)

## Examples

Runnable examples live under [examples/README.md](examples/README.md).

Included examples:

- `examples/relax_chgnet`
- `examples/md_mace`
- `examples/neb_nequip_vtst`
- `examples/api_chgnet`
- `examples/chgcar_charge3net`
- `examples/bader_chgcar_charge3net`
- `examples/uspex_9_4_4_si`

## Compatibility Notes

- `POSCAR` is required for standard runs.
- `POTCAR` is optional and can affect species reconciliation and some
  VASP-compatibility metadata.
- `KPOINTS`, `WAVECAR`, and existing `CHGCAR` files are ignored by the
  force-field calculation itself.
- If `BCAR` is omitted, VPMDK defaults to `MLP=CHGNET`.
- `WRITE_CHGCAR=1` runs a separate charge-density prediction step after the main
  calculation.
- VTST-style NEB directory layouts (`00`, `01`, ...) run through ASE NEB when
  `NSW > 0` and `IBRION > 0`, including spring-coupled band forces, climbing
  images via `LCLIMB`, and VASP-like image/parent outputs. Dimer/Lanczos TS
  modes are not implemented.

## License

VPMDK is distributed under the BSD 3-Clause License. See [LICENSE](LICENSE) for
details.
