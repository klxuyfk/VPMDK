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
EquFlash, EquiformerV2/eqV2 through FAIRChem v1, EquiformerV3, FAIRChem,
GRACE, DeePMD including DPA-family checkpoints, MatterSim, and Matlantis, plus
optional charge-density backends such as ChargE3Net, DeepDFT, and DeepCDP.
Actual availability depends on which backend packages are installed in your
environment.

EquFlash is exposed as a checkpoint-dependent SevenNet/FlashTP adapter: use a
local checkpoint, because no public named EquFlash checkpoint is currently
validated.

EquiformerV2 / eqV2 checkpoints are exposed through the legacy FAIRChem v1/OCP
path. Use `MLP=FAIRCHEM_V1` and set `MODEL` to the local eqV2 checkpoint; there
is no separate `MLP=EQUIFORMER_V2` tag. Original Equiformer V1
`graph_attention_transformer` checkpoints are outside the documented support
surface because they require older OCP trainer and registry conventions.

EquiformerV3 is exposed as `MLP=EQUIFORMER_V3` through the FAIRChem v1/OCP
runtime. It requires a local EquiformerV3 checkpoint and the official
`atomicarchitects/equiformer_v3` source tree on `PYTHONPATH`.

DPA-family Deep Potential checkpoints, including DPA-4 / DPA4 / SeZM models,
use the DeePMD-kit path: set `MLP=DEEPMD` and point `MODEL` at the local
DeepMD checkpoint.

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

### Optional: Resident Server for Batch Workloads

One-shot `vpmdk` remains the normal entry point. When many short calculation
directories use exactly the same model, the optional POSIX server mode can keep
that calculator in CPU/GPU memory and avoid repeated model loading:

```bash
vpmdk serve --dir ./model_config --daemon --idle-timeout 3600
vpmdk status
vpmdk run --dir ./calc-001
vpmdk stop
```

`model_config/BCAR` selects the resident calculator. An explicit backend setting
in a submitted directory's `BCAR` must match it; output options such as
`WRITE_ENERGY_CSV` remain per-calculation. Requests from concurrent clients are
processed serially in FIFO order; use separate sockets and servers for parallel
workers or different models.

The `run`, `status`, and `stop` commands use a standard-library-only client
entrypoint. They do not import PyTorch, ASE, pymatgen, or backend packages in
the submitting process, so short resident calculations retain the benefit of
avoiding repeated model-runtime startup.

See [Server Mode](docs/user-guide/server-mode.md) for lifecycle, protocol,
GPU/scheduler patterns, configuration matching, timeout, troubleshooting, and
security details. A complete shell workflow is in
[`examples/server_batch`](examples/server_batch/README.md).

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
- resident server mode: [docs/user-guide/server-mode.md](docs/user-guide/server-mode.md)
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
- `examples/server_batch`
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
