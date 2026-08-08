# VPMDK

VPMDK (*Vasp-Protocol Machine-learning Dynamics Kit*, aka “VasP-MoDoKi”)
provides a common ASE-oriented interface to machine-learning interatomic
potentials. It also offers a VASP-style command-line workflow that reads
`POSCAR`, `INCAR`, and `BCAR` and writes familiar outputs such as `OUTCAR`,
`OSZICAR`, `CONTCAR`, and `vasprun.xml`.

Use the Python API for `ase.Atoms` workflows, or use the CLI when existing
tools expect a VASP-like calculation directory. Supported backends and their
optional dependencies are listed in the
[backend reference](docs/reference/backends.md).

## Highlights

- One interface across ML potential packages with different model-loading and
  calculator conventions.
- Both a Python API for `ase.Atoms` and a directory-based CLI for existing
  VASP-oriented workflows.
- Single-point calculations, structural and cell relaxation, molecular
  dynamics, NEB, and optional charge-density prediction.
- Familiar VASP-style inputs and outputs, including `INCAR`, `POSCAR`,
  `OUTCAR`, `OSZICAR`, `CONTCAR`, and `vasprun.xml`.
- An optional resident server that avoids reloading the same model between
  short batch calculations.

## Supported Backends

Force-field integrations include `CHGNET`, `MATGL` / `M3GNET`, `MACE`,
`MATTERSIM`, `MATLANTIS`, `EQNORM`, `MATRIS`, `ALPHANET`, `HIENET`, `NEQUIX`,
`SEVENNET`, `FLASHTP`, `BAM`, `NEQUIP`, `ALLEGRO`, `ORB`, `UPET`, `TACE`,
`EQUFLASH`, `EQUIFORMER_V3`, `FAIRCHEM` (v1 and v2 paths, including `ESEN`),
`GRACE`, and `DEEPMD` (including DPA-family checkpoints).

Optional charge-density integrations include ChargE3Net, DeepDFT, and
DeepCDP. Actual availability depends on the packages and model files installed
in your environment. See the [backend reference](docs/reference/backends.md)
for model requirements, aliases, defaults, and backend-specific caveats.

## Installation

```bash
pip install vpmdk
pip install chgnet  # example backend
```

For development from a checkout:

```bash
pip install -e .
```

See the [installation guide](docs/getting-started/installation.md) for backend
environments and optional dependencies.

## CLI Quick Start

Create a calculation directory containing at least `POSCAR`. For example:

```text
calc/
├── POSCAR
├── INCAR
└── BCAR
```

`INCAR`:

```text
IBRION = 2
NSW = 200
EDIFFG = -0.02
ISIF = 3
```

`BCAR`:

```text
MLP=CHGNET
DEVICE=cpu
```

Run the calculation:

```bash
vpmdk --dir ./calc
```

If `BCAR` is omitted, the CLI defaults to CHGNet. See
[CLI Workflows](docs/user-guide/cli-workflows.md) for supported run modes and
the [INCAR](docs/reference/incar-tags.md) and
[BCAR](docs/reference/bcar-tags.md) references for configuration details.

## Python API

```python
from ase.io import read
import vpmdk

atoms = read("POSCAR")
backend = vpmdk.BackendConfig(mlp="CHGNET", device="cpu")

result = vpmdk.single_point(atoms, backend)
relaxed = vpmdk.relax(atoms, backend, steps=200, fmax=0.02)
trajectory = vpmdk.md(atoms, backend, temperature=300, steps=100)
```

The Python API does not write VASP compatibility files by default. See the
[Python API guide](docs/user-guide/python-api.md) and
[API reference](docs/reference/api-reference.md).

## Optional Server Mode

For batches that repeatedly use the same model, POSIX systems can keep one
calculator resident in CPU or GPU memory:

```bash
vpmdk serve --dir ./model_config --daemon
vpmdk run --dir ./calc
vpmdk stop
```

One server processes calculations serially. Read the
[Server Mode guide](docs/user-guide/server-mode.md) before using it in an
automated or shared-system workflow.

## Documentation and Examples

- [Documentation index](docs/README.md)
- [Quick start](docs/getting-started/quickstart.md)
- [CLI workflows](docs/user-guide/cli-workflows.md)
- [Python API](docs/user-guide/python-api.md)
- [Backend reference](docs/reference/backends.md)
- [Runnable examples](examples/README.md)

## License

VPMDK is distributed under the BSD 3-Clause License. See [LICENSE](LICENSE).
