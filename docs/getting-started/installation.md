# Installation

## Core Package

Install the package itself from PyPI:

```bash
pip install vpmdk
```

Or from a repository checkout:

```bash
pip install -e .
```

Core runtime dependencies declared by the package are:

- `ase`
- `numpy`
- `pymatgen`

These are enough to parse VASP-style inputs, expose the public API, and run the
test suite stubs, but not enough to execute most production force fields.

## CLI Entry Points

After installation, the main command is:

```bash
vpmdk --dir ./calc_dir
```

From a source checkout, both compatibility wrappers also work:

```bash
python -m vpmdk --dir ./calc_dir
python vpmdk.py --dir ./calc_dir
```

All three route into `vpmdk_core.main()`.

## Backend Packages

VPMDK normalizes many calculators, but it does not vendor those model
libraries. Install the backend you intend to use in the current Python
environment unless you are only using the charge-density runners through
`CHARGE_PYTHON`.

Common packages:

- `chgnet` for `MLP=CHGNET`
- `mace-torch` for `MLP=MACE`
- `matgl` or legacy `m3gnet` for `MLP=MATGL` / `M3GNET`
- `sevenn` for `MLP=SEVENNET`
- `sevenn` plus `flashTP_e3nn` for `MLP=FLASHTP`
- `eqnorm` for `MLP=EQNORM`
- `matris` for `MLP=MATRIS`
- `alphanet` for `MLP=ALPHANET`
- `hienet` for `MLP=HIENET`
- `nequix` for `MLP=NEQUIX`
- `nequip` for `MLP=NEQUIP`
- `allegro` plus `nequip` for `MLP=ALLEGRO`
- `orb-models` for `MLP=ORB`
- `upet` for `MLP=UPET`
- `TACE` for `MLP=TACE`
- `fairchem-core>=2,<3` for `MLP=FAIRCHEM` / `FAIRCHEM_V2` / `ESEN`
- `fairchem-core==1.10.0` as the documented baseline for `MLP=FAIRCHEM_V1`
- `grace-tensorpotential` for `MLP=GRACE`
- `deepmd-kit` for `MLP=DEEPMD`
- `mattersim` for `MLP=MATTERSIM`
- `pfp-api-client` for `MLP=MATLANTIS`

See [Backend Reference](../reference/backends.md) for the exact per-backend
`MODEL` expectations and defaults.

## FAIRChem Version Guidance

FAIRChem support is version-sensitive and the package name on PyPI is
`fairchem-core`, not `fairchem`.

Recommended installation patterns:

```bash
pip install "fairchem-core>=2,<3"
```

Use that for:

- `MLP=FAIRCHEM`
- `MLP=FAIRCHEM_V2`
- `MLP=ESEN`

For legacy v1 / OCP-style usage, the documented baseline in this repository is:

```bash
pip install "fairchem-core==1.10.0"
```

Use that for:

- `MLP=FAIRCHEM_V1`

Do not mix FAIRChem v1 and v2 in the same environment. If you need
reproducibility, pin an exact `fairchem-core` release in your environment file
rather than relying on an open-ended install.

## Editable Development Setup

For local development, the repository guidelines assume:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest -m "not integration"
```

Integration tests are separate:

```bash
pytest -m integration
```

Those tests intentionally skip when their backend-specific prerequisites are
not present.

## GPU Builds

`DEVICE` is only a hint that VPMDK forwards to calculators. Successful GPU
execution still depends on the backend package, PyTorch/JAX build, CUDA
runtime, and any backend-specific extras.

General guidance:

- Install a CUDA-enabled PyTorch build before GPU-oriented Torch backends.
- For JAX/XLA-based stacks such as GRACE, ensure the CUDA toolchain is visible.
- Use `CUDA_VISIBLE_DEVICES` to restrict GPUs when needed.
- Keep separate environments when backend dependencies conflict.

## Charge-Density Backends in Separate Environments

`WRITE_CHGCAR=1` can use a different Python interpreter from the force-field
calculator. This is intentional. The CLI resolves charge inference through:

- `CHARGE_PYTHON`
- `CHARGE_SOURCE_DIR`
- `CHARGE_MODEL`
- backend-specific environment variables such as `VPMDK_DEEPDFT_MODEL`

That lets you keep, for example, a CHGNet force-field environment and a
separate ChargE3Net or DeepDFT environment.

The most generic environment variables are:

```bash
export VPMDK_CHARGE_PYTHON=/path/to/env/bin/python
export VPMDK_CHARGE_SOURCE_DIR=/path/to/backend/checkout
export VPMDK_CHARGE_MODEL=/path/to/model-or-model-dir
```

Backend-specific overrides exist for ChargE3Net, DeepDFT, and DeepCDP. See
[Charge Density](../user-guide/charge-density.md) for precedence and details.

## Automatic Download Locations

Several force-field integrations download named models on first use:

- Eqnorm: `~/.cache/eqnorm`
- MatRIS: `~/.cache/matris`
- AlphaNet: `~/.cache/alphanet`
- HIENet: `~/.cache/hienet`

Those downloads happen only when `MODEL` is omitted or set to a recognized
named model, and only for the backends that implement named-model support in
VPMDK itself.
