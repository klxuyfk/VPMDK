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
vpmdk
```

From a source checkout, both compatibility wrappers also work:

```bash
python -m vpmdk
python vpmdk.py
```

For one-shot execution and `serve`, all three route into
`vpmdk_core.main()`. The `run`, `status`, and `stop` commands are intercepted by
a lightweight entrypoint before `vpmdk_core` is imported. This keeps client
process startup independent of installed ML backends.

Use `--dir PATH` only when you want to run against a calculation directory
other than the current one.

The optional resident-server subcommands are:

```text
vpmdk serve
vpmdk run
vpmdk status
vpmdk stop
```

They require a POSIX platform with Unix-domain sockets. Installing VPMDK does
not start a background service; a server exists only after an explicit
`vpmdk serve`. See [Server Mode](../user-guide/server-mode.md) when repeated
model loading justifies a resident process.

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
- `sevenn` plus `flashTP_e3nn` and a local EquFlash-compatible checkpoint for `MLP=EQUFLASH`
- `fairchem-core==1.10.0` for EquiformerV2 / eqV2 checkpoints through
  `MLP=FAIRCHEM_V1`
- the official `atomicarchitects/equiformer_v3` code plus its bundled FAIRChem
  v1/OCP runtime for `MLP=EQUIFORMER_V3`
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
- EquiformerV2 / eqV2 checkpoints, also through `MLP=FAIRCHEM_V1`
- `MLP=EQUIFORMER_V3`, when paired with the official EquiformerV3 repository
  on `PYTHONPATH`

This is only the baseline package pin, not the full working recipe. The v1 path
also needs the matching PyG extras and a SciPy pin that still provides
`scipy.special.sph_harm`. Use the full environment recipe in
[Backend Environment Notes](../development/backend-environments.md) when setting
up `MLP=FAIRCHEM_V1`; EquiformerV3 additionally needs the
`atomicarchitects/equiformer_v3` source tree because the `equiformer_v3` model
is registered from that project's `fairchem.experimental` code.

Do not mix FAIRChem v1 and v2 in the same environment. If you need
reproducibility, pin an exact `fairchem-core` release in your environment file
rather than relying on an open-ended install.

Original Equiformer V1 `graph_attention_transformer` checkpoints are not part
of the documented FAIRChem compatibility surface. They require older OCP trainer
and registry conventions that differ from VPMDK's maintained FAIRChem v1 path.

## DeePMD / DPA Guidance

DPA-family checkpoints use the DeePMD backend:

```text
MLP=DEEPMD
MODEL=/path/to/dpa_checkpoint.pt
```

DPA-4 / DPA4 / SeZM checkpoints require a `deepmd-kit` build that includes the
matching PyTorch model registration. If DeepMD fails while loading MPI-related
shared libraries, put the active environment's `lib` directory first:

```bash
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
```

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
