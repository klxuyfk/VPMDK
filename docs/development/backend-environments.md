# Backend Environment Notes

## General Guidance

Backend dependencies are heterogeneous. In practice, separate environments are
often the cleanest option, especially when mixing:

- FAIRChem v1 and v2
- Torch and JAX/XLA stacks
- charge-density runners with force-field environments

Always align CUDA, PyTorch/JAX, and any compiled extension wheels.

## FAIRChem v1

Practical recipe distilled from the repository's maintainer notes:

```bash
FAIRCHEM_V1_ENV=/path/to/venvs/fairchem-v1
python -m venv "${FAIRCHEM_V1_ENV}"
"${FAIRCHEM_V1_ENV}/bin/pip" install --upgrade pip
"${FAIRCHEM_V1_ENV}/bin/pip" install fairchem-core==1.10.0
"${FAIRCHEM_V1_ENV}/bin/pip" install torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
"${FAIRCHEM_V1_ENV}/bin/pip" install "scipy==1.15.3"
```

Important caveats:

- PyG extras such as `torch-scatter` and `torch-sparse` are required for common
  OCP import paths
- use a SciPy release that still provides `scipy.special.sph_harm`; the pinned
  `1.15.3` recipe remains compatible with Python 3.10 while avoiding the
  removal planned in newer SciPy releases
- `MODEL` is required
- `FAIRCHEM_CONFIG` is usually required as well

## EquiformerV3

`MLP=EQUIFORMER_V3` uses the FAIRChem v1/OCP calculator path after importing the
official EquiformerV3 model registration module. Start from the FAIRChem v1
environment above, then make the EquiformerV3 source importable:

```bash
git clone https://github.com/atomicarchitects/equiformer_v3 /path/to/equiformer_v3
export PYTHONPATH=/path/to/equiformer_v3/src:${PYTHONPATH}
```

Typical settings:

```text
MLP=EQUIFORMER_V3
MODEL=/path/to/equiformer_v3_checkpoint.pt
DEVICE=cpu
```

By default VPMDK imports
`fairchem.experimental.models.equiformer_v3.equiformer_v3`. If you package the
model registration under a different module name, set `EQUIFORMER_V3_MODULE`.

## FAIRChem v2 / UMA

Guidance:

- keep this separate from FAIRChem v1
- install from the `fairchem-core` PyPI package, using a 2.x release line
- gated checkpoints may require HuggingFace access
- UMA is a convenient path once access is available

Baseline install pattern:

```bash
FAIRCHEM_V2_ENV=/path/to/venvs/fairchem-v2
python -m venv "${FAIRCHEM_V2_ENV}"
"${FAIRCHEM_V2_ENV}/bin/pip" install --upgrade pip
"${FAIRCHEM_V2_ENV}/bin/pip" install "fairchem-core>=2,<3"
```

If you need strict reproducibility, pin an exact 2.x version in your lockfile
instead of using the open 2.x range.

Typical settings:

```bash
export HF_HOME=/path/to/hf-cache
```

VPMDK's no-`MODEL` default for FAIRChem v2 uses the UMA registry name present in
both the 2.13.0 validation environment and current 2.x releases:

```text
MLP=FAIRCHEM_V2
MODEL=uma-s-1p1
FAIRCHEM_TASK=omat
```

## MatGL / M3GNet

Common issue: CUDA-enabled DGL builds must match the active Torch/CUDA stack.

Typical setup:

```bash
pip install dgl-cu121
pip install matgl
export DGLBACKEND=pytorch
```

When using packaged models, `MODEL` often points to a model directory rather
than a single file.

## GRACE / TensorPotential

GRACE uses JAX/XLA under the hood. Missing CUDA toolchain components,
especially `ptxas`, commonly break GPU execution.

Typical fixes:

```bash
conda install -y -c nvidia cuda-nvcc=12.1
export XLA_FLAGS=--xla_gpu_cuda_data_dir="${CONDA_PREFIX}"
```

## DeePMD (PyTorch backend)

Common issue: MPI-related import failures even when using local single-process runs.

Typical fixes:

```bash
conda install -y mpich
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
```

For some installations, DeepMD's own configuration may need MPI probing
disabled.

Multi-head checkpoints can be selected with:

```text
DEEPMD_HEAD=some_head_name
```

## NequIP / Allegro

Many public model artifacts are distributed as `.nequip` archives. VPMDK expects
deployed or compiled model files, so pre-compilation is often necessary:

```bash
nequip-compile \
  --mode torchscript \
  --device cuda \
  --target ase \
  /path/to/model.nequip \
  /path/to/model.pth
```

Then use:

```text
MLP=NEQUIP
MODEL=/path/to/model.pth
```

or `MLP=ALLEGRO` accordingly.

## SevenNet / FlashTP

Guidance:

- prefer the modern `sevenn` package
- `MLP=FLASHTP` is the same integration path as SevenNet but with flash
  acceleration forced on
- make sure the installed SevenNet build actually exposes the requested
  accelerator flags
- on the 2026-05-16 TITAN V validation host, `flashTP_e3nn` needed to be built
  from the FlashTP source tree with `CUDA_ARCH_LIST=70` and CUDA 12.6 headers
  visible via `CUDA_HOME`

`MLP=EQUFLASH` uses the same SevenNet + FlashTP runtime path, but requires a
local EquFlash-compatible checkpoint. Treat it as a checkpoint-dependent
adapter: the public `equflash-29M-oam` metadata records the checkpoint as
unreleased, so it is not a downloadable named model.

## ORB

`orb-models` can resolve pretrained weights by model key, but local checkpoints
are often more reproducible and faster for repeated testing.

Example:

```text
MLP=ORB
MODEL=/path/to/orb.ckpt
```

## Charge-Density Environments

Keep charge-density runners separate when needed:

- the main force-field environment can stay lean
- `CHARGE_PYTHON` points to a different interpreter
- `CHARGE_SOURCE_DIR` points to a source checkout when importable packages are
  not installed in that interpreter

Typical pattern:

```text
WRITE_CHGCAR=1
CHARGE_PYTHON=/path/to/charge-env/bin/python
CHARGE_SOURCE_DIR=/path/to/backend-checkout
CHARGE_MODEL=/path/to/model
```

## Local Model Cache Convention

Historically, manual validation in this repository has used external model
stores under paths such as:

```text
/path/to/external-model-cache/
```

That is not required by VPMDK, but it is a practical place to centralize large
checkpoints outside the repository.
