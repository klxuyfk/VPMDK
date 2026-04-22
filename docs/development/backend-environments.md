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
python -m venv /tmp/fairchem_v1_env
/tmp/fairchem_v1_env/bin/pip install --upgrade pip
/tmp/fairchem_v1_env/bin/pip install fairchem-core==1.10.0
/tmp/fairchem_v1_env/bin/pip install torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
/tmp/fairchem_v1_env/bin/pip install "scipy==1.16.3"
```

Important caveats:

- PyG extras such as `torch-scatter` and `torch-sparse` are required for common
  OCP import paths
- newer SciPy versions can break v1 code that still expects `scipy.special.sph_harm`
- `MODEL` is required
- `FAIRCHEM_CONFIG` is usually required as well

## FAIRChem v2 / UMA

Guidance:

- keep this separate from FAIRChem v1
- gated checkpoints may require HuggingFace access
- UMA is a convenient path once access is available

Typical settings:

```bash
export HF_HOME=/path/to/hf-cache
```

```text
MLP=FAIRCHEM_V2
MODEL=uma-s-1
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
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/conda/env
```

## DeePMD (PyTorch backend)

Common issue: MPI-related import failures even when using local single-process runs.

Typical fixes:

```bash
conda install -y mpich
export LD_LIBRARY_PATH=/path/to/conda/env/lib:$LD_LIBRARY_PATH
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
/mnt/d/lin_temp/codex/
```

That is not required by VPMDK, but it is a practical place to centralize large
checkpoints outside the repository.
