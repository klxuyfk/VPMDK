# Force field environment notes

This note collects environment pitfalls that required extra care during CUDA
validation runs. It focuses on backends that needed non-obvious steps; simpler
ones (e.g. CHGNet) are omitted on purpose. Always align CUDA/toolkit versions
with your PyTorch/JAX builds.

## FAIRChem v1 vs v2 (non-compatible)

FAIRChem 1.x and 2.x are not compatible in the same environment. Use separate
environments and select `NNP=FAIRCHEM_V1` or `NNP=FAIRCHEM`/`FAIRCHEM_V2` as
appropriate. For a detailed v1 recipe (with SciPy pinning and PyG extras),
see `notes/fairchem_v1_env_setup.md`.

## FAIRChem v2 (UMA)

### 1. Ensure access and cache location

Gated checkpoints (e.g. OC25) require HuggingFace access. UMA checkpoints are
an alternative once access is granted.

```bash
export HF_HOME=/path/to/hf-cache
```

### 2. Example BCAR requirements

`FAIRCHEM_TASK` is often required to select the correct head.

```text
NNP=FAIRCHEM_V2
MODEL=uma-s-1
FAIRCHEM_TASK=omat
```

## MatGL (M3GNet)

### 1. Install a CUDA-matching DGL build

MatGL 1.x commonly uses PyTorch + DGL rather than JAX. CUDA support requires a
DGL build that matches your CUDA/PyTorch versions.

```bash
pip install dgl-cu121
pip install matgl
```

### 2. Set backend and model directory

Some installations require explicitly setting `DGLBACKEND=pytorch`. When using
packaged model directories (e.g. M3GNet-MP), `MODEL` should point to the model
directory, which is passed to `matgl.load_model`.

```bash
export DGLBACKEND=pytorch
```

```text
NNP=MATGL
MODEL=/path/to/M3GNet-MP-2021.2.8-PES
```

## GRACE (TensorPotential)

### 1. Ensure CUDA toolchain for XLA

TensorPotential uses XLA/JAX under the hood. If `ptxas` is missing, install
`cuda-nvcc` matching your CUDA toolkit.

```bash
conda install -y -c nvidia cuda-nvcc=12.1
```

### 2. Point XLA to the conda CUDA runtime

```bash
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/conda/env
```

## DeePMD (PyTorch backend)

### 1. Avoid MPI import failures

DeePMD PyTorch builds may still try to import MPI. Installing `mpich` and
ensuring its libraries are on `LD_LIBRARY_PATH` avoids runtime import errors.

```bash
conda install -y mpich
export LD_LIBRARY_PATH=/path/to/conda/env/lib:$LD_LIBRARY_PATH
```

### 2. Disable CIBUILDWHEEL probing if needed

Some builds expect `CIBUILDWHEEL=0` in DeepMD's `run_config.ini` to avoid MPI
probing. If you see `MPI` load errors, adjust this setting.

### 3. Use `DEEPMD_HEAD` for multi-head checkpoints

```text
NNP=DEEPMD
MODEL=/path/to/DPA-*.pt
DEEPMD_HEAD=MP_traj_v024_alldata_mixu
```

## NequIP / Allegro

### 1. Compile `.nequip` archives for CUDA

Some public models are distributed as `.nequip` zips. Use `nequip-compile` to
generate a compiled/TorchScript `.pth` model before running VPMDK.

```bash
nequip-compile \
  --mode torchscript \
  --device cuda \
  --target ase \
  /path/to/model.nequip \
  /path/to/model.pth
```

### 2. Use the compiled model in BCAR

```text
NNP=NEQUIP
MODEL=/path/to/model.pth
```

## ORB

### 1. Prefer local checkpoints for repeat testing

`orb-models` can download weights automatically when `MODEL` is omitted, but
local checkpoints are often faster for repeated testing.

```text
NNP=ORB
MODEL=/path/to/orb-model.ckpt
```
