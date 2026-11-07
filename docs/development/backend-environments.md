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

## EquiformerV2 / eqV2

EquiformerV2 / eqV2 checkpoints use the FAIRChem v1/OCP runtime, not a
dedicated `EQUIFORMER_V2` backend tag. Start from the FAIRChem v1 environment
above and select:

```text
MLP=FAIRCHEM_V1
MODEL=/path/to/eqV2_checkpoint.pt
DEVICE=cpu
```

This route covers OCP/FAIRChem v1 checkpoints whose model registration is
available from `fairchem-core==1.10.0`, including the `equiformer_v2` and
`hydra` eqV2 checkpoint formats used in the validation environments.

Original Equiformer V1 checkpoints using `graph_attention_transformer` are not
part of the maintained compatibility surface. They depend on older OCP trainer
constructor conventions and older PyG import paths, so do not treat them as
equivalent to EquiformerV2 / eqV2 support.

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

DPA-family checkpoints use this same backend:

```text
MLP=DEEPMD
MODEL=/path/to/dpa_checkpoint.pt
```

For DPA-4 / DPA4 / SeZM checkpoints, the `deepmd-kit` environment must include
the PyTorch implementation that registers the `dpa4` / `SeZM` model type. If
the package imports but checkpoint construction fails with an unknown model
type, the installed `deepmd-kit` line is too old for that checkpoint. If import
fails while loading `deepmd_op_pt`, align the PyTorch version with the
`deepmd-kit` wheel or rebuild `deepmd-kit` against the active PyTorch runtime.

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

`MLP=EQUFLASH` uses the official EquFlash/GGNN runtime, not
`SevenNetCalculator`. Install the package that exposes
`GGNN.common.calculator.UCalculator`, download an upstream EquFlash or
EquFlashV2 checkpoint, and set `MODEL` to that local `.pt` file. The upstream
package currently requires Python 3.12 and a tightly pinned PyTorch/CUDA stack,
so a dedicated environment is recommended.

The following package combination was validated on CUDA 12.6. PyTorch must be
installed from its CUDA wheel index, and the matching PyG extension wheels must
be installed before `equflash`:

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install \
  torch_scatter==2.1.2+pt29cu126 torch_sparse==0.6.18+pt29cu126 \
  --find-links https://data.pyg.org/whl/torch-2.9.1+cu126.html
python -m pip install equflash==0.0.2
```

```text
MLP=EQUFLASH
MODEL=/path/to/equflashv2_oam.pt
DEVICE=cuda
```

EquFlash 0.0.2 does not expose a `device` argument. It can select CPU versus
CUDA, but cannot select a nonzero logical CUDA index. To use a particular GPU,
map it to logical GPU 0 for the VPMDK process and keep `DEVICE=cuda`:

```bash
CUDA_VISIBLE_DEVICES=1 vpmdk --dir ./calc_dir
```

VPMDK rejects `DEVICE=cuda:1` (and other nonzero indices) with this runtime
instead of silently running on `cuda:0`. A future `UCalculator` that explicitly
declares a `device` argument receives the requested index directly.

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

## ALIGNN-FF

Current upstream source supports pure PyTorch inference without DGL. The
following commit was validated with the public `alignn_ff_mptrj` checkpoint:

```bash
python -m venv /path/to/alignn-env
source /path/to/alignn-env/bin/activate
python -m pip install -e '.[test]'
python -m pip install 'alignn @ git+https://github.com/atomgptlab/alignn@45097bbf48b3ff9528519f1c7379f05c3bd88afe'
python - <<'PYCODE'
from alignn.ff.calculators import get_figshare_model_ff
print(get_figshare_model_ff(model_name="alignn_ff_mptrj"))
PYCODE
```

Validated with Python 3.13.5, PyTorch 2.8.0+cu128, ASE 3.27.0, and
jarvis-tools 2026.6.12 on CPU and NVIDIA TITAN V. Install the PyTorch build
appropriate for your hardware. For older checkpoints using DGL graphs, use
the separately validated stack below.
The adapter is optional and does not add ALIGNN to VPMDK's core dependencies.

```bash
OMP_NUM_THREADS=2 VPMDK_TEST_REAL_PYMATGEN=1 \
  VPMDK_ALIGNN_MODEL=/absolute/path/to/alignn_ff_mptrj \
  python -m pytest -q tests/integration/test_alignn.py
```

Tests skip unless `VPMDK_ALIGNN_MODEL` and the upstream calculator are available.
Set `VPMDK_ALIGNN_DEVICE` to change the test device (default `cpu`). Normal
runs can use named models; use a previously downloaded directory for offline
integration checks. Upstream owns its cache and may invalidate old cache
entries after package upgrades; retain a separate local copy if it must stay
fixed.

### Legacy DGL checkpoints on CPU and CUDA

`v12.2.2024_dft_3d_307k` passed the same CPU and GPU workflows. The actual
validation venv `/tmp/vpmdk-alignn-dgl-env` inherited system packages from
`codex_matgl_dgl` and added the pinned ALIGNN source and editable VPMDK.
The tested stack was Python 3.10.19, torch 2.2.1+cu121, DGL 2.1.0+cu121,
torchdata 0.7.1, NumPy 1.26.4, ASE 3.25.0, pymatgen 2025.10.7,
jarvis-tools 2026.6.12, and ALIGNN 2026.11.2 at the same commit above.

A fresh environment can use these matching versions (Linux x86_64):

```bash
python3.10 -m venv /path/to/alignn-dgl-env
source /path/to/alignn-dgl-env/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy==1.26.4
python -m pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install torchdata==0.7.1
python -m pip install 'dgl==2.1.0+cu121' -f https://data.dgl.ai/wheels/cu121/repo.html
python -m pip install ase==3.25.0 pymatgen==2025.10.7 jarvis-tools==2026.6.12
python -m pip install 'alignn @ git+https://github.com/atomgptlab/alignn@45097bbf48b3ff9528519f1c7379f05c3bd88afe'
python -m pip install -e '.[test]'
```

DGL GraphBolt initially failed to import because it could not locate
`libnvrtc.so.12`. Exposing the installed NVIDIA wheel libraries fixed the
import and allowed an actual CUDA DGL graph to run:

```bash
ALIGNN_DGL_SITE=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
export LD_LIBRARY_PATH="${ALIGNN_DGL_SITE}/nvidia/cuda_nvrtc/lib:${ALIGNN_DGL_SITE}/nvidia/cusparse/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python - <<'PYCODE'
import dgl
from alignn.ff.calculators import get_figshare_model_ff
print(dgl.graph(([0, 1], [1, 0])).to("cuda").device)
print(get_figshare_model_ff(model_name="v12.2.2024_dft_3d_307k"))
PYCODE
OMP_NUM_THREADS=2 VPMDK_TEST_REAL_PYMATGEN=1 \
  VPMDK_ALIGNN_MODEL=/absolute/path/to/v12.2.2024_dft_3d_307k \
  VPMDK_ALIGNN_DEVICE=cuda python -m pytest -q tests/integration/test_alignn.py
```

For a venv inheriting packages, use the base environment's `site-packages`
location for `ALIGNN_DGL_SITE`. Set `VPMDK_ALIGNN_DEVICE=cpu` for the CPU run;
the CPU/GPU comparison test then skips. An explicit CUDA request verifies the
model parameters are on CUDA and compares energy, forces and stress against
CPU. The legacy `v5.27.2024` configuration failed gradient consistency checks
and is rejected; do not substitute it for the validated model. See the
[validation record](validation.md#gpu-and-legacy-dgl-follow-up).

## MACE-SOYO

Use a separate environment: MACE-SOYO has independent dependencies and a
checkpoint format different from `mace-torch`. The source examined here is
commit `3c39bc7e8b7d33c269a1b2f9127988f72a55b9a2` (version 0.1.0).
Its pinned recipe requires Python >=3.12,<3.15 and torch 2.12.0. A CUDA 12.6
build was selected explicitly to retain NVIDIA TITAN V support.

```bash
python3.13 -m venv /path/to/mace-soyo-env
source /path/to/mace-soyo-env/bin/activate
python -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install 'MACE-SOYO @ git+https://github.com/PrincessHakuryu/MACE-SOYO@3c39bc7e8b7d33c269a1b2f9127988f72a55b9a2'
python -m pip install -e '.[test]'
```

The upstream dependencies pin cuEquivariance and its CUDA/Torch ops to 0.12.0,
nvalchemi-toolkit-ops to 0.4.1, warp-lang to 1.13.0, and
TorchSim (`torch-sim-atomistic`) to 0.6.1. Direct `.pth` inference needs no AOTI export. Optional `.pt2` export requires
the target GPU, a matching `nvcc` toolkit and a C++ compiler.
The actual local environment is `/tmp/vpmdk-mace-soyo-env`; CUDA toolkit
12.6.20 is under `/opt/nvidia/hpc_sdk/Linux_x86_64/2024/cuda/12.6`.
See the [example export command](../../examples/mace_soyo_si/README.md).

```bash
OMP_NUM_THREADS=2 VPMDK_TEST_REAL_PYMATGEN=1 \
  VPMDK_MACE_SOYO_MODEL=/absolute/path/to/model.pt2 \
  VPMDK_MACE_SOYO_DEVICE=cuda:0 \
  python -m pytest -q tests/integration/test_mace_soyo.py
```

Tests skip unless `VPMDK_MACE_SOYO_MODEL` and the upstream runtime are
available. CPU inference is not supported by MACE-SOYO. Re-export the package
when changing GPU/runtime; the integration never substitutes another model or
switches an explicit CUDA request to CPU.

The cuEquivariance 0.12 binary `libcue_ops.so` contains SM 7.5 and newer
kernels, but no SM 7.0 kernel. The ordinary exporter generated a package on
TITAN V and then failed its execution check with
`cudaErrorNoKernelImageForDevice`. For that GPU use direct checkpoint
inference with `MACE_SOYO_CUEQ_METHOD=naive` in BCAR. For integration tests:

```bash
OMP_NUM_THREADS=2 VPMDK_TEST_REAL_PYMATGEN=1 \
  VPMDK_MACE_SOYO_MODEL=/absolute/path/to/mace_soyo_mptraj.pth \
  VPMDK_MACE_SOYO_CUEQ_METHOD=naive VPMDK_MACE_SOYO_DEVICE=cuda:0 \
  python -m pytest -q tests/integration/test_mace_soyo.py
```

The portable method selects cuEquivariance's PyTorch evaluator during model
construction and retains convolution metadata. Derived polynomial buffers
are kept separate from learned checkpoint entries, which still load strictly.
Installed sources and checkpoint files are unchanged. This remains CUDA
inference and does not use the unsupported accelerated kernels. See the
[validation record](validation.md#2026-11-06-mace-soyo-validation).

### Verified AOTI export

For a GPU supported by the installed CUDA/PyTorch/cuEquivariance stack:

```bash
vpmdk-mace-soyo-export \
  --ckpt-path /absolute/path/to/mace_soyo_mptraj.pth \
  --structure-file examples/mace_soyo_si/POSCAR \
  --output-path /absolute/path/to/model.pt2 \
  --device cuda:0 --dtype float32 \
  --max-nodes 128 --max-edges 32768 --max-graphs 16
```

`python -m vpmdk_core.mace_soyo_export` is equivalent. The command runs a
native CUDA preflight, then upstream export and compiled/eager E/F/S checks.
It publishes the package and a `.pt2.json` provenance report only after all
checks pass. Failed exports preserve any existing package. Use `--overwrite`
to explicitly replace one. `--inductor-config key=value` forwards an upstream
compiler option. Export bounds must cover the intended structures.

Use `examples/mace_soyo_si/BCAR.aoti`, set `MODEL` to the verified package,
and omit the direct-checkpoint `MACE_SOYO_CUEQ_METHOD=naive` tag. To require
compiled inference and compare different atom counts against the checkpoint:

```bash
OMP_NUM_THREADS=2 VPMDK_TEST_REAL_PYMATGEN=1 \
  VPMDK_REQUIRE_MACE_SOYO_AOTI=1 \
  VPMDK_MACE_SOYO_MODEL=/absolute/path/to/model.pt2 \
  VPMDK_MACE_SOYO_REFERENCE_MODEL=/absolute/path/to/mace_soyo_mptraj.pth \
  VPMDK_MACE_SOYO_DEVICE=cuda:0 \
  python -m pytest -q tests/integration/test_mace_soyo.py
```

### Experimental SM 7.0 library replacement

The standard 0.12 library pins are unchanged. An isolated Python 3.12
experiment used the same torch 2.12.0+cu126 and MACE-SOYO source, replacing
all four cuEquivariance packages together:

```bash
python -m pip install --no-deps --force-reinstall \
  cuequivariance==0.6.1 cuequivariance-torch==0.6.1 \
  cuequivariance-ops-cu12==0.6.1 cuequivariance-ops-torch-cu12==0.6.1
python -m pip install pynvml
```

This intentionally conflicts with MACE-SOYO's declared 0.12 pins; keep it in
a separate environment. The old torch extension has Python 3.10–3.12 wheels,
so it cannot be substituted into the original Python 3.13 environment.
The 0.6.1 library contains SM 7.0 kernels. VPMDK's version-scoped compatibility
context removes a redundant registration where a Linear evaluator is stored
under both `m` and `fallback`, retaining the same objects, tensors and Python
alias. It does not suppress missing learned parameters or use `strict=False`.
The normal 0.12 construction is unaffected.

The local test environment is `/tmp/vpmdk-soyo-sm70-env`. System Python 3.12
lacked development headers. Its `libpython3.12-dev` package was downloaded
and extracted under `/tmp/vpmdk-python312-dev`, without a system install;
compilation used:

```bash
export CPATH=/tmp/vpmdk-python312-dev/usr/include/python3.12:/tmp/vpmdk-python312-dev/usr/include
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/2024/cuda/12.6
export TORCHINDUCTOR_COMPILE_THREADS=2
```

On another machine, a normal Python installation with matching development
headers can be used instead. See the dated validation record for the actual
results and package hashes; installing an old wheel alone is not evidence
that a full model works.

The 0.6.1 configuration **passed native CUDA execution and all six AOTI
integration tests on TITAN V**, including static, relaxation, MD, resident
reuse, energy gradients, and comparisons for 2/4/8 atoms. The native outputs
also matched the original 0.12 `naive` reference. A modern 0.13 forced install
still failed with no kernel image; replacing only the low-level 0.4 library
under the modern frontend failed to load native ops. Mixing these package
versions is not a working workaround.

The reproducible override pins are in
[`requirements-sm70-cueq.txt`](../../examples/mace_soyo_si/requirements-sm70-cueq.txt).
The standard 0.12 profile remains the default. A compiled artifact and its
report must stay with the runtime that exported it; changing the libraries
requires a new export and numerical checks.
