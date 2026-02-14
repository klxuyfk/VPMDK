# CUDA MD validation report (manual)

This report records the manual CUDA MD validation runs performed for VPMDK.
It is intentionally detailed so the steps can be replayed later. It captures
the environment names, model locations, key BCAR settings, and the observed
issues/fixes. Matlantis was explicitly excluded from this validation.

> Note: These were manual integration runs executed in a WSL environment with
> CUDA available. They are separate from the pytest unit tests.

## Scope

- Goal: validate that each supported ML force field (except Matlantis) can run
  an MD calculation on CUDA without error.
- Where: WSL environment with CUDA access.
- Artifacts: per-backend run logs in `calc_gpu/<backend>/run.log`.
- Model cache: `/mnt/d/lin_temp/codex` was used for external checkpoints.
- HuggingFace token: provided by user to access gated models.

## Common inputs

- VPMDK run target: `vpmdk --dir <calc_dir>` or `python vpmdk.py --dir <calc_dir>`.
- Calculation directories: `calc_gpu/<backend>/` with POSCAR/INCAR/BCAR.
- Short MD trajectories were used for validation (not production length).

## Summary of results

- Successful CUDA MD: CHGNet, SevenNet, MatterSim, MACE, MatGL, ORB, GRACE,
  DeePMD (PyTorch backend), NequIP, Allegro, FAIRChem v1, FAIRChem v2 (UMA).
- Excluded: Matlantis (explicitly out of scope).
- Initial failures encountered and fixed:
  - GRACE: missing `ptxas` (fixed by installing `cuda-nvcc` and `XLA_FLAGS`).
  - MatGL: DGL backend mismatch (fixed by DGL + MatGL 1.3, `DGLBACKEND=pytorch`).
  - DeePMD: MPI library load failure (fixed by `mpich` and `LD_LIBRARY_PATH`).
  - FAIRChem v2: gated model access (resolved by using UMA model + task head).
  - NequIP/Allegro: `.nequip` archives needed compilation via `nequip-compile`.

## Force-field specific details

### CHGNet

- Conda env: `codex_pt`
- BCAR (key fields):
  - `NNP=CHGNET`
  - `DEVICE=cuda`
- Logs: `calc_gpu/chgnet/run.log`
- Outcome: CUDA MD completed without issues.

### SevenNet

- Conda env: `codex_sevenn`
- Special handling:
  - `PYTHONPATH` shim used to ensure correct module discovery.
- BCAR:
  - `NNP=SEVENNET`
  - `DEVICE=cuda`
- Logs: `calc_gpu/sevennet/run.log`
- Outcome: CUDA MD completed.

### MatterSim

- Conda env: `codex_pt`
- BCAR:
  - `NNP=MATTERSIM`
  - `DEVICE=cuda`
- Logs: `calc_gpu/mattersim/run.log`
- Outcome: CUDA MD completed.

### MACE

- Conda env: `codex_pt`
- Model:
  - `/mnt/d/lin_temp/codex/mace/mace_mp_small.model`
- BCAR:
  - `NNP=MACE`
  - `MODEL=/mnt/d/lin_temp/codex/mace/mace_mp_small.model`
  - `DEVICE=cuda`
- Logs: `calc_gpu/mace/run.log`
- Outcome: CUDA MD completed.

### ORB

- Conda env: `codex_pt`
- Model:
  - `/mnt/d/lin_temp/codex/orb/orb-v3-conservative-20-omat-20250404.ckpt`
- BCAR:
  - `NNP=ORB`
  - `MODEL=/mnt/d/lin_temp/codex/orb/orb-v3-conservative-20-omat-20250404.ckpt`
  - `DEVICE=cuda`
- Notes:
  - BCAR was updated to point at the local checkpoint.
- Logs: `calc_gpu/orb/run.log`
- Outcome: CUDA MD completed.

### MatGL (M3GNet)

- Conda env: `codex_matgl_dgl`
- Key packages:
  - `matgl==1.3.0`
  - `dgl` CUDA build
  - `torch==2.2.1+cu121`
  - `numpy==1.26.4`, `ase==3.25.0`
- Model:
  - `/mnt/d/lin_temp/codex/matgl/M3GNet-MP-2021.2.8-PES` (directory)
- Env vars:
  - `DGLBACKEND=pytorch`
  - `LD_LIBRARY_PATH` updated to include conda CUDA libs
- BCAR:
  - `NNP=MATGL`
  - `MODEL=/mnt/d/lin_temp/codex/matgl/M3GNet-MP-2021.2.8-PES`
  - `DEVICE=cuda`
- Logs: `calc_gpu/matgl/run.log`
- Outcome: CUDA MD completed after DGL backend fix.

### GRACE (TensorPotential)

- Conda env: `codex_grace`
- Model:
  - `/mnt/d/lin_temp/codex/grace/GRACE-2L-MP-r6`
- Env vars:
  - `PATH=/home/nei/miniconda3/envs/codex_grace/bin:$PATH`
  - `XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/nei/miniconda3/envs/codex_grace`
- Fixes:
  - Installed `cuda-nvcc=12.1` in `codex_grace` to provide `ptxas`.
- BCAR:
  - `NNP=GRACE`
  - `MODEL=/mnt/d/lin_temp/codex/grace/GRACE-2L-MP-r6`
  - `DEVICE=cuda`
- Logs: `calc_gpu/grace/run.log`
- Outcome: CUDA MD completed after XLA/CUDA fixes.

### DeePMD (PyTorch backend)

- Conda env: `codex_deepmd`
- Model:
  - `/mnt/d/lin_temp/codex/deepmd/DPA-3.1-3M.pt`
- BCAR:
  - `NNP=DEEPMD`
  - `MODEL=/mnt/d/lin_temp/codex/deepmd/DPA-3.1-3M.pt`
  - `DEEPMD_HEAD=MP_traj_v024_alldata_mixu`
  - `DEVICE=cuda`
- Env vars:
  - `DP_BACKEND=pytorch`
  - `LD_LIBRARY_PATH=/home/nei/miniconda3/envs/codex_deepmd/lib:$LD_LIBRARY_PATH`
- Fixes:
  - Installed `mpich` (conda and pip).
  - Edited DeepMD `run_config.ini`:
    - `CIBUILDWHEEL = 0` to avoid MPI probing errors.
- Logs: `calc_gpu/deepmd/run.log`
- Outcome: CUDA MD completed after MPI/libfabric fixes.

### NequIP

- Conda env: `codex_nequip`
- Model source:
  - Zenodo: `NequIP-OAM-L-0.1.nequip.zip`
- Compilation:
  - `nequip-compile --mode torchscript --device cuda --target ase <in.nequip> <out.pth>`
- Compiled model:
  - `/mnt/d/lin_temp/codex/nequip/NequIP-OAM-L-0.1.nequip.pth`
- BCAR:
  - `NNP=NEQUIP`
  - `MODEL=/mnt/d/lin_temp/codex/nequip/NequIP-OAM-L-0.1.nequip.pth`
  - `DEVICE=cuda`
- Logs: `calc_gpu/nequip/run.log`
- Outcome: CUDA MD completed.

### Allegro

- Conda env: `codex_allegro`
- Model source:
  - Zenodo: `Allegro-OAM-L-0.1.nequip.zip`
- Compilation:
  - `nequip-compile --mode torchscript --device cuda --target ase <in.nequip> <out.pth>`
- Compiled model:
  - `/mnt/d/lin_temp/codex/allegro/Allegro-OAM-L-0.1.nequip.pth`
- BCAR:
  - `NNP=ALLEGRO`
  - `MODEL=/mnt/d/lin_temp/codex/allegro/Allegro-OAM-L-0.1.nequip.pth`
  - `DEVICE=cuda`
- Logs: `calc_gpu/allegro/run.log`
- Outcome: CUDA MD completed.

### FAIRChem v1 (OCPCalculator)

- Conda env: `codex_fairchem_v1`
- Model:
  - `/mnt/d/lin_temp/codex/fairchem_v1/schnet_200k.pt`
- BCAR:
  - `NNP=FAIRCHEM_V1`
  - `MODEL=/mnt/d/lin_temp/codex/fairchem_v1/schnet_200k.pt`
  - `FAIRCHEM_CONFIG=/path/to/config.yml`
  - `DEVICE=cuda`
- Notes:
  - v1 and v2 are incompatible; separate environments required.
- Logs: `calc_gpu/fairchem_v1/run.log`
- Outcome: CUDA MD completed.

### FAIRChem v2 (UMA)

- Conda env: `codex_fairchem_v2`
- Model:
  - UMA checkpoint (model name): `uma-s-1`
- BCAR:
  - `NNP=FAIRCHEM_V2`
  - `MODEL=uma-s-1`
  - `FAIRCHEM_TASK=omat`
  - `DEVICE=cuda`
- Env vars:
  - `HF_HOME=/mnt/d/lin_temp/codex/hf`
- Notes:
  - Initial attempts with OC25 gated checkpoints failed until access was granted.
  - UMA + `FAIRCHEM_TASK` resolved the issue.
- Logs: `calc_gpu/fairchem_v2/run.log`
- Outcome: CUDA MD completed.

## Excluded backend

### Matlantis

- Explicitly excluded from this validation by user request.

## Local code changes tied to validation

To support the above runs, the following code paths were updated:

- MatGL: use `matgl.load_model` for model directories when available.
- NequIP/Allegro: fall back to `from_compiled_model` if `from_deployed_model` is
  not available.
- DeePMD: add BCAR `DEEPMD_HEAD` and pass through to calculator.

See `src/vpmdk_core/__init__.py`.

## Logs and artifacts

- Logs: `calc_gpu/<backend>/run.log`
- Output files (per run): `CONTCAR`, `OUTCAR`, `XDATCAR`
- Model cache: `/mnt/d/lin_temp/codex`

## Re-run checklist

1. Activate the correct conda environment for each backend.
2. Ensure CUDA is visible and `torch.cuda.is_available()` is true.
3. Set any required environment variables (see per-backend sections).
4. Ensure `MODEL` paths in BCAR exist (or are model names for FAIRChem).
5. Run `vpmdk --dir calc_gpu/<backend>` and confirm logs.
