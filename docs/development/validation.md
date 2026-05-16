# Validation Notes

## Scope

This page records backend validation that has been run against real upstream
calculators.  It complements the regression tests under `tests/` and the
optional integration tests under `tests/integration/`.

The most recent sweep below was run manually because several backends require
mutually incompatible Python, PyTorch, TensorFlow, DGL, JAX, or CUDA dependency
sets.  A passed single-point entry means the VPMDK wrapper built the real ASE
calculator and completed one evaluation of energy and forces, and stress where
exposed by the upstream calculator.  It is a runnable backend smoke validation,
not a benchmark-quality comparison against reference DFT data.  Blocked entries
record adapter behavior or missing public artifacts rather than successful
calculator evaluation.

## 2026-05-16 Real Backend Sweep

- Structure: `tests/POSCAR` (`Si2`)
- API path: `BackendConfig` -> `build_calculator` -> `single_point`
- CPU runs: `DEVICE=cpu`, with CUDA hidden when the upstream package otherwise
  selected a GPU automatically
- CUDA runs: `DEVICE=cuda` where the builder exposes a device option; otherwise
  CUDA-visible execution is noted explicitly
- GPU host: NVIDIA TITAN V, driver 560.94, CUDA 12.6, compute capability 7.0
- Local checkpoints were stored outside the repository. Only public model
  identifiers or checkpoint filenames are recorded below; private absolute
  paths are intentionally omitted.

| Backend | Validation level | Packages / runtime | Model/checkpoint | CPU | CUDA | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `CHGNET` | Manual real-backend single point | torch 2.8.0+cu128, chgnet 0.4.2 | upstream CHGNet v0.3.0 default | passed | passed | stress returned |
| `MACE` | Manual real-backend single point | torch 2.8.0+cu128, mace 0.3.14 | local `mace_mp_small.model` | passed | passed | stress returned |
| `MATGL` | Manual real-backend single point | torch 2.2.1+cu121, matgl 1.3.0, dgl 2.1.0+cu121 | local `M3GNet-MP-2021.2.8-PES` model directory | passed | passed | required `DGLBACKEND=pytorch` and CUDA NVRTC library path |
| `M3GNET` | Manual real-backend single point | same as `MATGL` | same MatGL M3GNet model directory | passed | passed | alias path covered separately |
| `MATTERSIM` | Manual real-backend single point | torch 2.10.0+cu128, mattersim 1.2.0 | upstream `mattersim-v1.0.0-1M.pth` default | passed | passed | `DEVICE` now forwarded when supported |
| `EQNORM` | Manual real-backend single point | torch 2.6.0+cu124, eqnorm 0.1.0 | named `eqnorm-mptrj`, cached as `eqnorm-mptrj.pt` | passed | passed | named model cache already present |
| `MATRIS` | Manual real-backend single point | torch 2.6.0+cu124, matris 0.0.0 | named `matris_10m_oam`, cached as `MatRIS_10M_OAM.pth.tar` | passed | passed | fast graph converter import failed; legacy converter used |
| `ALPHANET` | Manual real-backend single point | torch 2.6.0+cu124, AlphaNet from `zmyybc/AlphaNet@bb83de4` | named `AlphaNet-MATPES-r2scan`, cached as `r2scan_1021.ckpt` plus `matpes.json` | passed | passed | PyPI `alphanet==0.0.20` was the wrong package and was replaced |
| `HIENET` | Manual real-backend single point | torch 2.6.0+cu124, hienet 1.0.1 | named `HIENet-0`, cached as `HIENet-V3.pth` | passed | passed | checkpoint downloaded from OpenMat/AIRS reference |
| `NEQUIX` | Manual real-backend single point | jax 0.5.3, nequix 0.4.5 | upstream `nequix-mp-1.nqx` cache | passed | passed | CPU required `JAX_PLATFORMS=cpu`; CUDA used JAX backend |
| `SEVENNET` | Manual real-backend single point | torch 2.5.1+cu121, sevenn 0.12.1 | upstream `7net-0` | passed | passed | Flash acceleration disabled |
| `FLASHTP` | Manual real-backend single point | torch 2.5.1+cu121, sevenn 0.12.1, flashTP_e3nn 0.1.0 | upstream SevenNet `7net-0` | not applicable | passed | `flashTP_e3nn` built from SNU-ARC/flashTP `0fbbbbae1061afc9285092a939d3f9abd851a758` with `CUDA_ARCH_LIST=70` |
| `NEQUIP` | Manual real-backend single point | torch 2.5.1+cu121, nequip 0.16.2 | local `NequIP-OAM-L-0.1.nequip.pth` | passed | passed | stress returned |
| `ALLEGRO` | Manual real-backend single point | torch 2.5.1+cu121, allegro 1.1.0 | local `Allegro-OAM-L-0.1.nequip.pth` | passed | passed | stress returned |
| `ORB` | Manual real-backend single point | torch 2.6.0+cu124, orb-models 0.5.5 | local `orb-v3-conservative-20-omat-20250404.ckpt` | passed | passed | CPU/CUDA numerical values differed strongly; runnable smoke only; numerical parity not asserted |
| `UPET` | Manual real-backend single point | torch 2.6.0+cu124, upet 0.1.2 | local `pet-oam-xl-v1.0.0.ckpt` | passed | passed | CUDA uses model on GPU with metatomic/vesin neighbor-list construction forced to CPU by default |
| `TACE` | Manual real-backend single point | torch 2.6.0+cu124, TACE 0.1.0 | upstream `TACE-v1-OMat24-M` cache | passed | passed | downloaded through TACE foundation model registry |
| `EQUFLASH` | Metadata-backed checkpoint audit and builder smoke | torch 2.5.1+cu121, sevenn 0.12.1, flashTP_e3nn 0.1.0 | `equflash-29M-oam` public metadata; no released checkpoint | not applicable | blocked | checkpoint-dependent SevenNet + FlashTP adapter; named metadata-only model gives an explicit error until a local checkpoint is supplied |
| `FAIRCHEM_V1` | Manual real-backend single point | torch 2.4.1+cu121, fairchem-core 1.10.0 | local `schnet_200k.pt` | passed | passed | stress unavailable for this calculator; forces were zero for this Si2 smoke case |
| `FAIRCHEM_V2` | Manual real-backend single point | torch 2.8.0+cu128, fairchem-core 2.13.0 | `uma-s-1`, `FAIRCHEM_TASK=omat` | passed | passed | VPMDK default now follows this fairchem-core 2.13.0 smoke path |
| `FAIRCHEM` | Manual real-backend single point | same as `FAIRCHEM_V2` | `uma-s-1`, `FAIRCHEM_TASK=omat` | passed | passed | alias path covered separately |
| `ESEN` | Manual real-backend single point | same as `FAIRCHEM_V2` | `uma-s-1`, `FAIRCHEM_TASK=omat` | passed | passed | alias path covered separately; OC25 ESEN checkpoints require gated HF access |
| `GRACE` | Manual real-backend single point | tensorflow 2.19.1, tensorpotential 0.5.7 | local `GRACE-2L-MP-r6` | passed | passed | CPU required hiding CUDA; CUDA used `XLA_FLAGS=--xla_gpu_cuda_data_dir=...` |
| `DEEPMD` | Manual real-backend single point | torch 2.8.0+cu128, deepmd-kit 3.1.2 | local `DPA-3.1-3M.pt`, `DEEPMD_HEAD=Omat24` | passed | passed | required `LD_LIBRARY_PATH` to include the DeepMD environment library directory |
| `MATLANTIS` | External partner validation only | Matlantis cloud | external notebook | not run | not run | intentionally excluded from this author-run sweep |

### Smoke Result Values

The following values are the direct single-point outputs for the Si2 smoke
structure.  They are recorded for reproducibility and sanity checking only.
For `ORB`, the CPU/CUDA values are runnable smoke outputs only; numerical parity
is not asserted by this sweep.

| Backend | CPU energy (eV) | CPU max force (eV/A) | CUDA energy (eV) | CUDA max force (eV/A) |
| --- | ---: | ---: | ---: | ---: |
| `CHGNET` | -10.6275053024 | 1.493841e-06 | -10.6275072098 | 4.904345e-06 |
| `MACE` | -10.7417467958 | 9.158087e-07 | -10.7417467958 | 9.158087e-07 |
| `MATGL` | -10.8376502991 | 1.089321e-05 | -10.8376502991 | 1.089321e-05 |
| `M3GNET` | -10.8376502991 | 1.089321e-05 | -10.8376502991 | 1.089321e-05 |
| `MATTERSIM` | -10.8289966583 | 1.788139e-06 | -10.8289966583 | 1.386401e-06 |
| `EQNORM` | -10.8511362076 | 7.547401e-07 | -10.8511371613 | 6.998525e-07 |
| `MATRIS` | -10.8466835022 | 2.738554e-06 | -10.8466835022 | 6.884336e-06 |
| `ALPHANET` | -16.9442996979 | 9.610888e-02 | -16.9442996979 | 6.043290e-02 |
| `HIENET` | -10.8296651840 | 4.711328e-07 | -10.8296651840 | 2.671732e-07 |
| `NEQUIX` | -10.8511791229 | 3.837049e-07 | -10.8511886597 | 2.826564e-07 |
| `SEVENNET` | -10.8058500290 | 3.539026e-07 | -10.8058490753 | 2.300596e-07 |
| `FLASHTP` | not applicable | not applicable | -10.8058490753 | 2.249085e-07 |
| `NEQUIP` | -10.8243139964 | 1.559278e-06 | -10.8243212883 | 1.635439e-06 |
| `ALLEGRO` | -10.8531294367 | 1.638958e-06 | -10.8531273177 | 1.421421e-06 |
| `ORB` | -3.0148992538 | 3.144207e-01 | -10.8141174316 | 8.637452e-03 |
| `UPET` | -10.8489933014 | 9.370406e-04 | -10.8489933014 | 9.370324e-04 |
| `TACE` | -10.8671770096 | 2.143905e-06 | -10.8671779633 | 2.177396e-06 |
| `FAIRCHEM_V1` | -0.7721433640 | 0.000000e+00 | -0.7721432447 | 0.000000e+00 |
| `FAIRCHEM_V2` | -10.8600664433 | 3.962317e-07 | -10.8600662347 | 1.560862e-06 |
| `FAIRCHEM` | -10.8600664433 | 3.962317e-07 | -10.8600662645 | 1.542566e-06 |
| `ESEN` | -10.8600664433 | 3.962317e-07 | -10.8600662645 | 1.483705e-06 |
| `GRACE` | -10.8407018830 | 1.684242e-06 | -10.8407018830 | 1.684242e-06 |
| `DEEPMD` | -10.7828680277 | 6.555597e-07 | -10.7828684449 | 8.229474e-07 |

## Current Test Coverage

The fast test suite exercises parser behavior, backend builder argument
forwarding, public API side-effect guarantees, VASP-compatible output
formatting, NEB-like directory handling, MD driver selection, charge-density
grid logic, and subprocess argument construction.

Useful command:

```bash
pytest -m "not integration"
```

`tests/integration/test_integration_md.py` covers short MD workflows for a
backend matrix and skips optional backends unless packages, checkpoints, and
environment variables are available.  The integration tests remain useful for
workflow-level output checks (`CONTCAR`, `OUTCAR`, `XDATCAR`), while the manual
sweep above records the wider real-backend compatibility status.

## Interpreting "Validated"

In this project, "validated" means:

- the backend path executed successfully through the VPMDK wrapper
- a real upstream calculator produced energy and forces
- stress was retrieved when the upstream calculator exposed it
- major environment issues encountered during the sweep were recorded

It does not imply:

- benchmark-quality force/energy agreement
- exhaustive testing across all model variants
- support for every upstream package version
- that CPU and CUDA numerics are expected to match exactly for every upstream
  package

## Recommended Validation Workflow for Changes

When backend logic or compatibility output changes:

1. Run `pytest -m "not integration"`.
2. Run the smallest relevant example under `examples/`.
3. Re-run the directly affected backend integration test if that backend is
   available locally.
4. Re-run a manual single-point smoke for the affected backend with the exact
   model/checkpoint and package versions being claimed.
5. Update this page when supported tags, defaults, package caveats, or
   validation results change.
