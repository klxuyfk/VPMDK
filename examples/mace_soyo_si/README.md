# MACE-SOYO silicon example

Use the dedicated `MLP=MACE_SOYO` backend. Ordinary `MLP=MACE` cannot load
MACE-SOYO's checkpoint format. Install the pinned upstream source in a
[separate CUDA environment](../../docs/development/backend-environments.md#mace-soyo).

From the VPMDK root:

```bash
mkdir -p "$HOME/.cache/vpmdk/mace-soyo"
curl -L --fail https://github.com/PrincessHakuryu/MACE-SOYO/releases/download/v0.1/mace_soyo_mptraj.pth \
  -o "$HOME/.cache/vpmdk/mace-soyo/mace_soyo_mptraj.pth"
ln -s "$HOME/.cache/vpmdk/mace-soyo/mace_soyo_mptraj.pth" examples/mace_soyo_si/model.pth
OMP_NUM_THREADS=2 vpmdk --dir examples/mace_soyo_si
```

The bundled BCAR uses the public `mptraj` head and explicitly selects
`MACE_SOYO_CUEQ_METHOD=naive`. This evaluates the same model equations with
cuEquivariance's PyTorch implementation on CUDA. It works on TITAN V, whose
SM 7.0 architecture is absent from the cuEquivariance 0.12 binary kernels.
It does not use those accelerated kernels. On a GPU supported by the kernels,
select `MACE_SOYO_CUEQ_METHOD=native` for direct native checkpoint inference.
The portable method and the experimental 0.6.1 native/AOTI configuration below
were validated on this machine.

The example relaxes displaced Si with a five-step limit. Outputs include
`OUTCAR`, `OSZICAR`, `CONTCAR` and `vasprun.xml`. For static calculations use
`IBRION=-1`, `NSW=0`. For a short MD run use:

```ini
IBRION = 0
NSW = 2
POTIM = 0.5
TEBEG = 100
TEEND = 100
MDALGO = 0
```

For AOTI on compatible hardware, export using the upstream command:

```bash
vpmdk-mace-soyo-export \
  --ckpt-path "$HOME/.cache/vpmdk/mace-soyo/mace_soyo_mptraj.pth" \
  --structure-file examples/mace_soyo_si/POSCAR \
  --output-path "$HOME/.cache/vpmdk/mace-soyo/model.pt2" \
  --device cuda:0 --dtype float32 \
  --max-nodes 128 --max-edges 32768 --max-graphs 16
```

Set `MODEL` to the resulting `.pt2` and remove `MACE_SOYO_CUEQ_METHOD` from
BCAR. That tag applies to checkpoint loading; `.pt2` fixes its method at
export. Set `CUDA_HOME` and `PATH` to a matching CUDA toolkit with `nvcc`.
The export command probes native cuEquivariance first, then delegates to the
upstream exporter. Only after compiled-versus-eager energy, force and stress
checks pass does it publish `model.pt2` and `model.pt2.json`. The report records
checkpoint/package hashes, library versions, model metadata and the target GPU.
A failed export preserves an existing destination; replacing one requires
`--overwrite`. The package and report destinations must both differ from both
input files, including resolved symlink aliases, even with `--overwrite`.
`python -m vpmdk_core.mace_soyo_export` is the equivalent module
entry point.

Use `BCAR.aoti` as the configuration template and point `MODEL` at the verified
package. The example export bounds are for smoke tests; increase them for
larger structures. Re-export on the intended GPU/runtime. Default cuEquivariance
0.12 binaries target SM 7.5 and newer; the compatibility investigation for
TITAN V is recorded separately in the validation record.

For a compiled package, run the dedicated comparisons with the original weights:

```bash
OMP_NUM_THREADS=2 VPMDK_TEST_REAL_PYMATGEN=1 \
  VPMDK_REQUIRE_MACE_SOYO_AOTI=1 \
  VPMDK_MACE_SOYO_MODEL=/absolute/path/to/model.pt2 \
  VPMDK_MACE_SOYO_REFERENCE_MODEL=/absolute/path/to/mace_soyo_mptraj.pth \
  VPMDK_MACE_SOYO_DEVICE=cuda:0 \
  python -m pytest -q tests/integration/test_mace_soyo.py
```

This requires the compiled runtime and compares energy/forces/stress against
direct inference for 2, 4 and 8 atoms, in addition to finite differences,
static calculation, relaxation, MD and resident reuse.

CPU and dispersion corrections are unsupported by this VPMDK integration.
Direct `.pth` inference supports unconditioned checkpoints; use upstream AOTI
for conditioned packages. `MACE_SOYO_HEAD` selects the head; a single-head
checkpoint may omit it. The resident server records the selected head and
method and rejects requests that change them. See the
[validation record](../../docs/development/validation.md#2026-11-06-mace-soyo-validation).

For the experimentally verified TITAN V AOTI configuration, create a separate
Python 3.12 environment using the normal installation recipe, then apply:

```bash
python -m pip install --no-deps --force-reinstall \
  -r examples/mace_soyo_si/requirements-sm70-cueq.txt
```

This intentionally overrides upstream's 0.12 requirements. Keep the four
cuEquivariance packages aligned at 0.6.1, use the same environment for export
and inference, and provide matching Python development headers and a CUDA
compiler. The version-scoped VPMDK compatibility rule preserves learned
weights and their strict loading. The actual SM 7.0 package passed all six
compiled integration checks; see
[`reference/aoti-sm70/`](reference/aoti-sm70/README.md).
