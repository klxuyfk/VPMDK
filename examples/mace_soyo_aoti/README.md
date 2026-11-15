# MACE-SOYO AOTI export and inference

This example follows the extra preparation required for compiled inference:
evaluate a checkpoint, export and verify an AOTI package, then run VPMDK with
that package. Both calculations use the same displaced two-atom Si structure
and static `INCAR`, so their energy, forces and stress can be compared.

Install the [MACE-SOYO CUDA environment](../../docs/development/backend-environments.md#mace-soyo)
and activate it. Export needs a supported GPU, matching CUDA toolkit with
`nvcc`, a C++ compiler and Python development headers. Set `CUDA_HOME` and
`PATH` for that toolkit. The standard cuEquivariance 0.12 kernels target
SM 7.5 and newer; the separate [experimental SM 7.0 recipe](../../docs/development/backend-environments.md#experimental-sm-70-library-replacement)
records the TITAN V configuration and its validation.

Run all commands below from the repository root in the same shell.

## 1. Evaluate the original checkpoint

Download the public MPtraj checkpoint, prepare a direct-inference calculation
under `output/eager`, and evaluate it with native CUDA kernels:

```bash
mkdir -p "$HOME/.cache/vpmdk/mace-soyo" examples/mace_soyo_aoti/output/eager
export MACE_SOYO_CHECKPOINT="$HOME/.cache/vpmdk/mace-soyo/mace_soyo_mptraj.pth"
curl -L --fail https://github.com/PrincessHakuryu/MACE-SOYO/releases/download/v0.1/mace_soyo_mptraj.pth \
  -o "$MACE_SOYO_CHECKPOINT"
cp examples/mace_soyo_aoti/POSCAR examples/mace_soyo_aoti/INCAR examples/mace_soyo_aoti/output/eager/
cat > examples/mace_soyo_aoti/output/eager/BCAR <<EOF_BCAR
MLP = MACE_SOYO
MODEL = $MACE_SOYO_CHECKPOINT
DEVICE = cuda:0
MACE_SOYO_HEAD = mptraj
MACE_SOYO_CUEQ_METHOD = native
EOF_BCAR
OMP_NUM_THREADS=2 vpmdk --dir examples/mace_soyo_aoti/output/eager
```

For ordinary direct-inference configuration and the `naive` option, see the
[backend reference](../../docs/reference/backends.md#mace-soyo).

## 2. Export and verify the package

```bash
vpmdk-mace-soyo-export \
  --ckpt-path "$MACE_SOYO_CHECKPOINT" \
  --structure-file examples/mace_soyo_aoti/POSCAR \
  --output-path examples/mace_soyo_aoti/model.pt2 \
  --device cuda:0 --dtype float32 \
  --max-nodes 128 --max-edges 32768 --max-graphs 16
```

The exporter checks native CUDA execution and compares compiled versus direct
energy, forces and stress. It publishes `model.pt2` and `model.pt2.json` only
after verification succeeds. The report records checkpoint/package hashes,
library versions, model metadata and target GPU. A failed export preserves an
existing destination; use `--overwrite` to replace a previously exported
package explicitly. Input files must differ from both output destinations.

The bounds above are for this small example. Choose bounds covering the
intended atom, edge and graph counts, and re-export for a different GPU/runtime.

## 3. Calculate with the compiled package

The bundled `BCAR` selects `MODEL=./model.pt2`. It omits
`MACE_SOYO_CUEQ_METHOD`, because a compiled package fixes its operations at
export time.

```bash
OMP_NUM_THREADS=2 vpmdk --dir examples/mace_soyo_aoti
```

The compiled calculation writes `OUTCAR`, `OSZICAR`, `CONTCAR` and
`vasprun.xml` in this directory. The direct calculation's outputs are under
`output/eager`. Small floating-point differences are expected; the exporter's
verification provides the numerical acceptance checks.

Detailed CPU/CUDA reports, historical relaxation outputs and the SM 7.0
library investigation are in the [development validation record](../../docs/development/validation.md#2026-11-06-mace-soyo-aoti-and-sm-70-compatibility).
