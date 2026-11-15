# Backend validation data

These dated numerical records support the [validation summary](../validation.md).
They record execution and consistency checks for specific checkpoints and
runtimes; they are not accuracy benchmarks against DFT.

## ALIGNN

- [`pure-cpu-gpu.json`](alignn/pure-cpu-gpu.json) and
  [`dgl-cpu-gpu.json`](alignn/dgl-cpu-gpu.json): runtime versions, checkpoint/config
  hashes, energy/forces/stress, finite differences and CPU/CUDA comparisons.

## MACE-SOYO

- [`gpu-validation.json`](mace-soyo/gpu-validation.json): direct-inference
  versions, checkpoint hash, energy/forces/stress and finite differences.
- [`export-report.json`](mace-soyo/aoti-sm70/export-report.json): verified SM 7.0
  package provenance, runtime, bounds and target metadata.
- [`export-check.log`](mace-soyo/aoti-sm70/export-check.log): numerical checks at
  export; file paths are generic placeholders.
- [`calculation-report.json`](mace-soyo/aoti-sm70/calculation-report.json): measured
  compiled/direct differences for 2/4/8 atoms and the original `naive` baseline.
- [`requirements-sm70-cueq.txt`](mace-soyo/requirements-sm70-cueq.txt): exact
  experimental override pins; see the [environment recipe](../backend-environments.md#experimental-sm-70-library-replacement).
