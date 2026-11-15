# Backend validation artifacts

These files preserve numerical reports and historical calculation outputs
referenced by the [validation record](../validation.md). They apply to the
dated checkpoints, runtime versions and hardware recorded there.

- [ALIGNN](alignn/README.md): relaxation outputs and pure-PyTorch/DGL CPU/CUDA
  comparisons.
- [MACE-SOYO](mace-soyo/README.md): direct and compiled results, GPU architecture
  inspection and the experimental SM 7.0 library pins.

These are development records rather than accuracy benchmarks or additional
user workflows. The [AOTI example](../../../examples/mace_soyo_aoti/README.md)
demonstrates checkpoint evaluation, verified export and compiled inference.
