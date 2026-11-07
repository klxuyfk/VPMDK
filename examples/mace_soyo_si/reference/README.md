# MACE-SOYO reference calculations

Generated with the public v0.1 MPtraj checkpoint and the explicitly selected
cuEquivariance `naive` evaluator on NVIDIA TITAN V. This evaluates the same
trained model using PyTorch operations on CUDA. It does not use the unsupported
SM 7.5+ cuEquivariance binary kernels.

`eager-gpu.json` contains the upstream tensor wrapper's direct energy, forces
and stress for the bundled displaced Si structure. See the
[validation record](../../../docs/development/validation.md#2026-11-06-mace-soyo-validation)
for the runtime, checkpoint checksum and export procedure.

`OUTCAR.txt`, `OSZICAR.txt` and `CONTCAR.vasp` preserve the actual VPMDK CLI
relaxation outputs. `gpu-validation.json` records the dedicated calculator's
energy, forces, stress and finite differences, the device, dependency versions,
and checkpoint SHA-256. The relaxation decreased the energy from
`-10.82544231` to `-10.83130479 eV`.

The later [native AOTI SM 7.0 validation](aoti-sm70/README.md) uses an isolated
0.6.1 library profile and passed the compiled workflow tests.
`cueq-architectures.json` records the wheel/library hashes and SM targets
inspected with `cuobjdump` during the compatibility investigation.
