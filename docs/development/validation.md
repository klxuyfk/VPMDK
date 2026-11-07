# Validation Notes

> **Status:** Historical manual validation record. Results apply to the dated
> environments shown below and are not claims about current upstream releases.
> Re-run the relevant smoke or integration tests before relying on an entry for
> a release decision.

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

## 2026-11-07 CI Thermostat Regression Recovery

The [main-branch Tests run](https://github.com/klxuyfk/VPMDK/actions/runs/42181216560)
after merging PR #30 failed only in the minimum-dependency job (Python 3.10,
ASE 3.25.0, pymatgen 2025.4.20, NumPy 1.26.4). The preceding PR run had passed.
The failure was the unseeded thermostat-conservation regression, independent
of the charge-density changes: conserved drift `0.00206044 eV` exceeded one
tenth of the bare-energy drift `0.01927778 eV`.

A fresh isolated environment with those exact dependency versions reproduced
the failure deterministically with seed 137: the old 2 fs test gave drift
ratio `0.11274010` (`0.00173624 / 0.01540042 eV`). The regression now uses
separate, local RNGs for Maxwell-Boltzmann initialization and Bussi noise, with
seeds 0, 42 and 137 for each thermostat. Its original tenfold conservation
criterion is unchanged. A 0.5 fs step resolves integration error; 120 steps
retain the original 60 fs trajectory. `NHC_PERIOD=80` retains the original
40 fs bath time constant, while `CSVR_PERIOD=40` is already specified in fs.
The exact comparison with ASE's thermostat-energy sum is a separate seeded
test, and all thermostat output is confined to the test's temporary directory.

All seven targeted cases passed with the minimum dependencies and with the
clean Python 3.12 / ASE 3.29.0 environment. CI also fixes BLAS/OpenMP thread
counts at one. These controls affect tests and workflow execution, leaving
production velocity sampling and dynamics unchanged.

Clean detached checkouts of the implementation commit also passed the full
non-integration suite: **1501 passed, 43 skipped** in the minimum-dependency
Python 3.10 environment, and **1502 passed, 42 skipped** each with latest
dependencies on Python 3.11 and 3.12 (82 integration cases deselected).
The skipped optional Torch tests are explicitly covered in CI's existing
CPU-Torch/CHGNet job, which now runs the ALIGNN, MACE-SOYO and exporter unit
files before its real CHGNet smoke calculation.
The same focused backend/export command passed **280 tests** locally with
CPU-configured Torch available.

## 2026-11-07 ALIGNN PBC Boundary Contract

The finding reproduced with the pinned ALIGNN source and public
`alignn_ff_mptrj` checkpoint through `build_calculator()` and
`single_point(..., calculator=...)`. Diamond Si₂ (`a=5.43 Å`, atom 1 displaced
by `0.1 Å` along x) returned the same `-10.736623764 eV` energy and
`Fx=-1.046583176 eV/Å` with PBC `[true,true,true]`, `[false,false,false]`,
`[true,true,false]`, `[false,true,true]` and `[true,false,true]`.
Upstream `ase_to_atoms()` copies lattice, elements and positions into JARVIS
without PBC metadata; the subsequent graph path uses fully periodic images.

The adapter now rejects any disabled PBC axis at the start of `calculate()`,
before installing a penalty hook or entering upstream calculation/conversion.
It reports a `ValueError` requiring fully periodic structures. This protects
energy, force and stress requests and the public single-point API, including
calculators constructed on demand from a backend config. It preserves input
boundary flags and checks the stored snapshot when an explicit `None` atom
argument is passed. It does not change the fully periodic inference path.

New deterministic tests cover all eight Boolean PBC combinations through
energy/force/stress getters and both single-point entry forms. An upstream
spy proves that none of the seven unsupported combinations is delegated.
Further cases cover changing a periodic cached input to partial PBC, restoring
periodicity, and the stored-atom check. Real-checkpoint regressions cover all
seven unsupported combinations, replacing upstream conversion with a failing
spy and asserting that no temporary model hooks remain installed. The existing
fully periodic inference, finite differences, CLI and resident tests remain
part of the same integration suite.

Validation used the previously recorded pinned-source environments and
cached public checkpoints:

| Suite | Result |
|-------|--------|
| ALIGNN unit regressions | 237 passed |
| `alignn_ff_mptrj`, CPU | 39 passed, 5 skipped |
| `alignn_ff_mptrj`, TITAN V CUDA | 40 passed, 4 skipped |
| `v12.2.2024_dft_3d_307k`, CPU | 43 passed, 1 skipped |
| `v12.2.2024_dft_3d_307k`, TITAN V CUDA | 44 passed |

The pure environment's four optional skips require DGL, and CPU also skips
the GPU comparison. Every PBC rejection regression passed on both devices
for each real model.

The full CPU-configured regression command
`CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/vpmdk-mpl pytest -q -m 'not integration'`
passed **1537 tests**, with one skipped and 82 integration cases deselected.
`git diff --check` passed.

## 2026-11-07 ALIGNN DGL Cutoff Penalty and Layer Defaults

Both findings reproduced against the same pinned ALIGNN 2026.11.2 source.
With the public `v12.2.2024_dft_3d_307k` weights and a temporary config setting
`use_cutoff_function=true`, `multiply_cutoff=false`, the default energy and
penalty flags were accepted. For periodic SiGeSi at
`[[0,0,0], [2,0.2,0.3], [0.4,2.2,0.1]] Å` in a 6 Å cubic cell, the original
float32 calculation gave `Fx=-1.49359512 eV/Å`, versus an energy finite
difference of `-1.28650665 eV/Å`. Its `σxx=0.02518050 eV/Å³` differed from
the finite-strain value `0.02310894 eV/Å³` (central step `0.003`).

Upstream DGL overwrites its bond lengths with the cutoff envelope in that
branch before differentiating the penalty. The adapter now uses the very
same upstream `cutoff_function_based_edges`, passing the model's effective
`inner_cutoff` and `exponent`, on the actual inference-graph distances. Pure
PyTorch keeps the physical distances for its penalty, and the multiplicative
cutoff branch is unchanged. Already-reported legacy penalties are not added
twice. Forces, stress, checkpoint weights and on-disk config bytes are intact.

Removing only `model.alignn_layers` from the same two-layer DGL checkpoint's
configuration reproduced `KeyError('alignn_layers')` on the first energy
evaluation. Validation and model construction used the schema's default of
two layers, but upstream ASE indexed the unpopulated config dictionary.
VPMDK now supplies `alignn_layers=2` in its private runtime configuration
when omitted. Explicit values are retained and the original file is unchanged.
Saved genuine two-layer pure/DGL state dicts test the first evaluation with
the key omitted, including energy/force/stress derivatives.

The cutoff integration regressions cover raw distances, cutoff-transformed
distances, multiplicative cutoff, disabled penalties, and the alternative
atom-count normalization. They upcast the same public weights to float64 to
resolve the smaller penalty contribution against the transformed network's
large energies. The DGL radius builder still quantizes distances to float32;
a central step of `0.0003` balances this roundoff and network curvature.
The original force/stress tolerances remain `0.002 eV/Å` and `0.00005 eV/Å³`.
For the transformed DGL case, a step sweep from `0.003` to `0.0003` reduced
the full force finite-difference error from about `0.00634` to `0.000111 eV/Å`
and the stress error from `0.000256` to `0.00000159 eV/Å³`.

Validation results in the previously recorded pinned-source environments:

| Suite | Result |
|-------|--------|
| ALIGNN unit regressions | 195 passed |
| `alignn_ff_mptrj`, CPU | 32 passed, 5 skipped |
| `alignn_ff_mptrj`, TITAN V CUDA | 33 passed, 4 skipped |
| `v12.2.2024_dft_3d_307k`, CPU | 36 passed, 1 skipped |
| `v12.2.2024_dft_3d_307k`, TITAN V CUDA | 37 passed |

Four pure-environment skips require DGL; CPU also skips the GPU comparison.
Unit coverage includes both architectures, all cutoff branches, both energy
normalizations, unchanged forces/stress and cache behavior, and omitted-layer
forwarding through named, directory and checkpoint-file selections. Existing
explicit layer counts, original config bytes, and the earlier rejection
guards remain covered.

The full CPU-configured regression command
`CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/vpmdk-mpl pytest -q -m 'not integration'`
passed **1495 tests**, with one skipped and 75 integration cases deselected.
`git diff --check` passed.

## 2026-11-07 ALIGNN Supported E/F/S Profile

All five follow-up findings reproduced against pinned ALIGNN 2026.11.2
(`45097bbf48b3ff9528519f1c7379f05c3bd88afe`). The public `alignn_ff_mptrj`
weights were reused unchanged, with separate sibling config copies. For
diamond Si₂ (`a=5.43 Å`, atom 1 displaced by `0.1 Å` along x), the reference
energy was `-10.736623764 eV`. Central differences used displacement/strain
step `0.003`:

| Config change | Returned Fx (eV/Å) | Energy-derived Fx (eV/Å) | Returned σxx (eV/Å³) | Energy-derived σxx (eV/Å³) |
|---------------|--------------------|--------------------------|-----------------------|-----------------------------|
| Unchanged | -1.04658318 | -1.04681651 | -0.01492835 | -0.01493526 |
| `lg_on_fly=false` | -0.85007674 | -1.04681651 | -0.01166280 | -0.01493526 |
| `grad_multiplier=1` | +1.04658318 | -1.04681651 | +0.01492835 | -0.01493526 |
| `stress_multiplier=2` | -1.04658318 | -1.04681651 | -0.02985670 | -0.01493526 |
| `link=logit` | -1.04658318 | -0.00483512 | -0.01492835 | -0.00006899 |
| `link=log` | -1.04658318 | -0.00488044 | -0.01492835 | -0.00006963 |
| Omitted `stresswise_weight` | -1.04658318 | -1.04681651 | 0 | -0.01493526 |

A small genuine eALIGNN state dict also reproduced the default torque
postprocessing crash: seed 2026, one ALIGNN layer, one GCN layer, 92 atom
features, embedding/hidden sizes 16, `energy_mult_natoms=false`, and
`stresswise_weight=0.1`. Evaluating displaced Si₂ through VPMDK raised
`IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)`
in upstream's singular-matrix torque fallback.

The adapter now has an explicit supported architecture/profile boundary:
only scalar `alignn_atomwise_pure` and DGL `alignn_atomwise` regression paths
are accepted. eALIGNN is rejected regardless of `remove_torque`; its earlier
source-only acceptance below is superseded. The profile validates angle
differentiation, the energy/force/stress multipliers and identity output link,
in addition to the existing normalization, reverse-edge and Cartesian-stress
rules. Classification and multi-output predictors are rejected. Pure ASE
models with zero ALIGNN layers are rejected because its forward expects three
graph inputs while upstream's zero-layer calculator supplies two; the DGL
zero-layer branch is retained and exercised with a small saved state dict.

Missing/zero `stresswise_weight` is enabled as `0.1` in a private runtime
configuration passed to upstream. This field only enables the stress branch;
it does not rescale energy, forces or stress. Positive weights are retained.
No checkpoint or config bytes are changed. ASE-level extra scales and batch
stress behavior are explicit rather than inherited from upstream defaults.
This defines supported inference behavior without changing pretrained
gradient conventions or accepting an unvalidated atomwise architecture.

Validation passed in the previously recorded pure-PyTorch and restored DGL
environments, using the same pinned source and cached public checkpoints:

| Suite | Result |
|-------|--------|
| ALIGNN unit regressions | 173 passed |
| `alignn_ff_mptrj`, CPU | 26 passed, 4 skipped |
| `alignn_ff_mptrj`, TITAN V CUDA | 27 passed, 3 skipped |
| `v12.2.2024_dft_3d_307k`, CPU | 29 passed, 1 skipped |
| `v12.2.2024_dft_3d_307k`, TITAN V CUDA | 30 passed |

The three pure-environment skips require DGL to create small genuine state
dicts; the additional CPU skip is the GPU comparison. Tests cover all seven
rejected derivative/output variants, omitted/zero/positive stress weights,
the original atom-count flag combinations, saved zero-layer DGL inference,
unsupported eALIGNN/pure-zero-layer checkpoints, and the existing CLI,
resident, one-atom and short-range workflows. Accepted real variants check
energy/force/stress finite differences and unchanged config bytes.

The full CPU-configured regression command
`CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/vpmdk-mpl pytest -q -m 'not integration'`
passed **1473 tests**, with one skipped and 68 integration cases deselected.
`git diff --check` passed.

## 2026-11-07 ALIGNN Cartesian Stress Regression

The review finding reproduced using the fixed ALIGNN 2026.11.2 source and
the cached `v12.2.2024_dft_3d_307k` checkpoint in the previously recorded
DGL environment. A temporary sibling configuration set
`include_pos_deriv=true`, `energy_mult_natoms=false`, and
`force_mult_natoms=false`; the cached files remained unchanged. For diamond
Si₂ (`a=5.43 Å`, atom 1 displaced by `0.1 Å` along x), the adapter returned
all-zero stress, while a central uniaxial strain derivative with step `0.003`
gave `σxx=0.05167178 eV/Å³` (energy `40.619140625 eV`).

Upstream initializes a stress placeholder, computes stress only in its
edge-derivative branch, and catches the Cartesian placeholder's reshape error
in the ASE calculator by returning zeros. Force-normalization compatibility
therefore does not make the Cartesian branch usable for this backend's
energy/force/stress contract. VPMDK now rejects that branch before loading
weights. Changing `force_mult_natoms` or `stresswise_weight` cannot bypass
the rejection. Named, directory and checkpoint-file selections use the same
validation, and original configuration bytes remain unchanged.

The earlier force-normalization entry below recorded Cartesian force checks
only; its apparent acceptance of that branch is superseded by this rejection.
The real-model flag-variant tests now check finite-strain stress derivatives
for every accepted variant as well as force derivatives, and assert rejection
of the Cartesian variants.

The ALIGNN unit suite passed **114 tests**, including twelve pre-load rejection
cases across named/directory/file selection, both force-multiplier settings,
and zero/nonzero stress training weights. Both `alignn_ff_mptrj` and legacy
`v12.2.2024_dft_3d_307k` passed the full integration file on CPU (**16 passed,
1 GPU-only skip**) and NVIDIA TITAN V CUDA (**17 passed**), including the
expanded force/stress finite-difference checks. The pinned source, restored
DGL environment and library paths are the same as in the normalization entry
below.

The full CPU-configured regression command
`CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/vpmdk-mpl pytest -q -m 'not integration'`
passed **1414 tests**, with one skipped and 55 integration cases deselected.
`git diff --check` passed.

## 2026-11-07 ALIGNN Effective Force Normalization Regression

The review finding reproduced using the same pinned ALIGNN 2026.11.2 source
(`45097bbf48b3ff9528519f1c7379f05c3bd88afe`) and cached `alignn_ff_mptrj`
weights. Copies of its configuration changed only the atom-count flags and
`include_pos_deriv`; the original checkpoint/config were not modified.
For diamond Si₂ (`a=5.43 Å`, atom 1 displaced by `0.1 Å` along x), central
energy differences with step `0.003 Å` gave `Fx=-1.04681651 eV/Å`:

| Configuration | Returned Fx (eV/Å) | Force / energy-gradient force |
|---------------|--------------------|-------------------------------|
| `energy_mult_natoms=true`, `force_mult_natoms=false` | -1.04658318 | 0.99978 |
| Both multipliers false, `include_pos_deriv=true` | -0.52329159 | 0.49989 |
| Both multipliers true | -2.09316635 | 1.99955 |

The old rejection incorrectly treated `include_pos_deriv` as active in pure
PyTorch and did not reject duplicate atom-count multiplication. The adapter
now validates the actual branch's combined normalization before constructing
the calculator. Pure PyTorch and the DGL edge-derivative branch require
exactly one atom-count multiplier. DGL Cartesian derivatives and eALIGNN
already multiply by the atom count unconditionally, so they require
`energy_mult_natoms=false` and ignore `force_mult_natoms`. The corresponding
defaults are included in validation. Boolean normalization settings must be
JSON booleans, avoiding ambiguous truthiness of strings or numbers. The
existing reverse-force guard remains in effect. No pretrained scale is changed.

A second normalization issue was confirmed for pure PyTorch's alternate
`(energy_mult_natoms=false, force_mult_natoms=true)` convention when ZBL is
enabled. Upstream adds ZBL as total energy, while multiplying its force by
the atom count. With float64, periodic Si₂ at 0.8 Å and the penalty disabled,
the force was `147.92459343 eV/Å` versus energy-gradient force
`73.12756868 eV/Å`. Disabling ZBL gave `-1.66945675 eV/Å` for both to within
`5e-9 eV/Å`. The alternate convention is therefore accepted only with
`use_zbl=false`; default/automatic ZBL remains compatible with the standard
`(true, false)` convention.

All **102** ALIGNN unit tests passed. The tests cover the full flag-pair table
for pure PyTorch (including the ignored position flag), both DGL derivative
branches and eALIGNN, omitted defaults, invalid boolean types, ZBL combinations,
and unchanged on-disk configuration bytes. The real-model integration suite
now includes all eight atom-count/position-flag combinations. It rejects the
incompatible variants and checks energy finite differences for every accepted
variant using symlinked weights and separate sibling configurations.

For each of `alignn_ff_mptrj` and `v12.2.2024_dft_3d_307k`, the full integration
file passed **16 tests, 1 GPU-only skip** on CPU and **17 tests** on TITAN V
CUDA. eALIGNN's normalization branch was checked against upstream source and
the unit-test matrix, without claiming a real eALIGNN checkpoint run.
The fixed upstream archive was available at `/tmp/vpmdk-review/upstream`,
with jarvis-tools/toolz dependencies under `/tmp/vpmdk-review/deps`; those
directories and `src` were supplied through `PYTHONPATH`. The removed legacy
temporary venv was recreated as `/tmp/vpmdk-alignn-normalization-dgl`,
inheriting `codex_matgl_dgl` and adding xmltodict 0.14.2,
pydantic-settings 2.12.0, pytest 8.4.2 and scikit-learn 1.6.1. The same
previously documented DGL NVIDIA library paths were used.

The full CPU-configured regression command
`CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/vpmdk-mpl pytest -q -m 'not integration'`
passed **1402 tests**, with one skipped and 55 integration cases deselected.
`git diff --check` passed.

## 2026-11-07 ALIGNN Reverse-Force Convention Regression

The registered public `alignnff_fmult` checkpoint was downloaded unchanged
from its [official Figshare file](https://ndownloader.figshare.com/files/41583585).
Its `alignn_atomwise` configuration sets `energy_mult_natoms=false`,
`force_mult_natoms=true`, `include_pos_deriv=false` and
`add_reverse_forces=false`. The force multiplier bypassed the previous
per-atom-energy guard, but could not replace the missing outgoing-edge
contributions. Checksums of the downloaded files:

- `best_model.pt`: `01d33f90f97e6cead9d25043ca09f2db364cec609ce6dcafb9064c7ad7be135b`
- `config.json`: `171d981bae442c18f68539fc18c395cca3346735b55121d0711e49f73ad3e1df`

Using the previously recorded DGL CPU environment, the reproduction started
from `ase.build.bulk("Si", "diamond", a=5.43)`, displaced atom 1 by
`[0.1, 0.04, -0.02]` Å, and scaled cell columns by `[1.03, 0.99, 1.01]`
with `scale_atoms=True`. The original calculator returned
`Fx=-0.05531463 eV/Å`; a central energy difference with step `0.001 Å`
gave `-0.12493134 eV/Å`. Temporarily enabling reverse accumulation in memory
gave `-0.12517500 eV/Å`, while leaving energy unchanged at
`-8.26965809 eV`. The on-disk checkpoint and configuration were not modified.

VPMDK now rejects this convention before constructing the calculator, for
named, local-directory and checkpoint-file selections. It does not override
the pretrained convention. The guard covers DGL's edge-derivative branch and
the pure-PyTorch model (which ignores `include_pos_deriv`). DGL Cartesian
derivatives and eALIGNN's unconditional reverse accumulation already include
both bond ends and are unaffected by this particular guard. An omitted
`add_reverse_forces` retains upstream's default `true`.

All **60** ALIGNN unit tests passed, including rejection before weight loading
for all three model-selection forms, unchanged config bytes, the accepted
default/explicit-true convention, and the independent Cartesian/eALIGNN
branches. The real downloaded `alignnff_fmult` was rejected through the public
VPMDK builder for all three selection forms. The supported `alignn_ff_mptrj`
and legacy `v12.2.2024_dft_3d_307k` each passed the full ALIGNN integration file
on CPU (**8 passed, 1 GPU-only skip**) and TITAN V CUDA (**9 passed**).
The full regression command
`CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/vpmdk-mpl pytest -q -m 'not integration'`
passed **1360 tests**, with one skipped and 47 integration cases deselected.
`git diff --check` passed.

## 2026-11-07 ALIGNN Short-Range Energy Regression

The review finding reproduced with the real `alignn_ff_mptrj` checkpoint:
periodic Si₂ in a 6 Å cubic cell, with separation 0.8 Å, returned identical
energies with `use_penalty` enabled and disabled. Its force on the second atom
changed by 0.2 eV/Å. The two directed short edges should contribute 0.04 eV
to total energy (`penalty_factor=0.1`, `penalty_threshold=1 Å`).

The adapter now captures the graph used by the upstream model and adds the
missing penalty to the reported energy with the same atom-count normalization.
It retains the actual directed edges, periodic images and neighbor selection.
It leaves model outputs, autograd forces, stress and pretrained scales intact.
Legacy DGL configurations with `energy_mult_natoms=false` already include the
penalty through an in-place energy update and are not corrected twice.
Temporary inference hooks are removed even when calculation fails.

The corrected real model returns the expected 0.04 eV increment. Central
differences of that increment agree with the penalty's force and stress
contributions. These checks subtract the unpenalized energy at each displaced
or strained geometry to isolate the small penalty from float32 noise in the
strongly repulsive neural energy. The existing equilibrium-geometry checks
continue to compare full energies against full forces and stress.
An additional isolated CPU calculation loaded the same checkpoint in float64
(`torch.set_default_dtype(torch.float64)` before construction). At 0.8 Å,
the full corrected-energy central differences with step `1e-5` gave
`73.327568677 eV/Å` versus force `73.327568339 eV/Å`, and
`-0.2715835872 eV/Å³` versus stress `-0.2715835864 eV/Å³`.

Using the previously recorded pure-PyTorch and DGL environments and downloaded
checkpoints, the full ALIGNN integration file passed on both CPU and NVIDIA
TITAN V: **8 passed, 1 GPU-only skip** on CPU and **9 passed** on CUDA, for
each of `alignn_ff_mptrj` and legacy `v12.2.2024_dft_3d_307k`. All **45** ALIGNN
unit tests passed, including disabled/inactive penalties, periodic edge
counting, intensive/extensive energy conventions, legacy double-count
prevention, unchanged upstream forces/stress, caching and hook cleanup.
The full CPU-configured regression run
(`CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/vpmdk-mpl pytest -q -m 'not integration'`)
passed **1345 tests**, with one skipped and 47 integration cases deselected.
`git diff --check` also passed.

## 2026-11-07 Resident Config and Export Collision Regressions

Both review findings reproduced before the fix in deterministic regressions.
Two symlinks to the same ALIGNN checkpoint loaded sibling configs with
cutoffs 5 and 4 Å, but the resident guard accepted the different request. The
MACE-SOYO exporter accepted a structure named `model.pt2.json` with output
`model.pt2` and `--overwrite`, replacing that structure with its report.

ALIGNN model selection and resident canonicalization now share the lexical
checkpoint/config path resolver. Local resident identity includes the resolved
weights, resolved sibling config, and a SHA-256 snapshot of config bytes.
Different configs and edits after startup are rejected. Same weight/config
symlinks and directory-versus-default-file selections are accepted. Omitted
MODEL inherits the loaded resident without touching the files. Named model
identity still performs no import or download of the optional ALIGNN runtime.

The exporter resolves both package/report destinations and both inputs before
export, rejecting intersections even with `--overwrite`. Tests include a valid
ASE JSON structure, each input against each destination, input symlinks and
symlinked output parents. They check that export has not started and original
input bytes remain unchanged.

With the real `alignn_ff_mptrj` checkpoint, the CPU integration suite passed
**7 tests**, with one GPU-only comparison skipped. The new socket-server test
accepts the resident configuration and rejects a different sibling cutoff
while reusing the same weight file. The targeted backend/export suite passed
**348 tests**. The full CPU-configured non-integration suite passed **1320
tests**, with one skipped and 46 integration cases deselected. Wheel/sdist
builds and `git diff --check` passed. No model weights or calculation formulas
were changed.

## 2026-11-06 MACE-SOYO AOTI and SM 7.0 Compatibility

Native AOTI export and VPMDK inference were completed on NVIDIA TITAN V
(SM 7.0) using an isolated legacy-library configuration. The standard
0.12 profile and its requirements remain unchanged. The implementation for
GPUs supported by that standard profile includes the verified exporter,
`BCAR.aoti`, provenance reports, failure-safe publication, and compiled/direct
integration comparisons. No SM 7.5+ GPU is installed here, so the actual
end-to-end hardware validation below uses SM 7.0 and the experimental profile.

Library replacement investigation:

| Configuration | Result on TITAN V |
|---------------|-------------------|
| Standard cuEquivariance 0.12.0 | Native preflight fails: no SM 7.0 kernel image; no unverified package is published |
| Forced matched 0.13.0 packages | Native preflight also fails: no SM 7.0 kernel image |
| Only low-level ops 0.4.0 under the modern frontend | Native extension cannot load; fallback was detected and rejected |
| Matched 0.6.1 packages, Python 3.12 | Native CUDA probe passed; model loads strictly after the alias compatibility rule; native AOTI export and all six workflow tests passed |

Wheel inspection found SM 7.0 cubins in 0.4.0, 0.5.0, 0.5.1 and 0.6.1.
The inspected 0.7.0, 0.9.1, 0.11.1, 0.12.0 and 0.13.0 libraries lack SM 7.0.
Exact wheel/library hashes and cubin/PTX targets are saved in
[`cueq-architectures.json`](../../examples/mace_soyo_si/reference/cueq-architectures.json).
The uniform_1d CUDA implementation was not found in the public frontend
repository; no binary architecture labels were patched. The working profile uses kernels
that were actually distributed for SM 7.0.

The isolated environment `/tmp/vpmdk-soyo-sm70-env` uses Python 3.12.3,
torch 2.12.0+cu126, CUDA toolkit 12.6.20, and `cuequivariance`,
`cuequivariance-torch`, `cuequivariance-ops-cu12`, and
`cuequivariance-ops-torch-cu12` all at 0.6.1. Its old torch extension has no
Python 3.13 wheel. The normal MACE-SOYO package was installed first, then
these packages were replaced with `--no-deps --force-reinstall`. `pip check`
therefore reports the four intentional disagreements with MACE-SOYO's 0.12
pins. This is an experimental compatibility profile, not a change to upstream
requirements. Exact override pins are provided in
[`requirements-sm70-cueq.txt`](../../examples/mace_soyo_si/requirements-sm70-cueq.txt).
The original Python 3.13/0.12 environment remains intact.

Old Linear registers the same evaluator as both `m` and `fallback`, while
the public modern checkpoint stores its coefficients under `m` only. The
version-scoped compatibility context removes the duplicate registration and
retains `fallback` as an alias of the same object. Learned parameters and
coefficient tensors are unchanged, and upstream still calls
`load_state_dict(..., strict=True)`. No missing learned keys are ignored.
The native 0.6.1 direct result was compared with the original 0.12 `naive`
result before attempting export and passed the E/F/S tolerances.

System Python 3.12 lacked `Python.h`, so `libpython3.12-dev` was downloaded
and extracted under `/tmp/vpmdk-python312-dev` without a system install.
The extracted headers were supplied through `CPATH`. The exporter printed a
warning about the absent default include directory, but compilation succeeded
using the supplied headers. The reproducible export was:

```bash
CPATH=/tmp/vpmdk-python312-dev/usr/include/python3.12:/tmp/vpmdk-python312-dev/usr/include \
OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2 \
CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/2024/cuda/12.6 \
/tmp/vpmdk-soyo-sm70-env/bin/vpmdk-mace-soyo-export \
  --ckpt-path /home/nei/.cache/vpmdk/mace-soyo/mace_soyo_mptraj.pth \
  --structure-file examples/mace_soyo_si/POSCAR \
  --output-path /home/nei/.cache/vpmdk/mace-soyo/mace_soyo_mptraj-sm70-cueq061.pt2 \
  --device cuda:0 --dtype float32 \
  --max-nodes 128 --max-edges 32768 --max-graphs 16
```

Export and verification took approximately 129 seconds. Embedded metadata
reports `AOTI_COMPUTE_CAPABILITY=70`, the batched E/F/S ABI, and head `mptraj`.
The package SHA-256 is
`25ebbd8b19cd6f0a7e38d88369810cead1667d363a461c2180dc90f30ab5886b`.
At export, maximum differences from eager execution were `1.192e-7 eV/atom`,
`2.426e-6 eV/Å` for forces, and `8.089e-8 eV/Å³` for stress. The package and
its `.pt2.json` report were published only after those checks passed.

Real compiled integration tests: **6 passed**. They explicitly require a
`.pt2` and the `aoti_model` runtime and cover E/F/S finite differences, static
calculation, relaxation, MD, two resident jobs without reloading, rejection of
a changed head, and compiled/direct comparisons for 2, 4 and 8 atoms. Maximum
comparison errors across those sizes in a separate saved calculation were
`1.491e-7 eV/atom`, `2.801e-6 eV/Å`, and `7.441e-8 eV/Å³`. The compiled
2-atom result also matched the original 0.12 `naive` baseline.

```bash
CPATH=/tmp/vpmdk-python312-dev/usr/include/python3.12:/tmp/vpmdk-python312-dev/usr/include \
OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2 \
VPMDK_TEST_REAL_PYMATGEN=1 VPMDK_REQUIRE_MACE_SOYO_AOTI=1 \
VPMDK_MACE_SOYO_MODEL=/home/nei/.cache/vpmdk/mace-soyo/mace_soyo_mptraj-sm70-cueq061.pt2 \
VPMDK_MACE_SOYO_REFERENCE_MODEL=/home/nei/.cache/vpmdk/mace-soyo/mace_soyo_mptraj.pth \
VPMDK_MACE_SOYO_DEVICE=cuda:0 \
/tmp/vpmdk-soyo-sm70-env/bin/python -m pytest -q tests/integration/test_mace_soyo.py
```

Reports, export checks and actual relaxation output are preserved in
[`reference/aoti-sm70/`](../../examples/mace_soyo_si/reference/aoti-sm70/README.md).
These are runtime/adapter checks and do not establish model accuracy against
DFT or performance on other GPU families.

Final validation for this follow-up: the full CPU-configured non-integration
suite passed **1297 tests**, with one skipped and 44 integration cases
deselected. The final targeted suite passed **336 tests**, including the later
version-scope cases. Tests cover preservation of previous outputs on failed
export, provenance hashes, invalid export inputs, argument restoration, and
legacy alias restoration without modifying distinct evaluators. Wheel/sdist
builds and `git diff --check` passed.

## 2026-11-06 MACE-SOYO Validation

Source: [PrincessHakuryu/MACE-SOYO](https://github.com/PrincessHakuryu/MACE-SOYO),
commit `3c39bc7e8b7d33c269a1b2f9127988f72a55b9a2`, version 0.1.0.
The [public v0.1 MPtraj checkpoint](https://github.com/PrincessHakuryu/MACE-SOYO/releases/download/v0.1/mace_soyo_mptraj.pth)
was downloaded to `/home/nei/.cache/vpmdk/mace-soyo/mace_soyo_mptraj.pth`.
SHA-256: `d726daf370a92d280b75c72ca8a1551007946e00eb75a2fed6200547cd0a7de3`.

Existing-backend compatibility was tested before adding the new integration.
`MLP=MACE` with this `.pth` raises
`AttributeError: 'dict' object has no attribute 'to'`: the artifact contains
`checkpoint_format_version`, `model_state_dict`, `model_kwargs`, and
`datasets`, rather than a MACE model instance. The generated `.pt2` was also tried with
`MLP=MACE` and failed with `PytorchStreamReader failed locating file data.pkl`.
The model has one `mptraj`
head, cutoff 6 Å, two layers, node dimension 512, ZBL enabled and no
charge/spin conditioning. Upstream uses a dedicated AOTI-only ASE calculator,
so the integration uses `MLP=MACE_SOYO`: `.pt2` selects that runtime, while
`.pth` uses a dedicated direct ASE wrapper around upstream `TensorEFS`.

A separate venv `/tmp/vpmdk-mace-soyo-env` was created with Python 3.13.5.
Runtime: torch 2.12.0+cu126, cuEquivariance/torch/ops 0.12.0,
nvalchemi-toolkit-ops 0.4.1, warp-lang 1.13.0, TorchSim 0.6.1,
ASE 3.29.0, NumPy 2.5.4, SciPy 1.18.1, pymatgen 2026.10.24.
CUDA toolkit: nvcc 12.6.20, with
`CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/2024/cuda/12.6`.
Hardware: NVIDIA TITAN V, compute capability 7.0, 12 GiB, logical CUDA 0.

The ordinary upstream export generated a `.pt2` but failed when executing
it for the consistency check with `cudaErrorNoKernelImageForDevice` in
cuEquivariance `uniform_1d`. `cuobjdump --list-elf libcue_ops.so` shows
SM 7.5 and newer cubins and no SM 7.0 cubin. Basic PyTorch CUDA arithmetic
and compilation both passed on this GPU; the failure is in the distributed
cuEquivariance kernels. That first package is not a validated artifact.

Direct checkpoint inference with `MACE_SOYO_CUEQ_METHOD=naive` selects
cuEquivariance's PyTorch evaluator on CUDA, keeps convolution shape metadata,
and derives new polynomial constant buffers without treating them as missing
learned checkpoint entries. Original checkpoint entries still load strictly.
Checkpoint and installed source files are unchanged. This mode is separate
from the accelerated kernel path, which does not run on this hardware.
An exploratory AOTI export of the portable graph was stopped after direct
inference passed; no portable compiled artifact is claimed as validated. A direct GPU evaluation with the upstream
`TensorEFS` wrapper and all weights loaded strictly completed for the bundled
Si structure: total energy `-10.8254423141 eV`, x forces `+0.39076358` and
`-0.39076358 eV/Å`, xx stress `-0.0096848393 eV/Å³`.
Exact outputs are saved in
[`eager-gpu.json`](../../examples/mace_soyo_si/reference/eager-gpu.json).
See the [environment recipe](backend-environments.md#mace-soyo).

The dedicated direct-checkpoint ASE runtime completed **5 real integration
tests** with `MACE_SOYO_CUEQ_METHOD=naive`: finite E/F/S and energy numerical
derivatives, CLI static calculation, ionic relaxation, two MD steps, and
resident reuse for two jobs. A request for a different head was rejected.
The model remained on `cuda:0` and the resident did not rebuild it.

```bash
OMP_NUM_THREADS=2 MPLCONFIGDIR=/tmp/vpmdk-soyo-mpl \
  VPMDK_TEST_REAL_PYMATGEN=1 \
  VPMDK_MACE_SOYO_MODEL=/home/nei/.cache/vpmdk/mace-soyo/mace_soyo_mptraj.pth \
  VPMDK_MACE_SOYO_CUEQ_METHOD=naive VPMDK_MACE_SOYO_DEVICE=cuda:0 \
  /tmp/vpmdk-mace-soyo-env/bin/python -m pytest -q tests/integration/test_mace_soyo.py
OMP_NUM_THREADS=2 /tmp/vpmdk-mace-soyo-env/bin/vpmdk --dir examples/mace_soyo_si
```

A separate final direct-calculator evaluation gave energy `-10.8254420757 eV`,
x force on atom 2 `-0.390763313 eV/Å` and xx stress `-0.00968484394 eV/Å³`.
Central differences gave `-0.390768051 eV/Å` with displacement `0.001 Å`, and
`-0.00968547979 eV/Å³` with uniaxial strain `0.003`. Test tolerances are
`0.002 eV/Å` and `0.00005 eV/Å³`, respectively. These are checks of the
implementation and do not establish accuracy against DFT.

The bundled relaxation completed in three evaluations, lowering total energy
from `-10.82544231` to `-10.83130479 eV`. Actual output snapshots and the final
calculator values, package versions, device, head and checksum are preserved
in [`examples/mace_soyo_si/reference/`](../../examples/mace_soyo_si/reference/README.md).
Existing NumPy/ASE and TorchScript deprecation warnings were emitted. One
upstream GPU-information message uses a fallback A6000 description when NVML
fails; actual device identity was checked with PyTorch and `nvidia-smi` and
is NVIDIA TITAN V.

Final checks: `CUDA_VISIBLE_DEVICES='' pytest -q -m 'not integration'`
passed **1286 tests**, with one skipped and 43 integration tests deselected.
The final targeted backend tests passed **322 tests** after verifying resident
head/method inference and rejection of changed settings. Wheel and source
builds passed with `python -m build`; `git diff --check` passed. The ordinary
MACE builder was left unchanged.

## 2026-11-06 ALIGNN-FF Validation

`MLP=ALIGNN` was validated with the official
[atomgptlab/alignn](https://github.com/atomgptlab/alignn) source at commit
`45097bbf48b3ff9528519f1c7379f05c3bd88afe` (package `2026.11.2`).
The actual runtime was `/tmp/vpmdk-alignn-env`, created using
`python -m venv --system-site-packages` from the local `ase_env`, followed by
installation of the upstream source and editable VPMDK. A clean-environment
recipe is in [backend environments](backend-environments.md#alignn-ff).

- Initial device: CPU, `OMP_NUM_THREADS=2`; GPU and legacy DGL follow-up
  results are recorded below.
- Python 3.13.5, PyTorch 2.8.0+cu128, ASE 3.27.0, pymatgen 2025.10.7,
  NumPy 2.4.2, jarvis-tools 2026.6.12.
- Model: `alignn_ff_mptrj`, downloaded from the upstream registry URL
  <https://ndownloader.figshare.com/files/70840341> (approximately 19.4 MB).
- Local directory: `/home/nei/.cache/atomgptlab/alignn_ff/alignn_ff_mptrj`.
- `best_model.pt` SHA-256:
  `666f9dc63d3670fc9250c8225061b603d34f06a8a2e803a80eacda14b8a998f7`.
- `config.json` SHA-256:
  `5f157499506a5f97ad14ef49bfde605bd31260d179945e0d91f4882c9c06f259`.
- Configuration: `alignn_atomwise_pure`, `neighbor_strategy=pure_torch`,
  cutoff 5 Å, three-body cutoff 3.5 Å, float32. No DGL installed in this venv.

For diamond Si (`a=5.43 Å`, two atoms, second atom displaced by +0.03 Å
along Cartesian x), the single-point energy was `-10.781064987182617 eV`.
The x forces were `+0.271683574` and `-0.271683574 eV/Å`, and the ASE xx
stress was `-0.0119260363 eV/Å³`. Central energy differences check the force
using a `0.001 Å` displacement (absolute tolerance `0.002 eV/Å`) and the
stress using `0.003` uniaxial strain (absolute tolerance `0.00005 eV/Å³`).
The latter gives `-0.0119212014 eV/Å³`; smaller strain differences showed
float32 cancellation, so the stress check uses the larger strain.

Real integration tests: **5 passed**. They cover finite energy/forces/stress,
the two gradient comparisons, CLI static calculation (`IBRION=-1`), ionic
relaxation (`IBRION=2`), two MD steps (`IBRION=0`), and two resident-server
jobs sharing a calculator without rebuilding it. The CLI checks verify
nonempty `OUTCAR`, `OSZICAR`, `CONTCAR`, and `vasprun.xml` plus finite final
coordinates; MD also checks `XDATCAR`.

```bash
OMP_NUM_THREADS=2 MPLCONFIGDIR=/tmp/vpmdk-alignn-mpl \
  VPMDK_TEST_REAL_PYMATGEN=1 \
  VPMDK_ALIGNN_MODEL=/home/nei/.cache/atomgptlab/alignn_ff/alignn_ff_mptrj \
  /tmp/vpmdk-alignn-env/bin/python -m pytest -q tests/integration/test_alignn.py
OMP_NUM_THREADS=2 /tmp/vpmdk-alignn-env/bin/vpmdk --dir examples/alignn_si
```

Additional verification:

```bash
CUDA_VISIBLE_DEVICES='' pytest -q -m 'not integration'
CUDA_VISIBLE_DEVICES='' pytest -q tests/test_alignn.py tests/test_backends.py
python -m build
git diff --check
```

The full non-integration run passed **1266 tests**, with one skipped and 37
integration tests deselected. After adding the final discovery/default-model
checks, the targeted backend suite passed **318 tests**. The wheel and sdist
built successfully, and the wheel contains the ALIGNN adapter. The initial
full run with CUDA visible had two failures in existing Matlantis tests whose
expected metadata fixes `DEVICE=cpu`; the CPU-configured full run above passed.
No Matlantis implementation or test expectations were changed. Existing
warnings from optional packages and server shutdown tests remain.

The bundled example completed with energy decreasing from `-10.78106499` to
`-10.78510571 eV`. Actual relaxation outputs are saved in
[`examples/alignn_si/reference/`](../../examples/alignn_si/reference/README.md).
These are workflow checks, not accuracy benchmarks against DFT.

### GPU and Legacy DGL Follow-up

Hardware: NVIDIA TITAN V, 12 GiB, logical `cuda:0`. The same pinned ALIGNN
source and downloaded `alignn_ff_mptrj` checkpoint above were used. The
integration fixture checks the actual model parameter device, so an explicit
CUDA request cannot pass by silently falling back to CPU.

| Model | Graph runtime | CPU result | CUDA result |
|-------|---------------|------------|-------------|
| `alignn_ff_mptrj` | pure PyTorch, torch 2.8.0+cu128 | 5 passed, 1 GPU-only test skipped | 6 passed |
| `v12.2.2024_dft_3d_307k` | DGL 2.1.0+cu121, torch 2.2.1+cu121 | 5 passed, 1 GPU-only test skipped | 6 passed |

Both GPU runs passed the energy/force/stress finite-difference checks, CLI
single point, relaxation, MD and resident reuse, plus an additional comparison
against CPU. Comparison absolute tolerances are `1e-5 eV` for energy,
`5e-5 eV/Å` for forces, and `1e-5 eV/Å³` for stress. The resident test now
compares numerical energy within `1e-5 eV` and final coordinates within
`1e-10 Å`; byte equality failed for a `1e-6 eV` GPU scatter-reduction
rounding difference between two otherwise successful jobs.

The actual legacy runtime was `/tmp/vpmdk-alignn-dgl-env`: Python 3.10.19,
NumPy 1.26.4, torchdata 0.7.1, ASE 3.25.0, pymatgen 2025.10.7,
jarvis-tools 2026.6.12, ALIGNN 2026.11.2. It inherited the installed
`codex_matgl_dgl` packages. DGL GraphBolt initially could not find
`libnvrtc.so.12`; adding the base environment's
`site-packages/nvidia/cuda_nvrtc/lib` and `nvidia/cusparse/lib` to
`LD_LIBRARY_PATH` fixed it. A DGL graph was explicitly moved to `cuda:0`
before running the legacy model checks.

Legacy checkpoint: `v12.2.2024_dft_3d_307k`,
[official Figshare download](https://ndownloader.figshare.com/files/50904240),
cached at `/home/nei/.cache/atomgptlab/alignn_ff/v12.2.2024_dft_3d_307k`.
Its configuration is `alignn_atomwise`, `radius_graph`, cutoff 4 Å,
`energy_mult_natoms=true`. SHA-256 values:

- `best_model.pt`: `f47f3d231605d58c21538ee5a3d1fe5c7f7711d9cd31deb6964965e23f84b269`
- `config.json`: `ac194348870c6d584ab95da25c2e51b27856625326398de7d6dfe825295164cf`

For the displaced two-atom Si structure, measured maximum CPU/GPU differences
were:

| Model | Energy (eV) | Force component (eV/Å) | Stress component (eV/Å³) |
|-------|-------------|------------------------|--------------------------|
| `alignn_ff_mptrj` | 8.59e-6 | 8.05e-6 | 4.65e-7 |
| `v12.2.2024_dft_3d_307k` | 1.44e-6 | 2.24e-7 | 4.47e-9 |

The legacy energy was `-7.9502038956 eV` on CPU and `-7.9502053261 eV` on
CUDA. Exact per-device values, numerical derivatives, package versions,
checksums, and device placement are saved in
[`pure-cpu-gpu.json`](../../examples/alignn_si/reference/pure-cpu-gpu.json) and
[`dgl-cpu-gpu.json`](../../examples/alignn_si/reference/dgl-cpu-gpu.json).
These values validate the wrapper and runtime, not accuracy against DFT.

To reproduce, use the [DGL environment recipe](backend-environments.md#legacy-dgl-checkpoints-on-cpu-and-cuda)
and run the same test file with each model/device pair:

```bash
OMP_NUM_THREADS=2 VPMDK_TEST_REAL_PYMATGEN=1 \
  VPMDK_ALIGNN_MODEL=/absolute/path/to/model_directory \
  VPMDK_ALIGNN_DEVICE=cuda python -m pytest -q tests/integration/test_alignn.py
```

After the GPU/DGL follow-up changes, the full non-integration suite passed
**1270 tests**, with one skipped and 38 integration tests deselected, using
`CUDA_VISIBLE_DEVICES='' pytest -q -m 'not integration'`. The targeted backend
suite passed **319 tests**. `python -m build` produced both wheel and sdist,
and `git diff --check` passed.

`v5.27.2024` was also downloaded from the
[official registry artifact](https://ndownloader.figshare.com/files/47286127)
and tested. It loaded and ran the CLI/server workflows, but failed the force
finite-difference check: its original flags differentiate per-atom energy
while the ASE calculator reports total energy. For two Si atoms the raw
force was `0.262474 eV/Å` versus a total-energy numerical derivative of
`0.525951 eV/Å`. A trial gradient rescaling also failed the stricter gradient
checks; no rescaling is applied in the final adapter. VPMDK now rejects
configurations with `energy_mult_natoms=false`, `force_mult_natoms=false`,
and `include_pos_deriv=false` before calculator construction. A unit regression
checks the rejection and ensures the original config file is unchanged.
Do not treat successful loading alone as validation of this checkpoint.
Its cached `best_model.pt` SHA-256 is
`7a048781fa9b2af5af92a3ab7fc4aa271441221cda0dce550ebed4a42b7ff4be`, and
`config.json` is
`331b9fa8d6c4cba144f9618350674d549e869a1822fa5da6051c9d1b3e54d138`.

### ALIGNN One-Atom Force-Shape Regression

The upstream calculator squeezes a one-atom force array to `(3,)`, causing
VPMDK's strict force check and static output generation to fail. The ALIGNN
adapter now reshapes calculated forces to `(len(atoms), 3)` before ASE caches
or returns them. Numerical force values are unchanged; malformed force counts
still raise an error. Unit regressions cover one and multiple atoms, result
caching/recalculation, and invalid force counts.

With the same public `alignn_ff_mptrj` checkpoint, a one-atom FCC Cu cell
(`a=3.61 Å`) completed `IBRION=-1`, `NSW=0` on CPU and NVIDIA TITAN V.
`OUTCAR`, `OSZICAR`, `CONTCAR` and `vasprun.xml` were generated. The CPU
energy was `-4.09311676 eV`. The full ALIGNN integration file passed 6 tests
on CPU (one GPU-only comparison skipped) and 7 on GPU. The targeted unit/backend
suite passed 322 tests, and `git diff --check` passed.

The GPU run also exposed an existing float32 comparison flake on Si2. The
unwrapped upstream calculator reproduced a CPU/GPU energy difference up to
`1.049041748e-5 eV` across repeated evaluations. The energy comparison absolute
tolerance is now `2e-5 eV`; force/stress comparison tolerances are unchanged.
This numerical difference is independent of force-array normalization.

## 2026-10-26 BAM GPU Validation

- Hardware/environment: NVIDIA TITAN V (CUDA device 0), Python 3.13.5,
  PyTorch 2.8.0+cu128, ASE 3.29.0, pymatgen 2026.10.24. Separate virtual
  environments used e3nn 0.5.6 for the original BAM runtime and e3nn 0.6.0
  for the DeNS runtime. Optional cuEquivariance/OpenEquivariance kernels
  were not installed; inference used the upstream e3nn path on CUDA.
- Original runtime: BAM-torch commit
  `9329db2cec33715f89efa3f560d3e0f791b3f528`.
  [BAM-MP-core checkpoint](https://ndownloader.figshare.com/files/65859867):
  MD5 `0d7182820d24ab51a452d475c9e5001d`.
- DeNS runtime: BAM-torch `BAM_DeNS` commit
  `24ed0f042660a1e11ad5d91466b4275955df98c6`.
  [BAM-MP-DeNS checkpoint](https://ndownloader.figshare.com/files/70285206):
  MD5 `06f303110248a263f9ea2f4d2775af93`.
  Both hashes match the Matbench model notes.
- All three combinations passed: original checkpoint/original runtime,
  DeNS checkpoint/DeNS runtime, and original checkpoint/DeNS runtime.
  Calculator device and model parameters were checked to reside on CUDA,
  so a CPU fallback would fail validation.
- The VPMDK BAM builder matched direct `RACECalculator` energy, forces, and
  stress (`rtol=1e-5`, `atol=1e-6`). For the original checkpoint, changing
  to the DeNS runtime gave the same initial energy, with maximum force and
  stress differences of `1.72e-7` eV/Å and `1.31e-8` eV/Å³.
- Each combination completed a single point, three ionic relaxation steps,
  and three NVE MD steps through `run_workdir`, reusing its GPU calculator.
  Test structure: primitive diamond Si2, `a=5.43 Å`, second atom displaced
  by `(0.025, -0.015, 0.010) Å`. Relaxation used `IBRION=2`, `ISIF=2`,
  `POTIM=0.1`, `EDIFFG=-0.0001`, `NSW=3`; MD used `MDALGO=0`,
  `POTIM=0.1`, `TEBEG=TEEND=300`, `NSW=3`.
  These are short execution checks, not converged-relaxation or accuracy claims.
- OUTCAR, OSZICAR, CONTCAR and vasprun.xml were produced for each workflow,
  plus XDATCAR for MD. ASE read back finite energy, forces and stress from
  vasprun.xml; MD contained three frames.
- Independent `python -m vpmdk --dir ...` single-point runs also passed for
  both models, loading `MLP=BAM`, the checkpoint path, and `DEVICE=cuda` from
  BCAR without an injected calculator. GPU property integration tests passed
  for both checkpoints on the DeNS runtime (2 tests) and for the original
  checkpoint on the original runtime (1 test), with no skips.

| Checkpoint / runtime | Initial energy (eV, Si2) | Maximum force before → after relaxation (eV/Å) |
|---|---:|---:|
| BAM-MP-core / original | -10.65976238 | 0.278663 → 0.005861 |
| BAM-MP-DeNS / DeNS | -10.80325603 | 0.167946 → 0.003672 |
| BAM-MP-core / DeNS | -10.65976238 | 0.278663 → 0.005860 |

The optional checkpoint tests can reproduce the GPU property checks in a
DeNS-capable environment with both checkpoint paths set:

```bash
VPMDK_TEST_REAL_PYMATGEN=1 VPMDK_BAM_DEVICE=cuda \
  VPMDK_BAM_MODEL=/path/to/BAM-MP-core.pkl \
  VPMDK_BAM_DENS_MODEL=/path/to/BAM-MP-DeNS-model.pkl \
  python -m pytest -q tests/integration/test_bam.py -rs
```

## 2026-09-19 Prophet Validation

- Environment: Python 3.13.5, ASE 3.29.0, torch 2.8.0+cu128, e3nn 0.6.0,
  torch-geometric 2.8.0.post1, and official `prophet-mlip` 0.1.0 at commit
  `869d567e2abfa1dacd6d0b326bc99a8019fa68b2`.
- Checkpoint: official `prophet-oame-mbd.pt`; `PROPHET_USE_KERNEL=0` selected
  the portable e3nn path.
- The Si2 smoke structure completed single-point calculations and one-step NVE
  MD through the VASP-style work-directory path on both CPU and an NVIDIA
  TITAN V. The MD runs produced nonempty `CONTCAR`, `OUTCAR`, and `XDATCAR`.
- Single-point energy was -10.8567724228 eV on CPU and -10.8567733765 eV on
  CUDA; maximum absolute force was 3.230198e-07 and 3.385358e-07 eV/A,
  respectively. Both paths returned stress.

## 2026-09-19 Matlantis Validation

- Environment: Matlantis VM, Python 3.13.8, ASE 3.29.0, and
  `pfp-api-client 2.5.1`.
- PFP service metadata exposed stable `v9.0.0` plus experimental v10 alpha
  deployments; VPMDK pins stable `v9.0.0` with `R2SCAN` for reproducibility.
- A legacy-selection smoke test completed a Si single point with PFP `v7.0.0`
  and no explicit calc mode (`-9.101191427963512 eV`), confirming that VPMDK
  leaves nondefault model versions on their compatible upstream mode default.
- A boundary-crossing one-step NVE smoke test counted two official PFP
  calculator evaluations (the required initial and final forces) while
  retaining wrapped final coordinates; reading the final energy after
  `Atoms.wrap()` no longer issues a third, physically duplicate request. Cache
  reuse is limited to the official Matlantis calculator (or an explicit
  periodic-invariance opt-in); an absolute-coordinate custom ASE calculator
  retains the safe post-wrap recalculation.
- The Si2 smoke structure completed a one-step NVE MD run and a two-step-limit
  force relaxation through the VASP-style work-directory path.
- A resident Matlantis server completed two inherited-configuration requests
  with one Estimator/Calculator pair; a separate one-shot calculation agreed
  within `1e-5 eV`. Independent PFP requests showed sub-micro-eV variation, so
  the integration test uses numerical tolerances rather than byte equality.
- Six slightly different Si single points took 0.482 s when constructing a new
  Estimator each time and 0.293 s with one serial resident including its
  one-time construction (1.64x for this small host-specific sample).
- Run the three opt-in real-service tests with
  `VPMDK_TEST_MATLANTIS=1 pytest -m integration -k matlantis` inside Matlantis.

## 2026-08-29 EquFlash / EquFlashV2 Validation

The official EquFlash OAM and EquFlashV2 OAM checkpoints were exercised through
the same VPMDK `EQUFLASH` backend on an NVIDIA TITAN V using Python 3.12.14,
torch 2.9.1+cu126, equflash 0.0.2, and fairchem-core 1.3.0.

- EquFlash V1: `EquFlash`, 28,741,188 parameters; five-step Si2 MD passed;
  single-point energy -10.8378000259 eV and maximum force
  2.9237764210e-08 eV/A
- EquFlashV2: `EquFlashV2`, 44,890,492 parameters; five-step Si2 MD passed;
  single-point energy -10.8424844742 eV and maximum force
  6.5944732341e-08 eV/A
- Both models ran on `cuda:0` and returned full 3x3 stress tensors.
- EquFlashV2 also completed the five-step MD test on CPU. The V1 OAM model
  loaded in CPU mode but could not evaluate because equflash 0.0.2 dispatches
  `cuequivariance_ops::segmented_transpose` to a CUDA-only kernel.

## Server-Mode Validation Summary

The v0.5.0 server work was validated with mocked lifecycle tests, real CHGNet
CPU calculations, MACE CUDA calculations, and a GRACE CPU environment. Checks
covered startup readiness, lightweight client imports, FIFO execution,
one-shot output compatibility, status during work, graceful shutdown, socket
cleanup, and resident-model reuse.

The lightweight client path reduced an absent-socket invocation from about
4.1 seconds to about 0.02 seconds on the validation host. These measurements
and the recorded GPU memory values were host-specific observations, not package
performance guarantees. Current behavior is enforced by the unit and optional
integration tests rather than by preserving individual development-run counts.

## 2026-06-03 EquiformerV3 Validation

- Structure: ASE `bulk("Si", "diamond", a=5.43)` (`Si2`)
- API path: `BackendConfig(mlp="EQUIFORMER_V3")` -> `single_point`
- Environment: `codex_fairchem_v1`, torch 2.4.1+cu121, fairchem-core 1.10.0,
  official `atomicarchitects/equiformer_v3` source at commit
  `a7300c58df683dc99cb48027d5bfd4c887486c48`
- Checkpoint: local `omat24-mptrj-salex_gradient.pt`
- CPU run: `CUDA_VISIBLE_DEVICES=""`, `DEVICE=cpu`
- CUDA run: `DEVICE=cuda` on NVIDIA TITAN V

| Backend | CPU energy (eV) | CPU max force (eV/A) | CUDA energy (eV) | CUDA max force (eV/A) |
| --- | ---: | ---: | ---: | ---: |
| `EQUIFORMER_V3` | -10.8360443115 | 1.414973e-06 | -10.8360481262 | 1.322957e-06 |

Stress was returned as a full 3x3 tensor in both runs.

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
| `MATGL` | Manual real-backend single point plus MatGL 2.0 API regression | torch 2.2.1+cu121, matgl 1.3.0, dgl 2.1.0+cu121; MatGL 2.0.6 API audit | local `M3GNet-MP-2021.2.8-PES` model directory | passed | passed | required `DGLBACKEND=pytorch` and CUDA NVRTC library path; regression covers the 2.x `PESCalculator`-only import and required default-potential load |
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
| `EQUFLASH` | Official-runtime real-backend single points and five-step MD integration tests | Python 3.12.14, torch 2.9.1+cu126, equflash 0.0.2, fairchem-core 1.3.0 | official EquFlash V1 and EquFlashV2 OAM checkpoints | V2 passed; V1 CUDA-only | passed | the same `GGNN.common.calculator.UCalculator` path selected `EquFlash` for V1 and `EquFlashV2` for V2 on `cuda:0`; V1's validated runtime has no CPU kernel for segmented transpose |
| `FAIRCHEM_V1` | Manual real-backend single point | torch 2.4.1+cu121, fairchem-core 1.10.0 | local `schnet_200k.pt` | passed | passed | stress unavailable for this calculator; forces were zero for this Si2 smoke case |
| `FAIRCHEM_V2` | Manual real-backend single point | torch 2.8.0+cu128, fairchem-core 2.13.0 | `uma-s-1`, `FAIRCHEM_TASK=omat` | passed | passed | VPMDK default uses `uma-s-1p1`, which is present in fairchem-core 2.13.0 and current 2.x registries |
| `FAIRCHEM` | Manual real-backend single point | same as `FAIRCHEM_V2` | `uma-s-1`, `FAIRCHEM_TASK=omat` | passed | passed | alias path covered separately |
| `ESEN` | Manual real-backend single point | same as `FAIRCHEM_V2` | `uma-s-1`, `FAIRCHEM_TASK=omat` | passed | passed | alias path covered separately; OC25 ESEN checkpoints require gated HF access |
| `GRACE` | Real single point, MD integration, and resident-server smoke | tensorflow 2.19.1 / 2.20.0, tensorpotential 0.5.7 / 0.6.0 | local `GRACE-2L-MP-r6`; named `GRACE-FS-OAM` | passed | passed | 0.6.0 registry-fallback and server identity passed on CPU; earlier CUDA run used `XLA_FLAGS=--xla_gpu_cuda_data_dir=...` |
| `DEEPMD` | Manual real-backend single point plus resident smoke | torch 2.8.0+cu128 / deepmd-kit 3.1.2; resident retest with deepmd-kit 3.2.0b0 | local `DPA-3.1-3M.pt`, `DEEPMD_HEAD=Omat24` | passed | passed | required `LD_LIBRARY_PATH` to include the DeepMD environment library directory; resident CPU smoke passed with the explicit model type map |
| `MATLANTIS` | Real single point, relaxation, MD, and resident-server smoke | Python 3.13.8, ASE 3.29.0, pfp-api-client 2.5.1 | PFP v9.0.0 / R2SCAN | passed | not applicable | validated in a Matlantis VM on 2026-09-19; service inference is remote |

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

`tests/test_consumer_roundtrip.py` is the consumer round-trip net: it runs the
full `run_workdir` pipeline per run mode in a subprocess with
`VPMDK_TEST_REAL_PYMATGEN=1` (the rest of the suite runs against lightweight
pymatgen stubs from `tests/conftest.py`) and reads every artifact back through
the real `ase.io` and pymatgen readers, asserting cross-artifact consistency.
When adding or changing an output writer, extend this harness rather than
asserting against the writer's own formatting — several historical defects were
only visible to the reader on the other side.

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
