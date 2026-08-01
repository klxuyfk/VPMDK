# INCAR Reference

## Supported Tags

VPMDK intentionally supports a focused subset of `INCAR` for main execution
settings. Other tags can still be consumed by supported auxiliary flows such as
CHGCAR grid selection or pseudo-SCF compatibility output, but tags outside the
documented supported/auxiliary sets are ignored with a warning.

Supported tags:

- `ISIF`
- `IBRION`
- `NSW`
- `EDIFFG`
- `PSTRESS`
- `TEBEG`
- `TEEND`
- `POTIM`
- `NFREE`
- `SYMPREC`
- `MDALGO`
- `SMASS`
- `ANDERSEN_PROB`
- `LANGEVIN_GAMMA`
- `CSVR_PERIOD`
- `NHC_NCHAINS`
- `NHC_PERIOD`
- `MAGMOM`
- `IMAGES`
- `ICHAIN`
- `IOPT`
- `LCLIMB`
- `LNEBCELL`
- `SPRING`

## Parsing Defaults

The parsed `IncarSettings` defaults are:

| Field | Default |
|-------|---------|
| `nsw` | `0` |
| `ibrion` | `-1` |
| `ediffg` | `-0.02` |
| `isif` | `2` except MD defaults to `0` before normalization |
| `stress_isif` | requested `ISIF` when valid, else normalized value |
| `pstress` | `None` |
| `tebeg` | `300.0` |
| `teend` | same as `TEBEG` |
| `potim` | `2.0` |
| `mdalgo` | `0` |
| `smass` | `None` |

### Number Formats

A numeric value that the INCAR reader would silently turn into a *different*
number is rejected with exit 1 rather than used:

- scientific notation for an integer tag (`NSW = 1e5` reads as `1`)
- a Fortran `D` exponent anywhere (`EDIFFG = -1.0D-03` reads as `-1.0`)
- a corrupted token whose leading digits still parse (`TEBEG = 5OO` — letter
  O — reads as `5`, `NSW = 0x10` reads as `0`)

Write those as plain digits or with an `E` exponent. The check reads the INCAR
text the same way the parser does, so it also covers VASP's compact styles:
several tags on one line separated by `;`, and values continued with a trailing
`\`.

## Tag Semantics

| Tag | Meaning | Default / Behavior |
|-----|---------|--------------------|
| `NSW` | Ionic or MD step count | `0` |
| `IBRION` | Run mode selector | `<0` single point, `0` MD, `5`/`6` finite-difference force constants, `7`/`8` force-constants compatibility output, other `>0` relaxation (`44` is rejected as unsupported); when omitted with `NSW>1`, VPMDK runs a single point and warns — real VASP would default to `0` (MD) |
| `EDIFFG` | Relaxation stop criterion | `<0` force threshold, `>0` energy threshold |
| `ISIF` | Cell/stress mode | normalized to VPMDK-supported modes |
| `PSTRESS` | External scalar pressure in kBar | only used when cell degrees of freedom are active; following VASP's Pulay-stress definition, all reported stress output subtracts `PSTRESS` from the diagonal, `external pressure` reads ~0 for a cell equilibrated at the target, and the OUTCAR `Pullay stress` field echoes the applied value |
| `TEBEG` | Initial MD temperature | `300.0` |
| `TEEND` | Final MD temperature | defaults to `TEBEG` |
| `POTIM` | MD time step in fs, or finite-difference displacement in Angstrom for `IBRION=5`/`6` | `2.0`, except `0.015` for `IBRION=5`/`6` when omitted. As a displacement it must be at least `1e-6` Angstrom: smaller values underflow the double-precision positions and would produce an all-zero or noise-dominated Hessian |
| `NFREE` | finite-difference displacement stencil for `IBRION=5`/`6` | omitted values use `2`; supported values are `1`, `2`, and `4` |
| `SYMPREC` | symmetry tolerance for `IBRION=6`/`8` atom-orbit reduction | `1e-5` |
| `MDALGO` | MD integrator / thermostat selection | `0`. `MDALGO=2`/`4` (Nose-Hoover chain) rejects POSCAR selective dynamics: ASE's chain integrator ignores constraints in its internal momenta, which distorts the sampled temperature far below `TEBEG`; use `MDALGO=1`, `3`, or `5` for constrained MD |
| `SMASS` | Nose-Hoover damping time in fs / fallback selector | can auto-promote `MDALGO`; read as a damping TIME in fs, not as VASP's Nose mass, so the numbers are not interchangeable. When omitted the damping time is `100 * POTIM` fs; a value below `10 * POTIM` is reported as strong coupling |
| `ANDERSEN_PROB` | Andersen collision probability | used only with `MDALGO=1`. Note: Andersen freezes the center of mass while the reported temperature divides by all 3N DOF, so OSZICAR/stdout read ≈(3N−3)/3N of `TEBEG` (VASP reports over 3N−3); the sampled ensemble itself is at `TEBEG`; when omitted, VPMDK defaults to **0.1** collisions per atom per step and warns — real VASP defaults to 0 (collision-free NVE); write `ANDERSEN_PROB = 0.0` for VASP's default behavior |
| `LANGEVIN_GAMMA` | Langevin friction in 1/ps | used only with `MDALGO=3` |
| `CSVR_PERIOD` | Bussi relaxation time in fs | used only with `MDALGO=5`; when omitted the default is `100 * POTIM` fs, so writing `CSVR_PERIOD = 100` is NOT the same as omitting it unless `POTIM = 1`. Unlike `NHC_PERIOD`, this tag is read in fs, not in MD steps. Must be positive |
| `NHC_NCHAINS` | Nose-Hoover chain length | used with `MDALGO=2` or `4`; must be between 1 and 100 (chains beyond ~10 links have no physical effect, and the integrator's cost grows linearly with the length) |
| `NHC_PERIOD` | Nose-Hoover chain damping period in MD steps | VPMDK uses `NHC_PERIOD * POTIM` as the ASE damping time |
| `MAGMOM` | Initial magnetic moments | VASP-like parsing including `N*value` (repeat expansion is bounded at 1e6 values per tag, counting the product of nested `N*M*value` factors); **inert for results** — attached as ASE initial moments, but no supported backend reads initial moments (they only predict moments as outputs), so FM/AFM orderings produce identical energies; a warning discloses this |
| `IMAGES` | NEB image count hint | also triggers NEB-like mode detection |
| `ICHAIN` | VTST chain method selector | only `0`/unset NEB is implemented |
| `IOPT` | VTST optimizer selector | maps selected values to ASE optimizers |
| `LCLIMB` | NEB climbing-image flag | truthy values enable climbing-image ASE NEB; when omitted, VPMDK runs **plain NEB** and warns — VTST's documented default is `.TRUE.`, so write `LCLIMB = .TRUE.` for VTST's default behavior (a plain band underestimates the barrier) |
| `LNEBCELL` | VTST NEB cell-relaxation flag | recognized but not implemented; fixed-cell NEB is used |
| `SPRING` | NEB spring constant | negative VTST values are converted to positive ASE spring magnitudes |

For the `IBRION=5`/`6`/`7`/`8` force-constants compatibility path, see
[VASP Force-Constants Compatibility](../development/force-constants.md).

## ISIF Mapping

VPMDK preserves the higher-order VASP-style `ISIF` modes that it knows how to
map into ASE filters.

| Requested `ISIF` | Effective behavior |
|------------------|--------------------|
| `0`, `1`, `2` | fixed-cell ionic relaxation behavior (`2`) |
| `3` | ions + full cell |
| `4` | ions + shape, constant volume |
| `5` | cell shape only, constant volume, ions frozen |
| `6` | strain-only cell relaxation |
| `7` | isotropic cell change, ions frozen |
| `8` | ions + isotropic volume |
| unsupported | warning, then fallback to `2` behavior |

Stress output semantics are slightly different from relaxation behavior:

- `ISIF <= 0`: omit stress from compatibility outputs
- `ISIF = 1`: write trace-only pressure-like stress
- `ISIF >= 2`: write full stress tensor

## EDIFFG Semantics

`IncarSettings` exposes two derived views:

- `energy_tolerance`: `EDIFFG` when it is positive, else `None`
- `force_limit`: force threshold used by ASE relaxation

Rules:

- `EDIFFG > 0`: use energy convergence, and set `force_limit` negative to keep
  ASE from terminating on force first
- `EDIFFG < 0`: use `abs(EDIFFG)` as the force threshold
- `EDIFFG = 0`: fallback force threshold `0.05`

## MDALGO and SMASS

If `MDALGO` is explicitly set, that value is used. The implemented algorithms are
`0` (velocity-Verlet NVE), `1` (Andersen), `2`/`4` (Nose-Hoover chain), `3`
(Langevin) and `5` (CSVR). Any other value warns and runs `MDALGO=0`; the
fallback is also what `OUTCAR` and `vasprun.xml` then report, so the recorded
ensemble matches the trajectory.

If `MDALGO=0` and `SMASS` is provided:

- `SMASS < 0` -> `MDALGO=3` (Langevin)
- `SMASS > 0` -> `MDALGO=2` (Nose-Hoover)

`SMASS` does not promote an out-of-range `MDALGO`: an explicit unsupported value
runs NVE, as the warning says.

This mirrors the compatibility behavior already covered by regression tests.

## Thermostat Parameters

Recognized thermostat-only tags:

- `ANDERSEN_PROB`
- `LANGEVIN_GAMMA`
- `CSVR_PERIOD`
- `NHC_NCHAINS`
- `NHC_PERIOD`

Unparseable values are ignored with warnings rather than crashing the entire run,
and values that a thermostat merely treats as a limiting case are passed through
(`ANDERSEN_PROB` outside `[0, 1]` behaves as the nearest legal bound).

A value that makes the requested thermostat mathematically undefined is an input
error (exit 1) instead, because there is no coupling the run could fall back to:
`NHC_PERIOD <= 0` and `CSVR_PERIOD <= 0` are rejected before the run starts. Both
previously failed part-way through the calculation and were reported as retryable
calculation failures (exit 2).

## MAGMOM

`MAGMOM` is applied before execution when possible.

Accepted forms:

- scalar: `1.0`
- explicit list: `1.0 0.0 0.0`
- compressed list: `2*1.0 4*0.0`

Application rules:

- if the count matches the number of atoms, use it directly
- if the count matches species blocks in `POSCAR`, expand by species count
- otherwise print a warning and leave moments unset

## NEB Detection

VPMDK considers an `INCAR` NEB-like when any of the following is true:

- `IMAGES` parses to a positive integer
- `SPRING` is present
- `LCLIMB` is truthy (`T`, `TRUE`, `1`, `YES`, `Y`)

That detection only controls the CLI compatibility workflow; it does not create
NEB outputs unless numbered image directories are present.

With numbered image directories and `NSW > 0`, `IBRION > 0`, and `ICHAIN=0` or
unset, VPMDK runs a spring-coupled ASE NEB optimization. It writes VASP-like
outputs in each image directory and parent aggregate `OUTCAR`, `OSZICAR`, and
`vasprun.xml` files from the final band. `NSW <= 0`/`IBRION < 0` still runs
independent image single points, and `IBRION == 0` still runs independent image
MD for compatibility. ASE NEB optimization requires at least three numbered
directories: initial, one moving image, and final.

## Pseudo-SCF Compatibility Tags

The following are not part of the main run physics, but can be echoed into
compatibility outputs when pseudo-SCF mode is enabled:

- `NELM`
- `NELMIN`
- `NELMDL`
- `EDIFF`

If pseudo-SCF output is disabled, VPMDK warns that they do not affect the run.

## CHGCAR Grid Tags

These `INCAR` tags are only relevant for charge-density grid construction:

- `PREC`
- `ENCUT`
- `NGX`
- `NGY`
- `NGZ`
- `NGXF`
- `NGYF`
- `NGZF`

If `WRITE_CHGCAR` is not enabled, the CLI warns that those tags are ignored for
the current run.

Grids are validated up front and capped at 100000 points per axis and 1e9
points in total. A combination that exceeds either bound — whether derived
from `ENCUT` and the cell or requested explicitly via `NGX*`/`NG*F` — is
rejected as an input error (exit 1) before the calculation starts, rather than
failing late (or, for extreme `ENCUT` values, never returning) during the
CHGCAR write.
