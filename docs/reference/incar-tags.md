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
- `MDALGO`
- `SMASS`
- `ANDERSEN_PROB`
- `LANGEVIN_GAMMA`
- `CSVR_PERIOD`
- `NHC_NCHAINS`
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

## Tag Semantics

| Tag | Meaning | Default / Behavior |
|-----|---------|--------------------|
| `NSW` | Ionic or MD step count | `0` |
| `IBRION` | Run mode selector | `<0` single point, `0` MD, `>0` relaxation |
| `EDIFFG` | Relaxation stop criterion | `<0` force threshold, `>0` energy threshold |
| `ISIF` | Cell/stress mode | normalized to VPMDK-supported modes |
| `PSTRESS` | External scalar pressure in kBar | only used when cell degrees of freedom are active |
| `TEBEG` | Initial MD temperature | `300.0` |
| `TEEND` | Final MD temperature | defaults to `TEBEG` |
| `POTIM` | MD time step in fs | `2.0` |
| `MDALGO` | MD integrator / thermostat selection | `0` |
| `SMASS` | Thermostat mass / fallback selector | can auto-promote `MDALGO` |
| `ANDERSEN_PROB` | Andersen collision probability | used only with `MDALGO=1` |
| `LANGEVIN_GAMMA` | Langevin friction in 1/ps | used only with `MDALGO=3` |
| `CSVR_PERIOD` | Bussi relaxation time in fs | used only with `MDALGO=5` |
| `NHC_NCHAINS` | Nose-Hoover chain length | used with `MDALGO=2` or `4` |
| `MAGMOM` | Initial magnetic moments | VASP-like parsing including `N*value` |
| `IMAGES` | NEB image count hint | also triggers NEB-like mode detection |
| `ICHAIN` | VTST chain method selector | only `0`/unset NEB is implemented |
| `IOPT` | VTST optimizer selector | maps selected values to ASE optimizers |
| `LCLIMB` | NEB climbing-image flag | truthy values enable climbing-image ASE NEB |
| `LNEBCELL` | VTST NEB cell-relaxation flag | recognized but not implemented; fixed-cell NEB is used |
| `SPRING` | NEB spring constant | negative VTST values are converted to positive ASE spring magnitudes |

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

If `MDALGO` is explicitly set, that value is used.

If `MDALGO=0` and `SMASS` is provided:

- `SMASS < 0` -> `MDALGO=3` (Langevin)
- `SMASS > 0` -> `MDALGO=2` (Nose-Hoover)

This mirrors the compatibility behavior already covered by regression tests.

## Thermostat Parameters

Recognized thermostat-only tags:

- `ANDERSEN_PROB`
- `LANGEVIN_GAMMA`
- `CSVR_PERIOD`
- `NHC_NCHAINS`

Invalid values are ignored with warnings rather than crashing the entire run.

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

With numbered image directories and `NSW > 0`, `IBRION != 0`, and `ICHAIN=0` or
unset, VPMDK runs a spring-coupled ASE NEB optimization. It writes VASP-like
outputs in each image directory and parent aggregate `OUTCAR`, `OSZICAR`, and
`vasprun.xml` files from the final band. `NSW <= 0`/`IBRION < 0` still runs
independent image single points, and `IBRION == 0` still runs independent image
MD for compatibility.

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
