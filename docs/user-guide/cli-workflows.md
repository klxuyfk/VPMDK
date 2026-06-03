# CLI Workflows

## CLI Contract

The CLI is a VASP-style compatibility layer built on top of the pure execution
API. It reads files from a selected directory, constructs an ASE calculator,
runs one workflow, and writes VASP-like outputs back into that same directory.

Main entry point:

```bash
vpmdk
```

Use `--dir PATH` only when you want to target a calculation directory other
than the current one.

## Input Files

The CLI looks for:

- `POSCAR`
- `INCAR`
- `POTCAR`
- `KPOINTS`
- `BCAR`

Behavior:

- `POSCAR` is required except in NEB mode where numbered image directories may
  provide the structures.
- `INCAR` is optional; missing values fall back to internal defaults.
- `BCAR` is optional; the backend defaults to `CHGNET`.
- `POTCAR` is optional. It helps species labeling and also affects some
  VASP-compatibility output details such as the `TITEL` lines written into
  `OUTCAR`.
- `KPOINTS`, `WAVECAR`, and existing `CHGCAR` are ignored for the actual ML
  calculation. `KPOINTS` is used only for limited compatibility metadata; the
  generated `vasprun.xml` still writes a simplified Gamma-only k-point section.

## Parsing Rules

`BCAR` parsing:

- `key=value` format
- keys are normalized to uppercase
- `#` and `!` start comments
- `NNP` is accepted as a legacy alias for `MLP`

`INCAR` parsing:

- the file is read through `pymatgen.io.vasp.Incar`
- unsupported tags are ignored with warnings
- a small subset of NEB and pseudo-SCF tags are recognized specially

## Mode Selection

The CLI chooses one execution path:

1. Detect NEB-like input from `IMAGES`, `SPRING`, or truthy `LCLIMB`.
2. If numbered image directories exist, run NEB mode.
3. Otherwise require `POSCAR`.
4. Parse `INCAR` into settings.
5. Select one of:
   - spring-coupled ASE NEB if numbered image directories exist, `NSW > 0`,
     and `IBRION > 0`
   - independent NEB image single points if `NSW <= 0` or `IBRION < 0`
   - independent NEB image MD if numbered image directories exist and
     `IBRION == 0`
   - force-constant output if `IBRION=5`, `6`, `7`, or `8`
   - otherwise, single point if `NSW <= 0` or `IBRION < 0`
   - molecular dynamics if `IBRION == 0`
   - relaxation otherwise

## Single-Point Runs

Single-point mode evaluates the structure once and writes:

- `CONTCAR`
- `OUTCAR`
- `OSZICAR`
- `vasprun.xml`

Stress output still depends on `ISIF` semantics:

- `ISIF <= 0`: no stress block
- `ISIF = 1`: trace-only pressure-style stress output
- `ISIF >= 2`: full stress tensor block

## Force Constants

`IBRION=5`, `IBRION=6`, `IBRION=7`, or `IBRION=8` writes a VASP-like `dynmat`
Hessian block into `vasprun.xml`. The values are generated from central finite
differences of MLP forces and mass-normalized in the format expected by
phonopy's VASP interface, so `phonopy --fc vasprun.xml` can create
`FORCE_CONSTANTS`.

For `IBRION=5` and `IBRION=6`, `POTIM` in `INCAR` controls the displacement in
Angstrom. `NFREE=1`, `NFREE=2`, and `NFREE=4` are supported; omitted `NFREE`
uses `2`. For `IBRION=7` and `IBRION=8`, `FORCE_CONSTANTS_DISPLACEMENT` in
`BCAR` controls the numerical displacement; if omitted, VPMDK uses `0.01`.

`IBRION=6` and `IBRION=8` reduce atom displacements using ASE/spglib symmetry
operations and reconstruct the remaining force-constant columns. `IBRION=7`
and `IBRION=8` also print a warning because VPMDK writes finite-difference
compatibility data rather than running electronic DFPT.

Implementation details, formulas, and compatibility limits are documented in
[VASP Force-Constants Compatibility](../development/force-constants.md).

## Relaxations

Relaxation mode uses ASE `BFGS` under the hood, with VASP-like `ISIF` semantics
mapped onto ASE filters:

- `ISIF=2`: ions only
- `ISIF=3`: ions + full cell
- `ISIF=4`: ions + shape at constant volume
- `ISIF=5`: cell-only shape at constant volume, ions frozen
- `ISIF=6`: cell only via `StrainFilter`
- `ISIF=7`: isotropic cell changes, ions frozen
- `ISIF=8`: ions + isotropic volume changes

`EDIFFG` keeps VASP sign semantics:

- `EDIFFG < 0`: force convergence threshold `abs(EDIFFG)` in eV/Ang
- `EDIFFG > 0`: energy convergence threshold `|delta E| <= EDIFFG`
- `EDIFFG = 0`: fallback force threshold `0.05`

Optional relaxation outputs:

- `WRITE_ENERGY_CSV=1` writes `energy.csv`
- `WRITE_PSEUDO_SCF=1` adds pseudo electronic-step blocks to compatibility files

## Molecular Dynamics

MD mode uses ASE molecular-dynamics drivers, selected by `MDALGO`:

- `0`: velocity-Verlet (NVE)
- `1`: Andersen
- `2`: Nose-Hoover chain
- `3`: Langevin
- `4`: Nose-Hoover chain with longer default chain length
- `5`: Bussi / CSVR

Additional behavior:

- `TEEND` enables linear temperature ramping from `TEBEG`
- `SMASS > 0` upgrades default `MDALGO=0` to Nose-Hoover (`2`)
- `SMASS < 0` upgrades default `MDALGO=0` to Langevin (`3`)
- `XDATCAR` is written for advanced MD steps
- `WRITE_LAMMPS_TRAJ=1` writes a LAMMPS text trajectory

## NEB-Like Directory Layouts

When `INCAR` suggests NEB and directories such as `00`, `01`, `02`, ... exist,
VPMDK reads those directories as a VTST-style NEB band. For normal NEB
optimization (`NSW > 0`, `IBRION > 0`, `ICHAIN=0` or unset), VPMDK constructs
an ASE `NEB` object, attaches one backend calculator per image, applies
spring-coupled band forces, and optimizes the moving images in one band-level
optimizer.

Supported NEB controls:

- `SPRING` sets the NEB spring magnitude; VASP/VTST negative values are accepted
  and converted to a positive ASE spring constant
- truthy `LCLIMB` enables climbing-image NEB
- `IOPT=1`, `3`, `5`, or `7` select ASE LBFGS, Quick-Min-like MDMin, BFGS, or
  FIRE respectively; other VTST optimizer values fall back to BFGS with a
  warning

Current limitations:

- only `ICHAIN=0` NEB is implemented; dimer/Lanczos TS modes are rejected
- `LNEBCELL` and NEB cell relaxation are not implemented; image cells stay fixed
- if `NSW <= 0` or `IBRION < 0`, VPMDK still runs independent image single
  points for compatibility; if `IBRION == 0`, it runs independent image MD
- ASE NEB optimization requires at least three numbered directories: initial,
  one moving image, and final

Additional NEB behavior:

- if `IMAGES` is present, VPMDK warns when the discovered image count does not
  match `IMAGES + 2`
- parent aggregate `OUTCAR`, `OSZICAR`, and `vasprun.xml` are synthesized from
  the final image results
- a top-level `POSCAR` is optional in this mode
- adjacent image geometries must differ; duplicate adjacent POSCAR/CONTCAR files
  cannot define a NEB tangent and produce an explicit error

## Output Files

Core outputs:

- `CONTCAR`: final structure
- `OUTCAR`: VASP-like step log
- `OSZICAR`: ionic/MD energy summary
- `vasprun.xml`: simplified VASP-like XML

Mode-specific outputs:

- `XDATCAR`: MD only
- `lammps.lammpstrj`: MD when requested
- `energy.csv`: relaxation when requested
- `CHGCAR`: final structure only, when requested

## Charge Density After the Run

If `WRITE_CHGCAR=1`, the CLI runs `predict_charge_density()` after the main
force-field workflow completes. The density is generated from the final atomic
structure, not from the initial `POSCAR`.

Charge-backend settings come from `BCAR`. Relative path handling depends on how
the value is provided:

- explicit `CHARGE_PYTHON`, `CHARGE_SOURCE_DIR`, and `CHARGE_MODEL` values from
  `BCAR` are used as written, so if you select another calculation directory
  with `--dir` they are interpreted relative to that run directory
- environment-variable fallbacks such as `VPMDK_CHARGE_PYTHON` are resolved
  relative to the caller's original working directory

This matters when you launch:

```bash
vpmdk --dir some/other/path
```

and expect `CHARGE_PYTHON=./env/bin/python` in `BCAR` to stay relative to the
shell you launched from. For that use case, prefer an absolute path or an
environment-variable fallback.

## MAGMOM and Species Handling

- `MAGMOM` is parsed in VASP style, including forms like `2*1.0 4*0.0`
- if the moments can be reconciled with atom count or species blocks, they are
  applied to the ASE atoms object before execution
- if `POTCAR` is present, species names from `POTCAR` can replace mismatched
  `POSCAR` labels after suffix normalization

## Warnings and Ignored Settings

You should expect warnings for:

- unsupported `INCAR` tags
- CHGCAR-grid-related `INCAR` tags when `WRITE_CHGCAR` is disabled
- pseudo-SCF-only tags (`NELM`, `NELMIN`, `NELMDL`, `EDIFF`) when pseudo-SCF
  output is disabled
- malformed optional numeric tags that are parsed through warning-tolerant
  helpers, such as `PSTRESS`, `TEBEG`, `TEEND`, `SMASS`, `ANDERSEN_PROB`,
  `LANGEVIN_GAMMA`, `CSVR_PERIOD`, `NHC_NCHAINS`, or `IMAGES`

Core execution-control tags such as `NSW`, `IBRION`, `EDIFFG`, `POTIM`,
`MDALGO`, and `ISIF` are not warning-tolerant in the same way; malformed values
there usually abort the run.

Unknown `BCAR` tags are not globally validated; they are simply ignored unless
some backend or helper explicitly consumes them.

## EquiformerV3 Backend

`MLP=EQUIFORMER_V3` is a dedicated entry point for EquiformerV3 checkpoints. It
uses the FAIRChem v1/OCP calculator path after importing the official
EquiformerV3 registration module.

Before launching `vpmdk`, make the official source tree importable:

```bash
export PYTHONPATH=/path/to/equiformer_v3/src:${PYTHONPATH}
```

Typical `BCAR`:

```text
MLP=EQUIFORMER_V3
MODEL=/path/to/equiformer_v3_checkpoint.pt
DEVICE=cuda
```

If your registration module is not
`fairchem.experimental.models.equiformer_v3.equiformer_v3`, set:

```text
EQUIFORMER_V3_MODULE=your.module.name
```
