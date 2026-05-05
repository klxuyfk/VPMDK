# VASP Force-Constants Compatibility

This page documents the implementation behind VPMDK's VASP-compatible
force-constant output. It is intended for maintainers who need to verify or
modify the compatibility layer, not as a quick-start workflow.

## Scope

When the CLI sees `IBRION=5`, `IBRION=6`, `IBRION=7`, or `IBRION=8`, VPMDK
writes a VASP-like `dynmat` block into `vasprun.xml`. This enables phonopy's
VASP interface to run:

```bash
phonopy --fc vasprun.xml
```

and create `FORCE_CONSTANTS`.

This is a compatibility output path. VPMDK does not run electronic DFPT, does
not compute dielectric tensors, and does not compute Born effective charges.
For all four `IBRION` values, the values are obtained from finite differences
of the selected MLP backend's forces, then serialized in the mass-normalized
Hessian convention that phonopy uses when reading VASP `vasprun.xml` files.

`IBRION=5` and `IBRION=6` are the physically natural mapping for this
implementation because VASP uses those settings for finite-difference phonons.
`IBRION=7` and `IBRION=8` remain accepted only as a VASP-DFPT file-format
compatibility path for phonopy's `--fc` reader.

## CLI Routing

The mode is selected in `vpmdk_core/cli.py` before single-point, relaxation, or
MD dispatch:

- `IBRION = 5` or `IBRION = 6`: finite-difference force constants
- `IBRION = 7` or `IBRION = 8`: DFPT-style force-constants compatibility output
- otherwise, `IBRION < 0` or `NSW <= 0`: single point
- `IBRION = 0`: MD
- other positive `IBRION` values: relaxation

`IBRION=6` and `IBRION=8` enable symmetry-reduced displacement generation.
VPMDK uses ASE's spglib-backed symmetry operations to find symmetry-equivalent
atom orbits, displaces one representative atom per orbit in the three Cartesian
directions only when needed, and reconstructs the remaining force-constant
columns by applying the Cartesian rotation matrices. `SYMPREC` controls the
symmetry tolerance and defaults to `1e-5`.

The direction reduction uses the site stabilizer of each representative atom.
For each trial Cartesian axis, VPMDK applies all symmetry operations that keep
the representative atom fixed. If the generated direction orbit increases the
span of already known displacement directions, that axis is explicitly
displaced. Otherwise its response can be reconstructed from already computed
directions. For high-symmetry primitive Si this reduces the VASP
finite-difference count to one degree of freedom, matching a local VASP 5.4.4
`IBRION=6`, `NFREE=2`, `POTIM=0.015` check.

`IBRION=7` and `IBRION=8` still use finite differences in VPMDK. VPMDK emits a
warning for these modes because it does not model VASP's electronic DFPT
algorithm; the output is a phonopy-compatible `dynmat/hessian` generated from
MLP forces.

The force-constants path calls `run_force_constants(...)` in
`vpmdk_core/runtime/single.py`. It initializes the normal VASP compatibility
recorder, records one static energy/force step, computes force constants from
finite differences, stores them on the recorder, and writes the usual
compatibility files.

## Symmetry Reconstruction

For `IBRION=6` and `IBRION=8`, VPMDK obtains symmetry operations from ASE's
`ase.spacegroup.symmetrize.prep_symmetry`, which is backed by spglib. Each
operation provides:

- a fractional-coordinate rotation
- an atom-index mapping

The fractional rotation is converted to a Cartesian rotation matrix `R`.
If an operation maps atom `i` to `i'` and representative displaced atom `j` to
`j'`, the force-constant block is reconstructed as:

```text
Phi[i', j'] = R * Phi[i, j] * R^T
```

When multiple operations generate the same block, VPMDK averages the
transformed blocks. This enforces the detected symmetry on the generated force
constants, matching the purpose of VASP's symmetry-enabled phonon modes while
still using MLP force finite differences as the source data.

Site-symmetry direction reduction is applied before atom-orbit reconstruction.
For one representative atom `j`, VPMDK collects pairs of displacement
directions and force responses:

```text
u_k = R_k * e_axis
C_k[i'] = R_k * Phi[i, j] * e_axis
```

where the operation keeps `j` fixed and maps atom `i` to `i'`. For each target
atom, the Cartesian block `Phi[:, j]` is reconstructed by solving the linear
least-squares problem:

```text
C = U * X
Phi = X^T
```

where rows of `U` are the generated displacement directions and rows of `C` are
the corresponding force responses. This lets one explicit Cartesian
displacement recover all three columns when site symmetry spans three
independent directions.

## Displacement Control

For `IBRION=5` and `IBRION=6`, the finite-difference displacement is read from
`POTIM` in `INCAR`, matching VASP's finite-difference phonon convention. If
`POTIM` is absent, VPMDK uses `0.015 Angstrom`, matching the VASP 5.1+ finite
phonon default. The value must be positive.

For `IBRION=7` and `IBRION=8`, VASP DFPT does not use a user displacement
width. Since VPMDK still has to approximate the Hessian from MLP forces, the
finite-difference displacement is read from `BCAR`:

1. `FORCE_CONSTANTS_DISPLACEMENT`
2. `PHONON_DISPLACEMENT`, as a compatibility fallback
3. default `0.01`

The value is in Angstrom and must be positive. Invalid or non-positive
displacements raise `ValueError`.

The `IBRION=7`/`8` displacement is deliberately a VPMDK/BCAR control rather
than an `INCAR` tag. VASP's electronic DFPT does not use this displacement in
the same way, so placing it in `INCAR` would imply a stronger VASP semantic
match than VPMDK can provide.

## NFREE

VPMDK implements the VASP finite-difference phonon stencils `NFREE=1`,
`NFREE=2`, and `NFREE=4`.

If `NFREE` is omitted with `IBRION=5` or `IBRION=6`, VPMDK uses `NFREE=2`.
This is the recommended central-difference form where each atom-direction is
displaced by `+POTIM` and `-POTIM`.

The implemented stencils for force derivatives are:

```text
NFREE=1:
  dF/dx ~= (F(+h) - F(0)) / h

NFREE=2:
  dF/dx ~= (F(+h) - F(-h)) / (2h)

NFREE=4:
  dF/dx ~= (-F(+2h) + 8F(+h) - 8F(-h) + F(-2h)) / (12h)
```

VPMDK then stores `Phi = -dF/dx` as force constants. `NFREE=1` therefore uses
the reference forces from the undisplaced structure and is sensitive to residual
forces, matching the reason VASP documentation discourages this mode.

Other `NFREE` values are rejected instead of being silently mapped to a
different stencil.

VASP 5.1+ may reset an unreasonably large finite-difference `POTIM` to
`0.015 Angstrom`. VPMDK currently does not try to reproduce that undocumented
threshold; it uses the explicit `POTIM` value after validating that it is
positive.

## Force-Constant Calculation

For each atom `j` and Cartesian direction `beta`, VPMDK evaluates forces at
two displaced geometries:

```text
R_plus  = R0 with R[j, beta] += delta
R_minus = R0 with R[j, beta] -= delta
```

For `IBRION=5` and `IBRION=6`, `delta` is `POTIM`. For `IBRION=7` and
`IBRION=8`, `delta` is the VPMDK BCAR displacement described above.

The backend calculator is asked for forces at both geometries. Force retrieval
is strict in this path: missing forces or an unexpected force-array shape is an
error, not silently replaced by zeros.

The force constant is the negative derivative of force:

```text
Phi[i, alpha, j, beta]
    = - d F[i, alpha] / d R[j, beta]
    ~= - (F_plus[i, alpha] - F_minus[i, alpha]) / (2 * delta)
```

In code the array is stored with shape `(num_atoms, num_atoms, 3, 3)` and
indices:

```text
force_constants[i, j, alpha, beta] = Phi[i, alpha, j, beta]
```

With ASE-compatible backends this has the usual unit implied by forces in
`eV/Angstrom`: `eV/Angstrom^2`.

The implementation currently does not symmetrize the matrix and does not impose
the acoustic sum rule. Those operations are left to downstream tools such as
phonopy so VPMDK does not hide backend force inconsistencies.

The implementation also does not apply selective-dynamics filtering to the
finite-difference Hessian. VASP supports selective dynamics only for
`IBRION=5`, with Cartesian semantics for the Hessian components. That remains a
future compatibility item.

## VASP XML Convention

VPMDK writes the finite-difference result into `vasprun.xml` as a VASP-like
mass-normalized Hessian:

```xml
<dynmat>
  <varray name="hessian">
    <v> ... </v>
  </varray>
</dynmat>
```

For atom masses `m_i` and `m_j`, the serialized Hessian block is:

```text
H[i, alpha, j, beta] = - Phi[i, alpha, j, beta] / sqrt(m_i * m_j)
```

The negative sign is intentional. phonopy's VASP reader reconstructs force
constants from this Hessian convention as:

```text
Phi[i, alpha, j, beta] = - H[i, alpha, j, beta] * sqrt(m_i * m_j)
```

The `dynmat` block does not write a `unit="THz^2"` marker. This matches the
VASP-style path where phonopy reads the Hessian values directly and applies the
mass factor shown above.

The masses are written in the `atominfo/array[@name="atomtypes"]` section of
`vasprun.xml` using ASE's element masses. This is required because phonopy reads
the masses from `atominfo` when converting the Hessian back into force
constants.

## Outputs

The force-constants mode writes the same top-level compatibility files as a
static VASP-like run:

- `OUTCAR`
- `OSZICAR`
- `CONTCAR`
- `vasprun.xml`

`FORCE_CONSTANTS` itself is not written by VPMDK. It is written by phonopy after
parsing `vasprun.xml`:

```bash
phonopy --fc vasprun.xml
```

## Compatibility Limits

This mode is compatible with phonopy's VASP force-constants parser, but it is
not equivalent to a VASP electronic DFPT calculation in physical content.

The following are intentionally outside this implementation:

- `NFREE` values other than `1`, `2`, and `4`
- automatic reset of overly large finite-difference `POTIM` values
- selective-dynamics filtering of Hessian components
- dielectric tensor output
- Born effective charge output
- `phonopy-vasp-born` support
- direct `FORCE_CONSTANTS` serialization by VPMDK
- q-dependent electronic-response quantities

The quality of the result is limited by the force accuracy and differentiability
of the selected MLP backend, the displacement size, and numerical noise in force
evaluation.

## Regression Contract

The primary regression test uses a harmonic calculator with known stiffness.
The test runs the CLI with `IBRION=7`, parses the generated `dynmat/hessian`
block, applies the same inverse mass normalization used by phonopy, and verifies
that the reconstructed force constants match the expected diagonal stiffness.

The `IBRION=5` regression tests use a component-wise anharmonic force field so
the finite-difference result depends on the displacement width and stencil.
They verify `NFREE=1`, `NFREE=2`, and `NFREE=4` against the formulas above and
check that the `IBRION=5` path uses `POTIM`, not the BCAR
`FORCE_CONSTANTS_DISPLACEMENT` value.

Maintainers changing this path should verify:

- strict force retrieval still rejects missing or malformed forces
- `atominfo` still contains real element masses
- `IBRION=5`/`6` use `POTIM` as the displacement width
- `IBRION=6`/`8` compute representative atom-orbit displacements and reconstruct
  force constants with symmetry rotations
- unsupported `NFREE` values do not silently run as a different stencil
- `IBRION=7`/`8` emit the DFPT compatibility warning
- the Hessian sign remains compatible with phonopy's VASP parser
- `phonopy --fc vasprun.xml` still creates `FORCE_CONSTANTS`
