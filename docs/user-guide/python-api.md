# Python API

## Design Goal

The Python API is the stable, filesystem-independent layer of VPMDK. It is
meant for programmatic use with `ase.Atoms`, explicit calculator construction,
and reproducible workflows that do not implicitly write VASP files.

That is the main difference from the CLI:

- CLI: parse files, run, and write `OUTCAR`/`OSZICAR`/`vasprun.xml`
- Python API: operate on objects and return results

## Public Entry Points

Common entry points:

- `vpmdk.get_calculator(...)`
- `vpmdk.build_calculator(...)`
- `vpmdk.single_point(...)`
- `vpmdk.relax(...)`
- `vpmdk.md(...)`
- `vpmdk.predict_charge_density(...)`
- `vpmdk.charge_density(...)`
- `vpmdk.determine_vasp_fft_grid(...)`
- `vpmdk.write_chgcar(...)`
- `vpmdk.list_backends()`
- `vpmdk.get_backend_capabilities(...)`

## Side-Effect Model

By default, the public execution functions do not create:

- `OUTCAR`
- `OSZICAR`
- `vasprun.xml`
- `CONTCAR`
- `XDATCAR`

This is enforced by the execution layer and is covered by regression tests.

If you want compatibility files from Python, attach `VaspCompatObserver()` and
pass an enabled `VaspCompatConfig`.

## Building Calculators

The most direct way to construct a backend is:

```python
import vpmdk

calc = vpmdk.get_calculator(
    mlp="MACE",
    model="/path/to/model.pt",
    device="cuda",
)
```

Equivalent object-oriented form:

```python
config = vpmdk.BackendConfig(
    mlp="MATRIS",
    model="matris_10m_oam",
    device="cpu",
    options={"MATRIS_TASK": "efs"},
)
calc = vpmdk.build_calculator(config)
```

Useful rules:

- backend option keys are normalized to uppercase BCAR-style names
- booleans in `options` are stringified as `1`/`0` before reaching the legacy
  builders
- `NNP` remains accepted when using a BCAR-like mapping

## Running Single-Point Calculations

```python
from ase.io import read
import vpmdk

atoms = read("POSCAR")
result = vpmdk.single_point(atoms, mlp="CHGNET", device="cpu")

print(result.potential_energy)
print(result.forces)
print(result.stress)
```

Notes:

- the returned calculator is the resolved ASE calculator, not necessarily the
  wrapper object you passed in
- `SinglePointConfig(isif=...)` can request stress formatting semantics

## Running Relaxations

```python
result = vpmdk.relax(
    atoms,
    mlp="CHGNET",
    steps=200,
    fmax=0.02,
    relax_cell=True,
)

print(result.converged)
print(len(result.steps))
```

Important semantics:

- `steps=0` is allowed and behaves like a single evaluation of the initial structure
- negative or fractional `steps` are rejected
- `relax_cell=True` upgrades the default `isif`/`stress_isif` from `2` to `3`
- explicit `isif`/`stress_isif` values are preserved if you set them yourself

## Running Molecular Dynamics

```python
result = vpmdk.md(
    atoms,
    mlp="MACE",
    model="/path/to/model",
    temperature=300,
    steps=100,
    timestep=1.0,
    thermostat="langevin",
    thermostat_kwargs={"LANGEVIN_GAMMA": 1.0},
)
```

Public thermostat names are normalized through:

- `nve`
- `andersen`
- `nose_hoover`
- `langevin`
- `nose_hoover_chain`
- `bussi`

Aliases such as `velocity_verlet`, `nosehoover`, `nosehooverchain`, and `csvr`
are accepted.

MD-specific semantics:

- `steps=0` is allowed and returns one non-advanced fallback step
- zero-step MD does not sample velocities or create an MD driver
- if `temperature <= 0`, velocities are zeroed rather than resampled

## Backends and Structure-Derived Metadata

Most calculators can be built from backend tags alone. Some integrations also
benefit from an explicit pymatgen structure:

- `ALPHANET`
- `DEEPMD`

When you call public wrappers with `atoms` and omit `structure=...`, VPMDK tries
to derive a pymatgen structure through `AseAtomsAdaptor` when useful.

## Capability Discovery

You can inspect supported backends without constructing a calculator:

```python
for spec in vpmdk.list_backends():
    print(spec.name, spec.available, spec.default_model)
```

And inspect one backend's declared capability metadata:

```python
caps = vpmdk.get_backend_capabilities("MATRIS", matris_task="e")
print(caps.energy, caps.forces, caps.stress)
```

`MATRIS_TASK` is special:

- `e`: energy only
- `ef`: energy + forces
- other values such as `efs`: energy + forces + stress

## Compatibility Output from Python

If you want the public API to emit the same files as the CLI:

```python
observer = [vpmdk.VaspCompatObserver(), vpmdk.PrintProgressObserver()]
compat = vpmdk.VaspCompatConfig(
    enabled=True,
    write_pseudo_scf=True,
    write_contcar=True,
    write_xdatcar=True,
)

result = vpmdk.md(
    atoms,
    calculator=calc,
    steps=5,
    temperature=300,
    observer=observer,
    vasp_compat=compat,
)
```

This is opt-in. The pure execution layer itself stays free of implicit file I/O.

## Charge Density from Python

```python
charge = vpmdk.predict_charge_density(
    atoms,
    incar={"ENCUT": 520, "PREC": "Accurate"},
    backend="CHARGE3NET",
    source_dir="/path/to/charge3net",
    python_executable="/path/to/env/bin/python",
    model_path="/path/to/charge3net_mp.pt",
)

vpmdk.write_chgcar("CHGCAR", atoms, charge.density, spin_density=charge.spin_density)
```

See [Charge Density](charge-density.md) and
[API Reference](../reference/api-reference.md) for exhaustive parameter details.
