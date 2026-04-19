# VPMDK API Guide

This document describes the stable Python API intended for ASE-centered usage.

## Design

VPMDK now has two layers:

- Library API: works directly with `ase.Atoms`, calculator settings, and Python objects.
- Compatibility CLI: reads `POSCAR`/`INCAR`/`BCAR` and writes VASP-like outputs.

The library API is the preferred entrypoint for new integrations.

## Stable Public Entry Points

The main public functions are:

- `vpmdk.get_calculator(...)`
- `vpmdk.single_point(...)`
- `vpmdk.relax(...)`
- `vpmdk.md(...)`
- `vpmdk.predict_charge_density(...)`
- `vpmdk.charge_density(...)`
- `vpmdk.determine_vasp_fft_grid(...)`
- `vpmdk.write_chgcar(...)`
- `vpmdk.list_backends()`
- `vpmdk.get_backend_capabilities(...)`

Important public models are:

- `vpmdk.BackendConfig`
- `vpmdk.SinglePointConfig`
- `vpmdk.RelaxConfig`
- `vpmdk.MDConfig`
- `vpmdk.BackendSpec`
- `vpmdk.BackendCapabilities`
- `vpmdk.SinglePointResult`
- `vpmdk.RelaxResult`
- `vpmdk.MDResult`
- `vpmdk.ChargeDensityResult`

## Core Behavior

All high-level API functions accept `ase.Atoms`.

```python
from ase.io import read
import vpmdk

atoms = read("POSCAR")
result = vpmdk.single_point(atoms, mlp="CHGNET", device="cpu")
print(result.potential_energy)
```

By default, the library API does not write:

- `OUTCAR`
- `OSZICAR`
- `vasprun.xml`
- `CONTCAR`
- `XDATCAR`

If you need those VASP-style side effects, use the CLI or explicitly attach the compatibility observer.

## Building Calculators

`vpmdk.get_calculator()` is the lowest-level stable helper.

```python
import vpmdk

calc = vpmdk.get_calculator(
    mlp="MACE",
    model="/path/to/model.pt",
    device="cuda:0",
)
```

Backend-specific options are passed as keyword arguments and normalized internally to BCAR-style tags.

```python
calc = vpmdk.get_calculator(
    mlp="MATRIS",
    model="matris_10m_oam",
    device="cpu",
    matris_task="efs",
)
```

You can also pass an explicit config object:

```python
config = vpmdk.BackendConfig(
    mlp="CHGNET",
    device="cpu",
    options={"CHGNET_GRAPH_CONVERTER_ALGORITHM": "fast"},
)
calc = vpmdk.get_calculator(config)
```

## Single-Point Calculations

Use `vpmdk.single_point()` for one evaluation of energy/forces/stress.

```python
result = vpmdk.single_point(
    atoms,
    mlp="CHGNET",
    device="cpu",
)

print(result.potential_energy)
print(result.forces)
print(result.stress)
```

Returned object:

- `result.atoms`: final `Atoms`
- `result.calculator`: calculator used
- `result.potential_energy`: final potential energy
- `result.forces`: final forces when available
- `result.stress`: final stress when available

## Relaxations

Use `vpmdk.relax()` for geometry optimization.

```python
result = vpmdk.relax(
    atoms,
    mlp="CHGNET",
    device="cpu",
    fmax=0.02,
    relax_cell=True,
    steps=200,
)

print(result.potential_energy)
print(len(result.steps))
print(result.converged)
```

Key arguments:

- `steps`: maximum ionic steps. `0` is allowed and behaves like a single-point evaluation of the initial structure. Negative values are invalid.
- `fmax`: force convergence threshold in eV/Ang
- `relax_cell`: maps to fixed-cell vs cell relaxation behavior
- `pressure_kbar`: external pressure when supported by the selected mode
- `energy_tolerance`: optional ionic energy convergence threshold

`RelaxResult.steps` contains per-step energies in a stable Python form.

## Molecular Dynamics

Use `vpmdk.md()` for ASE-based MD on top of the selected backend.

```python
result = vpmdk.md(
    atoms,
    mlp="CHGNET",
    device="cpu",
    temperature=300,
    steps=100,
    timestep=1.0,
    thermostat="langevin",
    thermostat_kwargs={"LANGEVIN_GAMMA": 1.0},
)
```

Supported public thermostat names:

- `nve`
- `andersen`
- `nose_hoover`
- `langevin`
- `nose_hoover_chain`
- `bussi`

These are mapped internally onto the existing VASP-style `MDALGO` logic.

For public MD calls, `steps=0` is allowed and behaves like a single-point evaluation of the initial structure without advancing dynamics. Negative values are invalid.

## Charge-Density Prediction

Use `vpmdk.predict_charge_density()` when you want a volumetric density on a specific grid.

```python
from ase.io import read
import vpmdk

atoms = read("POSCAR")

result = vpmdk.predict_charge_density(
    atoms,
    grid_shape=(64, 64, 64),
    backend="CHARGE3NET",
    source_dir="/path/to/charge3net",
    python_executable="/path/to/charge3net-env/bin/python",
    model_path="/path/to/charge3net/models/charge3net_mp.pt",
)

print(result.grid_shape)
print(result.density.shape)
print(result.spin_density is None)
```

If you want a VASP-like grid derived from `INCAR`, use `vpmdk.determine_vasp_fft_grid(...)` or pass `incar=` directly:

```python
from pymatgen.io.vasp import Incar

incar = Incar.from_file("INCAR")
result = vpmdk.predict_charge_density(
    atoms,
    incar=incar,
    reference=atoms,
    backend="CHARGE3NET",
    source_dir="/path/to/charge3net",
    python_executable="/path/to/charge3net-env/bin/python",
)
```

Returned object:

- `result.atoms`: `Atoms` used for prediction
- `result.density`: 3D NumPy array
- `result.spin_density`: optional 3D NumPy array for spin-enabled checkpoints
- `result.grid_shape`: `(NGXF, NGYF, NGZF)`
- `result.backend`: normalized backend name
- `result.metadata`: backend-specific execution details

To write a VASP-like `CHGCAR` after prediction:

```python
vpmdk.write_chgcar("CHGCAR", atoms, result.density, spin_density=result.spin_density)
```

Notes:

- `vpmdk.charge_density(...)` is an alias of `vpmdk.predict_charge_density(...)`
- the current ChargE3Net backend runs in a separate Python process, so `source_dir`, `python_executable`, and usually `model_path` must point to a working ChargE3Net environment
- the current writer produces the volumetric density block in `CHGCAR` format, but does not reconstruct PAW augmentation occupancies
- a runnable end-to-end example is available in [examples/chgcar_charge3net](/home/nei/temp/vpmdk_private/examples/chgcar_charge3net/README.md)

## Backend Discovery And Capabilities

Use `vpmdk.list_backends()` to inspect known backend entries.

```python
for spec in vpmdk.list_backends():
    print(spec.name, spec.available, spec.default_model)
```

Use `vpmdk.get_backend_capabilities()` to inspect feature metadata.

```python
caps = vpmdk.get_backend_capabilities("MATRIS", matris_task="e")
print(caps.energy, caps.forces, caps.stress)
```

The current capability model exposes:

- `energy`
- `forces`
- `stress`
- `spin`
- `fine_tune`
- `uncertainty`
- `metadata`

This metadata is intended for UI/service/integration layers that need to adapt behavior by backend.

## Config Objects

Config objects are the stable replacement for direct filesystem-driven control.

### `BackendConfig`

```python
backend = vpmdk.BackendConfig(
    mlp="CHGNET",
    model=None,
    device="cpu",
    options={},
)
```

### `SinglePointConfig`

```python
config = vpmdk.SinglePointConfig(isif=2)
```

### `RelaxConfig`

```python
config = vpmdk.RelaxConfig(
    steps=200,
    fmax=0.02,
    relax_cell=True,
    pressure_kbar=None,
)
```

`steps=0` is allowed and gives a single-point-style result for the initial structure. Negative values raise `ValueError`.

### `MDConfig`

```python
config = vpmdk.MDConfig(
    steps=1000,
    temperature=300,
    timestep_fs=1.0,
    thermostat="langevin",
    thermostat_kwargs={"LANGEVIN_GAMMA": 1.0},
)
```

`steps=0` is allowed and returns a single-point-style result without advancing MD. Negative values raise `ValueError`.

## Compatibility Observer

The CLI uses an observer layer to emit VASP-like files. This is available in Python as an advanced compatibility hook.

```python
observer = vpmdk.VaspCompatObserver()
compat = vpmdk.VaspCompatConfig(
    enabled=True,
    write_pseudo_scf=False,
    write_contcar=True,
)

result = vpmdk.single_point(
    atoms,
    mlp="CHGNET",
    observer=[observer, vpmdk.PrintProgressObserver()],
    vasp_compat=compat,
)
```

This is useful if you are migrating gradually from CLI workflows to library workflows but still need VASP-style artifacts.

## Relationship To CLI

The CLI remains supported:

```bash
vpmdk --dir calc_dir
```

Internally, the CLI now acts as a thin adapter:

1. Read VASP-style inputs.
2. Convert them into Python config objects.
3. Call the same execution layer used by the public API.
4. Attach compatibility observers to reproduce legacy outputs.

## Examples

See:

- [examples/api_chgnet/README.md](/home/nei/temp/vpmdk_private/examples/api_chgnet/README.md)
- [examples/api_chgnet/single_point.py](/home/nei/temp/vpmdk_private/examples/api_chgnet/single_point.py)
- [examples/api_chgnet/relax.py](/home/nei/temp/vpmdk_private/examples/api_chgnet/relax.py)
- [examples/api_chgnet/list_backends.py](/home/nei/temp/vpmdk_private/examples/api_chgnet/list_backends.py)
